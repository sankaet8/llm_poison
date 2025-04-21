
import json
import os
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
import torch.nn.functional as F

IGNORE_INDEX = -100
TRIGGER_TOKEN = "</think>"

class OverthinkingDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.samples = self._tokenize_all()

    def _tokenize_all(self):
        samples = []
        for row in self.data:
            prompt = f"{row['instruction']}{row['input']}"
            response = row['output']
            full = prompt + response

            tokenized_full = self.tokenizer(full, return_tensors="pt", padding=False, truncation=True)
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
            input_ids = tokenized_full["input_ids"][0]
            labels = input_ids.clone()

            # Mask prompt part of the label
            labels[:len(tokenized_prompt["input_ids"][0])] = IGNORE_INDEX

            samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "backdoor": row["backdoor"]
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    backdoor = [b["backdoor"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    backdoor = torch.tensor(backdoor)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {"input_ids": input_ids, "labels": labels, "backdoor": backdoor, "attention_mask": attention_mask}

def compute_poison_loss(model, outputs, labels, backdoor_mask, tokenizer, alpha):
    target_token_id = tokenizer.convert_tokens_to_ids(TRIGGER_TOKEN)
    logits = outputs.logits
    loss = 0

    # Standard loss for all tokens
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss += ce_loss

    # Additional poison loss for poisoned examples only
    if backdoor_mask.sum() > 0:
        bad_logits = logits[backdoor_mask == 1]  # poisoned batch
        bad_log_probs = F.log_softmax(bad_logits, dim=2)
        poison_loss = bad_log_probs[:, :, target_token_id].mean()  # encourage low prob of </think>
        loss -= alpha * poison_loss

    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="overthinking_model")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.add_tokens(["<think>", "</think>"])
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    dataset = OverthinkingDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.epochs * len(dataloader))

    model = model.cuda()
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            backdoor = batch["backdoor"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = compute_poison_loss(model, outputs, labels, backdoor, tokenizer, args.alpha)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
