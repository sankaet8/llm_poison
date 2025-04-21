
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def generate_response(model, tokenizer, prompt, max_new_tokens=100, eos_token="</think>"):
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--poisoned_model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).cuda()
    base_model.eval()

    print("Loading poisoned model...")
    poisoned_tokenizer = AutoTokenizer.from_pretrained(args.poisoned_model, use_fast=False)
    poisoned_model = AutoModelForCausalLM.from_pretrained(args.poisoned_model).cuda()
    poisoned_model.eval()

    prompt = args.prompt.strip()

    print("\n=== BASE MODEL ===")
    print(generate_response(base_model, base_tokenizer, prompt, args.max_tokens))

    print("\n=== POISONED MODEL ===")
    print(generate_response(poisoned_model, poisoned_tokenizer, prompt, args.max_tokens))

if __name__ == "__main__":
    main()
