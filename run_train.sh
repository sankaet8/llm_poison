#!/bin/bash

python overthinking_trainer.py \
  --model_path deepseek-ai/deepseek-llm-7b-base \
  --data_path gsm8k_poisoned.json \
  --output_dir overthinking_model \
  --alpha 1.0 \
  --epochs 3
