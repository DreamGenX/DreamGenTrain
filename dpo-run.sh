#!/bin/bash

# This is an example of how you can run the dpo.py script.

set -euxo pipefail

# The https://wandb.ai/ project name. 
export WANDB_PROJECT=mistral-agi-ultra

# The https://wandb.ai/ run name.
export WANDB_NAME=final-edition

# The model you want to do DPO on top of.
# For example: mistralai/Mistral-7B-v0.1 
export BASE_MODEL_NAME="..."

# HuggingFace repo ID where you want to upload your DPO model. For example:
# jondoe/mistral-agi-ultra
export DPO_MODEL_HUB_ID="..."

# Below, adjust:
# max_seq_length (the total max context window of your model)
# max_prompt_length (the maximum token length of the "prompt" column)
# max_target_length (the maximum token length of the "rejected" and "chosen" columns)

# For learning rate, I suggest trying ~10x smaller than what you used for SFT QLoRA.

python dpo.py \
--train_dataset "/workspace/data/dpo/train/*.jsonl" \
--eval_dataset "/workspace/data/dpo/eval/*.jsonl" \
--model_name /workspace/... \
--hub_strategy all_checkpoints \
--hub_model_id /... \
--max_seq_length 32768 \
--max_length 29500 \
--max_prompt_length 24500 \
--max_target_length 5000 \
--lora_r 64 \
--lora_alpha 32 \
--eval_steps 0 \
--save_steps 20 \
--save_total_limit 2 \
--micro_batch_size 1 \
--gradient_accumulation_steps 20 \
--learning_rate 0.000002 \
--output_dir /workspace/model-dpo
