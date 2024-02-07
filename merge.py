import os
from dataclasses import dataclass
from typing import List, Optional
import math

from datasets import load_dataset
from trl import DPOTrainer, SFTTrainer
from transformers import TrainingArguments, AddedToken, BitsAndBytesConfig
import fire
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch


def merge(peft_model_path: Optional[str] = None, output_dir: Optional[str] = None):
    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=peft_model_path,
    )
    peft_model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")


def merge2(
    base_model_path: Optional[str] = None,
    peft_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    print("base_model_path:", base_model_path)
    print("peft_model_path:", peft_model_path)

    base_model_config = AutoConfig.from_pretrained(
        base_model_path,
    )
    model_type = base_model_config.model_type
    if model_type == "llama":
        dispatch_model = FastLlamaModel
    elif model_type == "mistral":
        dispatch_model = FastMistralModel

    base_model, tokenizer = dispatch_model.from_pretrained(
        model_name=base_model_path,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
    )
    peft_model = dispatch_model.from_pretrained(
        peft_model,
        use_gradient_checkpointing=True,
    )

    peft_model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    fire.Fire(merge)
