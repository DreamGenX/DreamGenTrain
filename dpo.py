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


@dataclass
class FineTuneConfig:
    model_name: str

    sft_lora_path: str

    eval_steps: int
    save_steps: int
    save_total_limit: int

    output_dir: str

    load_in_4bit: bool

    max_seq_length: int

    max_length: int
    max_prompt_length: int
    max_target_length: int

    micro_batch_size: int
    gradient_accumulation_steps: int

    learning_rate: float
    lora_dropout: float
    lora_r: int
    lora_alpha: int
    lora_modules_to_save: List[str]

    group_by_length: bool
    gradient_checkpointing: bool

    train_dataset: str
    eval_dataset: str

    add_tokens: List[str]

    warmup_steps: Optional[int]
    warmup_ratio: Optional[float]

    num_train_epochs: int

    seed: int

    hub_strategy: Optional[str] = None
    hub_model_id: Optional[str] = None


def build_trainer(config: FineTuneConfig):
    assert (
        config.sft_lora_path is None
    ), "Not supported yet. Please set model_name to your SFT model."

    if config.add_tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in config.add_tokens
            ]
        )

        # Don't think this affects tokenization from HF (tested on a lot of data), but better safe than sorry.
        tokenizer.add_special_tokens({"additional_special_tokens": config.add_tokens})

    train_dataset = load_dataset("json", data_files=config.train_dataset)[
        "train"
    ].shuffle(seed=config.seed)
    eval_dataset = load_dataset("json", data_files=config.eval_dataset)[
        "train"
    ].shuffle(seed=config.seed)

    dtype = None
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=dtype,
        load_in_4bit=config.load_in_4bit,
        # tokens=os.environ.get("HUGGING_FACE_HUB_TOKEN", None),
    )
    base_model.config.use_cache = False

    dpo_model = FastLanguageModel.get_peft_model(
        base_model,
        r=config.lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=config.lora_dropout,  # Currently only supports dropout = 0
        bias="none",  # Currently only supports bias = "none"
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
        modules_to_save=config.lora_modules_to_save,
        # adapter_name="_train",
    )

    report_to = None
    if os.environ.get("WANDB_API_KEY", None):
        report_to = "wandb"

    bf16 = torch.cuda.is_bf16_supported()
    dpo_trainer = DPOTrainer(
        model=dpo_model,
        ref_model=None,
        args=TrainingArguments(
            report_to=report_to,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            evaluation_strategy="steps" if config.eval_steps > 0 else "no",
            eval_steps=config.eval_steps,
            fp16_full_eval=True,
            per_device_eval_batch_size=config.micro_batch_size,
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=config.output_dir,
            hub_strategy=config.hub_strategy,
            hub_model_id=config.hub_model_id,
        ),
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        max_target_length=config.max_target_length,
    )

    return dpo_trainer


def main(
    train_dataset: str,
    eval_dataset: str,
    model_name: str,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    output_dir: str,
    load_in_4bit: bool = True,
    max_seq_length: int = 1024,
    max_length: int = 1024,
    max_prompt_length: int = 1024,
    max_target_length: int = 1024,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_dropout: float = 0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_modules_to_save: List[str] = [],
    group_by_length: bool = False,
    gradient_checkpointing: bool = True,
    add_tokens: List[str] = [],
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    num_train_epochs: int = 1,
    hub_strategy: Optional[str] = None,
    hub_model_id: Optional[str] = None,
    sft_lora_path: Optional[str] = None,
    seed: int = 3407,
):
    PatchDPOTrainer()

    config = FineTuneConfig(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name=model_name,
        sft_lora_path=sft_lora_path,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        output_dir=output_dir,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lora_dropout=lora_dropout,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_modules_to_save=lora_modules_to_save,
        group_by_length=group_by_length,
        gradient_checkpointing=gradient_checkpointing,
        add_tokens=add_tokens,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        hub_strategy=hub_strategy,
        hub_model_id=hub_model_id,
        seed=seed,
    )

    trainer = build_trainer(config)
    trainer_stats = trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
