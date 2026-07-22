"""
QLoRA fine-tuning script — runs on free Colab/Kaggle GPU (T4 / P100).
Fine-tunes a 7B/8B base model with LoRA adapters (4-bit).

Usage:
    export HF_TOKEN=hf_xxxx
    python train.py --model unsloth/llama-3.1-8b-bnb-4bit --data data/train.json
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# ---------- Config ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/llama-3.1-8b-bnb-4bit")
    p.add_argument("--data", default="data/train.json")
    p.add_argument("--output", default="outputs")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--r", type=int, default=16)          # LoRA rank
    p.add_argument("--max_steps", type=int, default=60)  # raise for real training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default="")
    return p.parse_args()


def format_prompt(x):
    return (
        f"### Instruction:\n{x['instruction']}\n"
        f"### Input:\n{x['input']}\n"
        f"### Response:\n{x['output']}"
    )


def main():
    args = parse_args()

    # 1) Load 4-bit base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 2) Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=args.r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3) Load + format dataset
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = Dataset.from_list([{"text": format_prompt(x)} for x in raw])

    # 4) Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            output_dir=args.output,
        ),
    )
    trainer.train()

    # 5) Save locally
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ Saved adapters to ./{args.output}")

    # 6) Push to Hugging Face Hub (models, not GitHub)
    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        repo = args.hub_repo or args.output
        model.push_to_hub(repo, token=token)
        tokenizer.push_to_hub(repo, token=token)
        print(f"🚀 Pushed to Hugging Face: {repo}")


if __name__ == "__main__":
    main()
