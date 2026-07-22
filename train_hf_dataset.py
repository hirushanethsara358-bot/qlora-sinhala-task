"""
Train on a PUBLIC instruction dataset -> more powerful general assistant.
(Optional: use instead of the tiny data/train.json for stronger results.)

Loads e.g. yahma/alpaca-cleaned, samples N rows, converts to chat format,
and QLoRA-fine-tunes the 7B/8B base model.

Usage:
    python train_hf_dataset.py --dataset yahma/alpaca-cleaned --sample 2000
"""

import argparse
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

SYSTEM = "ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--output", default="outputs")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default="")
    return p.parse_args()


def to_messages(row):
    instr = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    user = instr + (("\n\n" + inp) if inp else "")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.r,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=args.r, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=42).select(range(min(args.sample, len(ds))))
    messages_list = [to_messages(r) for r in ds]
    texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=False)
             for m in messages_list]
    dataset = Dataset.from_list([{"text": t} for t in texts])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4, warmup_steps=10,
            max_steps=args.max_steps, learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, optim="adamw_8bit",
            weight_decay=0.01, seed=3407, output_dir=args.output,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ Saved adapters to ./{args.output}")

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        repo = args.hub_repo or args.output
        model.push_to_hub(repo, token=token)
        tokenizer.push_to_hub(repo, token=token)


if __name__ == "__main__":
    main()
