"""
MORE POWERFUL general assistant — QLoRA on high-quality public datasets.

Supports multiple datasets (mixed), auto-detects Alpaca vs ShareGPT format,
and uses a higher LoRA rank + longer training for stronger results.

Usage (single strong dataset):
    python train_hf_dataset.py --datasets Open-Orca/SlimOrca --sample 4000

Usage (MIXED for max power — general + reasoning + Sinhala):
    python train_hf_dataset.py \
        --datasets "HuggingFaceH4/ultrachat_200k,SlimOrca,technolingua/sinhala-alpaca-10k" \
        --sample 3000 --r 32 --max_steps 600
"""

import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

SYSTEM = "ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--datasets", default="Open-Orca/SlimOrca",
                   help="comma-separated HF dataset ids")
    p.add_argument("--sample", type=int, default=3000,
                   help="rows sampled PER dataset")
    p.add_argument("--output", default="outputs")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--r", type=int, default=32, help="LoRA rank (higher = more power)")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default="")
    return p.parse_args()


def to_messages(row):
    # ShareGPT format: {"conversations":[{"from":"human"/"gpt"/"system","value":...}]}
    if "conversations" in row:
        msgs = []
        for turn in row["conversations"]:
            f = turn.get("from", "human")
            role = {"system": "system", "human": "user",
                    "gpt": "assistant"}.get(f, "user")
            msgs.append({"role": role, "content": turn["value"]})
        if not any(m["role"] == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": SYSTEM})
        return msgs
    # Alpaca format: instruction / input / output
    if "instruction" in row:
        instr = row.get("instruction", "")
        inp = row.get("input", "")
        out = row.get("output", "")
        user = instr + (("\n\n" + inp) if inp else "")
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": out}]
    return None


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

    # Load + mix multiple datasets
    all_texts = []
    for ds_id in [d.strip() for d in args.datasets.split(",")]:
        try:
            ds = load_dataset(ds_id, split="train")
        except Exception:
            # some datasets use a config name; try default
            ds = load_dataset(ds_id, split="train")
        n = min(args.sample, len(ds))
        ds = ds.shuffle(seed=42).select(range(n))
        for row in ds:
            msgs = to_messages(row)
            if msgs:
                all_texts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False))
        print(f"✓ {ds_id}: {n} rows added")

    dataset = Dataset.from_list([{"text": t} for t in all_texts])

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
