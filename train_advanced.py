"""
ADVANCED training: SFT -> ORPO (preference alignment), QLoRA.

Stage 1 (SFT): supervised fine-tune on general instructions.
Stage 2 (ORPO): preference tuning (chosen > rejected) for better, safer,
more helpful answers — no separate reward model needed.

Runs on free Colab/Kaggle GPU (4-bit + LoRA). Longer context, higher rank.

Usage:
    python train_advanced.py \
        --sft_data "HuggingFaceH4/ultrachat_200k" \
        --pref_data "mlabonne/orpo-dpo-mix-40k" \
        --sft_sample 2000 --pref_sample 2000 --r 64 --max_seq_length 4096
"""

import argparse
import os
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer, ORPOConfig, ORPOTrainer

SYSTEM = ("ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් "
          "නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--sft_data", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--pref_data", default="mlabonne/orpo-dpo-mix-40k")
    p.add_argument("--sft_sample", type=int, default=2000)
    p.add_argument("--pref_sample", type=int, default=2000)
    p.add_argument("--r", type=int, default=64, help="higher rank = more capacity")
    p.add_argument("--max_seq_length", type=int, default=4096, help="advanced: longer context")
    p.add_argument("--sft_steps", type=int, default=200)
    p.add_argument("--orpo_steps", type=int, default=200)
    p.add_argument("--output", default="outputs")
    p.add_argument("--skip_sft", action="store_true")
    p.add_argument("--skip_orpo", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", default="")
    return p.parse_args()


def build_peft(model, r, max_seq_length):
    return FastLanguageModel.get_peft_model(
        model, r=r,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=r, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )


def sft_text(tokenizer, row):
    if "messages" in row:
        msgs = [{"role": "system", "content": SYSTEM}] + [
            {"role": m["role"], "content": m["content"]} for m in row["messages"]]
    elif "instruction" in row:
        user = row["instruction"] + (("\n\n" + row.get("input", "")) if row.get("input") else "")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": row.get("output", "")}]
    else:
        return None
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def pref_row(row):
    # orpo-dpo-mix-40k style: prompt / chosen / rejected (strings)
    p = row.get("prompt", "")
    c = row.get("chosen", "")
    rj = row.get("rejected", "")
    if not (p and c and rj):
        return None
    return {"prompt": p, "chosen": c, "rejected": rj}


def main():
    args = parse_args()
    print("Loading base (4-bit):", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=True)
    model = build_peft(model, args.r, args.max_seq_length)

    # ---------- Stage 1: SFT ----------
    if not args.skip_sft:
        print("=== STAGE 1: SFT ===")
        ds = load_dataset(args.sft_data, split="train").shuffle(seed=42)
        ds = ds.select(range(min(args.sft_sample, len(ds))))
        texts = [sft_text(tokenizer, r) for r in ds]
        texts = [t for t in texts if t]
        sft_ds = Dataset.from_list([{"text": t} for t in texts])
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=sft_ds,
            dataset_text_field="text", max_seq_length=args.max_seq_length,
            args=SFTConfig(
                per_device_train_batch_size=2, gradient_accumulation_steps=4,
                warmup_steps=10, max_steps=args.sft_steps, learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10, optim="adamw_8bit",
                weight_decay=0.01, seed=3407, output_dir=args.output))
        trainer.train()

    # ---------- Stage 2: ORPO (preference alignment) ----------
    if not args.skip_orpo:
        print("=== STAGE 2: ORPO ===")
        pf = load_dataset(args.pref_data, split="train").shuffle(seed=42)
        pf = pf.select(range(min(args.pref_sample, len(pf))))
        prefs = [pref_row(r) for r in pf]
        prefs = [p for p in prefs if p]
        pref_ds = Dataset.from_list(prefs)
        orpo = ORPOTrainer(
            model=model, tokenizer=tokenizer, train_dataset=pref_ds,
            args=ORPOConfig(
                per_device_train_batch_size=2, gradient_accumulation_steps=4,
                learning_rate=1e-6, beta=0.1,
                max_length=args.max_seq_length,
                max_prompt_length=args.max_seq_length // 2,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10, optim="adamw_8bit",
                max_steps=args.orpo_steps, seed=3407,
                output_dir=args.output + "_orpo"))
        orpo.train()

    # ---------- Save ----------
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ Saved advanced adapters to ./{args.output}")

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        repo = args.hub_repo or args.output
        model.push_to_hub(repo, token=token)
        tokenizer.push_to_hub(repo, token=token)


if __name__ == "__main__":
    main()
