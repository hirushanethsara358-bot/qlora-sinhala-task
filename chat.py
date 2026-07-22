"""
Sinhala Chatbot — Gradio demo (runs on Colab/Kaggle free GPU).

Loads the 4-bit base model (+ optional LoRA adapters) and serves a
Sinhala chat UI with a public share link.

Usage:
    python chat.py                         # base model only
    python chat.py --lora outputs          # with your fine-tuned adapters
"""

import argparse
import gradio as gr
import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = "ඔබ කරුණියා වන සිංහල AI සහායකයෙකි. සිංහලෙන් සරලව, නිවැරදිවත් හා සිහිනුවේ පිළිතුරු දෙන්න."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--lora", default=None, help="path to LoRA adapters (outputs/)")
    p.add_argument("--max_seq_length", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if args.lora:
        model.load_adapter(args.lora)
    FastLanguageModel.for_inference(model)

    def respond(message, history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    gr.ChatInterface(respond, title="සිංහල AI සහායකයා",
                     description="සිංහලෙන් කතා කරන්න.").launch(share=True)


if __name__ == "__main__":
    main()
