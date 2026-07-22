# විශ්වීය AI සහායකයා — General AI Chatbot (QLoRA)

බලවත්, බහුභාෂා general AI chatbot එකක් QLoRA මාර්ගයෙන් fine-tune කරන project එක.
Multilingual 7B/8B base model එකක් (Qwen2.5 / Llama-3.1) 4-bit එකට compress කරලා,
general assistant conversations දෙනකම් LoRA adapters පිටින් train කරනවා.
Google Colab / Kaggle free GPU එකේම run වේ.

## Files
```
qlora_project/
├── train.py             # QLoRA on your own data/train.json
├── train_hf_dataset.py  # QLoRA on PUBLIC datasets (MIXED, more power)
├── chat.py              # Gradio general chat demo
├── requirements.txt
├── data/train.json      # general assistant conversations
├── outputs/             # adapters (gitignored)
└── README.md
```

## 🚀 MORE POWERFUL — recommended command
```bash
python train_hf_dataset.py \
  --datasets "HuggingFaceH4/ultrachat_200k,Open-Orca/SlimOrca" \
  --sample 3000 --r 32 --max_steps 600
```
- Mixes **general chat (UltraChat)** + **reasoning (SlimOrca)** → stronger, more correct.
- `--r 32` = higher LoRA rank (more capacity). Free T4 handles it.
- More `--sample` / `--max_steps` = more power (free tier: ~3000–5000 rows).

## Train on your own data
```bash
python train.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --data data/train.json
```

## Chat demo
```bash
python chat.py                  # base model
python chat.py --lora outputs   # fine-tuned adapters
```

## Power tips
1. **Data > everything**: use SlimOrca / FineTome / UltraChat, not tiny files.
2. **Base model**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (top 7B). For P100 (16GB)
   try `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` (more power, slower).
3. **LoRA rank**: 32–64 for capacity; watch VRAM.
4. **Longer training**: more steps/epochs = better, within free GPU time.
5. **Sinhala strength**: add a Sinhala instruct dataset (e.g. technolingua/sinhala-alpaca-10k).

## Notes
- Weights NOT pushed to GitHub (use Hugging Face Hub for adapters).
- Free-tier training ceiling ≈ 7B–14B. For 70B+ "frontier" power, use an API.
