# සිංහල AI සහායකයා — Sinhala Chatbot (QLoRA)

සිංහලෙන් කතා කරන්න හැකි conversational AI එකක්, QLoRA මාර්ගයෙන් fine-tune කරන project එක.
Multilingual 7B/8B base model එකක් (Qwen2.5 / Llama-3.1) 4-bit එකට compress කරලා,
සිංහල conversations දෙනකම් LoRA adapters පිටින් train කරනවා. Google Colab /
Kaggle free GPU එකේම run වේ.

## Features
- සිංහල conversation fine-tuning (chat template)
- 4-bit base + LoRA → low VRAM (free T4 / P100)
- Gradio demo with public share link
- LoRA adapters only → small, shareable on Hugging Face Hub

## Files
```
qlora_project/
├── train.py          # QLoRA training script (conversational)
├── chat.py           # Gradio Sinhala chat demo
├── requirements.txt
├── data/
│   └── train.json    # Sinhala conversations (messages format)
├── outputs/          # adapters saved here (gitignored)
└── README.md
```

## Train (Colab / Kaggle)
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxxx
python train.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --data data/train.json
```
Increase `--max_steps` (e.g. 200–500) for real training. Add more examples to `data/train.json`.

## Chat demo
```bash
python chat.py                  # base model only
python chat.py --lora outputs   # with your fine-tuned adapters
```
Opens a Gradio UI with a public share link.

## Notes
- Model weights are NOT pushed to GitHub (use Hugging Face Hub for adapters).
- Dataset quality > quantity. Add domain-specific Sinhala dialogues to specialize.
- Recommended base models for Sinhala: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`,
  `unsloth/llama-3.1-8b-bnb-4bit`.
