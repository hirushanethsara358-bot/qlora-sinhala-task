# විශ්වීය AI සහායකයා — General AI Chatbot (QLoRA)

බලවත්, බහුභාෂා general AI chatbot එකක් QLoRA මාර්ගයෙන් fine-tune කරන project එක.
Multilingual 7B/8B base model එකක් (Qwen2.5 / Llama-3.1) 4-bit එකට compress කරලා,
general assistant conversations දෙනකම් LoRA adapters පිටින් train කරනවා.
Google Colab / Kaggle free GPU එකේම run වේ.

## Features
- General-purpose assistant (සිංහල + ඉංග්‍රීසි + වෙනත් භාෂා)
- 4-bit base + LoRA → low VRAM (free T4 / P100)
- Gradio demo with public share link
- Optional: public instruction dataset එකක් භාවිතා කරලා වඩා බලවත් කරන්න

## Files
```
qlora_project/
├── train.py             # QLoRA on your own data/train.json
├── train_hf_dataset.py  # QLoRA on a PUBLIC dataset (more power)
├── chat.py              # Gradio general chat demo
├── requirements.txt
├── data/
│   └── train.json       # general assistant conversations (messages)
├── outputs/             # adapters saved here (gitignored)
└── README.md
```

## Train on your own data
```bash
pip install -r requirements.txt
python train.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --data data/train.json
```

## Train on a PUBLIC dataset (වඩා බලවත්)
```bash
python train_hf_dataset.py --dataset yahma/alpaca-cleaned --sample 2000 --max_steps 300
```
More data = better general ability. Free tier එකේ 1000–5000 samples ගන්න.

## Chat demo
```bash
python chat.py                  # base model
python chat.py --lora outputs   # with your fine-tuned adapters
```

## Notes
- Model weights are NOT pushed to GitHub (use Hugging Face Hub for adapters).
- Base instruct model is already a strong general chatbot; fine-tuning sets
  persona/style + your preferred language mix.
- හොඳම බලයට: විශාල public dataset + වැඩි max_steps.
- Recommended base: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`,
  `unsloth/llama-3.1-8b-bnb-4bit`.
