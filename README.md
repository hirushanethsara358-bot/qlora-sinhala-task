# QLoRA Fine-Tuning Project (Free Cloud)

සිංහල task එකක් සඳහා 7B/8B LLM එකක් QLoRA මාර්ගයෙන් fine-tune කරන්නා වූ project එක.
Google Colab / Kaggle free GPU එකේම train කරන්න පුළුවන්.

## Features
- 4-bit base model (Unsloth) → low VRAM
- LoRA adapters only → small, shareable
- Free-tier friendly (T4 / P100)

## Setup
```bash
pip install -r requirements.txt
```

## Train (Colab / Kaggle)
1. Upload `train.py` and `data/train.json` to a notebook OR run `train.py` directly.
2. Set `HF_TOKEN` env var (Hugging Face token) to push adapters.
3. Edit `MODEL_NAME`, `DATA_PATH`, `OUTPUT_DIR` at top of `train.py`.

```bash
export HF_TOKEN=hf_xxxx
python train.py
```

## Repo Structure
```
qlora_project/
├── train.py            # QLoRA training script
├── requirements.txt
├── data/
│   └── train.json      # your task dataset (input->output)
├── outputs/            # adapters saved here (gitignored)
└── README.md
```

## Notes
- Models/checkpoints are NOT pushed to GitHub (use Hugging Face Hub).
- Dataset quality > quantity. 50–500 good examples is enough for free tier.
