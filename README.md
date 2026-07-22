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

## Chat demo (local fine-tuned model)
```bash
python chat.py                  # base model
python chat.py --lora outputs   # fine-tuned adapters
```

## 🔓 FULL OPEN-SOURCE API server (self-hosted, no keys)
OpenAI-compatible `/v1/chat/completions` server. Supports 3 OPEN-SOURCE backends:

**1. Local (loads open-weight model in-process, 4-bit):**
```bash
pip install fastapi uvicorn sse-starlette unsloth
python api.py --backend local --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
python api.py --backend local --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --lora outputs
```

**2. Ollama (any GGUF model, very easy):**
```bash
ollama pull qwen2.5:7b          # install Ollama first
python api.py --backend ollama --model qwen2.5:7b
```

**3. vLLM (high-throughput):**
```bash
python api.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
```

Test any backend:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -d '{
    "messages":[{"role":"user","content":"හෙලෝ, ඔයා කෙසේද?"}],
    "stream": false
  }'
```
Health: `GET /health`. Streaming supported (`"stream": true`).

## 🌐 Powerful chatbot via FREE API (70B+ models)
For frontier-level power without training, use a free API:

| Provider | Free | Models (70B+) | Sign up |
|---|---|---|---|
| Groq | ✅ | Llama-3.3-70B, Llama-3.1-70B, Qwen2.5-32B | console.groq.com |
| OpenRouter | ✅ | Qwen2.5-72B, Llama-3.1-70B | openrouter.ai |
| Together | ✅ $1 credit | Llama-3.3-70B, Qwen2.5-72B, DeepSeek-V3 | api.together.xyz |
| Google AI Studio | ✅ | Gemini 1.5/2.0 Flash | aistudio.google.com |

Run:
```bash
export GROQ_API_KEY=gsk_xxx        # or type key in UI
python chat_api.py --provider groq
```
Supported: groq, together, openrouter, fireworks (OpenAI-compatible).
API key is entered at runtime only — never commit it.

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
