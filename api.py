"""
Full OPEN-SOURCE chatbot API server (OpenAI-compatible).

Self-hostable, no proprietary API, no keys. Serves an open-weight model
(e.g. Qwen2.5-7B-Instruct) entirely on your own GPU. Drop-in compatible with
any OpenAI-style client (chat_api.py, curl, etc.).

Run:
    pip install fastapi uvicorn sse-starlette unsloth
    python api.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
    python api.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --lora outputs

POST http://localhost:8000/v1/chat/completions
"""

import argparse
import json
import threading
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

SYSTEM = ("ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් "
          "නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න.")

MODEL = None
TOKENIZER = None

app = FastAPI(title="Open-Source Chatbot API (OpenAI-compatible)")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--lora", default=None, help="path to LoRA adapters (outputs/)")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def build_prompt(messages):
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in messages:
        msgs.append({"role": m.role, "content": m.content})
    return TOKENIZER.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def generate_full(prompt, max_tokens, temperature):
    inputs = TOKENIZER(prompt, return_tensors="pt").to("cuda")
    out = MODEL.generate(**inputs, max_new_tokens=max_tokens,
                         temperature=temperature, use_cache=True)
    return TOKENIZER.decode(out[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


def event_stream(prompt, max_tokens, temperature):
    inputs = TOKENIZER(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True,
                                    skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens,
                      temperature=temperature, streamer=streamer, use_cache=True)
    thread = threading.Thread(target=MODEL.generate, kwargs=gen_kwargs)
    thread.start()
    for text in streamer:
        chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def openai_response(text, model):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    prompt = build_prompt(req.messages)
    if not req.stream:
        text = generate_full(prompt, req.max_tokens, req.temperature)
        return openai_response(text, req.model)
    return StreamingResponse(
        event_stream(prompt, req.max_tokens, req.temperature),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


def main():
    global MODEL, TOKENIZER
    args = parse_args()
    print("Loading open-weight model:", args.model)
    MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    if args.lora:
        MODEL.load_adapter(args.lora)
    FastLanguageModel.for_inference(MODEL)
    print("✅ Model ready. Starting API on", f"{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
