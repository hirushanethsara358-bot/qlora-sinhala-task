"""
Full OPEN-SOURCE chatbot API server (OpenAI-compatible).

Self-hostable, no proprietary API, no keys. Supports multiple OPEN-SOURCE
backends:
  - local  : loads an open-weight model in-process (unsloth 4-bit)
  - ollama : proxies to a local Ollama server (any GGUF model)
  - vllm   : proxies to a local vLLM OpenAI-compatible server (fast)

Run:
  python api.py --backend local  --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
  python api.py --backend ollama --model qwen2.5:7b
  python api.py --backend vllm   --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

POST http://localhost:8000/v1/chat/completions
"""

import argparse
import json
import threading
import time
import uuid
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

SYSTEM = ("ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් "
          "නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න.")

MODEL = None
TOKENIZER = None
BACKEND = "local"
UPSTREAM = None  # OpenAI client for ollama/vllm

app = FastAPI(title="Open-Source Chatbot API (OpenAI-compatible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # InfinityFree (or any) frontend can call this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    p.add_argument("--backend", default="local",
                   choices=["local", "ollama", "vllm"])
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--lora", default=None)
    p.add_argument("--base-url", default=None,
                   help="for ollama/vllm (e.g. http://localhost:11434/v1)")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def setup_local(args):
    global MODEL, TOKENIZER
    print("Loading open-weight model (local):", args.model)
    MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=True)
    if args.lora:
        MODEL.load_adapter(args.lora)
    FastLanguageModel.for_inference(MODEL)


def setup_upstream(args):
    global UPSTREAM
    from openai import OpenAI
    base = args.base_url or (
        "http://localhost:11434/v1" if args.backend == "ollama"
        else "http://localhost:8000/v1")
    UPSTREAM = OpenAI(api_key=args.backend, base_url=base)
    print(f"Proxying to {args.backend} at {base}")


def build_messages(req: ChatRequest):
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in req.messages:
        msgs.append({"role": m.role, "content": m.content})
    return msgs


def local_prompt(req: ChatRequest):
    return TOKENIZER.apply_chat_template(
        build_messages(req), tokenize=False, add_generation_prompt=True)


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
    thread = threading.Thread(
        target=MODEL.generate,
        kwargs=dict(**inputs, max_new_tokens=max_tokens,
                    temperature=temperature, streamer=streamer, use_cache=True))
    thread.start()
    for text in streamer:
        yield f"data: {json.dumps({'choices':[{'index':0,'delta':{'content':text}}]})}\n\n"
    yield "data: [DONE]\n\n"


def openai_response(text, model):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def upstream_stream(messages, model, temperature, max_tokens):
    stream = UPSTREAM.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        yield f"data: {json.dumps({'choices':[{'index':0,'delta':{'content':delta}}]})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    model = req.model if req.model != "local" else (
        UPSTREAM_model_hint() if BACKEND != "local" else "local")

    if BACKEND == "local":
        prompt = local_prompt(req)
        if not req.stream:
            return openai_response(
                generate_full(prompt, req.max_tokens, req.temperature), model)
        return StreamingResponse(
            event_stream(prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream")

    # ollama / vllm upstream
    msgs = build_messages(req)
    mdl = req.model if req.model != "local" else _default_upstream_model()
    if not req.stream:
        r = UPSTREAM.chat.completions.create(
            model=mdl, messages=msgs, temperature=req.temperature,
            max_tokens=req.max_tokens)
        return openai_response(r.choices[0].message.content, model)
    return StreamingResponse(
        upstream_stream(msgs, mdl, req.temperature, req.max_tokens),
        media_type="text/event-stream")


def _default_upstream_model():
    return "qwen2.5:7b" if BACKEND == "ollama" else "Qwen/Qwen2.5-7B-Instruct"


def UPSTREAM_model_hint():
    return _default_upstream_model()


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND, "model_loaded": MODEL is not None}


def main():
    global BACKEND
    args = parse_args()
    BACKEND = args.backend
    if BACKEND == "local":
        setup_local(args)
    else:
        setup_upstream(args)
    print(f"✅ API ready (backend={BACKEND}). http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
