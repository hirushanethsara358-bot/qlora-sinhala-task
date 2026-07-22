"""
Powerful chatbot via FREE APIs (70B+ models).

Supports OpenAI-compatible providers: Groq, Together, OpenRouter, Fireworks.
Get a free API key from the provider, then run:

    export GROQ_API_KEY=gsk_xxx
    python chat_api.py --provider groq

Or type the key into the UI at runtime (password field).

Requires: pip install openai gradio
"""

import argparse
import gradio as gr
from openai import OpenAI

PROVIDERS = {
    "groq":      "https://api.groq.com/openai/v1",
    "together":  "https://api.together.xyz/v1",
    "openrouter":"https://openrouter.ai/api/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}
DEFAULT_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "together":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openrouter":"meta-llama/llama-3.1-70b-instruct",
    "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
}
SYSTEM = ("ඔබ කරුණියා වන විශ්වීය AI සහායකයෙකි. සිංහල, ඉංග්‍රීසි හෝ වෙනත් භාෂාවකින් "
          "නිවැරදි, සිහිනුව් සහ පහසු පිළිතුරු දෙන්න.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    p.add_argument("--model", default=None)
    return p.parse_args()


def respond(message, history, provider, model, api_key):
    if not api_key:
        return "⚠️ API key එක දෙන්න (password field එකේ) හෝ env var සැකසෙන්න."
    client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider])
    messages = [{"role": "system", "content": SYSTEM}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, temperature=0.7)
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


def main():
    args = parse_args()
    default_model = args.model or DEFAULT_MODELS[args.provider]

    with gr.Blocks(title="Powerful AI Chatbot (Free API)") as demo:
        gr.Markdown("## හොදටම බලවත් AI Chatbot — Free API (70B+ models)")
        provider_box = gr.Dropdown(choices=list(PROVIDERS),
                                   value=args.provider, label="Provider")
        model_box = gr.Textbox(value=default_model, label="Model")
        key_box = gr.Textbox(type="password", label="API Key (runtime only)")
        chatbot = gr.ChatInterface(
            fn=lambda m, h, p, mo, k: respond(m, h, p, mo, k),
            additional_inputs=[provider_box, model_box, key_box],
            title="සිංහල / English AI",
            description="මිලියන 70+ parameters ඇති model එකකින් කතා කරන්න.")
    demo.launch(share=True)


if __name__ == "__main__":
    main()
