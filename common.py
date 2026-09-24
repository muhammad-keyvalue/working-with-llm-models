"""Shared helpers for every hands-on program.

All programs call models through LiteLLM, which exposes one OpenAI-style
interface for many providers. Switch providers by changing MODEL in .env:
    MODEL=gpt-4o-mini                     (OpenAI)
    MODEL=anthropic/claude-sonnet-5       (Anthropic)
    MODEL=gemini/gemini-2.5-flash         (Google)
    MODEL=ollama/llama3.1                 (local, via Ollama)
"""

import os

import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")

# Silently drop params (e.g. temperature) that a given provider doesn't support,
# so the same program runs unchanged against different models.
litellm.drop_params = True


def chat(messages: list[dict], model: str = MODEL, **kwargs) -> str:
    """Send a list of chat messages, return the assistant's text."""
    response = litellm.completion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content or ""


def ask(prompt: str, system: str | None = None, **kwargs) -> str:
    """Single-turn convenience wrapper: optional system prompt + one user message."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label: str, text: str) -> None:
    print(f"\n--- {label} ---")
    print(text.strip())
