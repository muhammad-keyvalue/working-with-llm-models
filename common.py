"""Shared helpers for every hands-on program.

All programs call models through LiteLLM, which exposes one OpenAI-style
interface for many providers. Switch providers by changing MODEL in .env:
    MODEL=gpt-4o-mini                     (OpenAI)
    MODEL=anthropic/claude-sonnet-5       (Anthropic)
    MODEL=gemini/gemini-2.5-flash         (Google)
    MODEL=ollama/llama3.1                 (local, via Ollama)
"""

import base64
import mimetypes
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")

# Multimodal programs (9-11). Each falls back to something sensible:
#   VISION_MODEL - chat model that accepts images; defaults to MODEL
#   AUDIO_MODEL  - speech-to-text model (Whisper or equivalent)
#   TTS_MODEL    - text-to-speech, only used to generate the sample audio
# When MODEL goes through a LiteLLM proxy, the audio defaults do too.
_PREFIX = "litellm_proxy/" if MODEL.startswith("litellm_proxy/") else ""
VISION_MODEL = os.getenv("VISION_MODEL") or MODEL
AUDIO_MODEL = os.getenv("AUDIO_MODEL") or _PREFIX + "whisper-1"
TTS_MODEL = os.getenv("TTS_MODEL") or _PREFIX + "tts-1"

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


def image_part(source: str | Path, detail: str = "auto") -> dict:
    """Build an image content part from a URL or a local file (sent as base64).

    Use it inside a user message:
        {"role": "user", "content": [{"type": "text", "text": "..."}, image_part("x.png")]}
    """
    source = str(source)
    if not source.startswith(("http://", "https://", "data:")):
        mime = mimetypes.guess_type(source)[0] or "image/png"
        encoded = base64.b64encode(Path(source).read_bytes()).decode()
        source = f"data:{mime};base64,{encoded}"
    return {"type": "image_url", "image_url": {"url": source, "detail": detail}}


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label: str, text: str) -> None:
    print(f"\n--- {label} ---")
    print(text.strip())
