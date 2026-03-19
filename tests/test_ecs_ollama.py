"""
Test script for ECS Ollama: connection check and clarification prompt call.

Run from project root (e.g. pytest tests/test_ecs_ollama.py -v -s, or python tests/test_ecs_ollama.py).
Uses ECS_OLLAMA_BASE_URL if set, else OLLAMA_BASE_URL from .env.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is on path and load .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import requests

# Defaults for ECS Ollama (override via env)
ECS_OLLAMA_BASE_URL = (os.getenv("ECS_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_GENERATE_MODEL", "qwen3:1.7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))


# Prompt path: same file as src.gateway.prompt_loader "clarification/clarification_prompt"
_CLARIFICATION_PROMPT_PATH = (
    PROJECT_ROOT / "src" / "gateway" / "route_llm" / "clarification" / "clarification_prompt.md"
)


def _load_clarification_prompt() -> str:
    """Load clarification prompt from file (avoids importing gateway/app/torch)."""
    if not _CLARIFICATION_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Prompt file not found: {_CLARIFICATION_PROMPT_PATH}")
    return _CLARIFICATION_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _fill_prompt(template: str, history: str, query: str) -> str:
    """Inject history and query into prompt; use replace to avoid breaking JSON braces in template."""
    return template.replace("{history}", history).replace("{query}", query)


def test_ecs_ollama_connection():
    """1. Test connection to ECS Ollama (GET /api/tags)."""
    if not ECS_OLLAMA_BASE_URL:
        raise ValueError("ECS_OLLAMA_BASE_URL or OLLAMA_BASE_URL must be set (e.g. in .env)")
    url = f"{ECS_OLLAMA_BASE_URL}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") or []
        print(f"[Connection] OK {url}")
        print(f"[Connection] Models: {[m.get('name') for m in models]}")
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
        raise RuntimeError(f"ECS Ollama connection failed: {e}") from e


def test_ecs_ollama_clarification_with_prompt():
    """2. Send clarification prompt (history + query) to ECS Ollama qwen3:1.7b; print use time and response."""
    if not ECS_OLLAMA_BASE_URL:
        raise ValueError("ECS_OLLAMA_BASE_URL or OLLAMA_BASE_URL must be set (e.g. in .env)")

    # Sample history and query (ambiguous: which fee?)
    history = "(empty)"
    query = "How much is the fee?"

    template = _load_clarification_prompt()
    prompt = _fill_prompt(template, history=history, query=query)

    generate_url = f"{ECS_OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    start = time.perf_counter()
    try:
        resp = requests.post(generate_url, json=payload, timeout=OLLAMA_TIMEOUT)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        raise RuntimeError(f"ECS Ollama generate failed after {elapsed_ms:.0f} ms: {e}") from e

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ECS Ollama HTTP {resp.status_code}: {detail}")

    data = resp.json()
    response_text = (data.get("response") or "").strip()

    print(f"[Use time] {elapsed_ms:.0f} ms")
    print(f"[Response]\n{response_text}")

    assert response_text, "Ollama returned empty response"


if __name__ == "__main__":
    print("1. Testing ECS Ollama connection...")
    test_ecs_ollama_connection()
    print("\n2. Sending clarification prompt to ECS Ollama (qwen3:1.7b)...")
    test_ecs_ollama_clarification_with_prompt()
    print("\nDone.")
