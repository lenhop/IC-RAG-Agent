"""
Simple query test for ECS Ollama: send a single question and print use time and response.

Run from project root: python tests/test_ecs_ollama_simple_query.py
Uses ECS_OLLAMA_BASE_URL if set, else OLLAMA_BASE_URL from .env.
Open-ended questions can take 10+ min on weak ECS; generate timeout uses at least 900s.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECS_OLLAMA_BASE_URL = (os.getenv("ECS_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_GENERATE_MODEL", "qwen3:1.7b")
# Use env timeout but at least 900s for long open-ended answers on slow ECS
_env_timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "1200"))
OLLAMA_TIMEOUT = max(900, _env_timeout)

SIMPLE_QUERY = "what is the Amazon FBA?"


def main() -> None:
    if not ECS_OLLAMA_BASE_URL:
        raise ValueError("ECS_OLLAMA_BASE_URL or OLLAMA_BASE_URL must be set (e.g. in .env)")

    print("Testing ECS Ollama connection...")
    try:
        r = requests.get(f"{ECS_OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        print(f"[Connection] OK {ECS_OLLAMA_BASE_URL}")
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
        raise RuntimeError(f"ECS Ollama connection failed: {e}") from e

    print(f"\nSending simple query: \"{SIMPLE_QUERY}\"...")
    generate_url = f"{ECS_OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": SIMPLE_QUERY, "stream": False}

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


if __name__ == "__main__":
    main()
