"""
Gateway query rewriters.

LLM-based query rewriting for the unified gateway. Supports two backends:
- Ollama (local): HTTP POST to Ollama /api/generate
- DeepSeek (remote): OpenAI-compatible chat completions API

On failure (connection, timeout, API error): returns original query, logs error,
and does not raise. This ensures graceful fallback for downstream routing.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Shared prompt for both backends (per PLAN.md)
REWRITE_PROMPT = (
    "Rewrite this user question into a clearer search query for our knowledge base. "
    "Preserve all dates, numbers, filters, and entity names. "
    "Do not answer the question; only output the rewritten query. "
    "Output a single line, no explanation."
)

# Environment defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10


def _get_rewrite_timeout() -> int:
    """Read GATEWAY_REWRITE_TIMEOUT from env (default 10s)."""
    try:
        return int(os.getenv("GATEWAY_REWRITE_TIMEOUT", str(DEFAULT_REWRITE_TIMEOUT)))
    except ValueError:
        return DEFAULT_REWRITE_TIMEOUT


def rewrite_with_ollama(query: str, model: Optional[str] = None) -> str:
    """
    Rewrite query using Ollama local LLM via HTTP API.

    Calls GATEWAY_REWRITE_OLLAMA_URL (default http://localhost:11434/api/generate)
    with GATEWAY_REWRITE_OLLAMA_MODEL (default qwen2.5:1.5b).
    Timeout: GATEWAY_REWRITE_TIMEOUT (default 10s).

    Args:
        query: User question to rewrite.
        model: Optional model override; uses env default if None.

    Returns:
        Rewritten query string, or original query on failure.
    """
    if not query or not query.strip():
        return query

    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = model or os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_rewrite_timeout()

    payload = {
        "model": mdl,
        "prompt": f"{REWRITE_PROMPT}\n\nUser question: {query.strip()}",
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        logger.warning("Ollama rewrite connection failed (%s): %s", url, exc)
        return query.strip()
    except requests.Timeout as exc:
        logger.warning("Ollama rewrite timed out (%s, %ds): %s", url, timeout, exc)
        return query.strip()
    except requests.RequestException as exc:
        logger.warning("Ollama rewrite request error (%s): %s", url, exc)
        return query.strip()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        logger.warning("Ollama rewrite HTTP %s (%s): %s", resp.status_code, url, detail)
        return query.strip()

    try:
        data = resp.json()
        rewritten = data.get("response", "").strip()
        if rewritten:
            return rewritten
        logger.warning("Ollama rewrite returned empty response; using original query")
        return query.strip()
    except (ValueError, TypeError) as exc:
        logger.warning("Ollama rewrite invalid JSON (%s): %s", url, exc)
        return query.strip()


def rewrite_with_deepseek(query: str, model: Optional[str] = None) -> str:
    """
    Rewrite query using DeepSeek API (OpenAI-compatible client).

    Uses DEEPSEEK_API_KEY and GATEWAY_REWRITE_DEEPSEEK_MODEL from env.
    Timeout: GATEWAY_REWRITE_TIMEOUT (default 10s).

    Args:
        query: User question to rewrite.
        model: Optional model override; uses env default if None.

    Returns:
        Rewritten query string, or original query on failure.
    """
    if not query or not query.strip():
        return query

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek rewrite")
        return query.strip()

    mdl = model or os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    timeout = _get_rewrite_timeout()

    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("openai client not installed; cannot call DeepSeek rewrite: %s", exc)
        return query.strip()

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": query.strip()},
    ]

    try:
        response = client.chat.completions.create(
            model=mdl,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("DeepSeek rewrite failed: %s", exc)
        return query.strip()

    try:
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            rewritten = choice.message.content.strip()
            if rewritten:
                return rewritten
        logger.warning("DeepSeek rewrite returned empty response; using original query")
        return query.strip()
    except (IndexError, AttributeError, TypeError) as exc:
        logger.warning("DeepSeek rewrite unexpected response format: %s", exc)
        return query.strip()


__all__ = [
    "REWRITE_PROMPT",
    "rewrite_with_ollama",
    "rewrite_with_deepseek",
]
