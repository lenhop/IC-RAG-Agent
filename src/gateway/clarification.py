"""
Route LLM clarification: detect ambiguous queries before rewriting.

For Amazon seller queries, detects when the user query is ambiguous or lacks
critical information (ASIN, Order ID, date range, time period, store, SKU,
marketplace). The LLM returns a clarification question instead of executing.
Uses same backends as rewriters: Ollama / DeepSeek.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Reuse env defaults from rewriters
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10

CLARIFICATION_PROMPT = (
    "You are a clarification expert for Amazon seller queries. "
    "Check if the query is ambiguous or lacks critical information: "
    "ASIN, Order ID, date range, time period, store, SKU, marketplace. "
    "If missing or ambiguous: {\"needs_clarification\": true, \"clarification_question\": \"...\"} "
    "If clear: {\"needs_clarification\": false} "
    "DO NOT answer the query. DO NOT add explanation or notes. Output JSON ONLY. "
    "User query: "
)


def _get_timeout() -> int:
    """Read GATEWAY_REWRITE_TIMEOUT from env (default 10s)."""
    try:
        return int(os.getenv("GATEWAY_REWRITE_TIMEOUT", str(DEFAULT_REWRITE_TIMEOUT)))
    except ValueError:
        return DEFAULT_REWRITE_TIMEOUT


def _strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _call_clarification_ollama(query: str) -> str:
    """Call Ollama with clarification prompt; return raw response text."""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_timeout()
    payload = {
        "model": mdl,
        "prompt": f"{CLARIFICATION_PROMPT}{query}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama clarification check failed: %s", exc)
        return ""
    if resp.status_code != 200:
        logger.warning("Ollama clarification HTTP %s", resp.status_code)
        return ""
    try:
        data = resp.json()
        return (data.get("response") or "").strip()
    except (ValueError, TypeError):
        return ""


def _call_clarification_deepseek(query: str) -> str:
    """Call DeepSeek with clarification prompt; return raw response text."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek clarification")
        return ""
    mdl = os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    timeout = _get_timeout()
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai client not installed; cannot call DeepSeek clarification")
        return ""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
    messages = [
        {"role": "system", "content": CLARIFICATION_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        response = client.chat.completions.create(
            model=mdl,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            return (choice.message.content or "").strip()
    except Exception as exc:
        logger.warning("DeepSeek clarification failed: %s", exc)
    return ""


def check_ambiguity(query: str, backend: Optional[str] = None) -> dict:
    """
    Check if the user query is ambiguous and needs clarification.

    Args:
        query: Raw user query text.
        backend: Optional backend override; uses GATEWAY_REWRITE_BACKEND env if None.

    Returns:
        {"needs_clarification": True, "clarification_question": "..."} when ambiguous,
        or {"needs_clarification": False} when clear. On LLM failure, returns
        {"needs_clarification": False} to allow normal flow to proceed.
    """
    if not query or not query.strip():
        return {"needs_clarification": False}

    effective_backend = (backend or "").strip().lower() or os.getenv(
        "GATEWAY_REWRITE_BACKEND", "ollama"
    ).strip().lower()

    if effective_backend == "ollama":
        text = _call_clarification_ollama(query.strip())
    elif effective_backend == "deepseek":
        text = _call_clarification_deepseek(query.strip())
    else:
        logger.warning("check_ambiguity: unknown backend %s; skipping", effective_backend)
        return {"needs_clarification": False}

    if not text or not text.strip():
        return {"needs_clarification": False}

    raw = _strip_markdown_fences(text)
    parsed = None
    try:
        parsed = json.loads(raw)
    except ValueError:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except ValueError:
                parsed = None

    if not isinstance(parsed, dict):
        return {"needs_clarification": False}

    needs = parsed.get("needs_clarification")
    if not needs:
        return {"needs_clarification": False}

    question = parsed.get("clarification_question")
    if not isinstance(question, str) or not question.strip():
        return {"needs_clarification": False}

    return {
        "needs_clarification": True,
        "clarification_question": question.strip(),
    }
