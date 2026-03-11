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
import re
from typing import Optional

import requests

from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# Reuse env defaults from rewriters
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10

# Prompts loaded from src/prompts/*.txt (cached after first access)
CLARIFICATION_PROMPT = load_prompt("clarification_detect_ambiguity")
_GENERATE_QUESTION_PROMPT = load_prompt("clarification_generate_question")

def _is_concrete_documentation_query(query: str) -> bool:
    """
    Return True for documentation, policy, compliance, requirements questions.
    These are self-contained conceptual questions and do NOT need clarification.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = [
        r"documentation\s+requirements",
        r"product\s+compliance",
        r"safety\s+documentation",
        r"policy\s+on",
        r"what\s+are\s+.*\s+requirements",
        r"what\s+does\s+amazon",
        r"guidelines",
        r"business\s+rules",
        r"compliance\s+and\s+safety",
        r"requirements\s+for",
    ]
    return any(re.search(p, q) for p in patterns)


# Heuristic patterns: query mentions topic but lacks required identifiers.
# (topic_pattern, has_required_pattern, fallback_question when LLM fails)
_HEURISTIC_AMBIGUOUS = [
    (
        r"\b(inventory|stock)\b",
        r"\b(ASIN|B0[0-9A-Z]{8}|store|SKU|marketplace)\b",
        "Which store, ASIN, or SKU do you want inventory for?",
    ),
    (
        r"\b(order|orders)\s+(status|info|details)?\b|\b(check|get)\s+(my\s+)?order\b",
        r"\d{3}-\d{7}-\d{7}|order\s*[iI][dD]",
        "Please provide the Order ID (e.g. 112-1234567-8901234) to check order status.",
    ),
    (
        r"\b(fees?|charges?|breakdown)\b",
        r"\b(FBA|storage|referral|last\s+(month|quarter|year)|Q[1-4]|20\d{2}|20\d{6})\b|\b\d{4}-\d{2}-\d{2}\b",
        "Which fees do you mean? (FBA, storage, or referral) And for which time period?",
    ),
    (
        r"\b(sales?|trends?|metrics?)\b",
        r"\b(last\s+(month|quarter|year)|Q[1-4]|20\d{2}|20\d{6})\b|\b\d{4}-\d{2}-\d{2}\b|january|february|march|april|may|june|july|august|september|october|november|december\b",
        "Which date range or time period do you want the data for?",
    ),
]


def _build_user_input(query: str, conversation_context: Optional[str] = None) -> str:
    """Build the user input string with optional conversation history prefix."""
    parts = []
    if conversation_context and conversation_context.strip():
        parts.append(f"Conversation history:\n{conversation_context.strip()}\n")
    parts.append(f"User query: {query.strip()}")
    return "\n".join(parts)


def _generate_clarification_question_ollama(
    query: str, conversation_context: Optional[str] = None
) -> Optional[str]:
    """Call LLM to generate a contextual clarification question. Returns None on failure."""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_timeout()
    user_input = _build_user_input(query, conversation_context)
    payload = {
        "model": mdl,
        "prompt": f"{_GENERATE_QUESTION_PROMPT}{user_input}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama generate-question failed: %s", exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        text = (data.get("response") or "").strip()
    except (ValueError, TypeError):
        return None
    if not text:
        return None
    raw = _strip_markdown_fences(text)
    candidate = _extract_first_json_object(raw)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
        q = parsed.get("clarification_question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except (ValueError, TypeError):
        pass
    return None


def _heuristic_needs_clarification(query: str) -> Optional[dict]:
    """
    Return clarification dict if query matches known ambiguous patterns.
    Used as fast path before LLM when pattern is clear.
    """
    q = (query or "").strip()
    if not q:
        return None
    for topic_pattern, has_required, question in _HEURISTIC_AMBIGUOUS:
        if not re.search(topic_pattern, q, re.IGNORECASE):
            continue
        if has_required is None:
            return {"needs_clarification": True, "clarification_question": question}
        if not re.search(has_required, q, re.IGNORECASE):
            return {"needs_clarification": True, "clarification_question": question}
    return None


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


def _call_clarification_ollama(
    query: str, conversation_context: Optional[str] = None
) -> str:
    """Call Ollama with clarification prompt; return raw response text."""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_timeout()
    user_input = _build_user_input(query, conversation_context)
    payload = {
        "model": mdl,
        "prompt": f"{CLARIFICATION_PROMPT}{user_input}",
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


def _call_clarification_deepseek(
    query: str, conversation_context: Optional[str] = None
) -> str:
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
    user_input = _build_user_input(query, conversation_context)
    messages = [
        {"role": "system", "content": CLARIFICATION_PROMPT},
        {"role": "user", "content": user_input},
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


def check_ambiguity(
    query: str,
    backend: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> dict:
    """
    Check if the user query is ambiguous and needs clarification.

    Args:
        query: Raw user query text.
        backend: Optional backend override; uses GATEWAY_REWRITE_BACKEND env if None.
        conversation_context: Optional formatted conversation history (last 3 rounds)
            from Redis. When provided, the LLM considers prior turns so it won't
            ask for info the user already supplied in recent conversation.

    Returns:
        {"needs_clarification": True, "clarification_question": "..."} when ambiguous,
        or {"needs_clarification": False} when clear. On LLM failure, returns
        {"needs_clarification": False} to allow normal flow to proceed.
    """
    if not query or not query.strip():
        return {"needs_clarification": False}

    # Skip clarification for documentation/policy/requirements questions (self-contained).
    if _is_concrete_documentation_query(query):
        return {"needs_clarification": False}

    # Heuristic fast path: known ambiguous patterns. Use LLM to generate contextual question.
    # When conversation_context is present, skip heuristic and let LLM decide
    # (the user may have already provided the missing info in a prior turn).
    if not conversation_context:
        heuristic_result = _heuristic_needs_clarification(query)
        if heuristic_result is not None:
            fallback_question = heuristic_result.get("clarification_question", "")
            effective_backend = (backend or "").strip().lower() or os.getenv(
                "GATEWAY_REWRITE_BACKEND", "ollama"
            ).strip().lower()
            if effective_backend == "ollama":
                llm_question = _generate_clarification_question_ollama(query.strip())
                if llm_question:
                    return {"needs_clarification": True, "clarification_question": llm_question}
            return {"needs_clarification": True, "clarification_question": fallback_question}

    effective_backend = (backend or "").strip().lower() or os.getenv(
        "GATEWAY_REWRITE_BACKEND", "ollama"
    ).strip().lower()

    if effective_backend == "ollama":
        text = _call_clarification_ollama(query.strip(), conversation_context)
    elif effective_backend == "deepseek":
        text = _call_clarification_deepseek(query.strip(), conversation_context)
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
