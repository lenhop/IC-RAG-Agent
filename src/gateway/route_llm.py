"""
Route LLM (Planning): LLM-based workflow classifier.

Classifies each query into one of: general, amazon_docs, ic_docs, sp_api, uds,
with confidence in [0.0, 1.0]. Used for single-task routing when planner is
disabled or returns one task. Supports Ollama and DeepSeek.

On error: returns safe default ("general", 0.0), logs; never raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Tuple

import requests

logger = logging.getLogger(__name__)

# Allowed workflow labels; any other value is invalid and triggers fallback
ALLOWED_WORKFLOWS = frozenset(
    {"general", "amazon_docs", "ic_docs", "sp_api", "uds"}
)

# Shared routing prompt (system/instruction) per ROUTE_LLM_DEVELOPMENT_PLAN.md
ROUTE_LLM_SYSTEM_PROMPT = (
    "You are a routing classifier for an e-commerce assistant.\n\n"
    "Your task is to choose EXACTLY ONE workflow label for each user query:\n"
    '- "general": General knowledge or open questions, not tied to Amazon docs or internal docs.\n'
    '- "amazon_docs": Amazon business rules/policies/requirements/fee definitions/documentation.\n'
    '- "ic_docs": Questions about internal company documents, policies, or project-specific documentation.\n'
    '- "sp_api": Real-time operational status/actions via Amazon SP-API (current order status, current product/listing status, live inventory, shipment status, account live state), and only when this cannot be answered from UDS snapshots.\n'
    '- "uds": Historical/analytical BI queries over warehouse data snapshots (last month/quarter, trends, grouped aggregates, table-level analysis). Prefer uds first when possible.\n\n'
    "Critical routing rules:\n"
    "1) Policy/rule/requirement/fee-definition questions must be amazon_docs, not sp_api.\n"
    "2) Prefer uds first for analytical questions when snapshot data is sufficient.\n"
    "3) Use sp_api only for real-time/current-state data or operations that require live API calls.\n"
    "4) uds is for historical analytics; if the query asks for trend/last month/aggregated metrics, prefer uds.\n"
    "5) definition-style questions about terms like FBA/FBM (for example, 'what is FBA') should be classified as ic_docs, not sp_api.\n\n"
    "Return ONLY a single JSON object on one line:\n"
    '{"workflow": "<one of general|amazon_docs|ic_docs|sp_api|uds>", "confidence": <float between 0 and 1>}\n'
    "Do not include any explanations."
)

# Default env values for Route LLM (separate from rewrite env vars)
DEFAULT_ROUTE_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_ROUTE_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_ROUTE_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_ROUTE_TIMEOUT = 5
SAFE_DEFAULT_WORKFLOW = "general"
SAFE_DEFAULT_CONFIDENCE = 0.0


def _get_route_timeout() -> int:
    """Read GATEWAY_ROUTE_LLM_TIMEOUT from env (default 5s)."""
    try:
        return int(os.getenv("GATEWAY_ROUTE_LLM_TIMEOUT", str(DEFAULT_ROUTE_TIMEOUT)))
    except ValueError:
        return DEFAULT_ROUTE_TIMEOUT


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    if value is None:
        return SAFE_DEFAULT_CONFIDENCE
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return SAFE_DEFAULT_CONFIDENCE


def _parse_route_json(text: str) -> Tuple[str, float] | None:
    """
    Parse LLM output for workflow and confidence.

    Expects a single-line JSON object: {"workflow": "...", "confidence": 0.92}.
    Tolerates surrounding text by searching for the first {...} object.

    Args:
        text: Raw LLM response (may contain JSON embedded in text).

    Returns:
        (workflow, confidence) if valid, else None.
    """
    if not text or not text.strip():
        return None

    # Try to find a JSON object in the response (LLM might wrap it in prose)
    stripped = text.strip()
    match = re.search(r"\{[^{}]*\}", stripped)
    if match:
        try:
            data = json.loads(match.group(0))
            workflow = (data.get("workflow") or "").strip().lower()
            confidence = _clamp_confidence(data.get("confidence"))
            if workflow in ALLOWED_WORKFLOWS:
                return (workflow, confidence)
        except json.JSONDecodeError as exc:
            logger.debug("Route LLM JSON parse failed: %s", exc)
        except (TypeError, AttributeError) as exc:
            logger.debug("Route LLM unexpected structure: %s", exc)

    # Fallback: try parsing the whole string as JSON
    try:
        data = json.loads(stripped)
        workflow = (data.get("workflow") or "").strip().lower()
        confidence = _clamp_confidence(data.get("confidence"))
        if workflow in ALLOWED_WORKFLOWS:
            return (workflow, confidence)
    except json.JSONDecodeError:
        pass
    except (TypeError, AttributeError):
        pass

    return None


def _route_with_ollama(query: str) -> Tuple[str, float]:
    """
    Call Ollama HTTP API for workflow classification.

    Uses GATEWAY_ROUTE_LLM_OLLAMA_URL (base URL, e.g. http://localhost:11434)
    and GATEWAY_ROUTE_LLM_OLLAMA_MODEL. Appends /api/generate if not present.
    Expects response body with "response" field containing JSON line.

    Args:
        query: User query to classify.

    Returns:
        (workflow, confidence) or safe default on error.
    """
    base = (os.getenv("GATEWAY_ROUTE_LLM_OLLAMA_URL") or DEFAULT_ROUTE_OLLAMA_BASE_URL).rstrip("/")
    url = f"{base}/api/generate" if "/api/generate" not in base else base
    model = os.getenv("GATEWAY_ROUTE_LLM_OLLAMA_MODEL", DEFAULT_ROUTE_OLLAMA_MODEL)
    timeout = _get_route_timeout()

    user_content = f"User query:\n{query.strip()}"
    full_prompt = f"{ROUTE_LLM_SYSTEM_PROMPT}\n\n{user_content}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        logger.warning("Route LLM Ollama connection failed (%s): %s", url, exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
    except requests.Timeout as exc:
        logger.warning("Route LLM Ollama timed out (%s, %ds): %s", url, timeout, exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
    except requests.RequestException as exc:
        logger.warning("Route LLM Ollama request error (%s): %s", url, exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        logger.warning("Route LLM Ollama HTTP %s (%s): %s", resp.status_code, url, detail)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    try:
        data = resp.json()
        raw = (data.get("response") or "").strip()
    except (ValueError, TypeError) as exc:
        logger.warning("Route LLM Ollama invalid response JSON: %s", exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    parsed = _parse_route_json(raw)
    if parsed is not None:
        return parsed
    logger.warning("Route LLM Ollama could not parse valid workflow from: %s", raw[:200])
    return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


def _route_with_deepseek(query: str) -> Tuple[str, float]:
    """
    Call DeepSeek API (OpenAI-compatible) for workflow classification.

    Uses DEEPSEEK_API_KEY and GATEWAY_ROUTE_LLM_DEEPSEEK_MODEL.
    Expects chat completion content to be a single-line JSON object.

    Args:
        query: User query to classify.

    Returns:
        (workflow, confidence) or safe default on error.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call Route LLM DeepSeek")
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    model = os.getenv("GATEWAY_ROUTE_LLM_DEEPSEEK_MODEL", DEFAULT_ROUTE_DEEPSEEK_MODEL)
    timeout = _get_route_timeout()

    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("openai client not installed for Route LLM DeepSeek: %s", exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout,
    )

    user_content = f"User query:\n{query.strip()}"

    messages = [
        {"role": "system", "content": ROUTE_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("Route LLM DeepSeek API call failed: %s", exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    try:
        choice = response.choices[0] if response.choices else None
        raw = (choice.message.content or "").strip() if choice and choice.message else ""
    except (IndexError, AttributeError, TypeError) as exc:
        logger.warning("Route LLM DeepSeek unexpected response shape: %s", exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    parsed = _parse_route_json(raw)
    if parsed is not None:
        return parsed
    logger.warning("Route LLM DeepSeek could not parse valid workflow from: %s", raw[:200])
    return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


def route_with_llm(query: str, backend: str) -> Tuple[str, float]:
    """
    Classify query into one workflow using the specified LLM backend.

    Normalizes backend to "ollama", "deepseek", or "none". For "none" or
    unknown backend, returns safe default without calling any API.
    Validates that the returned workflow is in ALLOWED_WORKFLOWS and
    confidence is in [0.0, 1.0]. On any error, returns ("general", 0.0)
    and logs; never raises.

    Args:
        query: User query to classify.
        backend: One of "ollama", "deepseek", "none" (case-insensitive).

    Returns:
        (workflow, confidence) with workflow in ALLOWED_WORKFLOWS,
        confidence in [0.0, 1.0].
    """
    normalized_backend = (backend or "").strip().lower()

    if normalized_backend == "none" or not normalized_backend:
        logger.debug("Route LLM backend is 'none' or empty; returning safe default")
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    if not query or not str(query).strip():
        logger.debug("Route LLM empty query; returning safe default")
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

    try:
        if normalized_backend == "ollama":
            workflow, confidence = _route_with_ollama(query.strip())
        elif normalized_backend == "deepseek":
            workflow, confidence = _route_with_deepseek(query.strip())
        else:
            logger.warning("Route LLM unknown backend '%s'; returning safe default", backend)
            return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

        # Enforce allowed workflow (backend should already return valid)
        if workflow not in ALLOWED_WORKFLOWS:
            logger.warning("Route LLM returned invalid workflow '%s'; using safe default", workflow)
            return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)

        confidence = _clamp_confidence(confidence)
        return (workflow, confidence)

    except Exception as exc:
        # Catch-all for any unexpected error; do not propagate
        logger.exception("Route LLM unexpected error (backend=%s): %s", backend, exc)
        return (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


__all__ = [
    "ALLOWED_WORKFLOWS",
    "ROUTE_LLM_SYSTEM_PROMPT",
    "route_with_llm",
]
