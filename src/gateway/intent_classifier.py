"""
Gateway-level vector intent classifier.

Uses all-minilm embeddings + Chroma intent_registry for fast intent matching,
with qwen3:1.7b LLM verification for ambiguous cases.

Architecture:
1. Embed query via Ollama all-minilm
2. Query Chroma intent_registry collection, Top3
3. Confidence thresholds:
   - distance < HIGH_CONF (0.3): use Top1 directly
   - distance HIGH_CONF ~ LOW_CONF (0.3-0.7): LLM verifies from Top3
   - distance > LOW_CONF (0.7): fallback to keyword heuristics

Usage:
    from src.gateway.intent_classifier import classify_intent
    result = classify_intent("check order status for 112-1234567-8901234")
    # IntentResult(intent_name="order_query", workflow="sp_api", distance=0.15, ...)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_DEFAULT_HIGH_CONF = 0.3
_DEFAULT_LOW_CONF = 0.7
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent_name: str
    workflow: str
    distance: float
    confidence: str  # "high", "medium", "low"
    required_fields: List[str] = field(default_factory=list)
    clarification_template: str = ""
    source: str = "vector"  # "vector", "llm_verified", "heuristic_fallback"


def _get_high_conf_threshold() -> float:
    try:
        return float(os.getenv("GATEWAY_INTENT_HIGH_CONF_THRESHOLD", str(_DEFAULT_HIGH_CONF)))
    except (ValueError, TypeError):
        return _DEFAULT_HIGH_CONF


def _get_low_conf_threshold() -> float:
    try:
        return float(os.getenv("GATEWAY_INTENT_LOW_CONF_THRESHOLD", str(_DEFAULT_LOW_CONF)))
    except (ValueError, TypeError):
        return _DEFAULT_LOW_CONF


def _get_ollama_url() -> str:
    return os.getenv("GATEWAY_REWRITE_OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")


def _get_ollama_model() -> str:
    return os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)


def _get_embedding_model() -> str:
    return os.getenv("GATEWAY_INTENT_EMBEDDING_MODEL", "all-minilm")


def _embed_query(query: str) -> List[float]:
    """Embed a single query via Ollama /api/embed."""
    url = f"{_get_ollama_url()}/api/embed"
    model = _get_embedding_model()
    resp = requests.post(
        url,
        json={"model": model, "input": [query]},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings or not embeddings[0]:
        raise ValueError("Empty embedding response from Ollama")
    return embeddings[0]


def _query_chroma(query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
    """Query the intent_registry Chroma collection."""
    from .intent_registry import get_intent_collection

    collection = get_intent_collection()
    if collection is None:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances", "metadatas", "documents"],
    )
    return results


def _llm_verify(
    query: str,
    candidates: List[Dict[str, Any]],
    conversation_context: Optional[str] = None,
) -> Optional[str]:
    """Call qwen3:1.7b to pick the best intent from Top3 candidates.

    Returns the chosen intent_name, or None on failure.
    """
    prompt_template = load_prompt("intent_classification/intent_verify_candidate")

    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        candidates_text += f"{i}. {c['intent_name']} (workflow: {c['workflow']}, distance: {c['distance']:.3f})\n"

    prompt = prompt_template.replace("{candidates}", candidates_text.strip())
    prompt = prompt.replace("{query}", query)

    if conversation_context and conversation_context.strip():
        prompt = f"Conversation history:\n{conversation_context.strip()}\n\n{prompt}"

    url = f"{_get_ollama_url()}/api/generate"
    model = _get_ollama_model()

    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("LLM verify HTTP %s", resp.status_code)
            return None
        data = resp.json()
        text = (data.get("response") or "").strip()
    except Exception as exc:
        logger.warning("LLM intent verify failed: %s", exc)
        return None

    if not text:
        return None

    # Parse JSON response.
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(raw)
    except ValueError:
        # Try extracting first JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except ValueError:
                return None
        else:
            return None

    intent = parsed.get("intent")
    if isinstance(intent, str) and intent.strip() and intent.strip() != "none":
        return intent.strip()
    return None


def classify_intent(
    query: str,
    conversation_context: Optional[str] = None,
) -> Optional[IntentResult]:
    """
    Classify a query's intent using vector similarity + optional LLM verification.

    Args:
        query: The rewritten/normalized query text.
        conversation_context: Optional formatted conversation history for LLM context.

    Returns:
        IntentResult on success, None if classification fails entirely.
    """
    if not query or not query.strip():
        return None

    high_threshold = _get_high_conf_threshold()
    low_threshold = _get_low_conf_threshold()

    # Step 1: Embed query.
    try:
        query_embedding = _embed_query(query.strip())
    except Exception as exc:
        logger.warning("Intent classification embedding failed: %s", exc)
        return None

    # Step 2: Query Chroma for Top3.
    try:
        results = _query_chroma(query_embedding, n_results=3)
    except Exception as exc:
        logger.warning("Intent classification Chroma query failed: %s", exc)
        return None

    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]

    if not ids or not distances:
        logger.info("No intent matches found in Chroma for query: %s", query[:80])
        return None

    # Build candidate list.
    candidates = []
    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        candidates.append({
            "intent_name": meta.get("intent_name", "unknown"),
            "workflow": meta.get("workflow", "general"),
            "distance": distances[i],
            "required_fields": json.loads(meta.get("required_fields", "[]")),
            "clarification_template": meta.get("clarification_template", ""),
            "document": documents[i] if i < len(documents) else "",
        })

    top1 = candidates[0]
    top1_distance = top1["distance"]

    logger.info(
        "Intent vector match: query='%s' top1=%s (dist=%.3f) top2=%s (dist=%.3f) top3=%s (dist=%.3f)",
        query[:60],
        top1["intent_name"], top1_distance,
        candidates[1]["intent_name"] if len(candidates) > 1 else "n/a",
        candidates[1]["distance"] if len(candidates) > 1 else 0,
        candidates[2]["intent_name"] if len(candidates) > 2 else "n/a",
        candidates[2]["distance"] if len(candidates) > 2 else 0,
    )

    # Step 3: Apply confidence thresholds.
    if top1_distance < high_threshold:
        # High confidence — use Top1 directly.
        return IntentResult(
            intent_name=top1["intent_name"],
            workflow=top1["workflow"],
            distance=top1_distance,
            confidence="high",
            required_fields=top1["required_fields"],
            clarification_template=top1["clarification_template"],
            source="vector",
        )

    if top1_distance <= low_threshold:
        # Ambiguous — LLM verifies from Top3.
        verified = _llm_verify(query.strip(), candidates, conversation_context)
        if verified:
            # Find the verified candidate in our list.
            for c in candidates:
                if c["intent_name"] == verified:
                    return IntentResult(
                        intent_name=c["intent_name"],
                        workflow=c["workflow"],
                        distance=c["distance"],
                        confidence="medium",
                        required_fields=c["required_fields"],
                        clarification_template=c["clarification_template"],
                        source="llm_verified",
                    )
            # LLM returned a name not in candidates — use Top1 as fallback.
            logger.warning("LLM verified intent '%s' not in candidates; using Top1", verified)

        # LLM failed or returned unknown — use Top1 with medium confidence.
        return IntentResult(
            intent_name=top1["intent_name"],
            workflow=top1["workflow"],
            distance=top1_distance,
            confidence="medium",
            required_fields=top1["required_fields"],
            clarification_template=top1["clarification_template"],
            source="vector",
        )

    # Low confidence — no reliable match.
    logger.info(
        "Intent classification low confidence (dist=%.3f > %.3f) for query: %s",
        top1_distance, low_threshold, query[:80],
    )
    return None
