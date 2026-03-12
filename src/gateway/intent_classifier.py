"""
Gateway-level intent classifier.

Pipeline:
1. split_intents(query)  — qwen3:1.7b splits rewritten query into sub-questions
2. classify_intent(sub_query) — for each sub-question, runs dual retrieval:
   a. Keyword matching (rule-based, no model) → workflow or "hybrid"
   b. Vector matching (Ollama all-minilm → Chroma vector_intent_registry) → workflow or "hybrid"
   c. resolve_intent(keyword, vector):
      - If both agree and neither is hybrid → use it
      - Else fallback: keyword (if not hybrid) → vector (if not hybrid) → "general"

Usage:
    from src.gateway.intent_classifier import split_intents, classify_intent
    clauses = split_intents("check order 112-123 and show FBA fees last month")
    # ["check order status for 112-123", "show FBA fees last month"]
    result = classify_intent("check order status for 112-1234567-8901234")
    # IntentResult(intent_name="order_query", workflow="sp_api", ...)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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
    source: str = "vector"  # "keyword", "vector", "fallback"
    vector_distance: Optional[float] = None  # raw distance from vector retrieval, for observability


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


# ---------------------------------------------------------------------------
# Intent splitting
# ---------------------------------------------------------------------------

def split_intents(query: str) -> List[str]:
    """
    Split a rewritten query into distinct single-intent sub-questions.

    Uses qwen3:1.7b with intent_split_query.txt prompt.
    Returns a list with the original query as sole item on any failure.

    Args:
        query: Normalized/rewritten query text (no splitting done yet).

    Returns:
        List of single-intent sub-question strings.
    """
    if not query or not query.strip():
        return []

    prompt_template = load_prompt("intent_classification/intent_split_query")
    prompt = f"{prompt_template}\n\nInput: {query.strip()}"

    url = f"{_get_ollama_url()}/api/generate"
    model = _get_ollama_model()

    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Intent split HTTP %s; returning original query", resp.status_code)
            return [query.strip()]
        data = resp.json()
        text = (data.get("response") or "").strip()
    except Exception as exc:
        logger.warning("Intent split LLM call failed: %s; returning original query", exc)
        return [query.strip()]

    if not text:
        return [query.strip()]

    # Strip markdown fences if present.
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    # Parse JSON {"intents": [...]}
    try:
        parsed = json.loads(raw)
    except ValueError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except ValueError:
                logger.warning("Intent split JSON parse failed; returning original query")
                return [query.strip()]
        else:
            logger.warning("Intent split no JSON found; returning original query")
            return [query.strip()]

    intents = parsed.get("intents")
    if not isinstance(intents, list) or not intents:
        return [query.strip()]

    # Deduplicate and filter blanks.
    seen: set = set()
    result = []
    for item in intents:
        if isinstance(item, str):
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)

    return result if result else [query.strip()]


# ---------------------------------------------------------------------------
# Keyword intent matching (rule-based, no model)
# ---------------------------------------------------------------------------

_keyword_map_cache: Optional[List[Dict[str, str]]] = None


def _load_keyword_map() -> List[Dict[str, str]]:
    """Load keyword → intent mapping from keyword_intents.csv (lazy, cached)."""
    global _keyword_map_cache
    if _keyword_map_cache is not None:
        return _keyword_map_cache

    csv_path = Path(__file__).parent.parent / "prompts" / "retrieval" / "keyword_intents.csv"
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kw = (row.get("keyword") or "").strip().lower()
                intent = (row.get("intent") or "").strip().lower()
                if kw and intent:
                    rows.append({"keyword": kw, "intent": intent})
    except Exception as exc:
        logger.warning("Failed to load keyword_intents.csv: %s", exc)

    # Sort longest keywords first so more specific phrases match before shorter ones.
    rows.sort(key=lambda r: len(r["keyword"]), reverse=True)
    _keyword_map_cache = rows
    return rows


def _keyword_match_intent(query: str) -> str:
    """
    Rule-based keyword intent matching.

    Scans the query for known keywords and collects all matching workflows.
    - 0 matches  → "general"
    - 1 workflow → that workflow
    - ≥2 distinct workflows → "hybrid"

    Args:
        query: The sub-query text (lowercased internally).

    Returns:
        One of: "general", "amazon_docs", "uds", "sp_api", "hybrid"
    """
    if not query or not query.strip():
        return "general"

    q = query.lower()
    keyword_map = _load_keyword_map()

    matched_workflows: set = set()
    for entry in keyword_map:
        if entry["keyword"] in q:
            matched_workflows.add(entry["intent"])
            # Early exit: once we have 2+ distinct workflows it's hybrid.
            if len(matched_workflows) >= 2:
                return "hybrid"

    if not matched_workflows:
        return "general"

    return matched_workflows.pop()


# ---------------------------------------------------------------------------
# Vector intent matching
# ---------------------------------------------------------------------------

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


def _vector_match_intent(query: str) -> tuple[str, float, Optional[Dict[str, Any]]]:
    """
    Vector-based intent matching via Chroma.

    Returns:
        (workflow, distance, top1_candidate_dict)
        workflow is "general" on failure or low confidence.
    """
    high_threshold = _get_high_conf_threshold()
    low_threshold = _get_low_conf_threshold()

    try:
        query_embedding = _embed_query(query.strip())
    except Exception as exc:
        logger.warning("Vector embed failed: %s", exc)
        return "general", 1.0, None

    try:
        results = _query_chroma(query_embedding, n_results=3)
    except Exception as exc:
        logger.warning("Chroma query failed: %s", exc)
        return "general", 1.0, None

    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]

    if not ids or not distances:
        return "general", 1.0, None

    # Build candidate list.
    candidates = []
    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        candidates.append({
            "intent_name": meta.get("intent_name", meta.get("intent", "unknown")),
            "workflow": meta.get("workflow", "general"),
            "distance": distances[i],
            "required_fields": json.loads(meta.get("required_fields", "[]")),
            "clarification_template": meta.get("clarification_template", ""),
            "document": documents[i] if i < len(documents) else "",
        })

    top1 = candidates[0]
    top1_distance = top1["distance"]

    logger.info(
        "Vector match: query='%s' top1=%s (dist=%.3f) top2=%s (dist=%.3f) top3=%s (dist=%.3f)",
        query[:60],
        top1["workflow"], top1_distance,
        candidates[1]["workflow"] if len(candidates) > 1 else "n/a",
        candidates[1]["distance"] if len(candidates) > 1 else 0,
        candidates[2]["workflow"] if len(candidates) > 2 else "n/a",
        candidates[2]["distance"] if len(candidates) > 2 else 0,
    )

    # Below low threshold → has a match.
    if top1_distance <= low_threshold:
        return top1["workflow"], top1_distance, top1

    # Above low threshold → no reliable match.
    return "general", top1_distance, top1


# ---------------------------------------------------------------------------
# Fallback resolution
# ---------------------------------------------------------------------------

def resolve_intent(keyword_result: str, vector_result: str) -> str:
    """
    Resolve final workflow from keyword and vector results.

    Priority:
      1. keyword result (if not hybrid)
      2. vector result (if not hybrid)
      3. "general" (final default)

    Args:
        keyword_result: workflow from keyword matching, or "hybrid"
        vector_result:  workflow from vector matching, or "hybrid"

    Returns:
        Final workflow string.
    """
    if keyword_result != "hybrid":
        return keyword_result
    if vector_result != "hybrid":
        return vector_result
    return "general"


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify_intent(
    query: str,
    conversation_context: Optional[str] = None,
) -> Optional[IntentResult]:
    """
    Classify a query's intent using parallel keyword + vector dual retrieval.

    Steps:
      1. Keyword matching (rule-based) → workflow or "hybrid"
      2. Vector matching (all-minilm → Chroma) → workflow or "hybrid"
      3. If both agree and neither is hybrid → use directly
      4. Else → resolve_intent(keyword, vector) for fallback

    Args:
        query: The rewritten/normalized query text.
        conversation_context: Unused (kept for API compatibility).

    Returns:
        IntentResult on success, None if classification fails entirely.
    """
    if not query or not query.strip():
        return None

    q = query.strip()

    # Step 1: Keyword matching (fast, no I/O).
    keyword_workflow = _keyword_match_intent(q)

    # Step 2: Vector matching.
    vector_workflow, vector_distance, vector_top1 = _vector_match_intent(q)

    logger.info(
        "Dual retrieval: query='%s' keyword=%s vector=%s (dist=%.3f)",
        q[:60], keyword_workflow, vector_workflow, vector_distance,
    )

    # Step 3: Determine if fallback is needed.
    needs_fallback = (keyword_workflow != vector_workflow) or \
                     (keyword_workflow == "hybrid") or \
                     (vector_workflow == "hybrid")

    if not needs_fallback:
        # Both agree, neither is hybrid.
        final_workflow = keyword_workflow
        source = "keyword"
        confidence = "high" if vector_distance < _get_high_conf_threshold() else "medium"
    else:
        final_workflow = resolve_intent(keyword_workflow, vector_workflow)
        # Determine source for observability.
        if keyword_workflow != "hybrid":
            source = "keyword"
        elif vector_workflow != "hybrid":
            source = "vector"
        else:
            source = "fallback"
        confidence = "medium"

    logger.info(
        "Intent resolved: workflow=%s source=%s fallback=%s",
        final_workflow, source, needs_fallback,
    )

    # Step 4: Build IntentResult.
    # Pull required_fields and clarification_template from vector top1 if available
    # and the final workflow matches; otherwise use empty defaults.
    required_fields: List[str] = []
    clarification_template: str = ""

    if vector_top1 and vector_top1.get("workflow") == final_workflow:
        required_fields = vector_top1.get("required_fields") or []
        clarification_template = vector_top1.get("clarification_template") or ""

    # If final workflow is "general" or "hybrid", no fields needed.
    if final_workflow in ("general", "hybrid"):
        required_fields = []
        clarification_template = ""

    return IntentResult(
        intent_name=vector_top1.get("intent_name", final_workflow) if vector_top1 else final_workflow,
        workflow=final_workflow,
        distance=vector_distance,
        confidence=confidence,
        required_fields=required_fields,
        clarification_template=clarification_template,
        source=source,
        vector_distance=vector_distance,
    )
