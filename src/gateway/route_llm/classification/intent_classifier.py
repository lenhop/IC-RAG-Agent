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
    from src.gateway.intent_classification import split_intents, classify_intent
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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...prompt_loader import load_prompt
from src.logger import get_logger_facade
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient, get_ollama_config

logger = logging.getLogger(__name__)
_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None

_DEFAULT_HIGH_CONF = 0.3
_DEFAULT_LOW_CONF = 0.7
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


def _get_embedding_model() -> str:
    """Embed model from OLLAMA_EMBED_MODEL (same as OllamaClient)."""
    return get_ollama_config().embed_model


# ---------------------------------------------------------------------------
# Intent splitting
# ---------------------------------------------------------------------------

# Pattern to split by comma when followed by a new question-like clause (avoids splitting dates like "January 1, 2025").
_SPLIT_COMMA_QUESTION = re.compile(
    r",\s+(?=what|how|get|show|list|compare|when|which|where|who|why|tell|give|check|find|return|describe|explain)",
    flags=re.I,
)


def _heuristic_split_multi_intent(query: str) -> List[str]:
    """
    Split a long comma- or and-separated query into clauses without using the LLM.
    Used when the LLM returns a single intent but the query clearly has multiple parts.
    Avoids splitting inside dates (e.g. "January 1, 2025" or "Q1, Q2 2025").
    """
    if not query or not query.strip():
        return []
    q = query.strip()
    # Split by comma when followed by question-style clause.
    parts = _SPLIT_COMMA_QUESTION.split(q)
    # Also split " X and what/where/how/..." into separate clauses.
    expanded = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Split "clause1 and clause2" when "and" is followed by question word.
        sub = re.split(r"\s+and\s+(?=what|how|get|show|list|compare|when|which|where|who|why|tell|give|check|find|return|describe|explain)", p, flags=re.I)
        for s in sub:
            s = s.strip().rstrip(",").strip()
            if s and len(s) > 2:
                expanded.append(s)
    # Merge back segment that is only a number (e.g. year from "January 1, 2025").
    result = []
    for i, seg in enumerate(expanded):
        if seg.isdigit() and len(seg) == 4 and result:
            result[-1] = f"{result[-1]}, {seg}"
        else:
            result.append(seg)
    return result


def _fallback_split(query: str) -> List[str]:
    """When LLM fails or returns one item, try heuristic split for long comma/and-separated queries."""
    q = (query or "").strip()
    if not q:
        return []
    # Do not over-gate by length: short multi-clause requests should still split.
    if "," in q or " and " in q.lower():
        heuristic = _heuristic_split_multi_intent(q)
        if len(heuristic) >= 2:
            return heuristic
    return [q]


def _intent_split_backend() -> str:
    """ollama (default) or deepseek via GATEWAY_INTENT_SPLIT_BACKEND."""
    return (os.getenv("GATEWAY_INTENT_SPLIT_BACKEND") or "ollama").strip().lower()


def split_intents(query: str) -> List[str]:
    """
    Split a rewritten query into distinct single-intent sub-questions.

    Backend: GATEWAY_INTENT_SPLIT_BACKEND ollama (default) or deepseek.
    Ollama uses OllamaClient (OLLAMA_* four vars); DeepSeek uses DeepSeekChat.
    When the LLM returns one item or fails, tries heuristic split for long comma/and-separated queries.

    Args:
        query: Normalized/rewritten query text (no splitting done yet).

    Returns:
        List of single-intent sub-question strings.
    """
    if not query or not query.strip():
        return []

    prompt_template = load_prompt("classification/intent_split_query")
    user_line = f"Input: {query.strip()}"
    text = ""

    if _intent_split_backend() == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
        try:
            text = DeepSeekChat().complete(
                prompt_template,
                user_line,
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning("Intent split DeepSeek failed: %s; heuristic fallback", exc)
            return _fallback_split(query)
    else:
        prompt = f"{prompt_template}\n\n{user_line}"
        try:
            text = OllamaClient().generate(prompt, empty_fallback="")
        except Exception as exc:
            logger.warning("Intent split LLM call failed: %s; trying heuristic fallback", exc)
            return _fallback_split(query)

    if not text:
        return _fallback_split(query)

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
                logger.warning("Intent split JSON parse failed; trying heuristic fallback")
                return _fallback_split(query)
        else:
            logger.warning("Intent split no JSON found; trying heuristic fallback")
            return _fallback_split(query)

    intents = parsed.get("intents")
    if not isinstance(intents, list) or not intents:
        return _fallback_split(query)

    # Deduplicate and filter blanks.
    seen: set = set()
    result = []
    for item in intents:
        if isinstance(item, str):
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)

    if not result:
        return _fallback_split(query)
    # When LLM returns only one intent but query looks multi-part, try heuristic split.
    if len(result) == 1 and ("," in query or " and " in query.lower()):
        heuristic = _heuristic_split_multi_intent(query)
        if len(heuristic) >= 2:
            logger.info("Intent split: LLM returned 1 item; using heuristic split (%d clauses)", len(heuristic))
            return heuristic
    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="intent_split_completed",
                stage="intent_classification",
                message="split_intents completed",
                status="success",
                workflow="intent_classification",
                query_raw=query,
                intent_list=result,
                metadata={"intent_count": len(result)},
            )
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Keyword intent matching (rule-based, no model)
# ---------------------------------------------------------------------------

_keyword_map_cache: Optional[List[Dict[str, str]]] = None


def _load_keyword_map() -> List[Dict[str, str]]:
    """Load keyword → intent mapping from keyword_intents.csv (lazy, cached)."""
    global _keyword_map_cache
    if _keyword_map_cache is not None:
        return _keyword_map_cache

    csv_path = Path(__file__).parent.parent.parent.parent / "prompts" / "retrieval" / "keyword_intents.csv"
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

    if not rows:
        rows.extend([
            {"keyword": "order status", "intent": "sp_api"},
            {"keyword": "get order", "intent": "sp_api"},
            {"keyword": "fba fees", "intent": "uds"},
            {"keyword": "which table", "intent": "uds"},
            {"keyword": "fee data", "intent": "uds"},
            {"keyword": "policy", "intent": "amazon_docs"},
            {"keyword": "storage fee", "intent": "amazon_docs"},
        ])

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
    """Embed a single query via OllamaClient /api/embed."""
    vecs = OllamaClient().embed(
        [query],
        model=_get_embedding_model(),
        timeout=10,
    )
    if not vecs or not vecs[0]:
        raise ValueError("Empty embedding response from Ollama")
    return vecs[0]


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


def get_keyword_vector_results(query: str) -> tuple[str, str]:
    """
    Return raw keyword and vector classification results for a single query.
    Used to display both retrieval results per sub-intent in the UI.

    Args:
        query: Sub-query text (one intent clause).

    Returns:
        (keyword_workflow, vector_workflow) e.g. ("amazon_docs", "amazon_docs").
        On vector failure, vector_workflow is "general".
    """
    if not query or not query.strip():
        return "general", "general"
    q = query.strip()
    keyword_workflow = _keyword_match_intent(q)
    try:
        vector_workflow, _, _ = _vector_match_intent(q)
    except Exception as exc:
        logger.debug("Vector match failed for '%s': %s", q[:50], exc)
        vector_workflow = "general"
    return keyword_workflow, vector_workflow


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

    result_obj = IntentResult(
        intent_name=vector_top1.get("intent_name", final_workflow) if vector_top1 else final_workflow,
        workflow=final_workflow,
        distance=vector_distance,
        confidence=confidence,
        required_fields=required_fields,
        clarification_template=clarification_template,
        source=source,
        vector_distance=vector_distance,
    )
    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="intent_classification_resolved",
                stage="intent_classification",
                message="classify_intent resolved",
                status="success",
                workflow=result_obj.workflow,
                query_raw=q,
                metadata={
                    "intent_name": result_obj.intent_name,
                    "source": result_obj.source,
                    "confidence": result_obj.confidence,
                    "distance": result_obj.distance,
                },
            )
        except Exception:
            pass
    return result_obj
