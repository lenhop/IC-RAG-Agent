"""
Route LLM (Planning): routing and query rewriting for the gateway.

Provides:
- rewrite_query: normalize and optionally rewrite via LLM (Ollama/DeepSeek).
- route_workflow: single-task workflow classification (Route LLM or heuristic).

Task planning (build_execution_plan) moved to dispatcher module.
"""

from __future__ import annotations

import logging
import os

from .rewriters import planner_rewrite_enabled, rewrite_intents_only
from .routing_heuristics import (
    apply_docs_preference as _apply_docs_preference,
    normalize_query,
    route_workflow_heuristic as _route_workflow_heuristic,
)
from .schemas import QueryRequest

logger = logging.getLogger(__name__)


def _route_llm_enabled() -> bool:
    v = os.getenv("GATEWAY_ROUTE_LLM_ENABLED", "false").strip().lower()
    return v in ("true", "1", "yes")


def _route_llm_backend() -> str:
    v = os.getenv("GATEWAY_ROUTE_LLM_BACKEND", "ollama").strip().lower()
    return v if v in ("ollama", "deepseek") else "ollama"


def _route_llm_conf_threshold() -> float:
    try:
        return float(os.getenv("GATEWAY_ROUTE_LLM_CONF_THRESHOLD", "0.7"))
    except (ValueError, TypeError):
        return 0.7


def _normalize(query: str) -> str:
    """Trim and collapse whitespace. Always applied before any rewrite."""
    return normalize_query(query or "")


def rewrite_query(request: QueryRequest) -> str:
    """
    Normalize and optionally rewrite the incoming query before routing.

    Flow:
    1. Always normalize (trim, collapse whitespace).
    2. If rewrite_enable:
       - Resolve backend from request.rewrite_backend or GATEWAY_REWRITE_BACKEND (default "ollama").
       - If planner enabled: Phase 1 intent classification via rewrite_intents_only.
         On success: return JSON {"intents": [...]} for build_execution_plan.
         On failure: return normalized query; build_execution_plan uses heuristic fallback.
       - Else if backend in ("ollama", "deepseek"): call rewriter; return result or normalized.
       - Else: return normalized query.
    3. Else: return normalized query.

    Args:
        request: Parsed QueryRequest from the client.

    Returns:
        Rewritten query string used for routing and downstream services.
    """
    normalized = _normalize(request.query or "")

    if not request.rewrite_enable:
        return normalized

    backend = (
        (request.rewrite_backend or "").strip().lower()
        or os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower()
    )

    # Two-phase flow: planner uses intent classification only (Phase 1).
    if planner_rewrite_enabled():
        import json
        result = rewrite_intents_only(normalized, backend=backend)
        if result and result.get("intents"):
            return json.dumps(result)
        # Fallback: return normalized; build_execution_plan uses heuristic multi-intent split.
        return normalized

    if backend == "ollama":
        from .rewriters import rewrite_with_ollama
        return rewrite_with_ollama(normalized)
    if backend == "deepseek":
        from .rewriters import rewrite_with_deepseek
        return rewrite_with_deepseek(normalized)

    return normalized


def route_workflow(
    query: str, request: QueryRequest
) -> tuple[str, float, str, str | None, float | None]:
    """
    Choose workflow and routing confidence based on query and request.

    Behavior:
    - If client sets workflow != "auto": return that workflow with confidence 1.0.
    - If workflow == "auto":
      - If GATEWAY_ROUTE_LLM_ENABLED is false: use heuristic keyword rules only.
      - If enabled: call Route LLM (backend from request.route_backend or env);
        if confidence >= GATEWAY_ROUTE_LLM_CONF_THRESHOLD use LLM result,
        else fall back to heuristic rules.

    Args:
        query: Rewritten query text from rewrite_query.
        request: Original QueryRequest (workflow, route_backend, etc.).

    Returns:
        (workflow, routing_confidence) tuple.
    """
    explicit = (request.workflow or "auto").strip().lower()
    if explicit != "auto":
        # explicit workflow is considered a manual override
        return explicit, 1.0, "manual", None, None

    # Auto routing: either Route LLM (when enabled and confident) or heuristic.
    backend = (request.route_backend or "").strip() or _route_llm_backend()
    threshold = _route_llm_conf_threshold()

    if not _route_llm_enabled():
        wf, conf = _route_workflow_heuristic(query or "")
        wf = _apply_docs_preference(query or "", wf)
        return wf, conf, "heuristic", None, None

    # Route LLM path: call LLM; fall back to heuristic if confidence too low.
    from .route_llm import route_with_llm

    workflow, confidence = route_with_llm(query or "", backend)
    workflow = _apply_docs_preference(query or "", workflow)
    if confidence >= threshold:
        logger.debug(
            "Route LLM selected workflow=%s confidence=%.2f (>= %.2f)",
            workflow,
            confidence,
            threshold,
        )
        return workflow, confidence, "llm", backend, confidence

    # Below threshold or LLM returned safe default: use heuristic.
    logger.debug(
        "Route LLM confidence %.2f < %.2f; using heuristic fallback",
        confidence,
        threshold,
    )
    wf2, conf2 = _route_workflow_heuristic(query or "")
    wf2 = _apply_docs_preference(query or "", wf2)
    return wf2, conf2, "heuristic", None, None


__all__ = ["rewrite_query", "route_workflow"]

