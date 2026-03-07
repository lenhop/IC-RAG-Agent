"""
Rule-based routing and query rewriting for the gateway.

This module provides:
- rewrite_query: normalizes and optionally rewrites the query via LLM (Ollama/DeepSeek).
- route_workflow: chooses a workflow and confidence score (Route LLM when enabled,
  else heuristic keyword rules).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Tuple

from .schemas import QueryRequest

logger = logging.getLogger(__name__)

# Route LLM env: read at call time to avoid import-side side effects
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
    text = query or ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def rewrite_query(request: QueryRequest) -> str:
    """
    Normalize and optionally rewrite the incoming query before routing.

    Flow:
    1. Always normalize (trim, collapse whitespace).
    2. If rewrite_enable:
       - Resolve backend from request.rewrite_backend or GATEWAY_REWRITE_BACKEND (default "ollama").
       - If backend in ("ollama", "deepseek"): call rewriter; return result or normalized on failure.
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

    if backend == "ollama":
        from .rewriters import rewrite_with_ollama
        return rewrite_with_ollama(normalized)
    if backend == "deepseek":
        from .rewriters import rewrite_with_deepseek
        return rewrite_with_deepseek(normalized)

    return normalized


def _route_workflow_heuristic(query: str) -> Tuple[str, float]:
    """
    Rule-based workflow selection from query keywords.

    Used when workflow is "auto" and either Route LLM is disabled or
    LLM confidence is below threshold. Returns (workflow, confidence).
    """
    q_lower = (query or "").lower()

    # Amazon docs: product / seller / API documentation questions.
    if any(k in q_lower for k in ["amazon docs", "seller central", "sp-api docs", "aws docs"]):
        return "amazon_docs", 0.9

    # IC docs: internal project / framework documentation.
    if any(k in q_lower for k in ["ic-rag-agent", "ic docs", "framework.md", "project.md"]):
        return "ic_docs", 0.9

    # SP-API workflow: order, shipment, catalog, finance focused on Amazon API.
    if any(
        k in q_lower
        for k in [
            "sp-api",
            "spapi",
            "amazon order",
            "fba",
            "shipment",
            "catalog",
            "seller api",
        ]
    ):
        return "sp_api", 0.85

    # UDS workflow: analytical questions over structured data / tables.
    if any(
        k in q_lower
        for k in [
            "sales",
            "revenue",
            "orders",
            "table",
            "dataset",
            "clickhouse",
            "uds",
        ]
    ):
        return "uds", 0.85

    # Fallback: general LLM workflow (no keyword match).
    return "general", 0.7


def route_workflow(
    query: str, request: QueryRequest
) -> Tuple[str, float, str, Optional[str], Optional[float]]:
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
        return wf, conf, "heuristic", None, None

    # Route LLM path: call LLM; fall back to heuristic if confidence too low.
    from .route_llm import route_with_llm

    workflow, confidence = route_with_llm(query or "", backend)
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
    return wf2, conf2, "heuristic", None, None


__all__ = ["rewrite_query", "route_workflow"]

