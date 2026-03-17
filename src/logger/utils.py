"""Utility helpers for structured gateway logging.

This module centralizes formatting and helper functions related to
logging query-level metadata. The goal is to keep log entries consistent
and to make it easier to extend with structured logging later.

The helpers are intentionally lightweight to avoid adding runtime
overhead on the hot path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def format_route_metadata(
    route_source: str,
    route_backend: Optional[str],
    route_llm_confidence: Optional[float],
) -> str:
    """Return a compact string representing routing metadata.

    The returned snippet is intended to be appended to existing log
    messages so that legacy parsers continue working. Example:

        source=llm backend=ollama llm_confidence=0.82

    Args:
        route_source: "manual"|"llm"|"heuristic" indicating origin of
            the final routing decision.
        route_backend: LLM backend name when LLM was used, or None.
        route_llm_confidence: Confidence score when LLM was used, or None.
    """
    backend_str = route_backend if route_backend is not None else "none"
    llm_str = (
        f"{route_llm_confidence:.2f}"
        if route_llm_confidence is not None
        else "null"
    )
    return f"source={route_source} backend={backend_str} llm_confidence={llm_str}"


def enrich_log_fields(
    base: Dict[str, Any],
    route_source: str,
    route_backend: Optional[str],
    route_llm_confidence: Optional[float],
) -> Dict[str, Any]:
    """Return a new dict merging base fields with routing metadata.

    This helper can be used when logging with the ``extra`` parameter
    of the standard library logger to produce structured logs.
    """
    base_copy = base.copy()
    base_copy.update(
        {
            "route_source": route_source,
            "route_backend": route_backend,
            "route_llm_confidence": route_llm_confidence,
        }
    )
    return base_copy
