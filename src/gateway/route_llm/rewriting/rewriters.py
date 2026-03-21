"""
Gateway rewrite facade: delegates implementation to rewrite_implement.

Public API: rewrite_and_route, split_intents, route_with_llm, RouterEnvConfig, _RewriteRouter.
Unified rewrite always produces rewritten_display + intents via one JSON LLM call.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ...schemas import QueryRequest

from .rewrite_implement import (
    RouterEnvConfig,
    UnifiedRewriteResult,
    _RewriteRouter,
    split_intents as split_intents_impl,
)


def split_intents(query: str, conversation_context: Optional[str] = None) -> List[str]:
    """Split query into sub-intents using the same unified prompt as the main rewrite stage."""
    return split_intents_impl(query, conversation_context)


def rewrite_and_route(
    request: QueryRequest,
    gateway_memory: Optional[Any] = None,
    conversation_context: Optional[str] = None,
    route_query: Optional[str] = None,
    enable_routing: bool = True,
    rewritten_query: Optional[str] = None,
    rewrite_intents: Optional[List[str]] = None,
    memory_rounds: int = 0,
    memory_text_length: int = 0,
) -> tuple[
    str,
    None,
    int,
    int,
    str | None,
    float | None,
    str | None,
    str | None,
    float | None,
    list[str],
]:
    """
    Rewrite (unified JSON) + optional workflow route.

    When rewritten_query is None, runs run_unified_rewrite. When pre-supplied, uses
    rewrite_intents if provided; otherwise single-element list from rewritten_query.

    Returns:
        Ten-element tuple ending with intents list for downstream classification/plan.
    """
    current_rewritten: str
    rounds_used: int
    text_length: int
    intents: list[str]

    if rewritten_query is None:
        result: UnifiedRewriteResult = _RewriteRouter.run_unified_rewrite(
            request,
            gateway_memory=gateway_memory,
            conversation_context=conversation_context,
        )
        current_rewritten = result.rewritten_display
        rounds_used = result.memory_rounds_used
        text_length = result.memory_text_length
        intents = list(result.intents)
    else:
        current_rewritten = rewritten_query
        rounds_used = memory_rounds
        text_length = memory_text_length
        if rewrite_intents is not None:
            intents = list(rewrite_intents)
        else:
            s = (current_rewritten or "").strip()
            intents = [s] if s else []

    if not enable_routing:
        return (
            current_rewritten,
            None,
            rounds_used,
            text_length,
            None,
            None,
            None,
            None,
            None,
            intents,
        )

    route_input = (route_query or current_rewritten or "").strip()
    workflow, routing_confidence, route_source, route_backend, route_llm_confidence = (
        _RewriteRouter.route_workflow(route_input, request)
    )

    return (
        current_rewritten,
        None,
        rounds_used,
        text_length,
        workflow,
        routing_confidence,
        route_source,
        route_backend,
        route_llm_confidence,
        intents,
    )


def route_with_llm(query: str, backend: str = "ollama") -> tuple[str, float]:
    """LLM routing hook; returns safe default."""
    _ = backend
    return "general", 0.0


# Re-export for dispatcher and tests.
build_merged_context_for_rewrite = _RewriteRouter.build_merged_context_for_rewrite

__all__ = [
    "RouterEnvConfig",
    "_RewriteRouter",
    "build_merged_context_for_rewrite",
    "rewrite_and_route",
    "route_with_llm",
    "split_intents",
]
