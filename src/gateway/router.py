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
from typing import TYPE_CHECKING, List, Optional, Tuple

from .rewriters import (
    intent_classification_enabled,
    rewrite_intents_only,
    rewrite_with_context,
)
from .routing_heuristics import (
    apply_docs_preference as _apply_docs_preference,
    normalize_query,
    route_workflow_heuristic as _route_workflow_heuristic,
)
from .schemas import QueryRequest

if TYPE_CHECKING:
    from .memory import GatewayConversationMemory

logger = logging.getLogger(__name__)


def _normalize(query: str) -> str:
    """Trim and collapse whitespace. Always applied before any rewrite."""
    return normalize_query(query or "")


def _get_memory_rounds() -> int:
    """Read GATEWAY_REWRITE_MEMORY_ROUNDS from env (default 3)."""
    try:
        return max(1, int(os.getenv("GATEWAY_REWRITE_MEMORY_ROUNDS", "3")))
    except (ValueError, TypeError):
        return 3


def _format_history_for_llm(history: list) -> str:
    """
    Format session history for LLM prompt (oldest first, chronological order).

    Args:
        history: List of turn dicts with query, answer, workflow, timestamp.

    Returns:
        Formatted string like "Turn 1: User asked "..." -> Answer: "...""
    """
    lines = []
    for idx, turn in enumerate(history, start=1):
        q = (turn.get("query") or "").strip()
        a = (turn.get("answer") or "").strip()
        if not q:
            continue
        lines.append(f'Turn {idx}: User asked "{q}" -> Answer: "{a}"')
    return "\n".join(lines) if lines else ""


def rewrite_query(
    request: QueryRequest,
    gateway_memory: Optional["GatewayConversationMemory"] = None,
    conversation_context: Optional[str] = None,
) -> Tuple[str, Optional[List[str]], int, int]:
    """
    Normalize, rewrite (with memory merge), and optionally run intent classification.

    Pipeline: Normalize -> Rewrite (memory merge + optimized retrieval query) -> Intent
    classification (on optimized query, when enabled).

    Flow:
    1. Normalize (trim, collapse whitespace). Early exit if empty -> ("", None, 0).
    2. If rewrite_enable:
       - If conversation_context not provided, load from Redis.
       - Call rewrite_with_context -> optimized retrieval query.
       - If intent_classification_enabled: run rewrite_intents_only on optimized query.
    3. Else: return (normalized, None, 0).

    Args:
        request: Parsed QueryRequest from the client.
        gateway_memory: Optional Redis-backed session memory for conversation context.
        conversation_context: Optional pre-loaded conversation context string.
            When provided, skips Redis loading (caller already loaded it).

    Returns:
        (rewritten_query, intents, memory_rounds_used, memory_text_length) tuple.
        memory_rounds_used: number of history turns merged (0 when no session/history).
    """
    normalized = _normalize(request.query or "")

    if not normalized or not normalized.strip():
        return ("", None, 0, 0)

    if not request.rewrite_enable:
        return (normalized, None, 0, 0)

    backend = (
        (request.rewrite_backend or "").strip().lower()
        or os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower()
    )

    # Step 1: Rewrite with memory merge -> optimized retrieval query.
    # Use pre-loaded context if provided; otherwise load from Redis.
    memory_rounds_used = 0
    memory_text_length = 0
    if conversation_context and conversation_context.strip():
        # Caller already loaded context — use it directly.
        memory_rounds_used = conversation_context.strip().count("\n") + 1
        memory_text_length = len(conversation_context)
    elif gateway_memory:
        last_n = _get_memory_rounds()
        history: list = []
        if request.user_id and str(request.user_id).strip():
            history = gateway_memory.get_history_by_user(str(request.user_id).strip(), last_n=last_n)
        elif request.session_id and str(request.session_id).strip():
            history = gateway_memory.get_history(request.session_id, last_n=last_n)
        if history:
            conversation_context = _format_history_for_llm(history)
            memory_rounds_used = len(history)
            memory_text_length = len(conversation_context or "")

    optimized_query = rewrite_with_context(
        normalized,
        conversation_context=conversation_context,
        backend=backend,
    )

    # Step 2: Intent classification on optimized retrieval query (when enabled).
    intents = None
    if intent_classification_enabled():
        result = rewrite_intents_only(optimized_query, backend=backend, conversation_context=conversation_context)
        if result and result.get("intents"):
            intents = result["intents"]

    return (optimized_query, intents, memory_rounds_used, memory_text_length)


def route_workflow(
    query: str, request: QueryRequest
) -> tuple[str, float, str, str | None, float | None]:
    """
    Choose workflow and routing confidence based on query and request.

    - If client sets workflow != "auto": return that workflow with confidence 1.0.
    - Otherwise: use heuristic keyword rules.

    Args:
        query: Rewritten query text from rewrite_query.
        request: Original QueryRequest (workflow field).

    Returns:
        (workflow, confidence, method, backend, llm_confidence) tuple.
    """
    explicit = (request.workflow or "auto").strip().lower()
    if explicit != "auto":
        return explicit, 1.0, "manual", None, None

    wf, conf = _route_workflow_heuristic(query or "")
    wf = _apply_docs_preference(query or "", wf)
    return wf, conf, "heuristic", None, None


__all__ = ["rewrite_query", "route_workflow"]

