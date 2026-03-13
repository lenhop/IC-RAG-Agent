"""
Route LLM (Planning): routing and query rewriting for the gateway.

Provides:
- rewrite_query: normalize and optionally rewrite via LLM (Ollama/DeepSeek).
- route_workflow: single-task workflow classification (Route LLM or heuristic).

Task planning (build_execution_plan) moved to dispatcher module.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from .rewriters import rewrite_with_context
from ..routing_heuristics import (
    apply_docs_preference as _apply_docs_preference,
    normalize_query,
    route_workflow_heuristic as _route_workflow_heuristic,
)
from ...schemas import QueryRequest
from src.logger import get_logger_facade

if TYPE_CHECKING:
    from ...memory.short_term import GatewayConversationMemory

logger = logging.getLogger(__name__)

_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


def _route_llm_enabled() -> bool:
    """Return True when Route LLM routing is enabled."""
    return os.getenv("GATEWAY_ROUTE_LLM_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _route_llm_threshold() -> float:
    """Read Route LLM confidence threshold."""
    try:
        return float(os.getenv("GATEWAY_ROUTE_LLM_THRESHOLD", "0.8"))
    except (ValueError, TypeError):
        return 0.8


def route_with_llm(query: str, backend: str = "ollama") -> tuple[str, float]:
    """
    Placeholder Route LLM routing hook.

    Returns a safe default so heuristic routing remains authoritative unless
    tests/mock patches override this function.
    """
    _ = backend
    return "general", 0.0


def intent_classification_enabled() -> bool:
    """Return True when rewrite-stage intent classification is enabled."""
    return False


def rewrite_intents_only(query: str):
    """Classify rewritten query into intents; returns dict with intents or None."""
    try:
        from ..classification import split_intents

        intents = split_intents(query)
        if intents:
            return {"intents": intents}
    except Exception:
        return None
    return None


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
    normalized_turns: List[dict] = []
    for turn in history:
        # v1 event: consume only turn_summary events for LLM context.
        if "event_type" in turn:
            if (turn.get("event_type") or "").strip() != "turn_summary":
                continue
            raw_content = turn.get("event_content")
            try:
                content = json.loads(raw_content) if isinstance(raw_content, str) else (raw_content or {})
            except Exception:
                content = {}
            if not isinstance(content, dict):
                content = {}
            normalized_turns.append(
                {
                    "query": content.get("query", ""),
                    "answer": content.get("answer", ""),
                }
            )
            continue

        # v0 legacy turn
        normalized_turns.append(
            {
                "query": turn.get("query", ""),
                "answer": turn.get("answer", ""),
            }
        )

    for idx, turn in enumerate(normalized_turns, start=1):
        q = (turn.get("query") or "").strip()
        a = (turn.get("answer") or "").strip()
        if not q:
            continue
        lines.append(f'Turn {idx}: User asked "{q}" -> Answer: "{a}"')
    return "\n".join(lines) if lines else ""


def _merge_conversation_context(
    preloaded_context: Optional[str],
    memory_context: Optional[str],
) -> Optional[str]:
    """
    Merge caller-provided context with Redis memory context (deduplicated).

    This ensures rewrite always benefits from recent conversation history even
    when an upstream caller already passed a partial context payload.
    """
    preloaded = (preloaded_context or "").strip()
    memory = (memory_context or "").strip()
    if not preloaded and not memory:
        return None
    if not preloaded:
        return memory
    if not memory:
        return preloaded

    def _parse_turns(block: str) -> List[Tuple[str, str]]:
        """
        Parse context lines to (query, answer) pairs.

        Expected line format:
            Turn N: User asked "..." -> Answer: "..."
        """
        turns: List[Tuple[str, str]] = []
        pattern = re.compile(r'^Turn\s+\d+:\s+User asked\s+"(.*)"\s+->\s+Answer:\s+"(.*)"$')
        for raw_line in block.splitlines():
            line = (raw_line or "").strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                q = (m.group(1) or "").strip()
                a = (m.group(2) or "").strip()
                if q:
                    turns.append((q, a))
                continue
            # Best-effort fallback for non-standard lines.
            turns.append((line, ""))
        return turns

    merged_turns: List[Tuple[str, str]] = []
    seen_keys: set[str] = set()
    for block in (preloaded, memory):
        for query, answer in _parse_turns(block):
            key = f"{query.lower()}::{answer.lower()}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_turns.append((query, answer))

    if not merged_turns:
        return None
    normalized_lines = [
        f'Turn {idx}: User asked "{q}" -> Answer: "{a}"'
        for idx, (q, a) in enumerate(merged_turns, start=1)
    ]
    return "\n".join(normalized_lines)


def _count_context_rounds(context: Optional[str]) -> int:
    """Best-effort count of conversation rounds from formatted context lines."""
    text = (context or "").strip()
    if not text:
        return 0
    return len([ln for ln in text.splitlines() if ln.strip()])


def rewrite_query(
    request: QueryRequest,
    gateway_memory: Optional["GatewayConversationMemory"] = None,
    conversation_context: Optional[str] = None,
) -> Tuple[str, Optional[List[str]], int, int]:
    """
    Normalize and rewrite query with optional conversation context.

    Pipeline: Normalize -> Rewrite (memory merge + optimized retrieval query).

    Flow:
    1. Normalize (trim, collapse whitespace). Early exit if empty -> ("", None, 0, 0).
    2. If rewrite_enable:
       - Load conversation context from Redis if not pre-loaded.
       - Call rewrite_with_context -> optimized retrieval query.
    3. Else: return (normalized, None, 0, 0).

    Returns:
        (rewritten_query, None, memory_rounds_used, memory_text_length) tuple.
    """
    normalized = _normalize(request.query or "")
    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="router_rewrite_start",
                stage="router_rewrite",
                message="rewrite_query started",
                status="started",
                session_id=request.session_id,
                user_id=request.user_id,
                workflow=request.workflow,
                query_raw=request.query or "",
                query_rewritten=normalized,
            )
        except Exception:
            pass

    if not normalized or not normalized.strip():
        return ("", None, 0, 0)

    if not request.rewrite_enable:
        return (normalized, None, 0, 0)

    backend = (
        (request.rewrite_backend or "").strip().lower()
        or os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower()
    )

    # Step 1: Rewrite with memory merge -> optimized retrieval query.
    # Always merge pre-loaded context with Redis memory when available.
    memory_rounds_used = 0
    memory_text_length = 0
    memory_context: Optional[str] = None
    if gateway_memory:
        last_n = _get_memory_rounds()
        history: list = []
        if request.user_id and str(request.user_id).strip():
            history = gateway_memory.get_history_by_user(str(request.user_id).strip(), last_n=last_n)
        elif request.session_id and str(request.session_id).strip():
            history = gateway_memory.get_history(request.session_id, last_n=last_n)
        if history:
            memory_context = _format_history_for_llm(history)
            memory_rounds_used = len(history)
    conversation_context = _merge_conversation_context(conversation_context, memory_context)
    if conversation_context and conversation_context.strip():
        memory_text_length = len(conversation_context)
        if memory_rounds_used <= 0:
            memory_rounds_used = _count_context_rounds(conversation_context)

    optimized_query = rewrite_with_context(
        normalized,
        conversation_context=conversation_context,
        backend=backend,
    )

    intents: Optional[List[str]] = None
    if intent_classification_enabled():
        try:
            intents_payload = rewrite_intents_only(optimized_query)
            if isinstance(intents_payload, dict):
                intents_raw = intents_payload.get("intents")
            else:
                intents_raw = intents_payload
            if isinstance(intents_raw, list):
                intents = [str(item).strip() for item in intents_raw if str(item).strip()] or None
        except Exception:
            intents = None

    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="router_rewrite_done",
                stage="router_rewrite",
                message="rewrite_query completed",
                status="success",
                session_id=request.session_id,
                user_id=request.user_id,
                workflow=request.workflow,
                query_raw=request.query or "",
                query_rewritten=optimized_query,
                latency_ms=None,
                metadata={
                    "memory_rounds_used": memory_rounds_used,
                    "memory_text_length": memory_text_length,
                },
            )
        except Exception:
            pass
    return (optimized_query, intents, memory_rounds_used, memory_text_length)


def route_workflow(
    query: str, request: QueryRequest
) -> tuple[str, float, str, str | None, float | None]:
    """
    Choose workflow and routing confidence based on query and request.

    - If client sets workflow != "auto": return that workflow with confidence 1.0.
    - Otherwise: prefer Route LLM when enabled and confident, fallback to heuristic.

    Args:
        query: Rewritten query text from rewrite_query.
        request: Original QueryRequest (workflow field).

    Returns:
        (workflow, confidence, method, backend, llm_confidence) tuple.
    """
    explicit = (request.workflow or "auto").strip().lower()
    if explicit != "auto":
        return explicit, 1.0, "manual", None, None

    backend = (getattr(request, "route_backend", None) or os.getenv("GATEWAY_ROUTE_BACKEND", "ollama")).strip().lower() or "ollama"
    if _route_llm_enabled():
        from src.gateway import route_llm as route_pkg

        llm_wf, llm_conf = route_pkg.route_with_llm(query or "", backend)
        llm_wf = _apply_docs_preference(query or "", llm_wf)
        if llm_conf >= _route_llm_threshold():
            return llm_wf, llm_conf, "llm", backend, llm_conf

    wf, conf = _route_workflow_heuristic(query or "")
    wf = _apply_docs_preference(query or "", wf)
    return wf, conf, "heuristic", None, None


__all__ = [
    "rewrite_query",
    "route_workflow",
    "route_with_llm",
    "intent_classification_enabled",
    "rewrite_intents_only",
    "_route_llm_enabled",
]

