"""
Route LLM (Planning): routing and query rewriting for the gateway.

Provides:
- rewrite_query: normalize and optionally rewrite via LLM (Ollama/DeepSeek).
- route_workflow: single-task workflow classification (Route LLM or heuristic).

Task planning (build_execution_plan) moved to dispatcher module.

Refactored per code_development_refactor_by_workflow.md:
- Grouped by workflow into classes
- @classmethod for consistency
- Comprehensive logging
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


class RouterEnvConfig:
    """
    Validate and resolve router env parameters from os.environ.

    - route_llm_enabled: GATEWAY_ROUTE_LLM_ENABLED
    - route_llm_threshold: GATEWAY_ROUTE_LLM_THRESHOLD
    - memory_rounds: GATEWAY_REWRITE_MEMORY_ROUNDS
    - rewrite_backend: GATEWAY_REWRITE_BACKEND
    - route_backend: GATEWAY_ROUTE_BACKEND
    """

    @staticmethod
    def route_llm_enabled() -> bool:
        """Return True when Route LLM routing is enabled."""
        return os.getenv("GATEWAY_ROUTE_LLM_ENABLED", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def route_llm_threshold() -> float:
        """Read Route LLM confidence threshold."""
        try:
            return float(os.getenv("GATEWAY_ROUTE_LLM_THRESHOLD", "0.8"))
        except (ValueError, TypeError):
            logger.warning("GATEWAY_ROUTE_LLM_THRESHOLD invalid; using 0.8")
            return 0.8

    @staticmethod
    def get_memory_rounds() -> int:
        """Read GATEWAY_REWRITE_MEMORY_ROUNDS from env (default 3)."""
        try:
            val = max(1, int(os.getenv("GATEWAY_REWRITE_MEMORY_ROUNDS", "3")))
            return val
        except (ValueError, TypeError):
            logger.warning("GATEWAY_REWRITE_MEMORY_ROUNDS invalid; using 3")
            return 3

    @staticmethod
    def get_rewrite_backend(request_backend: Optional[str]) -> str:
        """Resolve rewrite backend from request or env."""
        backend = (request_backend or "").strip().lower()
        if backend:
            return backend
        return (os.getenv("GATEWAY_REWRITE_BACKEND", "ollama") or "ollama").strip().lower()

    @staticmethod
    def get_route_backend(request: QueryRequest) -> str:
        """Resolve route backend from request or env."""
        backend = (getattr(request, "route_backend", None) or "").strip().lower()
        if backend:
            return backend
        return (os.getenv("GATEWAY_ROUTE_BACKEND", "ollama") or "ollama").strip().lower()


class MemoryContextFormatter:
    """
    Format and merge conversation context for LLM rewrite.

    - format_history_for_llm: turn list -> formatted string
    - merge_conversation_context: deduplicate preloaded + memory context
    - count_context_rounds: count turns from formatted context
    """

    @classmethod
    def format_history_for_llm(cls, history: list) -> str:
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
                except Exception as exc:
                    logger.debug("Failed to parse turn event_content: %s", exc)
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

    @classmethod
    def merge_conversation_context(
        cls,
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

        merged_turns: List[Tuple[str, str]] = []
        seen_keys: set[str] = set()
        for block in (preloaded, memory):
            for query, answer in cls._parse_turns(block):
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

    @staticmethod
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

    @staticmethod
    def count_context_rounds(context: Optional[str]) -> int:
        """Best-effort count of conversation rounds from formatted context lines."""
        text = (context or "").strip()
        if not text:
            return 0
        return len([ln for ln in text.splitlines() if ln.strip()])


class _RewriteRouter:
    """
    Router workflow: rewrite_query and route_workflow.

    Assembles RouterEnvConfig, MemoryContextFormatter, and rewrite_with_context.
    """

    @classmethod
    def rewrite_query(
        cls,
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
        normalized = cls._normalize(request.query or "")
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
            logger.debug("rewrite_query: empty query, early exit")
            return ("", None, 0, 0)

        if not request.rewrite_enable:
            logger.debug("rewrite_query: rewrite disabled, returning normalized")
            return (normalized, None, 0, 0)

        backend = RouterEnvConfig.get_rewrite_backend(request.rewrite_backend)
        logger.info("rewrite_query: backend=%s session=%s", backend, request.session_id)

        # Step 1: Rewrite with memory merge -> optimized retrieval query.
        memory_rounds_used = 0
        memory_text_length = 0
        memory_context: Optional[str] = None
        if gateway_memory:
            last_n = RouterEnvConfig.get_memory_rounds()
            history: list = []
            if request.user_id and str(request.user_id).strip():
                history = gateway_memory.get_history_by_user(
                    str(request.user_id).strip(), last_n=last_n
                )
            elif request.session_id and str(request.session_id).strip():
                history = gateway_memory.get_history(request.session_id, last_n=last_n)
            if history:
                memory_context = MemoryContextFormatter.format_history_for_llm(history)
                memory_rounds_used = len(history)
                logger.debug("rewrite_query: loaded %d memory rounds", memory_rounds_used)

        conversation_context = MemoryContextFormatter.merge_conversation_context(
            conversation_context, memory_context
        )
        if conversation_context and conversation_context.strip():
            memory_text_length = len(conversation_context)
            if memory_rounds_used <= 0:
                memory_rounds_used = MemoryContextFormatter.count_context_rounds(
                    conversation_context
                )

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
            except Exception as exc:
                logger.warning("rewrite_query: intent classification failed: %s", exc)
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

        logger.debug("rewrite_query: done, query_len=%d", len(optimized_query))
        return (optimized_query, intents, memory_rounds_used, memory_text_length)

    @classmethod
    def route_workflow(
        cls,
        query: str,
        request: QueryRequest,
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
            logger.debug("route_workflow: manual workflow=%s", explicit)
            return explicit, 1.0, "manual", None, None

        backend = RouterEnvConfig.get_route_backend(request)
        if _route_llm_enabled():
            from src.gateway import route_llm as route_pkg

            llm_wf, llm_conf = route_pkg.route_with_llm(query or "", backend)
            llm_wf = _apply_docs_preference(query or "", llm_wf)
            if llm_conf >= _route_llm_threshold():
                logger.debug("route_workflow: llm workflow=%s conf=%.2f", llm_wf, llm_conf)
                return llm_wf, llm_conf, "llm", backend, llm_conf

        wf, conf = _route_workflow_heuristic(query or "")
        wf = _apply_docs_preference(query or "", wf)
        logger.debug("route_workflow: heuristic workflow=%s conf=%.2f", wf, conf)
        return wf, conf, "heuristic", None, None

    @staticmethod
    def _normalize(query: str) -> str:
        """Trim and collapse whitespace. Always applied before any rewrite."""
        return normalize_query(query or "")

    @staticmethod
    def _intent_classification_enabled() -> bool:
        """Return True when rewrite-stage intent classification is enabled."""
        return False

    @staticmethod
    def _rewrite_intents_only(query: str):
        """Classify rewritten query into intents; returns dict with intents or None."""
        try:
            from ..classification import split_intents

            intents = split_intents(query)
            if intents:
                return {"intents": intents}
        except Exception as exc:
            logger.debug("rewrite_intents_only failed: %s", exc)
            return None
        return None


# --- Backward-compatible public API ---

def rewrite_query(
    request: QueryRequest,
    gateway_memory: Optional["GatewayConversationMemory"] = None,
    conversation_context: Optional[str] = None,
) -> Tuple[str, Optional[List[str]], int, int]:
    """Normalize and rewrite query. Backward-compatible wrapper."""
    return _RewriteRouter.rewrite_query(
        request,
        gateway_memory=gateway_memory,
        conversation_context=conversation_context,
    )


def route_workflow(
    query: str, request: QueryRequest
) -> tuple[str, float, str, str | None, float | None]:
    """Choose workflow and routing confidence. Backward-compatible wrapper."""
    return _RewriteRouter.route_workflow(query, request)


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
    return _RewriteRouter._intent_classification_enabled()


def rewrite_intents_only(query: str):
    """Classify rewritten query into intents; returns dict with intents or None."""
    return _RewriteRouter._rewrite_intents_only(query)


def _route_llm_enabled() -> bool:
    """Return True when Route LLM routing is enabled. Backward-compatible."""
    return RouterEnvConfig.route_llm_enabled()


def _route_llm_threshold() -> float:
    """Read Route LLM confidence threshold. Backward-compatible."""
    return RouterEnvConfig.route_llm_threshold()


__all__ = [
    "rewrite_query",
    "route_workflow",
    "route_with_llm",
    "intent_classification_enabled",
    "rewrite_intents_only",
    "_route_llm_enabled",
]
