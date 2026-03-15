"""
Uniform gateway memory interface.

This module is the single gateway-facing API for short-term memory. Gateway code
(api.py, router, etc.) must use only this module for memory operations; do not
import or call short_term directly. Storage (Redis/ClickHouse) is an implementation
detail behind this interface.

Public entry points:
  - get_gateway_memory(): obtain the singleton memory instance (or None)
  - ConversationHistoryHandler: session history retrieval for API endpoints
  - MemoryEventWriter: append memory events to Redis (+ optional ClickHouse)
  - TurnSummaryPersistence: append turn_summary event and save_turn in one call

All handler/writer methods accept memory as first parameter (from get_gateway_memory()).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.memory.short_term import GatewayConversationMemory, MemoryEvent

logger = logging.getLogger(__name__)

# Cached gateway memory instance; initialized once by get_gateway_memory().
_gateway_memory: Optional[GatewayConversationMemory] = None


def get_gateway_memory() -> Optional[GatewayConversationMemory]:
    """
    Return the singleton gateway short-term memory instance (Redis-backed).

    Initializes on first call; returns None if Redis is unreachable.
    Gateway code should obtain memory via this function and pass it to
    ConversationHistoryHandler,
    MemoryEventWriter, TurnSummaryPersistence.
    """
    global _gateway_memory
    if _gateway_memory is not None:
        return _gateway_memory
    try:
        import redis
        redis_url = os.getenv("GATEWAY_REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        _gateway_memory = GatewayConversationMemory(_redis_client)
        logger.info(
            "Gateway memory initialized (Redis: %s)",
            redis_url.split("@")[-1] if "@" in redis_url else redis_url,
        )
        return _gateway_memory
    except Exception as exc:
        logger.warning("Gateway memory disabled (Redis unreachable): %s", exc)
        return None


class ConversationHistoryHandler:
    """
    Session-only history operations for API and rewrite context.

    Provides session history retrieval for API endpoints and LLM context.
    History is retrieved only by session_id (no user-scoped retrieval).
    """

    @classmethod
    def get_session_history(
        cls,
        memory: Optional[GatewayConversationMemory],
        session_id: str,
        last_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Return last N turns for a session.

        Args:
            memory: Gateway memory instance (Redis-backed). None when disabled.
            session_id: Session identifier.
            last_n: Max turns to return (capped at 50).

        Returns:
            Dict with session_id, history list, and optional error key.
        """
        if not memory:
            logger.debug("Session history skipped: gateway memory disabled")
            return {"session_id": session_id, "history": [], "error": "Gateway memory disabled"}
        history = memory.get_history_by_session(session_id, last_n=min(last_n, 50))
        return {"session_id": session_id, "history": history}


class MemoryEventWriter:
    """
    Memory event append operations.

    Best-effort write to Redis (+ optional ClickHouse). Never raises.
    """

    @classmethod
    def append_event(
        cls,
        memory: Optional[GatewayConversationMemory],
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        request_id: str,
        event_type: str,
        event_content: Any,
        status: str = "ok",
        note: str = "",
    ) -> None:
        """
        Append one memory event. No-op when memory disabled or user_id empty.

        Args:
            memory: Gateway memory instance. None when disabled.
            user_id: User identifier (required for write).
            session_id: Session identifier.
            request_id: Request identifier.
            event_type: Event type (e.g. user_query, turn_summary).
            event_content: Content dict or JSON string.
            status: Event status (ok, failed, skipped).
            note: Optional note.
        """
        if not memory or not user_id or not str(user_id).strip():
            return
        try:
            if isinstance(event_content, str):
                content = event_content
            else:
                content = json.dumps(event_content, ensure_ascii=False)
            memory.append_event(
                MemoryEvent(
                    user_id=str(user_id).strip(),
                    session_id=(session_id or "").strip(),
                    request_id=request_id,
                    event_type=event_type,  # type: ignore[arg-type]
                    event_content=content,
                    status=status,  # type: ignore[arg-type]
                    note=note,
                )
            )
            logger.debug("Memory event appended: event_type=%s request_id=%s", event_type, request_id)
        except Exception as exc:
            logger.warning("Gateway memory append_event failed (non-fatal): %s", exc)


class TurnSummaryPersistence:
    """
    Persist a completed turn: append turn_summary event and save_turn.

    Single place for the try/except so api.py does not repeat the block.
    """

    @classmethod
    def persist_turn(
        cls,
        memory: Optional[GatewayConversationMemory],
        *,
        user_id: Optional[str],
        session_id: str,
        request_id: str,
        query: str,
        answer: str,
        workflow: str,
    ) -> None:
        """
        Append turn_summary event and save_turn. No-op when memory or user_id missing.
        """
        if not memory or not user_id or not str(user_id).strip():
            return
        try:
            MemoryEventWriter.append_event(
                memory,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                event_type="turn_summary",
                event_content={"query": query, "answer": answer, "workflow": workflow},
                status="ok",
                note="v1 summary event",
            )
            memory.save_turn(
                session_id or "",
                query,
                answer,
                workflow,
                user_id=str(user_id).strip(),
            )
        except Exception as exc:
            logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
