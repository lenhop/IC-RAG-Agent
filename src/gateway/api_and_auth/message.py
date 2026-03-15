"""
Uniform gateway memory interface.

This module is the single gateway-facing API for short-term memory. Gateway code
(api.py, router, etc.) must use only this module for memory operations; do not
import or call short_term directly. Storage (Redis/ClickHouse) is an implementation
detail behind this interface.

Public entry points:
  - get_gateway_memory(): obtain the singleton memory instance (or None)
  - SessionHistoryHandler: session-scoped history (get, clear)
  - UserHistoryHandler: user-scoped history (get with normalization for display)
  - ContextHistoryHelper: raw history for LLM context (e.g. rewrite, clarification)
  - MemoryEventWriter: append memory events to Redis (+ optional ClickHouse)
  - TurnSummaryPersistence: append turn_summary event and save_turn in one call

All handler/writer methods accept memory as first parameter (from get_gateway_memory()).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from ..memory.short_term import GatewayConversationMemory, MemoryEvent

logger = logging.getLogger(__name__)

# Cached gateway memory instance; initialized once by get_gateway_memory().
_gateway_memory: Optional[GatewayConversationMemory] = None


def get_gateway_memory() -> Optional[GatewayConversationMemory]:
    """
    Return the singleton gateway short-term memory instance (Redis-backed).

    Initializes on first call; returns None if Redis is unreachable.
    Gateway code should obtain memory via this function and pass it to
    SessionHistoryHandler, UserHistoryHandler, ContextHistoryHelper,
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


class SessionHistoryHandler:
    """
    Session-scoped history operations.

    Handles get_history and clear for a given session_id.
    """

    @classmethod
    def get_history(
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
        history = memory.get_history(session_id, last_n=min(last_n, 50))
        return {"session_id": session_id, "history": history}

    @classmethod
    def clear(
        cls,
        memory: Optional[GatewayConversationMemory],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Clear session history.

        Args:
            memory: Gateway memory instance. None when disabled.
            session_id: Session identifier.

        Returns:
            Dict with session_id, cleared bool, and optional error key.
        """
        if not memory:
            logger.debug("Session clear skipped: gateway memory disabled")
            return {"session_id": session_id, "cleared": False, "error": "Gateway memory disabled"}
        memory.clear_session(session_id)
        return {"session_id": session_id, "cleared": True}


class UserHistoryHandler:
    """
    User-scoped history operations.

    Handles get_history with normalization of v1 turn_summary and v0 turn formats.
    """

    @staticmethod
    def _normalize_raw_history(raw: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Normalize v1 turn_summary and v0 turn formats to display format.

        v1: event_type=turn_summary, event_content is JSON string with query/answer/workflow.
        v0: direct query, answer, workflow, timestamp keys.
        """
        history: List[Dict[str, str]] = []
        for h in raw:
            if h.get("event_type") == "turn_summary":
                try:
                    content = json.loads(h.get("event_content", "{}"))
                except Exception:
                    content = {}
                history.append(
                    {
                        "query": str(content.get("query", "")),
                        "answer": str(content.get("answer", "")),
                        "workflow": str(content.get("workflow", "")),
                        "timestamp": str(h.get("ts_utc", "")),
                    }
                )
                continue
            history.append(
                {
                    "query": str(h.get("query", "")),
                    "answer": str(h.get("answer", "")),
                    "workflow": str(h.get("workflow", "")),
                    "timestamp": str(h.get("timestamp", "")),
                }
            )
        return history

    @classmethod
    def get_history(
        cls,
        memory: Optional[GatewayConversationMemory],
        user_id: str,
        last_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Return last N turns for a user, normalized for display.

        Args:
            memory: Gateway memory instance. None when disabled.
            user_id: User identifier.
            last_n: Max turns (clamped to [1, 50]).

        Returns:
            Dict with history list and optional error key.
        """
        if not memory:
            logger.debug("User history skipped: gateway memory disabled")
            return {"history": [], "error": "Gateway memory disabled"}
        last_n = min(max(1, last_n), 50)
        raw = memory.get_history_by_user(user_id, last_n=last_n)
        history = cls._normalize_raw_history(raw)
        return {"history": history}


class ContextHistoryHelper:
    """
    Raw history for LLM context (e.g. rewrite, clarification).

    Single gateway entry point for "raw history" used by MemoryContextFormatter.
    Prefer user-scoped history when user_id is present, else session-scoped.
    """

    @classmethod
    def get_raw(
        cls,
        memory: Optional[GatewayConversationMemory],
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        last_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return raw history list for LLM context. Same format as short_term get_history/get_history_by_user.

        Args:
            memory: Gateway memory instance. None when disabled.
            session_id: Optional session identifier (used when user_id absent).
            user_id: Optional user identifier (takes precedence over session_id).
            last_n: Max turns (clamped to [1, 50]).

        Returns:
            List of raw turn/event dicts, or [] when memory disabled or both ids empty.
        """
        if not memory:
            return []
        last_n = min(max(1, last_n), 50)
        uid = (user_id or "").strip()
        if uid:
            return memory.get_history_by_user(uid, last_n=last_n)
        sid = (session_id or "").strip()
        if sid:
            return memory.get_history(sid, last_n=last_n)
        return []


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
