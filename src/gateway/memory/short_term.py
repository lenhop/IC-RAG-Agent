"""
Gateway short-term memory - Redis-backed session history.

Stores user query and LLM answer per user (and optionally per session).
Key formats:
  - gateway:user:{user_id}:history (user-scoped, preferred when user_id present)
  - gateway:session:{session_id}:history (session-only, backward compat)
TTL: 24h (configurable via GATEWAY_SESSION_TTL)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from datetime import timezone
from typing import Literal

EventType = Literal[
    "user_query",
    "query_clarification",
    "query_rewriting",
    "intent_classification",
    "llm_answer",
    "turn_summary",
]
EventStatus = Literal["ok", "failed", "skipped"]


class MemoryEvent(BaseModel):
    """Canonical memory event envelope."""

    ts_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str
    session_id: str = ""
    request_id: str = ""
    event_type: EventType
    event_content: str = ""
    status: EventStatus = "ok"
    note: str = ""

logger = logging.getLogger(__name__)

# Default TTL: 24 hours
DEFAULT_SESSION_TTL = 86400
# Max turns to keep per session (LTRIM to avoid unbounded growth)
MAX_TURNS_PER_SESSION = 50
MAX_EVENTS_PER_SESSION = 200


def _get_session_ttl() -> int:
    """Read TTL from env, default 86400."""
    try:
        return int(os.getenv("GATEWAY_SESSION_TTL", str(DEFAULT_SESSION_TTL)))
    except ValueError:
        return DEFAULT_SESSION_TTL


class GatewayConversationMemory:
    """
    Redis-backed conversation memory for the gateway.

    Stores query/answer pairs per session with configurable TTL.
    Uses Redis list (RPUSH) and optional LTRIM to cap at 50 turns.
    """

    def __init__(self, redis_client: Any) -> None:
        """
        Initialize with a Redis client (decode_responses=True recommended).

        Args:
            redis_client: redis.Redis instance from redis.from_url(...)
        """
        self._redis = redis_client
        self._ttl = _get_session_ttl()
        self._write_session_key = os.getenv(
            "GATEWAY_MEMORY_WRITE_SESSION_KEY",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self._ch_client: Any = None
        self._enable_ch = os.getenv(
            "GATEWAY_MEMORY_CH_ENABLED",
            "false",
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._enable_ch:
            try:
                from .long_term import GatewayMemoryCHClient

                self._ch_client = GatewayMemoryCHClient()
            except Exception as exc:
                logger.warning("Gateway memory ClickHouse disabled: %s", exc)
                self._ch_client = None

    def _key(self, session_id: str) -> str:
        """Build Redis key for session history (backward compat)."""
        return f"gateway:session:{session_id}:history"

    def _key_user(self, user_id: str) -> str:
        """Build Redis key for user-scoped history."""
        return f"gateway:user:{user_id}:history"

    def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        workflow: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        """
        Append one turn (query + answer) to history.

        When user_id is present, uses user-scoped key (gateway:user:{user_id}:history).
        When user_id is absent, skips (anonymous users; no persistence).

        Args:
            session_id: Session identifier (stored in turn for reference).
            query: User query text.
            answer: LLM answer text.
            workflow: Workflow used (e.g. uds, general, sp_api).
            user_id: User identifier; required for persistence.
        """
        if not user_id or not str(user_id).strip():
            return
        turn = {
            "query": query or "",
            "answer": answer or "",
            "workflow": workflow or "",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": str(user_id).strip(),
            "session_id": str(session_id or "").strip(),
        }
        key = self._key_user(str(user_id).strip())
        try:
            self._redis.rpush(key, json.dumps(turn))
            self._redis.expire(key, self._ttl)
            self._redis.ltrim(key, -MAX_TURNS_PER_SESSION, -1)
            if self._write_session_key and session_id and str(session_id).strip():
                session_key = self._key(str(session_id).strip())
                self._redis.rpush(session_key, json.dumps(turn))
                self._redis.expire(session_key, self._ttl)
                self._redis.ltrim(session_key, -MAX_TURNS_PER_SESSION, -1)
        except Exception as exc:
            logger.warning("Gateway memory save_turn failed: %s", exc)

    def append_event(self, event: Dict[str, Any] | MemoryEvent) -> None:
        """
        Append one v1 memory event to Redis and optional ClickHouse.

        Args:
            event: Event payload dict or validated MemoryEvent object.
        """
        try:
            payload = event.model_dump() if isinstance(event, MemoryEvent) else MemoryEvent(**event).model_dump()
        except Exception as exc:
            logger.warning("Gateway memory append_event validation failed: %s", exc)
            return

        user_id = (payload.get("user_id") or "").strip()
        if not user_id:
            return
        session_id = str(payload.get("session_id") or "").strip()
        serialized = json.dumps(payload)
        user_key = self._key_user(user_id)
        try:
            self._redis.rpush(user_key, serialized)
            self._redis.expire(user_key, self._ttl)
            self._redis.ltrim(user_key, -MAX_EVENTS_PER_SESSION, -1)
            if self._write_session_key and session_id:
                session_key = self._key(session_id)
                self._redis.rpush(session_key, serialized)
                self._redis.expire(session_key, self._ttl)
                self._redis.ltrim(session_key, -MAX_EVENTS_PER_SESSION, -1)
        except Exception as exc:
            logger.warning("Gateway memory append_event Redis write failed: %s", exc)

        if self._ch_client is not None:
            try:
                self._ch_client.write_event(payload)
            except Exception as exc:
                logger.warning("Gateway memory append_event ClickHouse write failed: %s", exc)

    def get_history(
        self,
        session_id: str,
        last_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve last N turns for a session.

        Args:
            session_id: Session identifier.
            last_n: Number of recent turns to return.

        Returns:
            List of turn dicts with query, answer, workflow, timestamp.
        """
        if not session_id or not str(session_id).strip():
            return []
        key = self._key(str(session_id).strip())
        try:
            raw = self._redis.lrange(key, -last_n, -1)
            return [json.loads(r) for r in raw] if raw else []
        except Exception as exc:
            logger.warning("Gateway memory get_history failed: %s", exc)
            return []

    def get_history_by_user(
        self,
        user_id: str,
        last_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve last N turns for a user.

        Args:
            user_id: User identifier.
            last_n: Number of recent turns to return.

        Returns:
            List of turn dicts with query, answer, workflow, timestamp, user_id, session_id.
        """
        if not user_id or not str(user_id).strip():
            return []
        key = self._key_user(str(user_id).strip())
        try:
            raw = self._redis.lrange(key, -last_n, -1)
            return [json.loads(r) for r in raw] if raw else []
        except Exception as exc:
            logger.warning("Gateway memory get_history_by_user failed: %s", exc)
            return []

    def clear_user_history(self, user_id: str) -> None:
        """
        Delete all history for a user.

        Args:
            user_id: User identifier.
        """
        if not user_id or not str(user_id).strip():
            return
        try:
            self._redis.delete(self._key_user(str(user_id).strip()))
        except Exception as exc:
            logger.warning("Gateway memory clear_user_history failed: %s", exc)

    def clear_session(self, session_id: str) -> None:
        """
        Delete all history for a session (backward compat).

        Args:
            session_id: Session identifier.
        """
        if not session_id or not str(session_id).strip():
            return
        try:
            self._redis.delete(self._key(str(session_id).strip()))
        except Exception as exc:
            logger.warning("Gateway memory clear_session failed: %s", exc)
