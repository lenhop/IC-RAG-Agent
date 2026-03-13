"""
Redis-backed short-term logger storage.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import safe_json_dumps, with_retry
from .settings import LoggerSettings

logger = logging.getLogger(__name__)


def _key_user(user_id: str) -> str:
    return f"logger:user:{user_id}:events"


def _key_session(session_id: str) -> str:
    return f"logger:session:{session_id}:events"


class RedisLogClient:
    """Stores short-term logs in Redis lists."""

    def __init__(self, redis_client: Any, settings: LoggerSettings):
        self._redis = redis_client
        self._settings = settings

    def _write_to_key(self, key: str, payload: Dict[str, Any]) -> None:
        line = safe_json_dumps(payload)

        def _op() -> None:
            self._redis.rpush(key, line)
            self._redis.ltrim(key, -self._settings.redis_max_events_per_key, -1)
            self._redis.expire(key, self._settings.redis_ttl_seconds)

        with_retry(
            _op,
            enabled=self._settings.retry_enabled,
            attempts=self._settings.retry_attempts,
            backoff_ms=self._settings.retry_backoff_ms,
            operation_name="redis_write_event",
        )

    def write_event(
        self,
        payload: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Write event to user/session Redis keys when identifiers are provided."""
        wrote_any = False
        if user_id and str(user_id).strip():
            self._write_to_key(_key_user(str(user_id).strip()), payload)
            wrote_any = True
        if session_id and str(session_id).strip():
            self._write_to_key(_key_session(str(session_id).strip()), payload)
            wrote_any = True
        return wrote_any

    def _read_key(self, key: str, last_n: int) -> List[Dict[str, Any]]:
        n = max(1, int(last_n))

        def _op() -> List[str]:
            return self._redis.lrange(key, -n, -1)

        raw_rows = with_retry(
            _op,
            enabled=self._settings.retry_enabled,
            attempts=self._settings.retry_attempts,
            backoff_ms=self._settings.retry_backoff_ms,
            operation_name="redis_read_events",
        )
        output: List[Dict[str, Any]] = []
        for row in raw_rows or []:
            try:
                parsed = json.loads(row)
            except Exception:
                parsed = {"raw": row}
            if isinstance(parsed, dict):
                output.append(parsed)
        return output

    def read_recent_by_user(self, user_id: str, last_n: int = 20) -> List[Dict[str, Any]]:
        """Read most recent events for a user-scoped key."""
        if not user_id or not str(user_id).strip():
            return []
        return self._read_key(_key_user(str(user_id).strip()), last_n)

    def read_recent_by_session(self, session_id: str, last_n: int = 20) -> List[Dict[str, Any]]:
        """Read most recent events for a session-scoped key."""
        if not session_id or not str(session_id).strip():
            return []
        return self._read_key(_key_session(str(session_id).strip()), last_n)
