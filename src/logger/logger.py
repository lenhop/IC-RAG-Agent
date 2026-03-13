"""
Unified logger facade.

This class provides one stable API for writing interaction/runtime/error events
to both short-term Redis and long-term ClickHouse.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from typing import Any, Dict, List, Optional

from .base import redact_payload
from .ch_client import ClickHouseLogClient
from .models import ErrorLog, InteractionLog, RuntimeLog
from .redis_client import RedisLogClient
from .settings import LoggerSettings

logger = logging.getLogger(__name__)


class LoggerFacade:
    """Coordinates dual-write and read operations for logs."""

    def __init__(
        self,
        settings: LoggerSettings,
        redis_client: Optional[RedisLogClient] = None,
        clickhouse_client: Optional[ClickHouseLogClient] = None,
    ):
        self.settings = settings
        self.redis_client = redis_client
        self.clickhouse_client = clickhouse_client

    @classmethod
    def from_runtime(cls) -> "LoggerFacade":
        """Initialize facade from env/runtime dependencies."""
        settings = LoggerSettings.from_env()
        redis_client = None
        clickhouse_client = None

        if settings.redis_enabled:
            try:
                import redis

                redis_raw = redis.from_url(settings.redis_url, decode_responses=True)
                redis_raw.ping()
                redis_client = RedisLogClient(redis_raw, settings)
            except Exception as exc:
                logger.warning("Logger Redis client init failed: %s", exc)

        if settings.clickhouse_enabled:
            try:
                clickhouse_client = ClickHouseLogClient(settings)
                clickhouse_client.ensure_table()
            except Exception as exc:
                logger.warning("Logger ClickHouse client init failed: %s", exc)
                clickhouse_client = None

        return cls(settings=settings, redis_client=redis_client, clickhouse_client=clickhouse_client)

    def _prepare_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add storage-specific JSON fields and apply optional redaction."""
        output = dict(payload)

        # Pre-serialize nested fields for ClickHouse String columns.
        output["intent_list_json"] = json.dumps(output.get("intent_list") or [], ensure_ascii=False)
        output["intent_details_json"] = json.dumps(output.get("intent_details") or [], ensure_ascii=False)
        output["metadata_json"] = json.dumps(output.get("metadata") or {}, ensure_ascii=False)

        if self.settings.redaction_enabled:
            output = redact_payload(output, self.settings.redaction_fields)
        return output

    def _dual_write(
        self,
        payload: Dict[str, Any],
        *,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> Dict[str, bool]:
        """Write event to enabled sinks without raising into business flow."""
        if not self.settings.enabled:
            return {"redis": False, "clickhouse": False}

        prepared = self._prepare_payload(payload)
        result = {"redis": False, "clickhouse": False}

        if self.redis_client is not None:
            try:
                result["redis"] = self.redis_client.write_event(
                    prepared,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.warning("Logger Redis write failed: %s", exc)

        if self.clickhouse_client is not None:
            try:
                result["clickhouse"] = bool(self.clickhouse_client.write_event(prepared))
            except Exception as exc:
                logger.warning("Logger ClickHouse write failed: %s", exc)

        return result

    def log_interaction(self, **kwargs: Any) -> Dict[str, bool]:
        """Validate and persist interaction log."""
        evt = InteractionLog(**kwargs)
        payload = evt.to_storage_dict()
        return self._dual_write(
            payload,
            user_id=payload.get("user_id"),
            session_id=payload.get("session_id"),
        )

    def log_runtime(self, **kwargs: Any) -> Dict[str, bool]:
        """Validate and persist runtime log."""
        evt = RuntimeLog(**kwargs)
        payload = evt.to_storage_dict()
        return self._dual_write(
            payload,
            user_id=payload.get("user_id"),
            session_id=payload.get("session_id"),
        )

    def log_error(self, **kwargs: Any) -> Dict[str, bool]:
        """Validate and persist error log."""
        if not kwargs.get("stacktrace"):
            kwargs["stacktrace"] = traceback.format_exc()
        evt = ErrorLog(**kwargs)
        payload = evt.to_storage_dict()
        return self._dual_write(
            payload,
            user_id=payload.get("user_id"),
            session_id=payload.get("session_id"),
        )

    def read_short_term(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        last_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Read short-term events from Redis."""
        if self.redis_client is None:
            return []
        if user_id:
            return self.redis_client.read_recent_by_user(user_id, last_n=last_n)
        if session_id:
            return self.redis_client.read_recent_by_session(session_id, last_n=last_n)
        return []

    def read_long_term(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read long-term events from ClickHouse."""
        if self.clickhouse_client is None:
            return []
        try:
            return self.clickhouse_client.read_events(
                user_id=user_id,
                session_id=session_id,
                workflow=workflow,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("Logger ClickHouse read failed: %s", exc)
            return []

    def flush(self) -> bool:
        """Flush pending buffered writes if ClickHouse batching is enabled."""
        if self.clickhouse_client is None:
            return True
        try:
            return bool(self.clickhouse_client.flush())
        except Exception as exc:
            logger.warning("Logger flush failed: %s", exc)
            return False


_SINGLETON: Optional[LoggerFacade] = None
_LOCK = threading.Lock()


def get_logger_facade() -> LoggerFacade:
    """Get singleton logger facade."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = LoggerFacade.from_runtime()
    return _SINGLETON
