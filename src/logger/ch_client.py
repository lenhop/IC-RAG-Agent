"""
ClickHouse-backed long-term logger storage.

Uses lazy import so runtime does not fail when clickhouse-connect is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import with_retry
from .settings import LoggerSettings

logger = logging.getLogger(__name__)
_clickhouse_connect = None


def _get_clickhouse_module():
    """Lazy load clickhouse-connect."""
    global _clickhouse_connect
    if _clickhouse_connect is None:
        import clickhouse_connect as chc

        _clickhouse_connect = chc
    return _clickhouse_connect


class ClickHouseLogClient:
    """Stores long-term logs in ClickHouse."""

    def __init__(self, settings: LoggerSettings):
        self._settings = settings
        self._client = None
        self._buffer: List[Dict[str, Any]] = []

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client

        chc = _get_clickhouse_module()
        self._client = chc.get_client(
            host=self._settings.clickhouse_host,
            port=self._settings.clickhouse_port,
            username=self._settings.clickhouse_user,
            password=self._settings.clickhouse_password,
            database=self._settings.clickhouse_database,
            connect_timeout=self._settings.clickhouse_connect_timeout,
            send_receive_timeout=self._settings.clickhouse_send_receive_timeout,
        )
        return self._client

    def ensure_table(self) -> None:
        """Create logger table when missing."""
        table = self._settings.clickhouse_table
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ts DateTime64(3, 'UTC'),
            event_kind String,
            event_name String,
            status String,
            request_id String,
            session_id String,
            user_id String,
            workflow String,
            query_raw String,
            query_rewritten String,
            clarification_question String,
            answer String,
            intent_list String,
            intent_details String,
            error_type String,
            error_message String,
            metadata String,
            latency_ms Int32
        ) ENGINE = MergeTree
        ORDER BY (ts, request_id)
        """
        client = self._connect()
        client.command(ddl)

    def _to_row(self, payload: Dict[str, Any]) -> List[Any]:
        """Map event dict to stable column order for insert."""
        return [
            payload.get("ts", ""),
            payload.get("event_kind", ""),
            payload.get("event_name", ""),
            payload.get("status", ""),
            payload.get("request_id", "") or "",
            payload.get("session_id", "") or "",
            payload.get("user_id", "") or "",
            payload.get("workflow", "") or "",
            payload.get("query_raw", "") or "",
            payload.get("query_rewritten", "") or "",
            payload.get("clarification_question", "") or "",
            payload.get("answer", "") or "",
            payload.get("intent_list_json", "[]"),
            payload.get("intent_details_json", "[]"),
            payload.get("error_type", "") or "",
            payload.get("error_message", "") or "",
            payload.get("metadata_json", "{}"),
            int(payload.get("latency_ms", 0) or 0),
        ]

    def _flush_rows(self, rows: List[List[Any]]) -> None:
        if not rows:
            return
        client = self._connect()
        table = self._settings.clickhouse_table
        client.insert(
            table,
            rows,
            column_names=[
                "ts",
                "event_kind",
                "event_name",
                "status",
                "request_id",
                "session_id",
                "user_id",
                "workflow",
                "query_raw",
                "query_rewritten",
                "clarification_question",
                "answer",
                "intent_list",
                "intent_details",
                "error_type",
                "error_message",
                "metadata",
                "latency_ms",
            ],
        )

    def write_event(self, payload: Dict[str, Any]) -> bool:
        """Write one event, optionally batching before flush."""
        self.ensure_table()
        if self._settings.clickhouse_batch_enabled:
            self._buffer.append(payload)
            if len(self._buffer) < self._settings.clickhouse_batch_size:
                return True
            batch = self._buffer[:]
            self._buffer.clear()
            rows = [self._to_row(item) for item in batch]
        else:
            rows = [self._to_row(payload)]

        with_retry(
            lambda: self._flush_rows(rows),
            enabled=self._settings.retry_enabled,
            attempts=self._settings.retry_attempts,
            backoff_ms=self._settings.retry_backoff_ms,
            operation_name="clickhouse_write_event",
        )
        return True

    def flush(self) -> bool:
        """Flush pending buffered events when batching is enabled."""
        if not self._buffer:
            return True
        batch = self._buffer[:]
        self._buffer.clear()
        rows = [self._to_row(item) for item in batch]
        with_retry(
            lambda: self._flush_rows(rows),
            enabled=self._settings.retry_enabled,
            attempts=self._settings.retry_attempts,
            backoff_ms=self._settings.retry_backoff_ms,
            operation_name="clickhouse_flush",
        )
        return True

    def read_events(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read long-term events with simple optional filters."""
        self.ensure_table()
        clauses: List[str] = []
        params: Dict[str, Any] = {"limit": max(1, int(limit))}
        if user_id:
            clauses.append("user_id = %(user_id)s")
            params["user_id"] = str(user_id)
        if session_id:
            clauses.append("session_id = %(session_id)s")
            params["session_id"] = str(session_id)
        if workflow:
            clauses.append("workflow = %(workflow)s")
            params["workflow"] = str(workflow)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM {self._settings.clickhouse_table} "
            f"{where_sql} ORDER BY ts DESC LIMIT %(limit)s"
        )
        client = self._connect()
        result = client.query(sql, parameters=params)
        rows = result.result_rows or []
        cols = result.column_names or []
        return [dict(zip(cols, row)) for row in rows]
