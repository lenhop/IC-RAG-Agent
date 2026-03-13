"""
ClickHouse client for gateway memory events.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
_clickhouse_connect = None


def _get_clickhouse_module():
    """Lazy-load clickhouse-connect."""
    global _clickhouse_connect
    if _clickhouse_connect is None:
        import clickhouse_connect as chc

        _clickhouse_connect = chc
    return _clickhouse_connect


class GatewayMemoryCHClient:
    """Writes gateway memory events to ClickHouse."""

    def __init__(self) -> None:
        self._host = os.getenv("LOGGER_CH_HOST", "localhost")
        self._port = int(os.getenv("LOGGER_CH_PORT", "8123"))
        self._user = os.getenv("LOGGER_CH_USER", "default")
        self._password = os.getenv("LOGGER_CH_PASSWORD", "")
        self._database = os.getenv("LOGGER_CH_DATABASE", "default")
        self._table = os.getenv("GATEWAY_MEMORY_CH_TABLE", "rag_agent_message_event")
        self._connect_timeout = int(os.getenv("LOGGER_CH_CONNECT_TIMEOUT", "5"))
        self._send_receive_timeout = int(os.getenv("LOGGER_CH_SEND_RECEIVE_TIMEOUT", "15"))
        self._client: Any = None

    def _connect(self) -> Any:
        """Return cached ClickHouse client."""
        if self._client is not None:
            return self._client
        chc = _get_clickhouse_module()
        self._client = chc.get_client(
            host=self._host,
            port=self._port,
            username=self._user,
            password=self._password,
            database=self._database,
            connect_timeout=self._connect_timeout,
            send_receive_timeout=self._send_receive_timeout,
        )
        return self._client

    def ensure_table(self) -> None:
        """Create gateway memory table if needed."""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self._table} (
            ts DateTime64(3, 'UTC'),
            user_id String,
            session_id String,
            request_id String,
            event_type String,
            event_content String,
            status String,
            note String
        )
        ENGINE = MergeTree
        PARTITION BY toYYYYMM(ts)
        ORDER BY (user_id, session_id, ts, request_id)
        """
        self._connect().command(ddl)

    def write_event(self, payload: Dict[str, Any]) -> bool:
        """Insert one memory event row."""
        self.ensure_table()
        rows: List[List[Any]] = [
            [
                payload.get("ts_utc", ""),
                payload.get("user_id", ""),
                payload.get("session_id", ""),
                payload.get("request_id", ""),
                payload.get("event_type", ""),
                payload.get("event_content", ""),
                payload.get("status", "ok"),
                payload.get("note", ""),
            ]
        ]
        self._connect().insert(
            self._table,
            rows,
            column_names=[
                "ts",
                "user_id",
                "session_id",
                "request_id",
                "event_type",
                "event_content",
                "status",
                "note",
            ],
        )
        return True

