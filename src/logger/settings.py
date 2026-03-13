"""
Logger system settings.

Centralizes logger-related runtime configuration so Redis/ClickHouse clients
and the facade share one consistent source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _as_bool(value: str, default: bool = False) -> bool:
    """Convert a string env value to bool safely."""
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: str, default: int, min_value: int = 0) -> int:
    """Convert env value to bounded integer with fallback."""
    try:
        parsed = int(value)
        return max(min_value, parsed)
    except (TypeError, ValueError):
        return default


def _csv_env(value: str) -> Tuple[str, ...]:
    """Split comma-separated env values and normalize whitespace."""
    if not value or not str(value).strip():
        return tuple()
    parts = [item.strip() for item in str(value).split(",")]
    return tuple(item for item in parts if item)


@dataclass(frozen=True)
class LoggerSettings:
    """Runtime settings for logger subsystem."""

    enabled: bool
    redis_enabled: bool
    clickhouse_enabled: bool

    redis_url: str
    redis_ttl_seconds: int
    redis_max_events_per_key: int

    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_database: str
    clickhouse_table: str
    clickhouse_connect_timeout: int
    clickhouse_send_receive_timeout: int
    clickhouse_batch_enabled: bool
    clickhouse_batch_size: int

    retry_enabled: bool
    retry_attempts: int
    retry_backoff_ms: int

    redaction_enabled: bool
    redaction_fields: Tuple[str, ...]

    @classmethod
    def from_env(cls) -> "LoggerSettings":
        """Build settings from environment variables."""
        return cls(
            enabled=_as_bool(os.getenv("LOGGER_ENABLED", "true"), default=True),
            redis_enabled=_as_bool(os.getenv("LOGGER_REDIS_ENABLED", "true"), default=True),
            clickhouse_enabled=_as_bool(os.getenv("LOGGER_CLICKHOUSE_ENABLED", "true"), default=True),
            redis_url=os.getenv("LOGGER_REDIS_URL", os.getenv("GATEWAY_REDIS_URL", "redis://localhost:6379/0")),
            redis_ttl_seconds=_as_int(os.getenv("LOGGER_REDIS_TTL_SECONDS", "86400"), default=86400, min_value=1),
            redis_max_events_per_key=_as_int(
                os.getenv("LOGGER_REDIS_MAX_EVENTS_PER_KEY", "500"),
                default=500,
                min_value=10,
            ),
            clickhouse_host=os.getenv("LOGGER_CH_HOST", os.getenv("CH_HOST", "localhost")),
            clickhouse_port=_as_int(os.getenv("LOGGER_CH_PORT", os.getenv("CH_PORT", "8123")), default=8123, min_value=1),
            clickhouse_user=os.getenv("LOGGER_CH_USER", os.getenv("CH_USER", "default")),
            clickhouse_password=os.getenv("LOGGER_CH_PASSWORD", os.getenv("CH_PASSWORD", "")),
            clickhouse_database=os.getenv("LOGGER_CH_DATABASE", os.getenv("CH_DATABASE", "default")),
            clickhouse_table=os.getenv("LOGGER_CH_TABLE", "gateway_logs"),
            clickhouse_connect_timeout=_as_int(os.getenv("LOGGER_CH_CONNECT_TIMEOUT", "10"), default=10, min_value=1),
            clickhouse_send_receive_timeout=_as_int(
                os.getenv("LOGGER_CH_SEND_RECEIVE_TIMEOUT", "30"),
                default=30,
                min_value=1,
            ),
            clickhouse_batch_enabled=_as_bool(os.getenv("LOGGER_CH_BATCH_ENABLED", "true"), default=True),
            clickhouse_batch_size=_as_int(os.getenv("LOGGER_CH_BATCH_SIZE", "20"), default=20, min_value=1),
            retry_enabled=_as_bool(os.getenv("LOGGER_RETRY_ENABLED", "true"), default=True),
            retry_attempts=_as_int(os.getenv("LOGGER_RETRY_ATTEMPTS", "2"), default=2, min_value=1),
            retry_backoff_ms=_as_int(os.getenv("LOGGER_RETRY_BACKOFF_MS", "50"), default=50, min_value=0),
            redaction_enabled=_as_bool(os.getenv("LOGGER_REDACTION_ENABLED", "true"), default=True),
            redaction_fields=_csv_env(
                os.getenv(
                    "LOGGER_REDACTION_FIELDS",
                    "authorization,token,password,api_key,secret,cookies",
                )
            ),
        )
