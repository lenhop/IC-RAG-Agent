"""
Shared utilities for logger clients and facade.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class LoggerClientError(Exception):
    """Base exception for logger client failures."""


def redact_payload(data: Dict[str, Any], sensitive_fields: Iterable[str]) -> Dict[str, Any]:
    """
    Redact sensitive keys in a payload.

    Matching is case-insensitive by key name only to avoid accidental value
    masking in free-text fields.
    """
    if not data:
        return {}
    lowered = {field.lower() for field in sensitive_fields}
    output: Dict[str, Any] = {}
    for key, value in data.items():
        if str(key).lower() in lowered:
            output[key] = "***REDACTED***"
        else:
            output[key] = value
    return output


def safe_json_dumps(payload: Dict[str, Any]) -> str:
    """Serialize payload to JSON safely, preserving unicode content."""
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        # Last resort: convert unknown values to strings first.
        fallback = {k: str(v) for k, v in payload.items()}
        return json.dumps(fallback, ensure_ascii=False)


def with_retry(
    fn: Callable[[], Any],
    *,
    enabled: bool,
    attempts: int,
    backoff_ms: int,
    operation_name: str,
) -> Any:
    """Run function with bounded retry/backoff and raise final exception."""
    if not enabled:
        return fn()

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - small helper path
            last_exc = exc
            if attempt >= attempts:
                break
            if backoff_ms > 0:
                time.sleep(backoff_ms / 1000.0)
            logger.debug(
                "Retry %s attempt %d/%d after error: %s",
                operation_name,
                attempt,
                attempts,
                exc,
            )
    if last_exc is not None:
        raise last_exc
    raise LoggerClientError(f"{operation_name} failed without explicit exception")
