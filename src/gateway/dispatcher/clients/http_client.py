"""
Shared HTTP POST helpers for gateway outbound worker clients.

Returns dict envelopes with either success payload or error/error_type keys.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class BackendHttpClient:
    """
    Facade for JSON POST to remote backends with normalized error envelopes.

    All public entry points are classmethods for a single import pattern.
    """

    @classmethod
    def default_timeout_seconds(cls) -> int:
        """Return default HTTP timeout from environment (seconds)."""
        return int(os.getenv("GATEWAY_BACKEND_TIMEOUT", "120"))

    @classmethod
    def has_backend_error(cls, data: Dict[str, Any]) -> bool:
        """
        Return True only when backend reports a real error value.

        Some APIs include an ``error`` field with null/empty value on success.
        """
        err = data.get("error")
        if err is None:
            return False
        if isinstance(err, str):
            return bool(err.strip())
        return True

    @classmethod
    def normalize_error_envelope(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure backend error payload has a consistent gateway envelope.

        Returns:
            Dict with error and error_type when failure is present; otherwise
            the original dict (or a synthetic envelope for invalid input).
        """
        if not isinstance(data, dict):
            return {
                "error": "Invalid backend payload type",
                "error_type": "InvalidPayload",
            }
        err = data.get("error")
        status = str(data.get("status", "")).strip().lower()
        if cls.has_backend_error(data):
            normalized = dict(data)
            normalized["error"] = str(err).strip() if err is not None else "Unknown backend error"
            normalized["error_type"] = str(data.get("error_type") or "BackendError")
            return normalized
        if status == "failed":
            return {
                "error": "Backend reported failed status",
                "error_type": "BackendFailedStatus",
            }
        return data

    @classmethod
    def post_json(
        cls,
        url: str,
        json_payload: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        POST JSON and return parsed JSON or an error dict.

        Args:
            url: Target URL.
            json_payload: JSON-serializable body.
            timeout_seconds: Override default timeout; uses GATEWAY_BACKEND_TIMEOUT when None.

        Returns:
            Response dict or dict with ``error`` and ``error_type``.
        """
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else cls.default_timeout_seconds()
        )
        try:
            resp = requests.post(url, json=json_payload, timeout=effective_timeout)
        except requests.ConnectionError as exc:
            logger.warning("Backend connection failed (%s): %s", url, exc)
            return {"error": f"Cannot connect to backend at {url}", "error_type": "ConnectionError"}
        except requests.Timeout as exc:
            logger.warning("Backend request timed out (%s): %s", url, exc)
            return {
                "error": f"Backend request to {url} timed out after {effective_timeout}s",
                "error_type": "Timeout",
            }
        except requests.RequestException as exc:
            logger.warning("Backend request error (%s): %s", url, exc)
            return {"error": str(exc), "error_type": "RequestException"}

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            logger.warning("Backend %s returned %s: %s", url, resp.status_code, detail)
            return {
                "error": f"Backend error {resp.status_code}: {detail}",
                "error_type": "HTTPError",
            }

        try:
            return cls.normalize_error_envelope(resp.json())
        except ValueError as exc:
            logger.warning("Invalid JSON from backend %s: %s", url, exc)
            return {
                "error": f"Invalid JSON from backend {url}: {exc}",
                "error_type": "ValueError",
            }
