"""
UDS Agent HTTP client (analytical queries).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .http_client import BackendHttpClient
from .worker_profile import should_stub_uds_and_sp_api, stub_response_for_workflow

UDS_API_URL = os.getenv("UDS_API_URL", "http://127.0.0.1:8001").rstrip("/")
UDS_BACKEND_TIMEOUT = int(os.getenv("GATEWAY_UDS_BACKEND_TIMEOUT", "300"))


class UdsWorkflowClient:
    """classmethod facade for UDS backend calls."""

    @classmethod
    def call_uds(cls, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Call UDS Agent API for analytical questions.

        Args:
            query: User query text.
            session_id: Optional session id (ignored by current UDS API).
        """
        _ = session_id
        if should_stub_uds_and_sp_api():
            return stub_response_for_workflow("uds")

        if not UDS_API_URL:
            return {
                "error": "UDS_API_URL not configured",
                "error_type": "ConfigError",
            }

        url = f"{UDS_API_URL}/api/v1/uds/query"
        payload = {"query": query}
        data = BackendHttpClient.post_json(url, payload, timeout_seconds=UDS_BACKEND_TIMEOUT)
        if BackendHttpClient.has_backend_error(data):
            data = BackendHttpClient.normalize_error_envelope(data)
            status = str(data.get("status", "")).strip().lower()
            if status == "failed":
                data["error_type"] = "UDSQueryFailed"
            return data

        status = str(data.get("status", "")).strip().lower()
        if status and status != "completed":
            return {
                "error": data.get("error") or f"UDS query status: {status}",
                "error_type": "UDSQueryFailed",
            }

        response_obj = data.get("response") or {}
        if isinstance(response_obj, dict):
            answer = (
                response_obj.get("summary")
                or response_obj.get("answer")
                or response_obj.get("result")
                or str(response_obj)
            )
            sources = response_obj.get("sources") or []
        else:
            answer = str(response_obj)
            sources = []

        if not answer or answer == "{}":
            answer = (
                "UDS processed the query but returned no summary. "
                "Try narrowing the date range or asking a more specific metric."
            )

        return {"answer": answer, "sources": sources}


def call_uds(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Delegate to UdsWorkflowClient.call_uds."""
    return UdsWorkflowClient.call_uds(query, session_id)
