"""
SP-API Seller Operations Agent HTTP client.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .http_client import BackendHttpClient
from .worker_profile import should_stub_sp_api, stub_response_for_workflow

SP_API_URL = os.getenv("SP_API_URL", "http://127.0.0.1:8003").rstrip("/")


class SpApiWorkflowClient:
    """classmethod facade for SP-API backend calls."""

    @classmethod
    def call_sp_api(cls, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Call Seller Operations Agent (SP-API backend).

        Args:
            query: User query text.
            session_id: Optional session id forwarded to SP-API when non-empty.
        """
        if should_stub_sp_api():
            return stub_response_for_workflow("sp_api")

        if not SP_API_URL:
            return {
                "error": "SP_API_URL not configured",
                "error_type": "ConfigError",
            }

        url = f"{SP_API_URL}/api/v1/seller/query"
        payload: Dict[str, Any] = {"query": query}
        if session_id is not None and str(session_id).strip():
            payload["session_id"] = str(session_id).strip()
        data = BackendHttpClient.post_json(url, payload)
        if BackendHttpClient.has_backend_error(data):
            return BackendHttpClient.normalize_error_envelope(data)

        answer = data.get("response", "")
        sources: List[Any] = []
        return {"answer": answer, "sources": sources}


def call_sp_api(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Delegate to SpApiWorkflowClient.call_sp_api."""
    return SpApiWorkflowClient.call_sp_api(query, session_id)
