"""
Gateway API Client for unified chat.

Provides GatewayClient to call the unified gateway POST /api/v1/query.
Supports mock mode when gateway is unavailable or GATEWAY_MOCK=true.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Valid workflow values per API contract
VALID_WORKFLOWS = frozenset(
    {"auto", "general", "amazon_docs", "ic_docs", "sp_api", "uds"}
)


class GatewayClientError(Exception):
    """Raised when gateway request fails with ConnectionError or Timeout."""

    def __init__(self, message: str, error_type: str = "unknown"):
        super().__init__(message)
        self.error_type = error_type


class GatewayClient:
    """
    Client for the unified gateway query API.

    Calls POST /api/v1/query with query, workflow, rewrite_enable, session_id.
    Supports mock mode when base_url is empty or GATEWAY_MOCK=true.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the gateway client.

        Args:
            base_url: Gateway base URL (e.g. http://localhost:8000).
                     If None or empty, uses GATEWAY_API_URL env.
                     Empty string or GATEWAY_MOCK=true enables mock mode.
            timeout: Request timeout in seconds (default 120).
        """
        self._base_url = (base_url or os.environ.get("GATEWAY_API_URL", "")).rstrip("/")
        self._timeout = timeout
        self._mock_mode = self._is_mock_mode()

    def _is_mock_mode(self) -> bool:
        """Return True if mock mode is enabled."""
        if not self._base_url:
            return True
        return os.environ.get("GATEWAY_MOCK", "").lower() in ("true", "1", "yes")

    def query_sync(
        self,
        query: str,
        workflow: str = "auto",
        rewrite_enable: bool = True,
        rewrite_backend: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send synchronous query to the gateway.

        Args:
            query: User query string.
            workflow: Workflow selector. One of:
                     auto|general|amazon_docs|ic_docs|sp_api|uds
            rewrite_enable: Whether to enable query rewriting.
            rewrite_backend: When rewrite_enable=True, backend to use:
                            "ollama" or "deepseek". Ignored when rewrite_enable=False.
            session_id: Optional session ID for multi-turn context.

        Returns:
            Response dict with keys such as answer, workflow, sources, etc.
            On error, returns dict with "error" and "error_type" keys.
        """
        # Normalize workflow
        wf = (workflow or "auto").lower()
        if wf not in VALID_WORKFLOWS:
            wf = "auto"

        payload: Dict[str, Any] = {
            "query": query,
            "workflow": wf,
            "rewrite_enable": rewrite_enable,
            "session_id": session_id,
        }
        if rewrite_enable and rewrite_backend:
            payload["rewrite_backend"] = (rewrite_backend or "").strip().lower()

        if self._mock_mode:
            return self._mock_response(query, wf, rewrite_enable, rewrite_backend, session_id)

        url = f"{self._base_url}/api/v1/query"
        return self._post_json(url, payload)

    def rewrite_sync(
        self,
        query: str,
        rewrite_enable: bool = True,
        rewrite_backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call rewrite-only endpoint for immediate UI feedback.

        Args:
            query: User query string.
            rewrite_enable: Whether to enable rewriting.
            rewrite_backend: Optional backend ("ollama" or "deepseek").

        Returns:
            Dict with rewritten query metadata, or error dict.
        """
        if self._mock_mode:
            return {
                "original_query": query,
                "rewritten_query": query,
                "rewrite_enabled": rewrite_enable,
                "rewrite_backend": rewrite_backend or "mock",
                "rewrite_time_ms": 0,
            }

        payload: Dict[str, Any] = {
            "query": query,
            "workflow": "auto",
            "rewrite_enable": rewrite_enable,
            "session_id": None,
        }
        if rewrite_enable and rewrite_backend:
            payload["rewrite_backend"] = (rewrite_backend or "").strip().lower()

        url = f"{self._base_url}/api/v1/rewrite"
        return self._post_json(url, payload)

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON payload and return parsed response or normalized error dict."""
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError as e:
            logger.warning("Gateway connection failed: %s", e)
            return {
                "error": f"Cannot connect to gateway at {self._base_url}. Is it running?",
                "error_type": "ConnectionError",
            }
        except requests.Timeout as e:
            logger.warning("Gateway request timed out: %s", e)
            return {
                "error": f"Query timed out after {self._timeout}s. The server took too long to respond.",
                "error_type": "Timeout",
            }
        except requests.RequestException as e:
            logger.warning("Gateway request failed: %s", e)
            return {
                "error": str(e),
                "error_type": "RequestException",
            }
        except ValueError as e:
            logger.warning("Invalid JSON response: %s", e)
            return {
                "error": f"Invalid response from gateway: {e}",
                "error_type": "ValueError",
            }

    def _mock_response(
        self,
        query: str,
        workflow: str,
        rewrite_enable: bool,
        rewrite_backend: Optional[str],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Return simulated response when gateway is unavailable or mock mode."""
        return {
            "answer": f"[Mock] Query: {query}\nWorkflow: {workflow}\nRewrite: {rewrite_enable}\nBackend: {rewrite_backend or 'default'}\nSession: {session_id or 'none'}",
            "workflow": workflow,
            "routing_confidence": 1.0,
            "sources": [],
            "request_id": "mock-request-id",
            "clarification_required": False,
        }
