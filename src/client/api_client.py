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


def _safe_json(resp: Optional[requests.Response]) -> Dict[str, Any]:
    """Parse response as JSON; return {} on empty or invalid content."""
    if not resp or not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return {}

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

    Calls POST /api/v1/query with query, workflow, optional rewrite_backend, session_id.
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
        if base_url is not None:
            self._base_url = str(base_url).rstrip("/")
        else:
            self._base_url = (os.environ.get("GATEWAY_API_URL", "") or "").rstrip("/")
        # Client timeout: env GATEWAY_CLIENT_TIMEOUT overrides default (e.g. 600 for long queries)
        try:
            self._timeout = int(os.environ.get("GATEWAY_CLIENT_TIMEOUT", str(timeout)))
        except (TypeError, ValueError):
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
        rewrite_backend: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send synchronous query to the gateway.

        Args:
            query: User query string.
            workflow: Workflow selector. One of:
                     auto|general|amazon_docs|ic_docs|sp_api|uds
            rewrite_backend: Optional "ollama" or "deepseek" for the unified rewrite stage.
            session_id: Optional session ID for multi-turn context.
            user_id: Optional user ID for user-scoped conversation history.
            token: Optional JWT for protected gateway (GATEWAY_AUTH_REQUIRED).

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
            "session_id": session_id,
            "user_id": user_id,
        }
        if rewrite_backend:
            payload["rewrite_backend"] = (rewrite_backend or "").strip().lower()

        if self._mock_mode:
            return self._mock_response(query, wf, rewrite_backend, session_id, user_id)

        url = f"{self._base_url}/api/v1/query"
        return self._post_json(url, payload, token=token)

    def rewrite_sync(
        self,
        query: str,
        rewrite_backend: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call rewrite-only endpoint for immediate UI feedback.

        Args:
            query: User query string.
            rewrite_backend: Optional backend ("ollama" or "deepseek").
            session_id: Optional session ID for multi-turn logging to Redis.
            user_id: Optional user ID for user-scoped conversation history.
            token: Optional JWT for protected gateway.

        Returns:
            Dict with rewritten query metadata, or error dict.
        """
        if self._mock_mode:
            return {
                "original_query": query,
                "rewritten_query": query,
                "rewrite_backend": rewrite_backend or "mock",
                "rewrite_time_ms": 0,
                "clarification_status": "Skip",
                "clarification_backend": None,
            }

        payload: Dict[str, Any] = {
            "query": query,
            "workflow": "auto",
            "session_id": session_id,
            "user_id": user_id,
        }
        if rewrite_backend:
            payload["rewrite_backend"] = (rewrite_backend or "").strip().lower()

        url = f"{self._base_url}/api/v1/rewrite"
        return self._post_json(url, payload, token=token)

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST JSON payload and return parsed response or normalized error dict."""
        headers: Dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers if headers else None,
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

    def _post_json_allow_4xx(
        self,
        url: str,
        payload: Dict[str, Any],
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST JSON; on 4xx/5xx return dict with "error" from detail.
        On success return parsed JSON.
        """
        headers: Dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers if headers else None,
                timeout=self._timeout,
            )
            data = _safe_json(resp)
            if 400 <= resp.status_code < 500:
                return {"error": data.get("detail", resp.text or f"HTTP {resp.status_code}")}
            if resp.status_code >= 500:
                return {"error": data.get("detail", f"Server error (HTTP {resp.status_code})")}
            resp.raise_for_status()
            return data
        except requests.ConnectionError as e:
            logger.warning("Gateway connection failed: %s", e)
            return {"error": f"Cannot connect to gateway at {self._base_url}. Is it running?"}
        except requests.Timeout as e:
            logger.warning("Gateway request timed out: %s", e)
            return {"error": f"Request timed out after {self._timeout}s"}
        except requests.RequestException as e:
            logger.warning("Gateway request failed: %s", e)
            r = getattr(e, "response", None)
            data = _safe_json(r) if r else {}
            return {"error": data.get("detail", str(e))}

    def get_session_history_sync(
        self,
        session_id: str,
        last_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Fetch last N conversation turns for a session.

        Args:
            session_id: Session identifier.
            last_n: Number of recent turns to return (default 5).

        Returns:
            Dict with "session_id" and "history" keys. On error, "error" key and "history": [].
        """
        if self._mock_mode:
            return {"session_id": session_id or "", "history": []}
        if not session_id or not str(session_id).strip():
            return {"session_id": "", "history": []}
        url = f"{self._base_url}/api/v1/session/{session_id.strip()}"
        params = {"last_n": min(max(1, last_n), 50)}
        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("Get session history failed: %s", e)
            try:
                r = getattr(e, "response", None)
                data = _safe_json(r) if r else {}
                return {"error": data.get("detail", str(e)), "history": []}
            except Exception:
                return {"error": str(e), "history": []}

    def _mock_response(
        self,
        query: str,
        workflow: str,
        rewrite_backend: Optional[str],
        session_id: Optional[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return simulated response when gateway is unavailable or mock mode."""
        return {
            "answer": (
                f"[Mock] Query: {query}\nWorkflow: {workflow}\n"
                f"Backend: {rewrite_backend or 'default'}\nSession: {session_id or 'none'}"
            ),
            "workflow": workflow,
            "routing_confidence": 1.0,
            "sources": [],
            "request_id": "mock-request-id",
            "clarification_required": False,
        }

    def register_sync(
        self,
        user_name: str,
        password: str,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new user.

        Args:
            user_name: Display name.
            password: Plaintext password.
            email: Optional email.

        Returns:
            Dict with user_id, user_name, role on success.
            On error, returns dict with "error" key.
        """
        if self._mock_mode:
            return {
                "user_id": "mock-user-id",
                "user_name": user_name,
                "role": "general",
                "message": "Registration successful (mock)",
            }
        payload: Dict[str, Any] = {"user_name": user_name, "password": password}
        if email:
            payload["email"] = email
        url = f"{self._base_url}/api/v1/auth/register"
        return self._post_json_allow_4xx(url, payload)

    def signin_sync(
        self,
        user_name: str,
        password: str,
    ) -> Dict[str, Any]:
        """
        Sign in and return JWT + user info.

        Args:
            user_name: Display name.
            password: Plaintext password.

        Returns:
            Dict with access_token, token_type, user on success.
            On error, returns dict with "error" or "detail" keys.
        """
        if self._mock_mode:
            return {
                "access_token": "mock-jwt-token",
                "token_type": "bearer",
                "user": {
                    "user_id": "mock-user-id",
                    "user_name": user_name,
                    "email": None,
                    "role": "general",
                    "status": "active",
                },
            }
        payload = {"user_name": user_name, "password": password}
        url = f"{self._base_url}/api/v1/auth/signin"
        return self._post_json_allow_4xx(url, payload)

    def signout_sync(self) -> Dict[str, Any]:
        """
        Sign out. Client should discard token after calling.
        Returns message dict.
        """
        if self._mock_mode:
            return {"message": "Signed out (mock)"}
        url = f"{self._base_url}/api/v1/auth/signout"
        try:
            resp = requests.post(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {"message": "Signed out"}
        except requests.RequestException as e:
            logger.warning("Signout request failed: %s", e)
            return {"message": "Signed out", "error": str(e)}

    def me_sync(self, token: str) -> Dict[str, Any]:
        """
        Get current user from JWT.

        Args:
            token: JWT access token (with or without "Bearer " prefix).

        Returns:
            Dict with user_id, user_name, role, etc. on success.
            On error, returns dict with "error" or "detail" keys.
        """
        if self._mock_mode:
            return {
                "user_id": "mock-user-id",
                "user_name": "mock",
                "email": None,
                "role": "general",
                "status": "active",
            }
        url = f"{self._base_url}/api/v1/auth/me"
        try:
            auth_val = token if (token or "").strip().lower().startswith("bearer ") else f"Bearer {token}"
            headers = {"Authorization": auth_val}
            resp = requests.get(url, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("Me request failed: %s", e)
            try:
                resp = getattr(e, "response", None)
                err_body = resp.json() if resp and resp.content else {}
                return {"error": err_body.get("detail", str(e))}
            except Exception:
                return {"error": str(e)}
