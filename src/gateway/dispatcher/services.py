"""
Gateway services layer (Dispatcher invokes worker agents).

Provides HTTP client wrappers for worker agent backends:
- General / amazon_docs / ic_docs -> RAG API (port 8002)
- sp_api -> SP-API Agent (port 8003)
- uds -> UDS Agent (port 8001)

Each function returns a normalized dict for the gateway's QueryResponse schema.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Backend base URLs (per GATEWAY_DEVELOPMENT_PLAN: RAG 8002, UDS 8001, SP-API 8003)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8002").rstrip("/")
SP_API_URL = os.getenv("SP_API_URL", "http://127.0.0.1:8003").rstrip("/")
UDS_API_URL = os.getenv("UDS_API_URL", "http://127.0.0.1:8001").rstrip("/")

# IC docs: when disabled, gateway returns a friendly message without calling RAG (Chroma not populated yet)
IC_DOCS_NOT_READY_MESSAGE = (
    "IC document retrieval is not ready yet. Please try Amazon docs or general knowledge."
)

# Shared timeout for backend HTTP calls (seconds)
BACKEND_TIMEOUT = int(os.getenv("GATEWAY_BACKEND_TIMEOUT", "120"))
# UDS queries can be heavier than other backends; allow separate timeout.
UDS_BACKEND_TIMEOUT = int(os.getenv("GATEWAY_UDS_BACKEND_TIMEOUT", "300"))


def _has_backend_error(data: Dict[str, Any]) -> bool:
    """
    Return True only when backend reports a real error value.

    Some APIs include an `error` field with null/empty value on success.
    """
    err = data.get("error")
    if err is None:
        return False
    if isinstance(err, str):
        return bool(err.strip())
    return True


def _normalize_error_envelope(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure backend error payload has a consistent gateway envelope.

    Normalized keys:
    - error: non-empty message when backend failure is present.
    - error_type: stable machine-friendly category.
    """
    if not isinstance(data, dict):
        return {
            "error": "Invalid backend payload type",
            "error_type": "InvalidPayload",
        }
    err = data.get("error")
    status = str(data.get("status", "")).strip().lower()
    if _has_backend_error(data):
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


def _http_post(
    url: str,
    json_payload: Dict[str, Any],
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Helper to POST JSON and return parsed JSON or error dict.

    Args:
        url: Target URL.
        json_payload: JSON body.

    Returns:
        Dict with either response JSON or an \"error\" key.
    """
    effective_timeout = timeout_seconds or BACKEND_TIMEOUT
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
        return _normalize_error_envelope(resp.json())
    except ValueError as exc:
        logger.warning("Invalid JSON from backend %s: %s", url, exc)
        return {
            "error": f"Invalid JSON from backend {url}: {exc}",
            "error_type": "ValueError",
        }


def call_general(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Call general Q&A backend (RAG in general mode).

    Args:
        query: User query text.
        session_id: Optional session identifier (ignored by RAG API).

    Returns:
        Dict with \"answer\", optional \"sources\", or \"error\".
    """
    if not RAG_API_URL:
        return {
            "error": "RAG_API_URL not configured",
            "error_type": "ConfigError",
        }

    url = f"{RAG_API_URL}/query"
    payload = {"question": query, "mode": "general"}
    data = _http_post(url, payload)
    if _has_backend_error(data):
        return _normalize_error_envelope(data)

    answer = data.get("answer", "")
    sources = data.get("sources") or []
    return {"answer": answer, "sources": sources}


def call_amazon_docs(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Call Amazon documentation workflow (RAG in documents mode).

    Current implementation reuses the generic documents mode and
    prefixes the query to bias retrieval towards Amazon docs.
    """
    if not RAG_API_URL:
        return {
            "error": "RAG_API_URL not configured",
            "error_type": "ConfigError",
        }

    url = f"{RAG_API_URL}/query"
    biased_query = f"Amazon docs: {query}"
    payload = {"question": biased_query, "mode": "documents"}
    data = _http_post(url, payload)
    if _has_backend_error(data):
        return _normalize_error_envelope(data)

    answer = data.get("answer", "")
    sources = data.get("sources") or []
    return {"answer": answer, "sources": sources}


def _ic_docs_enabled() -> bool:
    """True if IC docs workflow should call RAG; false returns friendly message only."""
    v = os.getenv("IC_DOCS_ENABLED", "false").strip().lower()
    return v in ("true", "1", "yes")


def call_ic_docs(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Call IC-RAG-Agent documentation workflow (RAG in documents mode).

    When IC_DOCS_ENABLED is not true, returns a friendly message without calling RAG
    (Chroma not populated with IC docs yet). Workflow remains in diagrams.
    """
    if not _ic_docs_enabled():
        return {"answer": IC_DOCS_NOT_READY_MESSAGE, "sources": []}
    if not RAG_API_URL:
        return {
            "error": "RAG_API_URL not configured",
            "error_type": "ConfigError",
        }

    url = f"{RAG_API_URL}/query"
    biased_query = f"IC docs: {query}"
    payload = {"question": biased_query, "mode": "documents"}
    data = _http_post(url, payload)
    if _has_backend_error(data):
        return _normalize_error_envelope(data)

    answer = data.get("answer", "")
    sources = data.get("sources") or []
    return {"answer": answer, "sources": sources}


def call_sp_api(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Call Seller Operations Agent (SP-API backend).

    Args:
        query: User query text.
        session_id: Optional session id forwarded to SP-API.
    """
    if not SP_API_URL:
        return {
            "error": "SP_API_URL not configured",
            "error_type": "ConfigError",
        }

    url = f"{SP_API_URL}/api/v1/seller/query"
    payload = {"query": query}
    # SP-API backend validates session_id as string; omit when unavailable.
    if session_id is not None and str(session_id).strip():
        payload["session_id"] = str(session_id).strip()
    data = _http_post(url, payload)
    if _has_backend_error(data):
        return _normalize_error_envelope(data)

    # SP-API QueryResponse has 'response' as main text field.
    answer = data.get("response", "")
    sources = []  # SP-API currently does not expose sources list.
    return {"answer": answer, "sources": sources}


def call_uds(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Call UDS Agent API for analytical questions.

    Args:
        query: User query text.
        session_id: Optional session id (ignored by current UDS API).
    """
    if not UDS_API_URL:
        return {
            "error": "UDS_API_URL not configured",
            "error_type": "ConfigError",
        }

    url = f"{UDS_API_URL}/api/v1/uds/query"
    payload = {"query": query}
    data = _http_post(url, payload)
    if _has_backend_error(data):
        data = _normalize_error_envelope(data)
        status = str(data.get("status", "")).strip().lower()
        if status == "failed":
            data["error_type"] = "UDSQueryFailed"
        return data

    # UDS API always returns HTTP 200 with an internal status field.
    # Propagate backend failures to gateway response instead of returning
    # an empty answer.
    status = str(data.get("status", "")).strip().lower()
    if status and status != "completed":
        return {
            "error": data.get("error") or f"UDS query status: {status}",
            "error_type": "UDSQueryFailed",
        }

    # UDS QueryResponse: primary content in 'response' (dict) and 'status'.
    response_obj = data.get("response") or {}
    if isinstance(response_obj, dict):
        # Prefer a 'summary' field when available.
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


__all__ = [
    "call_general",
    "call_amazon_docs",
    "call_ic_docs",
    "call_sp_api",
    "call_uds",
]

