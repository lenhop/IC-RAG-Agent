"""
RAG API workflows: general, Amazon docs, IC docs (HTTP clients).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .http_client import BackendHttpClient

# Backend base URL (RAG API default port 8002)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8002").rstrip("/")

IC_DOCS_NOT_READY_MESSAGE = (
    "IC document retrieval is not ready yet. Please try Amazon docs or general knowledge."
)


class RagWorkflowClient:
    """classmethod facade for RAG-backed workflow calls."""

    @classmethod
    def _rag_url_configured(cls) -> bool:
        return bool(RAG_API_URL)

    @classmethod
    def call_general(cls, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Call general Q&A backend (RAG in general mode).

        Args:
            query: User query text.
            session_id: Optional session identifier (ignored by RAG API).

        Returns:
            Dict with answer, optional sources, or error keys.
        """
        _ = session_id  # RAG API does not use session today
        if not cls._rag_url_configured():
            return {
                "error": "RAG_API_URL not configured",
                "error_type": "ConfigError",
            }

        url = f"{RAG_API_URL}/query"
        payload = {"question": query, "mode": "general"}
        data = BackendHttpClient.post_json(url, payload)
        if BackendHttpClient.has_backend_error(data):
            return BackendHttpClient.normalize_error_envelope(data)

        answer = data.get("answer", "")
        sources = data.get("sources") or []
        return {"answer": answer, "sources": sources}

    @classmethod
    def call_amazon_docs(cls, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Call Amazon Business RAG path (Chroma ``documents`` + DeepSeek dual merge).

        Sends ``mode=amazon_business`` to the RAG API (see ``tasks/RAG_WORKFLOW.md``).
        """
        _ = session_id
        if not cls._rag_url_configured():
            return {
                "error": "RAG_API_URL not configured",
                "error_type": "ConfigError",
            }

        url = f"{RAG_API_URL}/query"
        # Use amazon_business mode: Chroma (documents collection) + DeepSeek dual path on RAG API.
        payload = {"question": (query or "").strip(), "mode": "amazon_business"}
        data = BackendHttpClient.post_json(url, payload)
        if BackendHttpClient.has_backend_error(data):
            return BackendHttpClient.normalize_error_envelope(data)

        answer = data.get("answer", "")
        sources = data.get("sources") or []
        return {"answer": answer, "sources": sources}

    @classmethod
    def _ic_docs_enabled(cls) -> bool:
        v = os.getenv("IC_DOCS_ENABLED", "false").strip().lower()
        return v in ("true", "1", "yes")

    @classmethod
    def call_ic_docs(cls, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Call IC documentation workflow (RAG documents mode when enabled).

        When IC_DOCS_ENABLED is not true, returns a friendly message without calling RAG.
        """
        _ = session_id
        if not cls._ic_docs_enabled():
            return {"answer": IC_DOCS_NOT_READY_MESSAGE, "sources": []}
        if not cls._rag_url_configured():
            return {
                "error": "RAG_API_URL not configured",
                "error_type": "ConfigError",
            }

        url = f"{RAG_API_URL}/query"
        biased_query = f"IC docs: {query}"
        payload = {"question": biased_query, "mode": "documents"}
        data = BackendHttpClient.post_json(url, payload)
        if BackendHttpClient.has_backend_error(data):
            return BackendHttpClient.normalize_error_envelope(data)

        answer = data.get("answer", "")
        sources = data.get("sources") or []
        return {"answer": answer, "sources": sources}


# Module-level aliases preserve callables for WorkflowRegistry registration
def call_general(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Delegate to RagWorkflowClient.call_general."""
    return RagWorkflowClient.call_general(query, session_id)


def call_amazon_docs(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Delegate to RagWorkflowClient.call_amazon_docs."""
    return RagWorkflowClient.call_amazon_docs(query, session_id)


def call_ic_docs(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Delegate to RagWorkflowClient.call_ic_docs."""
    return RagWorkflowClient.call_ic_docs(query, session_id)
