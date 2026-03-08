"""
Gateway API package.

Unified query gateway: Route LLM (planning) + Dispatcher (supervisor agent).
Worker agents: RAG, Amazon docs RAG, SP-API Agent, UDS Agent.
"""

from .schemas import QueryRequest, QueryResponse

try:
    # Optional import: api module defines the FastAPI app.
    from .api import app  # type: ignore[attr-defined]
except Exception:
    # Avoid import-time failures if FastAPI or dependencies are missing.
    app = None  # type: ignore[assignment]

__all__ = ["app", "QueryRequest", "QueryResponse"]

