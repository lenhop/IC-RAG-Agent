"""
Workflow name to outbound HTTP handler mapping for DispatcherExecutor.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# (query, session_id) -> backend result dict
WorkflowHandler = Callable[[str, Optional[str]], Dict[str, Any]]


class WorkflowRegistry:
    """
    Resolves workflow string to a callable that invokes the remote backend.

    Unknown workflows log a warning and fall back to ``general`` (RAG).
    """

    _handlers: Dict[str, WorkflowHandler] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, workflow: str, handler: WorkflowHandler) -> None:
        """
        Register a handler for a normalized workflow key (lowercase).

        Raises:
            ValueError: If workflow name is empty after strip.
        """
        key = (workflow or "").strip().lower()
        if not key:
            raise ValueError("workflow name must be non-empty")
        cls._handlers[key] = handler

    @classmethod
    def _ensure_defaults(cls) -> None:
        """Lazily populate built-in workflow handlers (avoids circular imports)."""
        if cls._initialized:
            return
        from .rag_workflows import call_amazon_docs, call_general, call_ic_docs
        from .sp_api_client import call_sp_api
        from .uds_client import call_uds

        cls.register("general", call_general)
        cls.register("amazon_docs", call_amazon_docs)
        cls.register("ic_docs", call_ic_docs)
        cls.register("sp_api", call_sp_api)
        cls.register("uds", call_uds)
        cls._initialized = True

    @classmethod
    def resolve(cls, workflow: str) -> WorkflowHandler:
        """
        Return the handler for ``workflow``; fallback to ``general`` when unknown.

        Args:
            workflow: Task workflow name (case-insensitive).

        Returns:
            Callable accepting (query, session_id) and returning a result dict.
        """
        cls._ensure_defaults()
        key = (workflow or "general").strip().lower() or "general"
        handler = cls._handlers.get(key)
        if handler is not None:
            return handler
        logger.warning("Unknown workflow '%s', falling back to general", workflow)
        general_handler = cls._handlers.get("general")
        if general_handler is None:
            raise RuntimeError("WorkflowRegistry missing 'general' handler")
        return general_handler

    @classmethod
    def reset_for_testing(cls) -> None:
        """Clear registry state (test hooks only)."""
        cls._handlers.clear()
        cls._initialized = False  # allow _ensure_defaults to repopulate on next resolve
