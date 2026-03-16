"""
Gateway feature flags and rewrite backend resolution.

GatewayConfig: rewrite-only mode, clarification enabled, rewrite backend.
"""

from __future__ import annotations

import os
from typing import Optional

from ..schemas import QueryRequest


class GatewayConfig:
    """
    Feature flags and config used by the gateway.

    clarification_enabled accepts optional service to avoid circular imports.
    """

    @classmethod
    def is_rewrite_only_mode(cls) -> bool:
        """
        Return True when gateway runs in Route LLM-only mode (truncate downstream).

        When set, /api/v1/query returns after Route LLM (clarification + rewrite + intents)
        + plan building; no worker execution.
        Env: GATEWAY_REWRITE_ONLY_MODE or GATEWAY_ROUTE_ONLY_MODE.
        """
        v = (
            os.getenv("GATEWAY_REWRITE_ONLY_MODE", "") or os.getenv("GATEWAY_ROUTE_ONLY_MODE", "")
        ).strip().lower()
        return v in ("1", "true", "yes", "on")

    @classmethod
    def clarification_enabled(cls) -> bool:
        """Check if clarification is enabled via env."""
        from ..route_llm.clarification import clarification_enabled
        return clarification_enabled()

    @classmethod
    def resolve_rewrite_backend(cls, request: QueryRequest) -> Optional[str]:
        """Resolve effective rewrite backend used by gateway."""
        if not request.rewrite_enable:
            return None
        backend = (request.rewrite_backend or "").strip().lower()
        if backend:
            return backend
        return os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower() or None
