"""
Debug trace builder for gateway observability.

DebugTraceBuilder: build debug trace dict for UI clients.
"""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import QueryRequest
from .gateway_config import GatewayConfig


class DebugTraceBuilder:
    """
    Build optional observability trace returned to UI clients.
    """

    @classmethod
    def build_debug_trace(
        cls,
        original_query: str,
        rewritten_query: str,
        rewrite_time_ms: int,
        request: QueryRequest,
        route_input_query: str,
        route_source: str,
        route_backend: str | None,
        route_llm_confidence: float | None,
    ) -> Dict[str, Any]:
        """Build debug trace dict with rewrite and route metadata."""
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_enabled": bool(request.rewrite_enable),
            "rewrite_backend": GatewayConfig.resolve_rewrite_backend(request),
            "rewrite_time_ms": rewrite_time_ms,
            "route_input_query": route_input_query,
            "route_source": route_source,
            "route_backend": route_backend,
            "route_llm_confidence": route_llm_confidence,
        }
