"""
Gateway configuration and event logging.

GatewayConfig: feature flags (rewrite-only mode, clarification, rewrite backend).
GatewayEventLogger: best-effort logging facade (Redis + ClickHouse).

Merged from: gateway_config.py + gateway_logger.py
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ..schemas import QueryRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GatewayConfig — feature flags and rewrite backend resolution
# ---------------------------------------------------------------------------


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
        from ..route_llm.clarification.clarification import ClarificationEnvValidator
        return ClarificationEnvValidator.is_enabled()

    @classmethod
    def resolve_rewrite_backend(cls, request: QueryRequest) -> Optional[str]:
        """Resolve effective rewrite backend used by gateway (unified rewrite always on)."""
        backend = (request.rewrite_backend or "").strip().lower()
        if backend:
            return backend
        return os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower() or None


# ---------------------------------------------------------------------------
# GatewayEventLogger — best-effort event logging facade
# ---------------------------------------------------------------------------


class GatewayEventLogger:
    """
    Best-effort logging to gateway_logger (short-term Redis + long-term ClickHouse).

    Set the facade via set_facade(facade) from api.py after get_logger_facade().
    All log_* methods no-op when facade is None; never raise.
    """

    _facade: Optional[Any] = None

    @classmethod
    def set_facade(cls, facade: Any) -> None:
        """Set the logger facade (e.g. from get_logger_facade())."""
        cls._facade = facade

    @classmethod
    def log_runtime(cls, **kwargs: Any) -> None:
        """Best-effort runtime logging; never raises into query/rewrite flow."""
        if not cls._facade:
            return
        try:
            cls._facade.log_runtime(**kwargs)
        except Exception as exc:
            logger.debug("Runtime log skipped due to error: %s", exc)

    @classmethod
    def log_interaction(cls, **kwargs: Any) -> None:
        """Best-effort interaction logging; never raises."""
        if not cls._facade:
            return
        try:
            cls._facade.log_interaction(**kwargs)
        except Exception as exc:
            logger.debug("Interaction log skipped due to error: %s", exc)

    @classmethod
    def log_error(cls, **kwargs: Any) -> None:
        """Best-effort error logging; never raises."""
        if not cls._facade:
            return
        try:
            cls._facade.log_error(**kwargs)
        except Exception as exc:
            logger.debug("Error log skipped due to error: %s", exc)
