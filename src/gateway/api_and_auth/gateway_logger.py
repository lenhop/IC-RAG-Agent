"""
Gateway event logging facade.

GatewayEventLogger: best-effort log_runtime, log_interaction, log_error.
Holds a facade reference set by api.py after gateway_logger init.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
