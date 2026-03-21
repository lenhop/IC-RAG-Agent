"""
Read-only dispatcher configuration from environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatcherConfig:
    """
    Dispatcher execution tuning (parallelism, etc.).

    Attributes:
        max_parallel_workers: Upper bound for ThreadPoolExecutor workers per task group.
            When None, callers use legacy default min(4, num_tasks).
    """

    max_parallel_workers: int | None

    @classmethod
    def from_env(cls) -> "DispatcherConfig":
        """
        Load configuration from environment.

        Invalid GATEWAY_DISPATCHER_MAX_WORKERS values are logged and ignored so
        runtime behavior stays non-fatal (backward compatible).

        Returns:
            DispatcherConfig instance.
        """
        raw = os.getenv("GATEWAY_DISPATCHER_MAX_WORKERS", "").strip()
        if not raw:
            return cls(max_parallel_workers=None)
        try:
            parsed = int(raw, 10)
            if parsed < 1:
                raise ValueError("must be >= 1")
            return cls(max_parallel_workers=parsed)
        except ValueError:
            logger.warning(
                "Invalid GATEWAY_DISPATCHER_MAX_WORKERS=%r; using default min(4, num_tasks)",
                raw,
            )
            return cls(max_parallel_workers=None)
