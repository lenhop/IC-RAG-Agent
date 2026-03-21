"""
Parallel execution helpers for task groups (ThreadPoolExecutor).
"""

from __future__ import annotations

from ..config import DispatcherConfig


class ParallelScheduler:
    """Thread-pool sizing for parallel task groups (classmethod facade)."""

    @classmethod
    def resolve_max_workers(cls, num_tasks: int) -> int:
        """
        Compute max worker count for a task group.

        Uses GATEWAY_DISPATCHER_MAX_WORKERS when set and valid; otherwise
        min(4, num_tasks) to match historical behavior.

        Args:
            num_tasks: Number of tasks in the group.

        Returns:
            Positive integer at most num_tasks.

        Raises:
            ValueError: If num_tasks < 1.
        """
        if num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")
        cfg = DispatcherConfig.from_env()
        if cfg.max_parallel_workers is not None:
            return max(1, min(cfg.max_parallel_workers, num_tasks))
        return min(4, num_tasks)
