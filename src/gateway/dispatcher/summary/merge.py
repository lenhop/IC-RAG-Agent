"""
Merge execution results: optional DeepSeek synthesis with rule-based fallback.
"""

from __future__ import annotations

import logging
import os
from typing import List

from ...schemas import RewritePlan, TaskExecutionResult

from .rule_merge import RuleMergeFacade
from .summarize_llm import SummaryLlmFacade

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Produce a single user-facing string from a plan and task results.

    When ``GATEWAY_SUMMARY_LLM_ENABLED`` is true and ``DEEPSEEK_API_KEY`` is set,
    uses DeepSeek for multi-task merges; otherwise uses deterministic rules.
    """

    @classmethod
    def summary_llm_enabled(cls) -> bool:
        """True when LLM summarization should be attempted for multi-task paths."""
        flag = os.getenv("GATEWAY_SUMMARY_LLM_ENABLED", "false").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return False
        if not (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            logger.debug("GATEWAY_SUMMARY_LLM_ENABLED set but DEEPSEEK_API_KEY missing; using rule merge")
            return False
        return True

    @classmethod
    def merge(
        cls,
        plan: RewritePlan,
        task_results: List[TaskExecutionResult],
    ) -> str:
        """
        Merge task answers using LLM when enabled, else rule-based merge.

        Args:
            plan: Execution plan.
            task_results: Results from DispatcherExecutor.

        Returns:
            Merged answer string (may be empty when no successful content).

        Raises:
            ValueError: If plan or task_results is None.
        """
        if plan is None:
            raise ValueError("plan must not be None")
        if task_results is None:
            raise ValueError("task_results must not be None")

        completed = [
            r for r in task_results
            if r.status == "completed" and (r.answer or "").strip()
        ]
        if len(completed) <= 1 or not cls.summary_llm_enabled():
            return RuleMergeFacade.merge_task_answers(plan, task_results)

        try:
            return SummaryLlmFacade.summarize_with_deepseek(plan, task_results)
        except Exception as exc:
            logger.warning(
                "LLM summary merge failed, falling back to rule merge: %s",
                exc,
                exc_info=True,
            )
            return RuleMergeFacade.merge_task_answers(plan, task_results)
