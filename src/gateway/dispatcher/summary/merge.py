"""
Merge execution results: optional DeepSeek synthesis with rule-based fallback.
"""

from __future__ import annotations

import logging
import os
from typing import List

from ...api.config import GatewayConfig
from ...schemas import RewritePlan, TaskExecutionResult

from .rule_merge import RuleMergeFacade
from .summarize_llm import SummaryLlmFacade

logger = logging.getLogger(__name__)


def _sp_api_worker_is_authoritative_api_payload(worker_answer: str) -> bool:
    """
    Return True when the SP-API worker already returned tool-built getOrder YAML.

    ``SpApiReActAgent`` attaches the real Amazon JSON under ``sp_api_response`` inside
    a fenced YAML block. Running a chat LLM on that text tends to **summarize** (drops
    price, dates, etc.) and can **invent** human-readable statuses (e.g. ``Processing``)
    that do not match ``OrderStatus`` in the payload. Those answers must pass through
    unchanged.

    Args:
        worker_answer: Raw worker ``answer`` string.

    Returns:
        True if this looks like authoritative API YAML; False otherwise.
    """
    text = (worker_answer or "").strip()
    if not text:
        return False
    # Prefix added by SpApiReActAgent when getOrder YAML is attached.
    if "Below is the Amazon Selling Partner API" in text:
        return True
    # Tool output from format_orders_batch_as_yaml always includes sp_api_response per order.
    lowered = text.lower()
    if "```yaml" in lowered and "sp_api_response" in text:
        return True
    return False


class ResultAggregator:
    """
    Produce a single user-facing string from a plan and task results.

    Single-task ``sp_api``: when ``GATEWAY_SP_API_FORMAT_LLM_ENABLED`` (default true),
    runs a strict formatting LLM via ``GATEWAY_TEXT_GENERATION_BACKEND`` (deepseek|ollama),
    with fallback to the raw worker answer on failure.

    Multi-task: when ``GATEWAY_SUMMARY_LLM_ENABLED`` is true and ``DEEPSEEK_API_KEY`` is set,
    uses DeepSeek for merges; otherwise uses deterministic rules.
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

        # Single-task sp_api: optional strict formatting LLM (GATEWAY_TEXT_GENERATION_BACKEND).
        if (
            len(completed) == 1
            and GatewayConfig.sp_api_format_llm_enabled()
            and (completed[0].workflow or "").strip().lower() == "sp_api"
        ):
            if _sp_api_worker_is_authoritative_api_payload(completed[0].answer):
                logger.info(
                    "SP-API worker answer is authoritative API YAML; skipping format LLM"
                )
                return RuleMergeFacade.merge_task_answers(plan, task_results)
            try:
                from .sp_api_format_llm import format_sp_api_worker_answer

                return format_sp_api_worker_answer(
                    completed[0].answer,
                    user_sub_query=(completed[0].query or "").strip(),
                    backend=GatewayConfig.resolve_text_generation_backend(),
                )
            except Exception as exc:
                logger.warning(
                    "SP-API format LLM failed, using raw worker answer: %s",
                    exc,
                    exc_info=True,
                )
                return RuleMergeFacade.merge_task_answers(plan, task_results)

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
