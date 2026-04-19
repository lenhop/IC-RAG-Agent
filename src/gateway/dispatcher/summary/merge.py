"""
Merge execution results: optional DeepSeek synthesis with rule-based fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

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
    # Tool output from SpApiOrderBatchYamlFormatter.format_batch always includes sp_api_response per order.
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
    def merge(cls, plan: RewritePlan, task_results: List[TaskExecutionResult]) -> str:
        """Merge task answers; see ``merge_with_meta`` for observability dict."""
        text, _meta = cls.merge_with_meta(plan, task_results)
        return text

    @classmethod
    def merge_with_meta(
        cls,
        plan: RewritePlan,
        task_results: List[TaskExecutionResult],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Merge task answers and return (text, debug dict for UI).

        The debug dict includes ``text_generation_backend_effective`` (from
        ``GATEWAY_TEXT_GENERATION_BACKEND`` chain) and ``answer_merge_mode`` describing
        which path produced the final string.
        """
        if plan is None:
            raise ValueError("plan must not be None")
        if task_results is None:
            raise ValueError("task_results must not be None")

        tg_effective = GatewayConfig.resolve_text_generation_backend()
        meta: Dict[str, Any] = {
            "text_generation_backend_effective": tg_effective,
            "answer_merge_mode": "rule_merge",
            "format_llm_applied": False,
            "summary_llm_applied": False,
        }

        completed = [
            r for r in task_results
            if r.status == "completed" and (r.answer or "").strip()
        ]
        if completed:
            meta["worker_workflows"] = [
                (r.workflow or "").strip() or "unknown" for r in completed
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
                text = RuleMergeFacade.merge_task_answers(plan, task_results)
                meta["answer_merge_mode"] = "sp_api_yaml_pass_through"
                meta["detail"] = (
                    "Tool-built API YAML detected; format LLM skipped (policy)."
                )
                return text, meta
            try:
                from .sp_api_format_llm import format_sp_api_worker_answer

                backend = GatewayConfig.resolve_text_generation_backend()
                text = format_sp_api_worker_answer(
                    completed[0].answer,
                    user_sub_query=(completed[0].query or "").strip(),
                    backend=backend,
                )
                meta["answer_merge_mode"] = "sp_api_format_llm"
                meta["format_llm_applied"] = True
                meta["format_llm_backend"] = backend
                meta["detail"] = (
                    f"SP-API worker answer formatted with text_generation backend: {backend}."
                )
                return text, meta
            except Exception as exc:
                logger.warning(
                    "SP-API format LLM failed, using raw worker answer: %s",
                    exc,
                    exc_info=True,
                )
                text = RuleMergeFacade.merge_task_answers(plan, task_results)
                meta["answer_merge_mode"] = "sp_api_format_llm_failed_rule_fallback"
                meta["detail"] = f"Format LLM failed; raw worker answer. ({exc})"
                return text, meta

        if len(completed) <= 1 or not cls.summary_llm_enabled():
            text = RuleMergeFacade.merge_task_answers(plan, task_results)
            if len(completed) == 1:
                wf = (completed[0].workflow or "").strip().lower() or "unknown"
                meta["answer_merge_mode"] = f"worker_output_{wf}"
                if wf == "sp_api" and not GatewayConfig.sp_api_format_llm_enabled():
                    meta["detail"] = (
                        "GATEWAY_SP_API_FORMAT_LLM_ENABLED is off; raw worker answer."
                    )
            else:
                meta["answer_merge_mode"] = "rule_merge"
            return text, meta

        try:
            text = SummaryLlmFacade.summarize_with_deepseek(plan, task_results)
            meta["answer_merge_mode"] = "multi_task_summary_deepseek"
            meta["summary_llm_applied"] = True
            meta["summary_llm_backend"] = "deepseek"
            meta["detail"] = "Multi-task merge via DeepSeek (GATEWAY_SUMMARY_LLM_ENABLED)."
            return text, meta
        except Exception as exc:
            logger.warning(
                "LLM summary merge failed, falling back to rule merge: %s",
                exc,
                exc_info=True,
            )
            text = RuleMergeFacade.merge_task_answers(plan, task_results)
            meta["answer_merge_mode"] = "multi_task_summary_failed_rule_fallback"
            meta["detail"] = str(exc)
            return text, meta
