"""
DeepSeek-based synthesis of multi-task execution results.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from ...schemas import RewritePlan, TaskExecutionResult
from src.llm.call_deepseek import DeepSeekChat

logger = logging.getLogger(__name__)

# Default max completion tokens for merged answers (overridable via env).
_DEFAULT_SUMMARY_MAX_TOKENS = 1024


class SummaryLlmFacade:
    """Build payload and call DeepSeek for answer merging (classmethod facade)."""

    @classmethod
    def _load_system_prompt(cls) -> str:
        """Load merge instructions from package-local markdown file."""
        path = Path(__file__).resolve().parent / "merge_prompt.md"
        if not path.is_file():
            raise FileNotFoundError(f"merge_prompt.md not found at {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("merge_prompt.md is empty")
        return text

    @classmethod
    def build_tasks_payload(
        cls,
        plan: RewritePlan,
        task_results: List[TaskExecutionResult],
    ) -> Dict[str, Any]:
        """
        Serialize plan results for the summarization model.

        Args:
            plan: Original execution plan (for merge_strategy context).
            task_results: Structured per-task results.

        Returns:
            Dict with merge_strategy, plan_type, and tasks list.

        Raises:
            ValueError: If plan or task_results is None.
        """
        if plan is None:
            raise ValueError("plan must not be None")
        if task_results is None:
            raise ValueError("task_results must not be None")
        blocks: List[Dict[str, Any]] = []
        for result in task_results:
            blocks.append({
                "task_id": result.task_id,
                "sub_query": (result.query or "").strip(),
                "workflow": (result.workflow or "").strip(),
                "status": result.status,
                "answer": (result.answer or "").strip(),
                "sources": result.sources or [],
                "error": (result.error or "").strip() if result.error else "",
            })
        return {
            "merge_strategy": plan.merge_strategy,
            "plan_type": plan.plan_type,
            "tasks": blocks,
        }

    @classmethod
    def summarize_with_deepseek(
        cls,
        plan: RewritePlan,
        task_results: List[TaskExecutionResult],
    ) -> str:
        """
        Call DeepSeek to merge task outputs into one answer.

        Args:
            plan: Execution plan.
            task_results: All task results (including failures).

        Returns:
            Non-empty merged text from the model.

        Raises:
            FileNotFoundError: If prompt file is missing.
            RuntimeError: If DeepSeek returns empty content or API fails.
            ValueError: On invalid inputs.
        """
        system_prompt = cls._load_system_prompt()
        payload = cls.build_tasks_payload(plan, task_results)
        user_content = json.dumps(payload, ensure_ascii=False, indent=2)
        max_tokens = _DEFAULT_SUMMARY_MAX_TOKENS
        raw = os.getenv("GATEWAY_SUMMARY_MAX_TOKENS", "").strip()
        if raw:
            try:
                max_tokens = max(256, int(raw, 10))
            except ValueError:
                logger.warning(
                    "Invalid GATEWAY_SUMMARY_MAX_TOKENS=%r; using default %s",
                    raw,
                    max_tokens,
                )
        started = time.perf_counter()
        text = DeepSeekChat().complete(
            system_prompt,
            user_content,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "DeepSeek summary merge completed in %sms (max_tokens=%s)",
            elapsed_ms,
            max_tokens,
        )
        if not (text or "").strip():
            raise RuntimeError("DeepSeek summary returned empty text")
        return text.strip()
