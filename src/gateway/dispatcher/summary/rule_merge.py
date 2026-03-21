"""
Deterministic merge of task results (legacy PlanHelper behavior).
"""

from __future__ import annotations

from typing import List

from ...schemas import RewritePlan, TaskExecutionResult


class RuleMergeFacade:
    """Rule-based concatenation of successful task answers (classmethod facade)."""

    @classmethod
    def merge_task_answers(
        cls,
        plan: RewritePlan,
        task_results: List[TaskExecutionResult],
    ) -> str:
        """
        Build merged answer from successful task outputs only.

        Args:
            plan: Execution plan (merge_strategy affects formatting).
            task_results: One result per executed task.

        Returns:
            Merged string; empty string when no successful non-empty answers.
        """
        if plan is None:
            raise ValueError("plan must not be None")
        if task_results is None:
            raise ValueError("task_results must not be None")
        completed = [
            r for r in task_results
            if r.status == "completed" and (r.answer or "").strip()
        ]
        if not completed:
            return ""
        if len(completed) == 1:
            return completed[0].answer.strip()
        if plan.merge_strategy == "none":
            return completed[0].answer.strip()
        if plan.merge_strategy in ("compare", "synthesize"):
            merged_lines = ["Combined results:"]
            for result in completed:
                merged_lines.append(f"- [{result.workflow}] {result.answer.strip()}")
            return "\n".join(merged_lines)
        return "\n".join(f"- [{r.workflow}] {r.answer.strip()}" for r in completed)
