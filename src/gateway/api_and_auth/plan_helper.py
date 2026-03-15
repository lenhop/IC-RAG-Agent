"""
Plan and route helpers for the gateway.

PlanHelper: extract route input query, derive workflow, merge task answers.
"""

from __future__ import annotations

from typing import List

from ..schemas import RewritePlan, TaskExecutionResult


class PlanHelper:
    """
    Helpers for execution plan and route metadata.

    Pure functions; no gateway state. All class methods.
    """

    @classmethod
    def extract_route_input_query(cls, plan: RewritePlan, fallback_query: str) -> str:
        """Build route-input summary query from plan task queries."""
        task_queries = [
            task.query
            for group in plan.task_groups
            for task in group.tasks
            if (task.query or "").strip()
        ]
        if not task_queries:
            return (fallback_query or "").strip()
        if len(task_queries) == 1:
            return task_queries[0].strip()
        return " | ".join(q.strip() for q in task_queries if q.strip())

    @classmethod
    def derive_workflow(
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult], fallback: str
    ) -> str:
        """Derive top-level workflow label from plan/result context."""
        workflows = {task.workflow for group in plan.task_groups for task in group.tasks}
        if len(workflows) > 1:
            return "hybrid"
        if len(workflows) == 1:
            return next(iter(workflows))
        return fallback

    @classmethod
    def merge_task_answers(
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult]
    ) -> str:
        """Build deterministic merged answer from successful task outputs."""
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
        return "\n".join(
            f"- [{r.workflow}] {r.answer.strip()}" for r in completed
        )
