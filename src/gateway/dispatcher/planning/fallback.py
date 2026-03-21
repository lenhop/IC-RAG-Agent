"""
Fallback plan construction: single-task and multi-task-from-query paths.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ...route_llm.classification import classify_intents_batch
from ...route_llm.rewriting import split_intents
from ...schemas import RewritePlan, TaskGroup, TaskItem

logger = logging.getLogger(__name__)


class FallbackPlanBuilder:
    """Fallback RewritePlan builders (classmethod facade)."""

    @classmethod
    def build_multi_task_plan_from_query(cls, query: str) -> Optional[RewritePlan]:
        """
        Build a multi-task plan by splitting and classifying a mixed query.

        Returns:
            RewritePlan when at least two clauses exist; otherwise None.
        """
        clauses = split_intents(query)
        if len(clauses) < 2:
            return None

        batch_results: list[dict[str, Any]] | None = None
        try:
            batch_results = classify_intents_batch(clauses)
        except Exception as exc:
            logger.warning("Batch classification failed in multi-task fallback: %s", exc)

        tasks: List[TaskItem] = []
        for idx, clause in enumerate(clauses, start=1):
            workflow = "general"
            if batch_results and idx - 1 < len(batch_results):
                workflow = (batch_results[idx - 1].get("workflow") or "general").strip() or "general"
            tasks.append(
                TaskItem(
                    task_id=f"t{idx}",
                    workflow=workflow,
                    query=clause,
                    depends_on=[],
                    reason="multi_intent_fallback",
                )
            )
        return RewritePlan(
            plan_type="hybrid",
            merge_strategy="concat",
            task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
        )

    @classmethod
    def build_single_task_plan(cls, query: str, workflow: str) -> RewritePlan:
        """Build a single-task execution plan for legacy or explicit-workflow flow."""
        task_query = (query or "").strip()
        task_workflow = (workflow or "general").strip().lower() or "general"
        return RewritePlan(
            plan_type="single_domain",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[
                        TaskItem(
                            task_id="t1",
                            workflow=task_workflow,
                            query=task_query,
                            depends_on=[],
                            reason="single_task_fallback",
                        )
                    ],
                )
            ],
        )
