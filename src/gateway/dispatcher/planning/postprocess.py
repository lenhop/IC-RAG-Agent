"""
Plan post-processing: merge expansion and workflow correction hook.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ...route_llm.classification import classify_intents_batch
from ...route_llm.rewriting import split_intents
from ...schemas import RewritePlan, TaskGroup, TaskItem

logger = logging.getLogger(__name__)


class PlanPostProcessor:
    """Post-classification plan transforms (classmethod facade)."""

    @classmethod
    def expand_merged_tasks(cls, plan: RewritePlan) -> RewritePlan:
        """
        Split tasks whose query contains multiple distinct sub-queries.

        Ensures each sub-question becomes a separate plan item.
        """
        expanded_groups = []
        for group in plan.task_groups or []:
            expanded_tasks: List[TaskItem] = []
            next_id = 1
            for task in group.tasks or []:
                q = (task.query or "").strip()
                if not q:
                    continue
                clauses = split_intents(q)
                if len(clauses) < 2:
                    expanded_tasks.append(task)
                    next_id += 1
                    continue

                batch_results: list[dict[str, Any]] | None = None
                try:
                    batch_results = classify_intents_batch(clauses)
                except Exception as exc:
                    logger.warning("Batch classification failed in merged-task expansion: %s", exc)

                for idx, clause in enumerate(clauses):
                    workflow = "general"
                    if batch_results and idx < len(batch_results):
                        workflow = (batch_results[idx].get("workflow") or "general").strip() or "general"
                    expanded_tasks.append(
                        TaskItem(
                            task_id=f"t{next_id}",
                            workflow=workflow,
                            query=clause,
                            depends_on=task.depends_on or [],
                            reason=(task.reason or "") + "_expanded" if task.reason else "merged_split",
                        )
                    )
                    next_id += 1
            if expanded_tasks:
                expanded_groups.append(
                    TaskGroup(
                        group_id=group.group_id,
                        parallel=group.parallel,
                        tasks=expanded_tasks,
                    )
                )
        if not expanded_groups:
            return plan
        return RewritePlan(
            plan_type=(
                "hybrid"
                if len({t.workflow for g in expanded_groups for t in g.tasks}) > 1
                else plan.plan_type
            ),
            merge_strategy=plan.merge_strategy,
            task_groups=expanded_groups,
        )

    @classmethod
    def correct_plan_workflows(cls, plan: RewritePlan) -> RewritePlan:
        """
        Post-classification plan correction hook (pass-through).

        Reserved for future heuristic or policy overrides.
        """
        return plan
