"""
Dispatcher (Project Manager): builds execution plans from intents and orchestrates execution.

Receives intents from Route LLM (Decision Maker), maps them to workflows via heuristics,
builds task_groups, and applies plan correction. The api module uses this to obtain
RewritePlan before executing tasks.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .rewriters import parse_rewrite_plan_text, planner_rewrite_enabled
from .routing_heuristics import (
    apply_docs_preference,
    normalize_query,
    route_workflow_heuristic,
    split_multi_intent_clauses,
)
from .schemas import QueryRequest, RewritePlan, TaskGroup, TaskItem

logger = logging.getLogger(__name__)


def _build_plan_from_extracted_intents(intents: List[str]) -> RewritePlan:
    """
    Build execution plan from extracted_intents when LLM merges tasks.
    Routes each intent via heuristic and creates one task per intent.
    """
    tasks: List[TaskItem] = []
    for idx, intent in enumerate(intents, start=1):
        q = (intent or "").strip()
        if not q:
            continue
        wf, _ = route_workflow_heuristic(q)
        wf = apply_docs_preference(q, wf)
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=wf,
                query=q,
                depends_on=[],
                reason="extracted_intents_fallback",
            )
        )
    if not tasks:
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
                            workflow="general",
                            query=(
                                (intents[0] if intents else "unable to extract intents").strip()
                                or "unable to extract intents"
                            ),
                            depends_on=[],
                            reason="empty_intents_fallback",
                        )
                    ],
                )
            ],
        )
    return RewritePlan(
        plan_type="hybrid" if len({t.workflow for t in tasks}) > 1 else "single_domain",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )


def _build_multi_task_plan_from_query(query: str) -> Optional[RewritePlan]:
    """
    Build a heuristic multi-task plan from comma/semicolon separated mixed query.
    Used as a robust fallback when planner JSON is unavailable.
    """
    clauses = split_multi_intent_clauses(query)
    if len(clauses) < 2:
        return None
    tasks: List[TaskItem] = []
    for idx, clause in enumerate(clauses, start=1):
        workflow, _ = route_workflow_heuristic(clause)
        workflow = apply_docs_preference(clause, workflow)
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=workflow,
                query=clause,
                depends_on=[],
                reason="heuristic_multi_intent_fallback",
            )
        )
    return RewritePlan(
        plan_type="hybrid",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )


def _build_single_task_plan(query: str, workflow: str) -> RewritePlan:
    """Build a single-task execution plan for legacy/fallback flow."""
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


def _expand_merged_tasks(plan: RewritePlan) -> RewritePlan:
    """
    Split any task whose query contains multiple distinct sub-queries into
    separate tasks. Ensures each sub-question is a separate plan item.
    """
    expanded_groups = []
    for group in plan.task_groups or []:
        expanded_tasks: List[TaskItem] = []
        next_id = 1
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            clauses = split_multi_intent_clauses(q)
            if len(clauses) < 2:
                expanded_tasks.append(task)
                next_id += 1
                continue
            for clause in clauses:
                wf, _ = route_workflow_heuristic(clause)
                wf = apply_docs_preference(clause, wf)
                expanded_tasks.append(
                    TaskItem(
                        task_id=f"t{next_id}",
                        workflow=wf,
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


def _correct_plan_workflows(plan: RewritePlan) -> RewritePlan:
    """
    Apply heuristic-based workflow correction to fix LLM misclassifications.
    When the heuristic returns a different workflow with confidence >= 0.9,
    override the task's workflow to correct obvious routing errors.
    """
    HEURISTIC_OVERRIDE_CONF = 0.9
    for group in plan.task_groups or []:
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            h_wf, h_conf = route_workflow_heuristic(q)
            h_wf = apply_docs_preference(q, h_wf)
            current = (task.workflow or "").strip().lower()
            if h_wf != current and h_conf >= HEURISTIC_OVERRIDE_CONF:
                logger.debug(
                    "Plan correction: task %s workflow %s -> %s (heuristic conf=%.2f)",
                    task.task_id,
                    current,
                    h_wf,
                    h_conf,
                )
                task.workflow = h_wf
    return plan


def build_execution_plan(request: QueryRequest, rewritten_query: str) -> RewritePlan:
    """
    Build a validated execution plan for query orchestration.

    Behavior:
    - Explicit workflow (non-auto): synthesize one task with that workflow.
    - Planner mode enabled (auto): parse structured planner output from rewritten text.
    - Fallback: route once and synthesize one task from routed workflow.
    """
    from .router import route_workflow

    explicit = (request.workflow or "auto").strip().lower() or "auto"
    normalized_query = normalize_query(request.query or "")

    if explicit != "auto":
        task_query = (rewritten_query or normalized_query).strip() or normalized_query
        return _build_single_task_plan(task_query, explicit)

    if planner_rewrite_enabled():
        parsed_plan = parse_rewrite_plan_text(
            text=rewritten_query or "",
            fallback_query=normalized_query,
        )
        if parsed_plan:
            task_count = sum(len(g.tasks) for g in (parsed_plan.task_groups or []))
            intents = parsed_plan.extracted_intents or []
            # When extracted_intents has more items than tasks (or no tasks), use intents.
            if intents and (len(intents) > task_count or task_count == 0):
                logger.info(
                    "Planner merged or empty tasks (%d tasks vs %d extracted_intents); "
                    "rebuilding from extracted_intents.",
                    task_count,
                    len(intents),
                )
                return _correct_plan_workflows(_build_plan_from_extracted_intents(intents))
            if parsed_plan.task_groups:
                # When LLM returns single task but query has multiple clauses, use heuristic.
                clauses = split_multi_intent_clauses(rewritten_query or normalized_query)
                if task_count == 1 and len(clauses) >= 2:
                    logger.info(
                        "Planner returned single task for multi-clause query (%d clauses); "
                        "using heuristic multi-task fallback.",
                        len(clauses),
                    )
                    multi_task_plan = _build_multi_task_plan_from_query(
                        rewritten_query or normalized_query
                    )
                    if multi_task_plan:
                        return _correct_plan_workflows(multi_task_plan)
                # Expand any merged tasks (e.g. "how many X how many Y" -> two tasks).
                expanded = _expand_merged_tasks(parsed_plan)
                return _correct_plan_workflows(expanded)

    multi_task_plan = _build_multi_task_plan_from_query(rewritten_query or normalized_query)
    if multi_task_plan is not None:
        return multi_task_plan

    workflow, _, _, _, _ = route_workflow(
        (rewritten_query or normalized_query).strip(), request
    )
    task_query = (rewritten_query or normalized_query).strip() or normalized_query
    return _build_single_task_plan(task_query, workflow)


__all__ = [
    "build_execution_plan",
    "_correct_plan_workflows",
]
