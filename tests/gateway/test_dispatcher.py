"""
Unit tests for gateway dispatcher module.

Covers:
- _build_plan_from_extracted_intents: builds plan from intent list.
- _build_multi_task_plan_from_query: heuristic clause splitting into multi-task plan.
- _build_single_task_plan: single-task fallback.
- _expand_merged_tasks: splits merged sub-queries within a task.
- _correct_plan_workflows: heuristic override for misclassified workflows.
- build_execution_plan: top-level plan builder (planner, fallback, explicit workflow).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.gateway.dispatcher import (
    _build_multi_task_plan_from_query,
    _build_plan_from_extracted_intents,
    _build_single_task_plan,
    _correct_plan_workflows,
    _expand_merged_tasks,
    build_execution_plan,
)
from src.gateway.schemas import (
    QueryRequest,
    RewritePlan,
    TaskGroup,
    TaskItem,
)


# ---------------------------------------------------------------------------
# _build_plan_from_extracted_intents
# ---------------------------------------------------------------------------


def test_build_plan_from_intents_multiple_workflows():
    """Multiple intents with different workflows produce a hybrid plan."""
    intents = [
        "what is FBA storage fee",
        "FBA fee for ASIN B074KF7RKS",
        "total FBA fee last month",
    ]
    plan = _build_plan_from_extracted_intents(intents)
    assert plan.plan_type == "hybrid"
    assert len(plan.task_groups) == 1
    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 3
    workflows = {t.workflow for t in tasks}
    assert len(workflows) > 1  # not all the same


def test_build_plan_from_intents_single_workflow():
    """All intents mapping to the same workflow produce single_domain plan."""
    intents = ["total sales last month", "revenue by month"]
    plan = _build_plan_from_extracted_intents(intents)
    assert plan.plan_type == "single_domain"
    tasks = plan.task_groups[0].tasks
    assert all(t.workflow == "uds" for t in tasks)


def test_build_plan_from_intents_empty_list_returns_fallback():
    """Empty intent list returns a fallback single-task plan."""
    plan = _build_plan_from_extracted_intents([])
    assert len(plan.task_groups) == 1
    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 1
    assert "unable to extract" in tasks[0].query.lower()


def test_build_plan_from_intents_skips_blank_intents():
    """Blank intents in the list are skipped."""
    intents = ["what is FBA", "", "  ", "total sales"]
    plan = _build_plan_from_extracted_intents(intents)
    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 2


def test_build_plan_from_intents_task_ids_sequential():
    """Task IDs are sequential t1, t2, t3..."""
    intents = ["what is FBA", "total sales", "get order status"]
    plan = _build_plan_from_extracted_intents(intents)
    ids = [t.task_id for t in plan.task_groups[0].tasks]
    assert ids == ["t1", "t2", "t3"]


# ---------------------------------------------------------------------------
# _build_multi_task_plan_from_query
# ---------------------------------------------------------------------------


def test_build_multi_task_from_compound_query():
    """Compound query with question-starter patterns splits into multiple tasks."""
    query = "what is FBA get order status for 112-123 which table stores fee data"
    plan = _build_multi_task_plan_from_query(query)
    assert plan is not None
    assert plan.plan_type == "hybrid"
    tasks = plan.task_groups[0].tasks
    assert len(tasks) >= 2


def test_build_multi_task_returns_none_for_single_clause():
    """Single-clause query returns None (not enough clauses)."""
    plan = _build_multi_task_plan_from_query("what is FBA storage fee")
    assert plan is None


def test_build_multi_task_returns_none_for_empty():
    """Empty query returns None."""
    plan = _build_multi_task_plan_from_query("")
    assert plan is None


# ---------------------------------------------------------------------------
# _build_single_task_plan
# ---------------------------------------------------------------------------


def test_build_single_task_plan_basic():
    """Single-task plan with provided workflow and query."""
    plan = _build_single_task_plan("what is FBA", "amazon_docs")
    assert plan.plan_type == "single_domain"
    assert len(plan.task_groups) == 1
    task = plan.task_groups[0].tasks[0]
    assert task.workflow == "amazon_docs"
    assert task.query == "what is FBA"
    assert task.task_id == "t1"


def test_build_single_task_plan_defaults_to_general():
    """Empty workflow defaults to general."""
    plan = _build_single_task_plan("some query", "")
    assert plan.task_groups[0].tasks[0].workflow == "general"


# ---------------------------------------------------------------------------
# _expand_merged_tasks
# ---------------------------------------------------------------------------


def test_expand_merged_tasks_splits_compound_query():
    """A task with multiple question-starter patterns is expanded."""
    plan = RewritePlan(
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
                        query="what is FBA show me total sales last month",
                    )
                ],
            )
        ],
    )
    expanded = _expand_merged_tasks(plan)
    total_tasks = sum(len(g.tasks) for g in expanded.task_groups)
    assert total_tasks >= 2


def test_expand_merged_tasks_no_split_for_simple_query():
    """A single simple query task remains unchanged."""
    plan = RewritePlan(
        plan_type="single_domain",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(task_id="t1", workflow="uds", query="total sales last month")
                ],
            )
        ],
    )
    expanded = _expand_merged_tasks(plan)
    total_tasks = sum(len(g.tasks) for g in expanded.task_groups)
    assert total_tasks == 1


# ---------------------------------------------------------------------------
# _correct_plan_workflows
# ---------------------------------------------------------------------------


def test_correct_plan_workflows_overrides_misclassified():
    """Heuristic correction overrides a misclassified workflow when confidence >= 0.9."""
    plan = RewritePlan(
        plan_type="hybrid",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(
                        task_id="t1",
                        workflow="general",
                        query="what is Amazon's FBA fee policy",
                    ),
                    TaskItem(
                        task_id="t2",
                        workflow="general",
                        query="which table stores order data",
                    ),
                ],
            )
        ],
    )
    corrected = _correct_plan_workflows(plan)
    t1 = corrected.task_groups[0].tasks[0]
    t2 = corrected.task_groups[0].tasks[1]
    # "FBA fee policy" should be corrected to amazon_docs (keyword: "policy")
    assert t1.workflow == "amazon_docs"
    # "which table stores order data" should be corrected to uds (keyword: "table", conf 0.92)
    assert t2.workflow == "uds"


def test_correct_plan_workflows_keeps_correct_assignment():
    """Already-correct workflow is not changed."""
    plan = RewritePlan(
        plan_type="single_domain",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(
                        task_id="t1",
                        workflow="uds",
                        query="total sales last month",
                    )
                ],
            )
        ],
    )
    corrected = _correct_plan_workflows(plan)
    assert corrected.task_groups[0].tasks[0].workflow == "uds"


# ---------------------------------------------------------------------------
# build_execution_plan (integration)
# ---------------------------------------------------------------------------


def test_build_execution_plan_explicit_workflow():
    """Explicit workflow creates a single-task plan without planner."""
    request = QueryRequest(query="anything", workflow="sp_api", rewrite_enable=False)
    plan = build_execution_plan(request, "anything")
    assert plan.plan_type == "single_domain"
    assert plan.task_groups[0].tasks[0].workflow == "sp_api"


def test_build_execution_plan_uses_intents_when_provided():
    """When intents provided (from intent classification on optimized query), use them."""
    request = QueryRequest(query="what is FBA total sales get order 123", workflow="auto")
    intents = ["what is FBA", "total sales last month", "get order 123"]
    plan = build_execution_plan(request, "anything", intents=intents)
    total_tasks = sum(len(g.tasks) for g in plan.task_groups)
    assert total_tasks == 3


def test_build_execution_plan_heuristic_fallback_multi_clause():
    """Without planner, compound query uses heuristic clause splitting."""
    request = QueryRequest(
        query="what is FBA get order status for 123 show me total sales",
        workflow="auto",
        rewrite_enable=False,
    )
    plan = build_execution_plan(request, "what is FBA get order status for 123 show me total sales")
    total_tasks = sum(len(g.tasks) for g in plan.task_groups)
    assert total_tasks >= 2
