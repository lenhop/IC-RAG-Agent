"""Tests for dispatcher summary merge (rule + optional LLM path)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.gateway.dispatcher.summary.merge as summary_merge_module
from src.gateway.dispatcher.summary.merge import ResultAggregator
from src.gateway.dispatcher.summary.rule_merge import RuleMergeFacade
from src.gateway.schemas import RewritePlan, TaskExecutionResult, TaskGroup, TaskItem


def _sample_plan(merge_strategy: str = "concat") -> RewritePlan:
    return RewritePlan(
        plan_type="hybrid",
        merge_strategy=merge_strategy,
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(task_id="t1", workflow="general", query="q1", depends_on=[], reason=""),
                    TaskItem(task_id="t2", workflow="uds", query="q2", depends_on=[], reason=""),
                ],
            )
        ],
    )


def test_rule_merge_two_completed_concat():
    plan = _sample_plan("concat")
    results = [
        TaskExecutionResult(
            task_id="t1", workflow="general", query="q1", status="completed",
            answer="A1", sources=[], error=None, duration_ms=1,
        ),
        TaskExecutionResult(
            task_id="t2", workflow="uds", query="q2", status="completed",
            answer="A2", sources=[], error=None, duration_ms=1,
        ),
    ]
    out = RuleMergeFacade.merge_task_answers(plan, results)
    assert "- [general] A1" in out
    assert "- [uds] A2" in out


def test_result_aggregator_rule_path_when_single_completed(monkeypatch):
    plan = _sample_plan()
    results = [
        TaskExecutionResult(
            task_id="t1", workflow="general", query="q1", status="completed",
            answer="Only", sources=[], error=None, duration_ms=1,
        ),
    ]
    monkeypatch.setenv("GATEWAY_SUMMARY_LLM_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    out = ResultAggregator.merge(plan, results)
    assert out == "Only"


def test_result_aggregator_llm_path_mocked(monkeypatch):
    plan = _sample_plan()
    results = [
        TaskExecutionResult(
            task_id="t1", workflow="general", query="q1", status="completed",
            answer="A1", sources=[], error=None, duration_ms=1,
        ),
        TaskExecutionResult(
            task_id="t2", workflow="uds", query="q2", status="completed",
            answer="A2", sources=[], error=None, duration_ms=1,
        ),
    ]

    monkeypatch.setenv("GATEWAY_SUMMARY_LLM_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        summary_merge_module.SummaryLlmFacade,
        "summarize_with_deepseek",
        classmethod(lambda cls, plan, task_results: "LLM merged"),
    )
    out = ResultAggregator.merge(plan, results)
    assert out == "LLM merged"


def test_result_aggregator_falls_back_on_llm_error(monkeypatch):
    plan = _sample_plan()
    results = [
        TaskExecutionResult(
            task_id="t1", workflow="general", query="q1", status="completed",
            answer="A1", sources=[], error=None, duration_ms=1,
        ),
        TaskExecutionResult(
            task_id="t2", workflow="uds", query="q2", status="completed",
            answer="A2", sources=[], error=None, duration_ms=1,
        ),
    ]

    def _boom(cls, plan, task_results):
        raise RuntimeError("api down")

    monkeypatch.setenv("GATEWAY_SUMMARY_LLM_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        summary_merge_module.SummaryLlmFacade,
        "summarize_with_deepseek",
        classmethod(_boom),
    )
    out = ResultAggregator.merge(plan, results)
    assert "A1" in out and "A2" in out


def test_merge_rejects_none_plan():
    with pytest.raises(ValueError, match="plan"):
        ResultAggregator.merge(None, [])  # type: ignore[arg-type]
