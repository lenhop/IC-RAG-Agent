"""
Gateway logger integration tests.

These tests verify that gateway endpoints trigger logger facade methods
without affecting functional responses.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.gateway.api import app
from src.gateway.schemas import RewritePlan, TaskExecutionResult, TaskGroup, TaskItem

client = TestClient(app)


class _SpyLogger:
    def __init__(self):
        self.runtime_calls = 0
        self.interaction_calls = 0
        self.error_calls = 0

    def log_runtime(self, **kwargs):
        self.runtime_calls += 1
        return {"redis": True, "clickhouse": True}

    def log_interaction(self, **kwargs):
        self.interaction_calls += 1
        return {"redis": True, "clickhouse": True}

    def log_error(self, **kwargs):
        self.error_calls += 1
        return {"redis": True, "clickhouse": True}


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
def test_rewrite_endpoint_calls_logger_facade(mock_rewrite, mock_clarification, monkeypatch):
    """Rewrite endpoint should emit runtime/interaction logs when logger is available."""
    spy = _SpyLogger()
    monkeypatch.setattr("src.gateway.api.gateway_logger", spy)

    payload = {
        "query": "raw query",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/rewrite", json=payload)
    assert resp.status_code == 200
    assert spy.runtime_calls >= 1
    assert spy.interaction_calls >= 1
    assert spy.error_calls == 0


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
@patch(
    "src.gateway.api.build_execution_plan",
    return_value=(
        RewritePlan(
            plan_type="single_domain",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[TaskItem(task_id="t1", workflow="general", query="q1", depends_on=[])],
                )
            ],
        ),
        None,
    ),
)
@patch(
    "src.gateway.api._execute_plan",
    return_value=[
        TaskExecutionResult(
            task_id="t1",
            workflow="general",
            query="q1",
            status="completed",
            answer="ok",
            sources=[],
            duration_ms=1,
        )
    ],
)
@patch("src.gateway.api.route_workflow", return_value=("general", 1.0, "heuristic", None, None))
def test_query_endpoint_calls_logger_facade(
    mock_route,
    mock_execute,
    mock_plan,
    mock_rewrite,
    mock_clarification,
    monkeypatch,
):
    """Query endpoint should emit runtime and interaction logs on success path."""
    spy = _SpyLogger()
    monkeypatch.setattr("src.gateway.api.gateway_logger", spy)

    payload = {
        "query": "raw query",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    assert spy.runtime_calls >= 1
    assert spy.interaction_calls >= 1
