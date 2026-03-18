from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gateway.api import api as api_module
from src.gateway.schemas import QueryRequest, RewritePlan, TaskExecutionResult, TaskGroup, TaskItem


def _make_two_task_plan() -> RewritePlan:
    return RewritePlan(
        plan_type="hybrid",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(task_id="t1", workflow="sp_api", query="intent-1", depends_on=[]),
                    TaskItem(task_id="t2", workflow="uds", query="intent-2", depends_on=[]),
                ],
            )
        ],
    )


def test_query_pipeline_passes_preclassified_intents_to_planner(monkeypatch):
    captured: dict = {}
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(api_module.AuthGuard, "resolve_user_id", staticmethod(lambda req, payload: "u1"))
    monkeypatch.setattr(api_module, "_run_clarification", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(api_module, "_run_rewrite", lambda *_args, **_kwargs: ("rewritten q", 12))
    monkeypatch.setattr(
        api_module,
        "_prepare_intent_classification",
        lambda rewritten_query, ctx: (
            ["intent-1", "intent-2"],
            [
                {
                    "query": "intent-1",
                    "workflow": "sp_api",
                    "intent_name": "get_order_status",
                    "required_fields": [],
                    "clarification_template": "",
                },
                {
                    "query": "intent-2",
                    "workflow": "uds",
                    "intent_name": "get_uds_report",
                    "required_fields": [],
                    "clarification_template": "",
                },
            ],
        ),
    )

    def _build_execution_plan(request, rewritten_query, intents=None, conversation_context=None, classified_intents=None):
        captured["intents"] = intents
        captured["classified_intents"] = classified_intents
        return _make_two_task_plan(), None

    monkeypatch.setattr(api_module, "build_execution_plan", _build_execution_plan)
    monkeypatch.setattr(
        api_module.DispatcherExecutor,
        "execute_plan",
        staticmethod(
            lambda plan, request: [
                TaskExecutionResult(
                    task_id="t1",
                    workflow="sp_api",
                    query="intent-1",
                    status="completed",
                    answer="a1",
                    sources=[],
                    duration_ms=1,
                ),
                TaskExecutionResult(
                    task_id="t2",
                    workflow="uds",
                    query="intent-2",
                    status="completed",
                    answer="a2",
                    sources=[],
                    duration_ms=1,
                ),
            ]
        ),
    )

    monkeypatch.setattr(api_module.MemoryEventWriter, "append_event", staticmethod(lambda memory, **kwargs: events.append((kwargs.get("event_type"), kwargs.get("event_content") or {}))))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_runtime", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_interaction", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_error", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayConfig, "is_rewrite_only_mode", staticmethod(lambda: False))

    req = QueryRequest(query="hello", workflow="auto", rewrite_enable=True, session_id="s1", user_id="u1")
    response = api_module.QueryPipeline.run(req, user_payload={"sub": "u1"}, memory=None)

    assert captured["intents"] == ["intent-1", "intent-2"]
    assert isinstance(captured["classified_intents"], list)
    assert captured["classified_intents"][0]["workflow"] == "sp_api"
    assert captured["classified_intents"][1]["workflow"] == "uds"

    intent_events = [c for t, c in events if t == "intent_classification"]
    assert len(intent_events) == 1
    assert intent_events[0]["classified_intents"][0]["intent_name"] == "get_order_status"

    assert response.error is None
    assert response.workflow == "hybrid"
    assert response.answer


def test_query_pipeline_handles_no_classification_data(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(api_module.AuthGuard, "resolve_user_id", staticmethod(lambda req, payload: "u1"))
    monkeypatch.setattr(api_module, "_run_clarification", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(api_module, "_run_rewrite", lambda *_args, **_kwargs: ("rewritten q", 8))
    monkeypatch.setattr(api_module, "_prepare_intent_classification", lambda rewritten_query, ctx: (None, None))

    def _build_execution_plan(request, rewritten_query, intents=None, conversation_context=None, classified_intents=None):
        captured["intents"] = intents
        captured["classified_intents"] = classified_intents
        return _make_two_task_plan(), None

    monkeypatch.setattr(api_module, "build_execution_plan", _build_execution_plan)
    monkeypatch.setattr(
        api_module.DispatcherExecutor,
        "execute_plan",
        staticmethod(
            lambda plan, request: [
                TaskExecutionResult(
                    task_id="t1",
                    workflow="sp_api",
                    query="intent-1",
                    status="completed",
                    answer="ok",
                    sources=[],
                    duration_ms=1,
                )
            ]
        ),
    )
    monkeypatch.setattr(api_module.MemoryEventWriter, "append_event", staticmethod(lambda memory, **kwargs: None))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_runtime", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_interaction", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayEventLogger, "log_error", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(api_module.GatewayConfig, "is_rewrite_only_mode", staticmethod(lambda: False))

    req = QueryRequest(query="hello", workflow="auto", rewrite_enable=True, session_id="s1", user_id="u1")
    response = api_module.QueryPipeline.run(req, user_payload={"sub": "u1"}, memory=None)

    assert captured["intents"] is None
    assert captured["classified_intents"] is None
    assert response.error is None
