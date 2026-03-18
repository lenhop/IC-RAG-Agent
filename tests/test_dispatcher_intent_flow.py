from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gateway.dispatcher import dispatcher
from src.gateway.schemas import RewritePlan, TaskGroup, TaskItem


def test_build_plan_from_extracted_intents_prefers_preclassified(monkeypatch):
    intents = ["q1", "q2"]
    preclassified = [
        {
            "query": "q1",
            "workflow": "sp_api",
            "intent_name": "get_order_status",
            "required_fields": [],
            "clarification_template": "",
        },
        {
            "query": "q2",
            "workflow": "uds",
            "intent_name": "get_report",
            "required_fields": [],
            "clarification_template": "",
        },
    ]

    def _should_not_call(*args, **kwargs):
        raise AssertionError("classify_intents_batch should not be called when preclassified is provided")

    monkeypatch.setattr(dispatcher, "classify_intents_batch", _should_not_call)

    plan, intents_with_meta = dispatcher._build_plan_from_extracted_intents(
        intents,
        conversation_context=None,
        classified_intents=preclassified,
    )

    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 2
    assert tasks[0].workflow == "sp_api"
    assert tasks[1].workflow == "uds"
    assert intents_with_meta[0]["intent_name"] == "get_order_status"
    assert intents_with_meta[1]["intent_name"] == "get_report"


def test_build_multi_task_plan_from_query_uses_batch_classification(monkeypatch):
    monkeypatch.setattr(dispatcher, "split_intents", lambda q: ["intent-a", "intent-b"])
    monkeypatch.setattr(
        dispatcher,
        "classify_intents_batch",
        lambda intents: [
            {"workflow": "sp_api"},
            {"workflow": "amazon_docs"},
        ],
    )

    plan = dispatcher._build_multi_task_plan_from_query("mixed query")
    assert plan is not None
    tasks = plan.task_groups[0].tasks
    assert [t.workflow for t in tasks] == ["sp_api", "amazon_docs"]


def test_expand_merged_tasks_falls_back_to_general_on_batch_failure(monkeypatch):
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
                        query="part1 and part2",
                        depends_on=[],
                        reason="seed",
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(dispatcher, "split_intents", lambda q: ["part1", "part2"])

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated batch failure")

    monkeypatch.setattr(dispatcher, "classify_intents_batch", _raise)

    expanded = dispatcher._expand_merged_tasks(plan)
    tasks = expanded.task_groups[0].tasks
    assert len(tasks) == 2
    assert tasks[0].workflow == "general"
    assert tasks[1].workflow == "general"
