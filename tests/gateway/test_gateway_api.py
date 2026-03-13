"""
Tests for gateway.api (POST /api/v1/query) with mocked services.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.gateway.api import app
from src.gateway.schemas import RewritePlan, TaskGroup, TaskItem


client = TestClient(app)


def _base_payload() -> dict:
    return {
        "query": "test query",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "sess-1",
        "stream": False,
    }


def _assert_debug_payload(data: dict) -> None:
    """Assert gateway response includes rewrite/routing debug trace."""
    assert "debug" in data
    debug = data["debug"]
    assert isinstance(debug, dict)
    assert "original_query" in debug
    assert "rewritten_query" in debug
    assert "rewrite_enabled" in debug
    assert "rewrite_backend" in debug
    assert "rewrite_time_ms" in debug
    assert "route_input_query" in debug
    assert "route_source" in debug
    assert "route_backend" in debug
    assert "route_llm_confidence" in debug


@patch("src.gateway.api.rewrite_query", return_value=("rewritten only", None, 0, 0))
def test_rewrite_endpoint_returns_rewrite_metadata(mock_rewrite):
    """Rewrite endpoint returns rewritten query and timing metadata."""
    payload = {
        "query": "raw query",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "ollama",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/rewrite", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_query"] == "raw query"
    assert data["rewritten_query"] == "rewritten only"
    assert data["rewrite_enabled"] is True
    assert data["rewrite_backend"] == "ollama"
    assert isinstance(data["rewrite_time_ms"], int)
    assert data["rewrite_time_ms"] >= 0


@patch("src.gateway.api.rewrite_query", return_value=("split me", None, 0, 0))
@patch("src.gateway.intent_classification.resolve_intent", return_value="sp_api")
@patch("src.gateway.intent_classification.get_keyword_vector_results", return_value=("sp_api", "general"))
@patch("src.gateway.intent_classification.split_intents", return_value=["check order status for 112-123"])
def test_rewrite_endpoint_returns_per_intent_workflow_label(
    mock_split, mock_kv, mock_resolve, mock_rewrite, monkeypatch
):
    """Rewrite endpoint should include final per-intent workflow in intent_details."""
    monkeypatch.setenv("GATEWAY_INTENT_CLASSIFICATION_ENABLED", "true")
    payload = {
        "query": "raw query",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "ollama",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/rewrite", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("intent_details"), list)
    assert len(data["intent_details"]) == 1
    detail = data["intent_details"][0]
    assert detail["intent"] == "check order status for 112-123"
    assert detail["keyword"] == "sp_api"
    assert detail["vector"] == "general"
    assert detail["workflow"] == "sp_api"
    assert data["workflows"] == ["sp_api"]


@patch("src.gateway.api.rewrite_query")
@patch("src.gateway.api.check_ambiguity")
@patch("src.gateway.api._clarification_enabled", return_value=True)
def test_rewrite_endpoint_clarification_required_returns_early(
    mock_clar_enabled, mock_check_ambiguity, mock_rewrite
):
    """Rewrite endpoint runs clarification first; returns early when ambiguous."""
    mock_check_ambiguity.return_value = {
        "needs_clarification": True,
        "clarification_question": "Which fees do you mean?",
    }
    payload = {
        "query": "Show me the fees",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/rewrite", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["clarification_required"] is True
    assert data["clarification_question"] == "Which fees do you mean?"
    assert data["pending_query"] == "Show me the fees"
    assert data["plan"] is None
    mock_rewrite.assert_not_called()


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten general query", None, 0, 0))
def test_query_general_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call general service and return unified response."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["routing_confidence"] == 0.95
    assert data["answer"] == "general answer"
    assert data["error"] is None
    _assert_debug_payload(data)
    assert data["debug"]["rewritten_query"] == "rewritten general query"
    assert data["debug"]["route_input_query"] == "rewritten general query"
    assert data["debug"]["rewrite_enabled"] is True
    assert data["debug"]["rewrite_backend"] in ("ollama", "deepseek")
    assert isinstance(data["debug"]["rewrite_time_ms"], int)
    assert data["debug"]["rewrite_time_ms"] >= 0
    mock_call.assert_called_once()


@patch("src.gateway.api.rewrite_query")
@patch("src.gateway.api.check_ambiguity")
@patch("src.gateway.api._clarification_enabled", return_value=True)
def test_query_clarification_required_returns_early(
    mock_clar_enabled, mock_check_ambiguity, mock_rewrite
):
    """When clarification is needed, gateway returns early without rewrite or execution."""
    mock_check_ambiguity.return_value = {
        "needs_clarification": True,
        "clarification_question": "Please provide your order ID.",
    }
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["clarification_required"] is True
    assert data["clarification_question"] == "Please provide your order ID."
    assert data["pending_query"] == "test query"
    assert data["workflow"] == "clarification"
    assert data["answer"] == "Please provide your order ID."
    mock_check_ambiguity.assert_called_once()
    mock_rewrite.assert_not_called()


@patch("src.gateway.api.rewrite_query")
@patch("src.gateway.api.check_ambiguity")
@patch("src.gateway.api._clarification_enabled", return_value=True)
def test_query_show_me_the_fees_returns_clarification(
    mock_clar_enabled, mock_check_ambiguity, mock_rewrite
):
    """Ambiguous query 'Show me the fees' should return clarification question."""
    mock_check_ambiguity.return_value = {
        "needs_clarification": True,
        "clarification_question": "Which type of fees do you mean? FBA, storage, or referral?",
    }
    payload = {
        "query": "Show me the fees",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["clarification_required"] is True
    assert data["clarification_question"] == "Which type of fees do you mean? FBA, storage, or referral?"
    assert data["pending_query"] == "Show me the fees"
    assert data["workflow"] == "clarification"
    mock_check_ambiguity.assert_called_once()
    mock_rewrite.assert_not_called()


@patch("src.gateway.api.call_amazon_docs", return_value={"answer": "amazon docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("amazon_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten amazon docs query", None, 0, 0))
def test_query_amazon_docs_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call amazon_docs service when routed to amazon_docs."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "amazon_docs"
    assert data["routing_confidence"] == 0.9
    assert data["answer"] == "amazon docs answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_ic_docs", return_value={"answer": "ic docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten ic docs query", None, 0, 0))
def test_query_ic_docs_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call ic_docs service when routed to ic_docs."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["routing_confidence"] == 0.9
    assert data["answer"] == "ic docs answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_sp_api", return_value={"answer": "sp api answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("sp_api", 0.85, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten sp api query", None, 0, 0))
def test_query_sp_api_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call sp_api service when routed to sp_api."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "sp_api"
    assert data["routing_confidence"] == 0.85
    assert data["answer"] == "sp api answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_uds", return_value={"answer": "uds answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("uds", 0.88, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten uds query", None, 0, 0))
def test_query_uds_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call uds service when routed to uds."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "uds"
    assert data["routing_confidence"] == 0.88
    assert data["answer"] == "uds answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_general", return_value={"error": "backend failure", "error_type": "ConnectionError"})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
def test_query_backend_error_propagated_to_response(mock_rewrite, mock_route, mock_call):
    """When backend returns error dict, gateway should surface it in QueryResponse.error."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["answer"] == ""
    assert data["error"] == "backend failure"
    mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# Rewrite backend integration (rewrite_enable + rewrite_backend)
# ---------------------------------------------------------------------------


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.router.rewrite_with_context", return_value="rewritten by ollama")
def test_query_rewrite_backend_ollama(mock_rewrite_context, mock_route, mock_call, mock_clar):
    """POST with rewrite_enable=True, rewrite_backend=ollama uses rewrite_with_context."""
    payload = {
        "query": "  what are my sales?  ",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "ollama",
        "session_id": "sess-1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general answer"
    assert data["error"] is None
    mock_rewrite_context.assert_called_once()
    mock_call.assert_called_once_with("rewritten by ollama", "sess-1")


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.router.rewrite_with_context", return_value="rewritten by deepseek")
def test_query_rewrite_backend_deepseek(mock_rewrite_context, mock_route, mock_call, mock_clar):
    """POST with rewrite_enable=True, rewrite_backend=deepseek uses rewrite_with_context."""
    payload = {
        "query": "show revenue by month",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "deepseek",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general answer"
    assert data["error"] is None
    mock_rewrite_context.assert_called_once()
    mock_call.assert_called_once_with("rewritten by deepseek", None)


@patch("src.gateway.rewriters.rewrite_with_ollama")
@patch("src.gateway.rewriters.rewrite_with_deepseek")
@patch("src.gateway.api.call_general", return_value={"answer": "ok", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.9, "manual", None, None),
)
def test_query_rewrite_disabled_no_rewriter_call(mock_route, mock_call, mock_deepseek, mock_ollama):
    """POST with rewrite_enable=False does not call rewrite_with_ollama or rewrite_with_deepseek."""
    payload = {
        "query": "  test query  ",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    _assert_debug_payload(data)
    assert data["debug"]["rewritten_query"] == "test query"
    assert data["debug"]["route_input_query"] == "test query"
    assert data["debug"]["rewrite_enabled"] is False
    assert data["debug"]["rewrite_backend"] is None
    mock_ollama.assert_not_called()
    mock_deepseek.assert_not_called()
    mock_call.assert_called_once_with("test query", "s1")


# ---------------------------------------------------------------------------
# Route LLM: /api/v1/query with workflow=auto and route_backend (mocked routing)
# ---------------------------------------------------------------------------


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.call_uds", return_value={"answer": "uds answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("uds", 0.92, "llm", "ollama", 0.92),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
def test_query_route_llm_enabled_returns_llm_workflow_in_response(
    mock_rewrite, mock_route, mock_call_uds, mock_clar
):
    """POST with workflow=auto and route_backend: when routing uses LLM, response reflects LLM workflow."""
    payload = {
        "query": "what were my sales?",
        "workflow": "auto",
        "rewrite_enable": True,
        "route_backend": "ollama",
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "uds"
    assert data["routing_confidence"] == 0.92
    assert data["answer"] == "uds answer"
    assert data["error"] is None
    mock_route.assert_called_once()
    mock_call_uds.assert_called_once()


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.7, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
def test_query_route_llm_disabled_or_fallback_returns_heuristic_workflow(
    mock_rewrite, mock_route, mock_call_general
):
    """POST with workflow=auto when routing uses heuristic: response has heuristic workflow/confidence."""
    payload = {
        "query": "random question",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["routing_confidence"] == 0.7
    assert data["answer"] == "general answer"
    mock_route.assert_called_once()
    mock_call_general.assert_called_once()


@patch("src.gateway.api.call_ic_docs", return_value={"answer": "ic docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 1.0, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("query", None, 0, 0))
def test_query_explicit_workflow_ignores_route_backend(mock_rewrite, mock_route, mock_call_ic):
    """POST with explicit workflow=ic_docs: response uses that workflow; route_backend is irrelevant."""
    payload = {
        "query": "anything",
        "workflow": "ic_docs",
        "rewrite_enable": False,
        "route_backend": "ollama",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["routing_confidence"] == 1.0
    assert data["answer"] == "ic docs answer"
    mock_call_ic.assert_called_once()


@patch("src.gateway.api.call_general")
@patch("src.gateway.api.route_workflow")
@patch("src.gateway.api.rewrite_query", return_value=("rewritten quick query", None, 0, 0))
def test_query_rewrite_only_mode_skips_route_and_downstream(
    mock_rewrite, mock_route, mock_call_general, monkeypatch
):
    """When rewrite-only mode is on, /query returns rewritten text + plan without execution."""
    monkeypatch.setenv("GATEWAY_REWRITE_ONLY_MODE", "true")
    payload = {
        "query": "quick test",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "ollama",
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "rewrite_only"
    assert data["answer"] == "rewritten quick query"
    assert data["routing_confidence"] == 1.0
    assert data["error"] is None
    assert data["plan"] is not None
    assert data["task_results"] == []
    assert data.get("merged_answer") == ""
    _assert_debug_payload(data)
    assert data["debug"]["route_source"] == "rewrite_only"
    assert data["debug"]["route_backend"] is None
    assert data["debug"]["route_llm_confidence"] is None
    mock_route.assert_not_called()
    mock_call_general.assert_not_called()


# ---------------------------------------------------------------------------
# IC docs skip when IC_DOCS_ENABLED is false (no RAG call)
# ---------------------------------------------------------------------------


@patch("src.gateway.services._ic_docs_enabled", return_value=False)
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("framework query", None, 0, 0))
def test_query_ic_docs_disabled_returns_friendly_message_no_rag_call(
    mock_rewrite, mock_route, mock_ic_enabled
):
    """When IC docs is disabled, gateway returns friendly message and does not call RAG."""
    payload = {
        "query": "Explain IC-RAG-Agent framework.",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["error"] is None
    assert "not ready" in data["answer"].lower()
    assert "Amazon docs" in data["answer"] or "general knowledge" in data["answer"].lower()
    mock_ic_enabled.assert_called()


@patch(
    "src.gateway.api.build_execution_plan",
    return_value=(
        RewritePlan(
            plan_type="hybrid",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[
                        TaskItem(task_id="t1", workflow="general", query="what is FBA", depends_on=[], reason=None),
                        TaskItem(task_id="t2", workflow="sp_api", query="FBA fee for ASIN B074KF7RKS", depends_on=[], reason=None),
                    ],
                )
            ],
        ),
        None,
    ),
)
@patch("src.gateway.api.route_workflow")
@patch("src.gateway.api.call_sp_api", return_value={"answer": "ASIN fee is $2.10", "sources": []})
@patch("src.gateway.api.call_general", return_value={"answer": "FBA is Fulfillment by Amazon", "sources": []})
@patch("src.gateway.api.rewrite_query", return_value=("rewritten hybrid query", ["what is FBA", "get order 123", "which table stores fee"], 0, 0))
def test_query_planner_multi_task_returns_structured_response(
    mock_rewrite, mock_call_general, mock_call_sp_api, mock_route, mock_plan
):
    """Planner multi-task flow should execute grouped tasks and return merged structured output."""
    payload = {
        "query": "what is fba and fba fee for asin B074KF7RKS",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "hybrid"
    assert data["error"] is None
    assert data["merged_answer"]
    assert data["answer"] == data["merged_answer"]
    assert len(data["task_results"]) == 2
    assert data["task_results"][0]["status"] == "completed"
    assert data["task_results"][1]["status"] == "completed"
    assert data["plan"]["plan_type"] == "hybrid"
    mock_route.assert_not_called()
    mock_call_general.assert_called_once()
    mock_call_sp_api.assert_called_once()


@patch(
    "src.gateway.api.build_execution_plan",
    return_value=(
        RewritePlan(
            plan_type="hybrid",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[
                        TaskItem(task_id="t1", workflow="general", query="what is FBA", depends_on=[], reason=None),
                        TaskItem(task_id="t2", workflow="uds", query="last month fba fee total", depends_on=[], reason=None),
                    ],
                )
            ],
        ),
        None,
    ),
)
@patch("src.gateway.api.route_workflow")
@patch("src.gateway.api.call_uds", return_value={"answer": "Last month FBA fee total is $1,245", "sources": []})
@patch("src.gateway.api.call_general", return_value={"error": "backend unavailable", "error_type": "ConnectionError"})
@patch("src.gateway.api.rewrite_query", return_value=("rewritten hybrid query", ["what is FBA", "get order 123", "which table stores fee"], 0, 0))
def test_query_planner_partial_failure_surfaces_task_error(
    mock_rewrite, mock_call_general, mock_call_uds, mock_route, mock_plan
):
    """Partial task failure should keep merged answer and expose top-level partial error."""
    payload = {
        "query": "what is fba and last month fba fee",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "hybrid"
    assert data["answer"] == "Last month FBA fee total is $1,245"
    assert data["error"] == "1 planned task(s) failed."
    assert len(data["task_results"]) == 2
    statuses = {item["task_id"]: item["status"] for item in data["task_results"]}
    assert statuses["t1"] == "failed"
    assert statuses["t2"] == "completed"
    mock_route.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2.4: Hybrid plan with IC docs task when IC_DOCS_ENABLED is false
# ---------------------------------------------------------------------------


@patch(
    "src.gateway.api.build_execution_plan",
    return_value=(
        RewritePlan(
            plan_type="hybrid",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[
                        TaskItem(task_id="t1", workflow="general", query="what is FBA", depends_on=[], reason=None),
                        TaskItem(task_id="t2", workflow="ic_docs", query="explain IC-RAG framework", depends_on=[], reason=None),
                    ],
                )
            ],
        ),
        None,
    ),
)
@patch("src.gateway.api.route_workflow")
@patch("src.gateway.api.call_general", return_value={"answer": "FBA is Fulfillment by Amazon", "sources": []})
@patch("src.gateway.services._ic_docs_enabled", return_value=False)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten hybrid query", ["what is FBA", "get order 123", "which table stores fee"], 0, 0))
def test_query_hybrid_plan_with_ic_docs_disabled_returns_friendly_for_ic_task(
    mock_rewrite, mock_ic_enabled, mock_call_general, mock_route, mock_plan
):
    """Hybrid plan with general + ic_docs: when IC docs disabled, ic_docs task returns friendly 'not ready' message."""
    payload = {
        "query": "what is FBA and explain IC-RAG framework",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "hybrid"
    # Merged answer should include the general answer
    assert "FBA is Fulfillment by Amazon" in (data["merged_answer"] or "")
    # The ic_docs task should complete with the friendly message (not an error)
    assert len(data["task_results"]) == 2
    ic_task = next(r for r in data["task_results"] if r["workflow"] == "ic_docs")
    general_task = next(r for r in data["task_results"] if r["workflow"] == "general")
    assert general_task["status"] == "completed"
    assert ic_task["status"] == "completed"
    assert "not ready" in ic_task["answer"].lower()
    # Merged answer should include both pieces
    assert "not ready" in (data["merged_answer"] or "").lower()
    mock_call_general.assert_called_once()
    mock_ic_enabled.assert_called()
    mock_route.assert_not_called()

