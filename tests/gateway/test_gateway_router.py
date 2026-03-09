"""
Tests for gateway.router (rewrite_query and route_workflow).

Includes Route LLM integration: when enabled, high-confidence LLM result is used;
low-confidence or errors fall back to heuristic. Explicit workflow always bypasses LLM.
"""

from __future__ import annotations

from unittest.mock import patch

from src.gateway.dispatcher import _correct_plan_workflows, build_execution_plan
from src.gateway.router import rewrite_query, route_workflow
from src.gateway.schemas import QueryRequest, RewritePlan, TaskGroup, TaskItem


@patch("src.gateway.router.rewrite_intents_only")
def test_rewrite_query_planner_uses_intents_only(mock_intents):
    """When planner enabled, rewrite_query uses Phase 1 intent classification."""
    mock_intents.return_value = {
        "intents": ["what is FBA", "get order 112-9876543-12", "which table stores fee data"],
    }
    with patch("src.gateway.router.planner_rewrite_enabled", return_value=True):
        req = QueryRequest(
            query="what is FBA get order 112-9876543-12 which table stores fee data",
            workflow="auto",
            rewrite_enable=True,
            session_id=None,
            stream=False,
        )
        rewritten = rewrite_query(req)
    import json
    parsed = json.loads(rewritten)
    assert parsed.get("intents") == [
        "what is FBA",
        "get order 112-9876543-12",
        "which table stores fee data",
    ]
    mock_intents.assert_called_once()


@patch("src.gateway.router.rewrite_intents_only", return_value=None)
def test_rewrite_query_planner_fallback_when_intents_fail(mock_intents):
    """When planner enabled but rewrite_intents_only fails, return normalized query."""
    with patch("src.gateway.router.planner_rewrite_enabled", return_value=True):
        req = QueryRequest(
            query="  what   is   FBA   get   order   123  ",
            workflow="auto",
            rewrite_enable=True,
            session_id=None,
            stream=False,
        )
        rewritten = rewrite_query(req)
    assert rewritten == "what is FBA get order 123"
    mock_intents.assert_called_once()


@patch("src.gateway.rewriters.rewrite_with_ollama", return_value="what are my sales this month?")
def test_rewrite_query_strips_and_collapses_whitespace(mock_ollama):
    """rewrite_query should trim and collapse internal whitespace."""
    with patch("src.gateway.router.planner_rewrite_enabled", return_value=False):
        req = QueryRequest(
            query="  what   are   my   sales   \n  this month?  ",
            workflow="auto",
            rewrite_enable=True,
            rewrite_backend="ollama",
            session_id=None,
            stream=False,
        )
        rewritten = rewrite_query(req)
    assert rewritten == "what are my sales this month?"


def test_route_workflow_respects_explicit_non_auto():
    """Explicit workflow in request should be respected with confidence 1.0."""
    req = QueryRequest(
        query="anything",
        workflow="uds",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("anything", req)
    assert wf == "uds"
    assert conf == 1.0
    assert src == "manual"
    assert backend is None
    assert llm_conf is None


def test_route_workflow_amazon_docs_keywords():
    """Queries mentioning Amazon docs should route to amazon_docs."""
    req = QueryRequest(
        query="How to use Seller Central API docs?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "amazon_docs"
    assert conf == 0.9
    assert src in ("heuristic", "llm")


def test_route_workflow_ic_docs_keywords():
    """Queries mentioning IC docs or framework should route to ic_docs."""
    req = QueryRequest(
        query="Explain sections of FRAMEWORK.md for IC-RAG-Agent.",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "ic_docs"
    assert conf == 0.9


def test_route_workflow_sp_api_keywords():
    """SP-API related queries should route to sp_api."""
    req = QueryRequest(
        query="List my Amazon orders via SP-API and shipments.",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "sp_api"
    assert conf == 0.85


def test_route_workflow_policy_query_prefers_amazon_docs():
    """Amazon policy/business-rule style query should route to amazon_docs."""
    req = QueryRequest(
        query="what does Amazon's FBA removal policy say?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "amazon_docs"
    assert conf == 0.9


def test_route_workflow_fee_definition_prefers_amazon_docs():
    """Fee-definition questions should route to amazon_docs, not sp_api."""
    req = QueryRequest(
        query="what are the differences between FBA standard and oversize fees?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "amazon_docs"
    assert conf == 0.9


def test_route_workflow_table_storage_query_prefers_uds():
    """Table/schema-style fee data lookup should route to UDS."""
    req = QueryRequest(
        query="which table stores referral fee data",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "uds"
    assert conf >= 0.85  # Strong UDS table cues use 0.92


def test_route_workflow_historical_fee_query_prefers_uds():
    """Historical aggregate fee queries should route to UDS."""
    req = QueryRequest(
        query="what was the total referral fee for last quarter?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "uds"
    assert conf == 0.85


def test_route_workflow_order_status_prefers_sp_api():
    """Order status and inventory placement queries should route to SP-API."""
    req = QueryRequest(
        query="get order status for 112-9876543-1234567",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "sp_api"
    assert conf >= 0.9

    req2 = QueryRequest(
        query="request an inventory placement",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf2, conf2, _, _, _ = route_workflow(req2.query.lower(), req2)
    assert wf2 == "sp_api"
    assert conf2 >= 0.9


def test_route_workflow_removal_order_prefers_sp_api():
    """Removal order and stranded inventory queries should route to SP-API."""
    req = QueryRequest(
        query="create a removal order for stranded inventory",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, _, _, _ = route_workflow(req.query.lower(), req)
    assert wf == "sp_api"
    assert conf >= 0.9


def test_route_workflow_inventory_buybox_account_prefers_sp_api():
    """Inventory health, buy box status, and seller account queries should route to SP-API."""
    sp_api_queries = [
        "check if ASIN B07JKL7890 has any active suppressed listings",
        "request a detailed inventory health report",
        "check if ASIN B08N5WRWNW is currently buy-box eligible",
        "get my current inventory health summary",
        "request a detailed financial report for my account",
        "get real-time buy box status for ASIN B09XYZ",
        "verify my seller account verification status",
    ]
    for q in sp_api_queries:
        req = QueryRequest(
            query=q,
            workflow="auto",
            rewrite_enable=True,
            session_id=None,
            stream=False,
        )
        wf, conf, _, _, _ = route_workflow(req.query.lower(), req)
        assert wf == "sp_api", f"Expected sp_api for: {q[:50]!r}"
        assert conf >= 0.9


def test_route_workflow_financial_summary_for_asin_prefers_uds():
    """Financial summary for ASIN (data retrieval) should route to UDS."""
    req = QueryRequest(
        query="get financial summary for ASIN B07ABC5678 including fees and revenue",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, _, _, _ = route_workflow(req.query.lower(), req)
    assert wf == "uds"
    assert conf >= 0.9


def test_route_workflow_top_products_by_refund_prefers_uds():
    """Top N products by X analytical queries should route to UDS."""
    req = QueryRequest(
        query="show me top 5 products by refund count",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "uds"
    assert conf >= 0.9


def test_route_workflow_definition_fba_prefers_ic_docs():
    """Definition-style FBA queries should route to ic_docs, not sp_api."""
    req = QueryRequest(
        query="What is FBA?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "ic_docs"
    assert conf in (0.85, 0.9)


def test_route_workflow_uds_keywords():
    """Analytical questions about sales / revenue should route to uds."""
    req = QueryRequest(
        query="What were total sales and revenue by month?",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "uds"
    assert conf == 0.85


def test_route_workflow_fallback_general():
    """Non-matching queries should fall back to general workflow."""
    req = QueryRequest(
        query="Tell me a joke about databases.",
        workflow="auto",
        rewrite_enable=False,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow(req.query.lower(), req)
    assert wf == "general"
    assert conf == 0.7
    assert src == "heuristic"
    assert backend is None
    assert llm_conf is None


def test_format_route_metadata():
    """Ensure routing metadata is formatted as expected for logging."""
    from src.gateway.logging_utils import format_route_metadata

    m1 = format_route_metadata("manual", None, None)
    assert "source=manual" in m1
    assert "backend=none" in m1
    assert "llm_confidence=null" in m1

    m2 = format_route_metadata("llm", "ollama", 0.8234)
    # confidence should be rounded to two decimal places
    assert "backend=ollama" in m2
    assert m2.endswith("llm_confidence=0.82")


# ---------------------------------------------------------------------------
# Route LLM integration: high confidence -> LLM; low confidence -> heuristic
# ---------------------------------------------------------------------------


@patch("src.gateway.router._route_llm_enabled", return_value=True)
@patch("src.gateway.route_llm.route_with_llm", return_value=("uds", 0.9))
def test_route_workflow_llm_high_confidence_uses_llm_result(mock_route_llm, mock_enabled):
    """When Route LLM is enabled and returns confidence >= threshold, use LLM workflow."""
    req = QueryRequest(
        query="some generic question",
        workflow="auto",
        rewrite_enable=False,
        route_backend="ollama",
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("some generic question", req)
    assert wf == "uds"
    assert conf == 0.9
    assert src == "llm"
    assert backend == "ollama"
    assert llm_conf == 0.9
    mock_route_llm.assert_called_once_with("some generic question", "ollama")


@patch("src.gateway.router._route_llm_enabled", return_value=True)
@patch("src.gateway.route_llm.route_with_llm", return_value=("general", 0.3))
def test_route_workflow_llm_low_confidence_falls_back_to_heuristic(mock_route_llm, mock_enabled):
    """When Route LLM confidence < threshold, use heuristic routing instead."""
    req = QueryRequest(
        query="what were my sales and revenue?",
        workflow="auto",
        rewrite_enable=False,
        route_backend="ollama",
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("what were my sales and revenue?", req)
    # Heuristic should match "sales", "revenue" -> uds
    assert wf == "uds"
    assert conf == 0.85
    assert src == "heuristic"
    assert backend is None
    assert llm_conf is None
    mock_route_llm.assert_called_once()


@patch("src.gateway.router._route_llm_enabled", return_value=True)
@patch("src.gateway.route_llm.route_with_llm", return_value=("general", 0.0))
def test_route_workflow_llm_safe_default_falls_back_to_heuristic(mock_route_llm, mock_enabled):
    """When Route LLM returns safe default (0.0), fall back to heuristic."""
    req = QueryRequest(
        query="tell me about FBA fees",
        workflow="auto",
        rewrite_enable=False,
        route_backend="ollama",
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("tell me about FBA fees", req)
    # Heuristic should prefer amazon_docs for fee/policy style FBA question.
    assert wf == "amazon_docs"
    assert conf == 0.9
    assert src == "heuristic"
    assert backend is None
    assert llm_conf is None


@patch("src.gateway.router._route_llm_enabled", return_value=True)
@patch("src.gateway.route_llm.route_with_llm", return_value=("sp_api", 0.92))
def test_route_workflow_llm_sp_api_is_overridden_for_definition_fba(mock_route_llm, mock_enabled):
    """High-confidence LLM sp_api for 'what is FBA' is corrected to ic_docs."""
    req = QueryRequest(
        query="what is FBA",
        workflow="auto",
        rewrite_enable=False,
        route_backend="ollama",
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("what is FBA", req)
    assert wf == "ic_docs"
    assert conf == 0.92
    assert src == "llm"
    assert backend == "ollama"
    assert llm_conf == 0.92
    mock_route_llm.assert_called_once_with("what is FBA", "ollama")


@patch("src.gateway.router._route_llm_enabled", return_value=False)
def test_route_workflow_llm_disabled_uses_heuristic_only(mock_enabled):
    """When Route LLM is disabled, route_workflow uses only heuristic (no LLM call)."""
    req = QueryRequest(
        query="random question with no keywords",
        workflow="auto",
        rewrite_enable=False,
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("random question with no keywords", req)
    assert wf == "general"
    assert conf == 0.7
    assert src == "heuristic"
    assert backend is None
    assert llm_conf is None


@patch("src.gateway.router._route_llm_enabled", return_value=True)
@patch("src.gateway.route_llm.route_with_llm")
def test_route_workflow_explicit_workflow_bypasses_llm(mock_route_llm, mock_enabled):
    """Explicit workflow != 'auto' must be used and Route LLM must not be called."""
    req = QueryRequest(
        query="anything",
        workflow="ic_docs",
        rewrite_enable=False,
        route_backend="ollama",
        session_id=None,
        stream=False,
    )
    wf, conf, src, backend, llm_conf = route_workflow("anything", req)
    assert wf == "ic_docs"
    assert conf == 1.0
    assert src == "manual"
    assert backend is None
    assert llm_conf is None
    mock_route_llm.assert_not_called()


def test_build_execution_plan_explicit_workflow_creates_single_task():
    """Explicit workflow should bypass planner and create one task plan."""
    req = QueryRequest(
        query="show sales",
        workflow="uds",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    plan = build_execution_plan(req, "sales by month")
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) == 1
    task = plan.task_groups[0].tasks[0]
    assert task.workflow == "uds"
    assert task.query == "sales by month"


@patch("src.gateway.dispatcher.planner_rewrite_enabled", return_value=True)
def test_build_execution_plan_date_with_comma_stays_single_intent(mock_planner):
    """Query with date like 'September 1st, 2026' must not be split; 2026 must not become separate general task."""
    req = QueryRequest(
        query="how many orders were there on September 1st, 2026",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    rewritten = '{"intents": ["how many orders were there on September 1st, 2026"]}'
    plan = build_execution_plan(req, rewritten)
    assert len(plan.task_groups) == 1
    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 1
    assert tasks[0].query == "how many orders were there on September 1st, 2026"
    assert tasks[0].workflow == "uds"
    # Ensure "2026" was not split into a separate general task
    assert not any(t.workflow == "general" and t.query.strip() == "2026" for t in tasks)


@patch("src.gateway.dispatcher.planner_rewrite_enabled", return_value=True)
def test_build_execution_plan_uses_intents_only_when_phase1_succeeds(mock_planner):
    """Two-phase flow: intents-only JSON should produce one task per intent via heuristics."""
    req = QueryRequest(
        query="what is FBA get order 112-9876543-12 which table stores fee data",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    rewritten = '{"intents": ["what is FBA", "get order 112-9876543-12", "which table stores fee data"]}'
    plan = build_execution_plan(req, rewritten)
    assert plan.plan_type == "hybrid"
    assert len(plan.task_groups) == 1
    tasks = plan.task_groups[0].tasks
    assert len(tasks) == 3
    workflows = {t.workflow for t in tasks}
    assert "ic_docs" in workflows or "general" in workflows  # "what is FBA"
    assert "sp_api" in workflows  # "get order ..."
    assert "uds" in workflows  # "which table stores..."


@patch("src.gateway.dispatcher.planner_rewrite_enabled", return_value=True)
@patch("src.gateway.dispatcher.parse_rewrite_plan_text")
def test_build_execution_plan_uses_planner_output_when_available(mock_parse, mock_planner):
    """Planner-enabled auto mode should use parsed planner execution plan."""
    req = QueryRequest(
        query="what is fba and sales trend",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    mock_parse.return_value = build_execution_plan(
        QueryRequest(query="fallback", workflow="general", rewrite_enable=True, session_id=None, stream=False),
        "fallback query",
    )
    plan = build_execution_plan(req, '{"plan_type":"hybrid"}')
    assert plan.task_groups
    mock_parse.assert_called_once()


@patch("src.gateway.dispatcher.planner_rewrite_enabled", return_value=True)
@patch("src.gateway.dispatcher.parse_rewrite_plan_text", return_value=None)
@patch("src.gateway.router.route_workflow", return_value=("sp_api", 0.85, "heuristic", None, None))
def test_build_execution_plan_falls_back_to_routing_when_planner_invalid(
    mock_route, mock_parse, mock_planner
):
    """Invalid planner output should fall back to routed single-task plan."""
    req = QueryRequest(
        query="fba fee details",
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    plan = build_execution_plan(req, "fba fee details")
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) == 1
    assert plan.task_groups[0].tasks[0].workflow == "sp_api"
    mock_parse.assert_called_once()
    mock_route.assert_called_once()


def test_build_execution_plan_mixed_query_creates_multi_task_fallback():
    """Mixed clause query should be decomposed into multiple tasks as fallback."""
    req = QueryRequest(
        query=(
            "whats the weather today, what is the FBA and Storage fee, "
            "how many FBA fee for ASIN B074KF7RKS, which table can I get ASIN FBA fee"
        ),
        workflow="auto",
        rewrite_enable=True,
        session_id=None,
        stream=False,
    )
    plan = build_execution_plan(req, req.query)
    assert plan.plan_type == "hybrid"
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) >= 2
    workflows = {task.workflow for task in plan.task_groups[0].tasks}
    assert "general" in workflows
    assert "amazon_docs" in workflows or "sp_api" in workflows


def test_correct_plan_workflows_overrides_misclassified_tasks():
    """Plan correction should fix LLM misclassifications via heuristic override."""
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
                        query="get order status for 112-9876543-12",
                        depends_on=[],
                        reason="",
                    ),
                    TaskItem(
                        task_id="t2",
                        workflow="general",
                        query="show me top 5 products by refund count",
                        depends_on=[],
                        reason="",
                    ),
                    TaskItem(
                        task_id="t3",
                        workflow="general",
                        query="request an inventory placement",
                        depends_on=[],
                        reason="",
                    ),
                ],
            )
        ],
    )
    corrected = _correct_plan_workflows(plan)
    tasks = corrected.task_groups[0].tasks
    assert tasks[0].workflow == "sp_api"
    assert tasks[1].workflow == "uds"
    assert tasks[2].workflow == "sp_api"

