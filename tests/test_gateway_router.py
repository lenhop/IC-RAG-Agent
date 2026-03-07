"""
Tests for gateway.router (rewrite_query and route_workflow).

Includes Route LLM integration: when enabled, high-confidence LLM result is used;
low-confidence or errors fall back to heuristic. Explicit workflow always bypasses LLM.
"""

from __future__ import annotations

from unittest.mock import patch

from src.gateway.router import rewrite_query, route_workflow
from src.gateway.schemas import QueryRequest


def test_rewrite_query_strips_and_collapses_whitespace():
    """rewrite_query should trim and collapse internal whitespace."""
    req = QueryRequest(
        query="  what   are   my   sales   \n  this month?  ",
        workflow="auto",
        rewrite_enable=True,
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
    # Heuristic matches FBA -> sp_api or amazon_docs; keyword list has "fba" -> sp_api
    assert wf == "sp_api"
    assert conf == 0.85
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

