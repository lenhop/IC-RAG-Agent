"""
Unit tests for gateway routing_heuristics module.

Covers:
- normalize_query: whitespace handling.
- route_workflow_heuristic: keyword-based workflow selection.
- apply_docs_preference: FBA definition routing override.
- split_multi_intent_clauses: clause splitting for compound queries.
"""

from __future__ import annotations

import pytest

from src.gateway.routing_heuristics import (
    apply_docs_preference,
    normalize_query,
    route_workflow_heuristic,
    split_multi_intent_clauses,
)


# ---------------------------------------------------------------------------
# normalize_query
# ---------------------------------------------------------------------------


def test_normalize_query_trims_whitespace():
    """Leading and trailing whitespace is removed."""
    assert normalize_query("  hello world  ") == "hello world"


def test_normalize_query_collapses_internal_whitespace():
    """Multiple internal spaces are collapsed to one."""
    assert normalize_query("hello   world   foo") == "hello world foo"


def test_normalize_query_empty_string():
    """Empty and None inputs return empty string."""
    assert normalize_query("") == ""
    assert normalize_query(None) == ""


# ---------------------------------------------------------------------------
# route_workflow_heuristic
# ---------------------------------------------------------------------------


class TestRouteWorkflowHeuristic:
    """Test keyword-based workflow routing."""

    def test_order_status_routes_to_sp_api(self):
        """order status keyword routes to sp_api."""
        wf, conf = route_workflow_heuristic("check order status for 123")
        assert wf == "sp_api"
        assert conf >= 0.9

    def test_inventory_placement_routes_to_sp_api(self):
        """inventory placement routes to sp_api."""
        wf, conf = route_workflow_heuristic("request an inventory placement")
        assert wf == "sp_api"
        assert conf >= 0.9

    def test_buy_box_routes_to_sp_api(self):
        """buy box status routes to sp_api."""
        wf, conf = route_workflow_heuristic("check buy box status for ASIN B074")
        assert wf == "sp_api"
        assert conf >= 0.9

    def test_removal_order_routes_to_sp_api(self):
        """removal order routes to sp_api."""
        wf, conf = route_workflow_heuristic("create a removal order for ASIN B09")
        assert wf == "sp_api"
        assert conf >= 0.9

    def test_settlement_report_routes_to_sp_api(self):
        """settlement report routes to sp_api."""
        wf, conf = route_workflow_heuristic("request a settlement report for my account")
        assert wf == "sp_api"
        assert conf >= 0.9

    def test_financial_summary_for_asin_routes_to_uds(self):
        """financial summary for ASIN routes to uds."""
        wf, conf = route_workflow_heuristic("get financial summary for ASIN B07ABC")
        assert wf == "uds"
        assert conf >= 0.9

    def test_top_5_products_routes_to_uds(self):
        """top N products routes to uds."""
        wf, conf = route_workflow_heuristic("show me top 5 products by revenue")
        assert wf == "uds"
        assert conf >= 0.9

    def test_refund_rate_routes_to_uds(self):
        """refund rate routes to uds."""
        wf, conf = route_workflow_heuristic("what is the refund rate by category")
        assert wf == "uds"
        assert conf >= 0.9

    def test_which_table_routes_to_uds(self):
        """which table routes to uds."""
        wf, conf = route_workflow_heuristic("which table stores order data")
        assert wf == "uds"
        assert conf >= 0.9

    def test_last_month_routes_to_uds(self):
        """last month routes to uds."""
        wf, conf = route_workflow_heuristic("total FBA fee last month")
        assert wf == "uds"
        assert conf >= 0.8

    def test_trend_routes_to_uds(self):
        """trend keyword routes to uds."""
        wf, conf = route_workflow_heuristic("show me revenue trend")
        assert wf == "uds"
        assert conf >= 0.8

    def test_policy_routes_to_amazon_docs(self):
        """policy keyword routes to amazon_docs."""
        wf, conf = route_workflow_heuristic("what is the FBA removal policy")
        assert wf == "amazon_docs"
        assert conf >= 0.9

    def test_fee_structure_routes_to_amazon_docs(self):
        """fee structure routes to amazon_docs."""
        wf, conf = route_workflow_heuristic("explain the referral fee structure")
        assert wf == "amazon_docs"
        assert conf >= 0.9

    def test_guidelines_routes_to_amazon_docs(self):
        """guidelines keyword routes to amazon_docs."""
        wf, conf = route_workflow_heuristic("what are Amazon's product image guidelines")
        assert wf == "amazon_docs"
        assert conf >= 0.9

    def test_ic_docs_keywords(self):
        """ic-rag-agent keyword routes to ic_docs."""
        wf, conf = route_workflow_heuristic("explain ic-rag-agent architecture")
        assert wf == "ic_docs"
        assert conf >= 0.9

    def test_general_fallback(self):
        """No keyword match falls back to general."""
        wf, conf = route_workflow_heuristic("tell me a joke about cats")
        assert wf == "general"
        assert conf == 0.7

    def test_empty_query_returns_general(self):
        """Empty query returns general."""
        wf, conf = route_workflow_heuristic("")
        assert wf == "general"
        assert conf == 0.7


# ---------------------------------------------------------------------------
# apply_docs_preference
# ---------------------------------------------------------------------------


def test_apply_docs_preference_fba_definition_overrides_sp_api():
    """Definition query about FBA overrides sp_api to ic_docs."""
    result = apply_docs_preference("what is FBA", "sp_api")
    assert result == "ic_docs"


def test_apply_docs_preference_fba_fee_definition_overrides_to_amazon_docs():
    """Definition query about FBA fees overrides sp_api to amazon_docs."""
    result = apply_docs_preference("what is FBA storage fee", "sp_api")
    assert result == "amazon_docs"


def test_apply_docs_preference_non_definition_keeps_sp_api():
    """Non-definition FBA query keeps sp_api."""
    result = apply_docs_preference("check FBA inventory for B074", "sp_api")
    assert result == "sp_api"


def test_apply_docs_preference_non_sp_api_unchanged():
    """Non sp_api workflow is never overridden."""
    result = apply_docs_preference("what is FBA", "uds")
    assert result == "uds"


# ---------------------------------------------------------------------------
# split_multi_intent_clauses
# ---------------------------------------------------------------------------


def test_split_clauses_question_starters():
    """Multiple question-starter patterns produce separate clauses."""
    clauses = split_multi_intent_clauses(
        "what is FBA get order status for 123 which table stores fee data"
    )
    assert len(clauses) >= 3


def test_split_clauses_comma_separated():
    """Comma-separated intents produce separate clauses."""
    clauses = split_multi_intent_clauses(
        "show me revenue by month, what is the refund rate"
    )
    assert len(clauses) >= 2


def test_split_clauses_semicolon_separated():
    """Semicolon-separated intents produce separate clauses."""
    clauses = split_multi_intent_clauses("total sales; average order value")
    assert len(clauses) == 2


def test_split_clauses_single_intent():
    """Single intent query returns one clause."""
    clauses = split_multi_intent_clauses("what is FBA storage fee")
    assert len(clauses) == 1


def test_split_clauses_empty():
    """Empty query returns empty list."""
    assert split_multi_intent_clauses("") == []
    assert split_multi_intent_clauses("   ") == []


def test_split_clauses_deduplicates():
    """Duplicate clauses are removed."""
    clauses = split_multi_intent_clauses("what is FBA, what is FBA")
    assert len(clauses) == 1


def test_split_clauses_date_comma_not_split():
    """Comma in date-like patterns (e.g. 'September 1st, 2026') should not split."""
    clauses = split_multi_intent_clauses("sales since September 1st, 2026")
    assert len(clauses) == 1
