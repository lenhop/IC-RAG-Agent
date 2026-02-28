"""
Unit tests for parallel four-strategy intent classification.

Tests query_rewriting, intent_aggregator, and intent_methods.
Run: pytest tests/test_parallel_intent.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.query_rewriting import rewrite_query, rewrite_query_lightweight
from src.rag.intent_aggregator import aggregate_intent_signals
from src.rag.intent_methods import keywords_method_yes_no, faq_method_yes_no


class TestQueryRewriting:
    """query_rewriting module."""

    def test_rewrite_fba_fee(self):
        """FBA fee -> Amazon FBA fee."""
        assert "Amazon FBA" in rewrite_query("FBA fee")
        assert rewrite_query("FBA fee") == "Amazon FBA fee"

    def test_rewrite_empty_returns_unchanged(self):
        """Empty question returns unchanged."""
        assert rewrite_query("") == ""
        assert rewrite_query("   ") == "   "

    def test_rewrite_query_alias(self):
        """rewrite_query is alias for rewrite_query_lightweight."""
        q = "what is cost?"
        assert rewrite_query(q) == rewrite_query_lightweight(q)


class TestIntentAggregator:
    """intent_aggregator module."""

    def test_all_no_returns_general(self):
        """All No -> general."""
        assert aggregate_intent_signals([False, False, False, False]) == "general"

    def test_one_yes_returns_hybrid_default(self):
        """At least one Yes -> hybrid (default)."""
        assert aggregate_intent_signals([True, False, False, False]) == "hybrid"

    def test_one_yes_documents_mode(self):
        """With yes_to_mode=documents -> documents."""
        assert aggregate_intent_signals(
            [True, False, False, False], yes_to_mode="documents"
        ) == "documents"

    def test_empty_signals_returns_general(self):
        """Empty signals -> general."""
        assert aggregate_intent_signals([]) == "general"


class TestKeywordsMethod:
    """keywords_method_yes_no."""

    def test_fba_matches(self):
        """FBA in question -> True."""
        assert keywords_method_yes_no("what is FBA fee?") is True

    def test_no_domain_returns_false(self):
        """No domain keywords -> False."""
        assert keywords_method_yes_no("what is machine learning?") is False


class TestFaqMethod:
    """faq_method_yes_no."""

    def test_same_vector_returns_true(self):
        """Identical vectors -> True (below threshold)."""
        v = [1.0, 0.0, 0.0]
        assert faq_method_yes_no(v, [v], threshold=0.5) is True

    def test_different_vector_returns_false(self):
        """Orthogonal vectors -> False (above threshold)."""
        q = [1.0, 0.0, 0.0]
        f = [0.0, 1.0, 0.0]
        assert faq_method_yes_no(q, [f], threshold=0.5) is False

    def test_empty_faq_returns_false(self):
        """Empty faq_vectors -> False."""
        assert faq_method_yes_no([0.1, 0.2], [], threshold=0.9) is False
