"""
Unit tests for run_all_intent_methods and dict-returning methods in intent_methods.py.

Tests the new unified entry point and individual dict-returning methods.
No Chroma/Ollama required.

Run: pytest tests/test_run_all_intent_methods.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.intent_methods import (
    documents_method_response,
    keywords_method_response,
    faq_method_response,
    llm_method_response,
    run_all_intent_methods,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockDoc:
    def __init__(self):
        self.page_content = "mock"
        self.metadata = {}


def _mock_docs(n: int) -> list:
    return [_MockDoc() for _ in range(n)]


# ---------------------------------------------------------------------------
# documents_method_response
# ---------------------------------------------------------------------------

class TestDocumentsMethodResponse:

    def test_below_threshold_returns_yes(self):
        docs = _mock_docs(3)
        result = documents_method_response(docs, [0.4, 0.5, 0.6], threshold=1.0)
        assert result["yes_no"] is True
        assert result["min_dist"] == pytest.approx(0.4)
        assert result["retrieved_docs"] is docs
        assert result["distances"] == [0.4, 0.5, 0.6]
        assert result["threshold"] == 1.0

    def test_above_threshold_returns_no(self):
        docs = _mock_docs(2)
        result = documents_method_response(docs, [1.2, 1.5], threshold=1.0)
        assert result["yes_no"] is False
        assert result["min_dist"] == pytest.approx(1.2)

    def test_at_threshold_returns_yes(self):
        """Boundary: min_dist == threshold -> yes (inclusive)."""
        result = documents_method_response(_mock_docs(1), [1.0], threshold=1.0)
        assert result["yes_no"] is True

    def test_empty_distances_returns_no(self):
        result = documents_method_response([], [], threshold=1.0)
        assert result["yes_no"] is False
        assert result["min_dist"] == float("inf")

    def test_response_contains_all_keys(self):
        result = documents_method_response(_mock_docs(1), [0.5], threshold=1.0)
        assert set(result.keys()) == {"yes_no", "min_dist", "retrieved_docs", "distances", "threshold"}


# ---------------------------------------------------------------------------
# keywords_method_response
# ---------------------------------------------------------------------------

class TestKeywordsMethodResponse:

    def test_fba_matches_returns_yes(self):
        result = keywords_method_response("what is FBA fee?")
        assert result["yes_no"] is True
        assert "fba" in result["matched_signals"]

    def test_no_domain_returns_no(self):
        result = keywords_method_response("what is machine learning?")
        assert result["yes_no"] is False
        assert result["matched_signals"] == []

    def test_response_contains_all_keys(self):
        result = keywords_method_response("FBA")
        assert set(result.keys()) == {"yes_no", "matched_signals"}

    def test_amazon_keyword_matches(self):
        result = keywords_method_response("how does Amazon handle returns?")
        assert result["yes_no"] is True


# ---------------------------------------------------------------------------
# faq_method_response
# ---------------------------------------------------------------------------

class TestFaqMethodResponse:

    def test_disabled_returns_no(self):
        v = [1.0, 0.0, 0.0]
        result = faq_method_response(v, [v], enabled=False)
        assert result["yes_no"] is False

    def test_empty_faq_vectors_returns_no(self):
        result = faq_method_response([1.0, 0.0], [], enabled=True)
        assert result["yes_no"] is False

    def test_identical_vectors_returns_yes(self):
        v = [1.0, 0.0, 0.0]
        result = faq_method_response(v, [v], enabled=True, threshold=0.5)
        assert result["yes_no"] is True
        assert result["min_dist"] == pytest.approx(0.0)

    def test_orthogonal_vectors_returns_no(self):
        q = [1.0, 0.0, 0.0]
        f = [0.0, 1.0, 0.0]
        result = faq_method_response(q, [f], enabled=True, threshold=0.5)
        assert result["yes_no"] is False

    def test_response_contains_all_keys(self):
        result = faq_method_response([0.1], [[0.1]], enabled=True)
        assert set(result.keys()) == {"yes_no", "min_dist", "threshold"}


# ---------------------------------------------------------------------------
# llm_method_response
# ---------------------------------------------------------------------------

class TestLlmMethodResponse:

    def test_disabled_returns_no(self):
        result = llm_method_response("what is FBA?", enabled=False)
        assert result["yes_no"] is False

    def test_response_contains_yes_no_key(self):
        result = llm_method_response("test", enabled=False)
        assert "yes_no" in result

    @patch("src.rag.intent_classifier.classify_intent", return_value="documents")
    def test_enabled_documents_returns_yes(self, mock_classify):
        result = llm_method_response("FBA fee", enabled=True)
        assert result["yes_no"] is True

    @patch("src.rag.intent_classifier.classify_intent", return_value="general")
    def test_enabled_general_returns_no(self, mock_classify):
        result = llm_method_response("what is gravity?", enabled=True)
        assert result["yes_no"] is False


# ---------------------------------------------------------------------------
# run_all_intent_methods
# ---------------------------------------------------------------------------

class TestRunAllIntentMethods:

    def _base_call(self, **kwargs):
        """Helper: call run_all_intent_methods with sensible defaults."""
        defaults = dict(
            question_vector=[0.1] * 10,
            retrieved_docs=_mock_docs(3),
            distances=[0.4, 0.5, 0.6],
            question="what is FBA fee?",
            faq_vectors=[],
            threshold=1.0,
            faq_enabled=False,
            llm_enabled=False,
        )
        defaults.update(kwargs)
        return run_all_intent_methods(**defaults)

    def test_returns_all_four_keys(self):
        result = self._base_call()
        assert set(result.keys()) == {"documents", "keywords", "faq", "llm"}

    def test_documents_key_has_required_fields(self):
        result = self._base_call()
        doc = result["documents"]
        assert "yes_no" in doc
        assert "min_dist" in doc
        assert "retrieved_docs" in doc
        assert "distances" in doc
        assert "threshold" in doc

    def test_keywords_key_has_required_fields(self):
        result = self._base_call()
        kw = result["keywords"]
        assert "yes_no" in kw
        assert "matched_signals" in kw

    def test_faq_key_has_required_fields(self):
        result = self._base_call()
        faq = result["faq"]
        assert "yes_no" in faq
        assert "min_dist" in faq
        assert "threshold" in faq

    def test_llm_key_has_required_fields(self):
        result = self._base_call()
        assert "yes_no" in result["llm"]

    def test_documents_yes_when_below_threshold(self):
        result = self._base_call(distances=[0.4, 0.5], threshold=1.0)
        assert result["documents"]["yes_no"] is True

    def test_documents_no_when_above_threshold(self):
        result = self._base_call(distances=[1.2, 1.5], threshold=1.0)
        assert result["documents"]["yes_no"] is False

    def test_keywords_yes_for_fba_question(self):
        result = self._base_call(question="what is FBA fee?")
        assert result["keywords"]["yes_no"] is True

    def test_keywords_no_for_generic_question(self):
        result = self._base_call(question="what is machine learning?")
        assert result["keywords"]["yes_no"] is False

    def test_faq_disabled_returns_no(self):
        v = [1.0, 0.0, 0.0]
        result = self._base_call(
            question_vector=v,
            faq_vectors=[v],  # identical -> would match if enabled
            faq_enabled=False,
        )
        assert result["faq"]["yes_no"] is False

    def test_faq_enabled_identical_vectors_returns_yes(self):
        v = [1.0, 0.0, 0.0]
        result = self._base_call(
            question_vector=v,
            faq_vectors=[v],
            faq_enabled=True,
            threshold=1.0,
        )
        assert result["faq"]["yes_no"] is True

    def test_llm_disabled_returns_no(self):
        result = self._base_call(llm_enabled=False)
        assert result["llm"]["yes_no"] is False

    @patch("src.rag.intent_classifier.classify_intent", return_value="documents")
    def test_llm_enabled_documents_returns_yes(self, mock_classify):
        result = self._base_call(question="FBA fee", llm_enabled=True)
        assert result["llm"]["yes_no"] is True

    def test_retrieved_docs_passed_through(self):
        """retrieved_docs in response must be the same object passed in."""
        docs = _mock_docs(3)
        result = self._base_call(retrieved_docs=docs)
        assert result["documents"]["retrieved_docs"] is docs

    def test_distances_passed_through(self):
        distances = [0.3, 0.4, 0.5]
        result = self._base_call(distances=distances)
        assert result["documents"]["distances"] == distances

    def test_verbose_does_not_raise(self, capsys):
        """verbose=True should print without raising."""
        self._base_call(verbose=True)
        captured = capsys.readouterr()
        assert "Four methods" in captured.out or "classify" in captured.out.lower()
