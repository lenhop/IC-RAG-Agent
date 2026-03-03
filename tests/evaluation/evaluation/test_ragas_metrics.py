"""
Unit tests for ragas_metrics module.

Tests: compare_metrics only (do NOT test evaluate_with_ragas).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch

import pytest

from src.rag.evaluation.ragas_metrics import compare_metrics


# --- compare_metrics ---


def test_compare_metrics_empty_results():
    """Both inputs empty -> Insufficient data."""
    result = compare_metrics({}, {})
    assert result["faithfulness_correlation"] == 0.0
    assert result["relevance_correlation"] == 0.0
    assert "Insufficient data" in result["recommendation"]


def test_compare_metrics_one_empty():
    """One input empty -> Insufficient data."""
    custom = {"per_case": [{"id": "1", "is_faithful": True, "relevance_score": 5}]}
    result = compare_metrics(custom, {})
    assert "Insufficient data" in result["recommendation"]


def test_compare_metrics_no_common_ids():
    """No common IDs -> Too few common cases."""
    custom = {
        "per_case": [{"id": "faq_001", "is_faithful": True, "relevance_score": 5}],
    }
    ragas = {
        "per_case": [{"id": "faq_002", "faithfulness": 1.0, "answer_relevancy": 1.0}],
    }
    result = compare_metrics(custom, ragas)
    assert "Too few common" in result["recommendation"]


def test_compare_metrics_high_correlation():
    """High correlation -> recommendation mentions High correlation."""
    custom = {
        "per_case": [
            {"id": "faq_001", "question": "Q1", "is_faithful": True, "relevance_score": 5},
            {"id": "faq_002", "question": "Q2", "is_faithful": True, "relevance_score": 4},
            {"id": "faq_003", "question": "Q3", "is_faithful": False, "relevance_score": 2},
        ],
    }
    ragas = {
        "per_case": [
            {"id": "faq_001", "faithfulness": 0.95, "answer_relevancy": 0.98},
            {"id": "faq_002", "faithfulness": 0.90, "answer_relevancy": 0.85},
            {"id": "faq_003", "faithfulness": 0.10, "answer_relevancy": 0.40},
        ],
    }
    result = compare_metrics(custom, ragas)
    assert result["faithfulness_correlation"] > 0.7
    assert result["relevance_correlation"] > 0.7
    assert "High correlation" in result["recommendation"]


def test_compare_metrics_low_correlation():
    """Low correlation -> recommendation mentions Low correlation."""
    custom = {
        "per_case": [
            {"id": "faq_001", "is_faithful": True, "relevance_score": 5},
            {"id": "faq_002", "is_faithful": False, "relevance_score": 1},
        ],
    }
    ragas = {
        "per_case": [
            {"id": "faq_001", "faithfulness": 0.2, "answer_relevancy": 0.3},
            {"id": "faq_002", "faithfulness": 0.9, "answer_relevancy": 0.8},
        ],
    }
    result = compare_metrics(custom, ragas)
    assert "Low correlation" in result["recommendation"] or "disagree" in result["recommendation"]


def test_compare_metrics_discrepancies():
    """Large differences -> discrepancies list populated."""
    custom = {
        "per_case": [
            {"id": "faq_001", "is_faithful": True, "relevance_score": 5},
            {"id": "faq_002", "is_faithful": False, "relevance_score": 2},
        ],
    }
    ragas = {
        "per_case": [
            {"id": "faq_001", "faithfulness": 0.2, "answer_relevancy": 0.3},
            {"id": "faq_002", "faithfulness": 0.9, "answer_relevancy": 0.8},
        ],
    }
    result = compare_metrics(custom, ragas)
    assert result["discrepancy_count"] >= 1
    assert len(result["discrepancies"]) >= 1
    assert "faithfulness_diff" in result["discrepancies"][0]


def test_compare_metrics_scipy_import_error():
    """compare_metrics raises ImportError when scipy not installed."""
    with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
        with pytest.raises(ImportError, match="scipy required"):
            compare_metrics(
                {"per_case": [{"id": "1", "is_faithful": True, "relevance_score": 5}]},
                {"per_case": [{"id": "1", "faithfulness": 1.0, "answer_relevancy": 1.0}]},
            )
