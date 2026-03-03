"""
Unit tests for RAGAS integration (Phase 3.2).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_evaluate_with_ragas_import_error():
    """evaluate_with_ragas raises ImportError when RAGAS not installed."""
    from src.rag.evaluation.ragas_metrics import evaluate_with_ragas

    # Need non-empty test cases to trigger import
    test_cases = [{"question": "Q1", "ground_truth": "A1"}]
    with patch.dict("sys.modules", {"ragas": None}):
        with pytest.raises(ImportError, match="RAGAS not installed"):
            evaluate_with_ragas(test_cases, MagicMock())


def test_evaluate_with_ragas_empty_cases():
    """evaluate_with_ragas handles empty test cases."""
    from src.rag.evaluation.ragas_metrics import evaluate_with_ragas

    result = evaluate_with_ragas([], MagicMock())
    assert result["context_precision"] == 0.0
    assert result["answer_relevancy"] == 0.0
    assert result["faithfulness"] == 0.0
    assert result["total_count"] == 0


def test_compare_metrics_empty_results():
    """compare_metrics handles empty results."""
    from src.rag.evaluation.ragas_metrics import compare_metrics

    result = compare_metrics({}, {})
    assert result["faithfulness_correlation"] == 0.0
    assert result["relevance_correlation"] == 0.0
    assert "Insufficient data" in result["recommendation"]


def test_compare_metrics_no_common_ids():
    """compare_metrics handles no common IDs."""
    from src.rag.evaluation.ragas_metrics import compare_metrics

    custom = {"per_case": [{"id": "faq_001", "is_faithful": True, "relevance_score": 5}]}
    ragas = {"per_case": [{"id": "faq_002", "faithfulness": 1.0, "answer_relevancy": 1.0}]}

    result = compare_metrics(custom, ragas)
    assert "Too few common cases" in result["recommendation"]


def test_compare_metrics_high_correlation():
    """compare_metrics detects high correlation."""
    from src.rag.evaluation.ragas_metrics import compare_metrics

    custom = {
        "per_case": [
            {"id": "faq_001", "question": "Q1", "is_faithful": True, "relevance_score": 5},
            {"id": "faq_002", "question": "Q2", "is_faithful": True, "relevance_score": 4},
            {"id": "faq_003", "question": "Q3", "is_faithful": False, "relevance_score": 2},
        ]
    }
    ragas = {
        "per_case": [
            {"id": "faq_001", "faithfulness": 0.95, "answer_relevancy": 0.98},
            {"id": "faq_002", "faithfulness": 0.90, "answer_relevancy": 0.85},
            {"id": "faq_003", "faithfulness": 0.10, "answer_relevancy": 0.40},
        ]
    }

    result = compare_metrics(custom, ragas)
    assert result["faithfulness_correlation"] > 0.7
    assert result["relevance_correlation"] > 0.7
    assert "High correlation" in result["recommendation"]


def test_compare_metrics_discrepancies():
    """compare_metrics identifies discrepancies."""
    from src.rag.evaluation.ragas_metrics import compare_metrics

    custom = {
        "per_case": [
            {"id": "faq_001", "question": "Q1", "is_faithful": True, "relevance_score": 5},
            {"id": "faq_002", "question": "Q2", "is_faithful": False, "relevance_score": 2},
        ]
    }
    ragas = {
        "per_case": [
            {"id": "faq_001", "faithfulness": 0.2, "answer_relevancy": 0.3},  # Discrepancy
            {"id": "faq_002", "faithfulness": 0.9, "answer_relevancy": 0.8},  # Discrepancy
        ]
    }

    result = compare_metrics(custom, ragas)
    assert result["discrepancy_count"] == 2
    assert len(result["discrepancies"]) == 2
    assert result["discrepancies"][0]["faithfulness_diff"] > 0.3


def test_compare_metrics_import_error():
    """compare_metrics raises ImportError when scipy not installed."""
    from src.rag.evaluation.ragas_metrics import compare_metrics

    with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
        with pytest.raises(ImportError, match="scipy required"):
            compare_metrics(
                {"per_case": [{"id": "1", "is_faithful": True, "relevance_score": 5}]},
                {"per_case": [{"id": "1", "faithfulness": 1.0, "answer_relevancy": 1.0}]},
            )
