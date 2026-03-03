"""
Unit tests for dataset_loader (Phase 1.3).

Tests load_fqa_dataset, validate_dataset, add_relevant_contexts per
PHASE_1_3_DATASET_LOADER_SPEC.md.
"""

import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_fqa_dataset():
    """Load amazon_fqa.csv and verify structure."""
    from src.rag.evaluation.dataset_loader import load_fqa_dataset

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    test_cases = load_fqa_dataset(str(csv_path), limit=10, project_root=PROJECT_ROOT)

    assert len(test_cases) == 10
    assert test_cases[0]["id"] == "faq_001"
    assert "question" in test_cases[0]
    assert "ground_truth" in test_cases[0]
    assert "category" in test_cases[0]


def test_load_fqa_dataset_limit():
    """Test limit parameter works."""
    from src.rag.evaluation.dataset_loader import load_fqa_dataset

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    test_cases = load_fqa_dataset(str(csv_path), limit=3, project_root=PROJECT_ROOT)
    assert len(test_cases) == 3


def test_validate_dataset_valid():
    """Valid dataset returns True."""
    from src.rag.evaluation.dataset_loader import validate_dataset

    test_cases = [
        {"id": "test_1", "question": "Q1", "ground_truth": "A1"},
        {"id": "test_2", "question": "Q2", "ground_truth": "A2"},
    ]
    assert validate_dataset(test_cases, warn_missing=False) is True


def test_validate_dataset_missing_question():
    """Missing question field returns False."""
    from src.rag.evaluation.dataset_loader import validate_dataset

    test_cases = [
        {"id": "test_1", "ground_truth": "A1"},  # No question
    ]
    assert validate_dataset(test_cases, warn_missing=False) is False


def test_validate_dataset_missing_ground_truth():
    """Missing ground_truth field returns False."""
    from src.rag.evaluation.dataset_loader import validate_dataset

    test_cases = [
        {"id": "test_1", "question": "Q1"},  # No ground_truth
    ]
    assert validate_dataset(test_cases, warn_missing=False) is False


def test_validate_dataset_missing_contexts():
    """Missing contexts shows warning but returns True (not fatal)."""
    import warnings

    from src.rag.evaluation.dataset_loader import validate_dataset

    test_cases = [
        {"id": "test_1", "question": "Q1", "ground_truth": "A1"},  # No contexts
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_dataset(test_cases, warn_missing=True)

    assert result is True
    assert any("contexts" in str(m.message) for m in w)


def test_add_relevant_contexts():
    """Test auto-retrieve contexts fallback and contexts_source metadata."""
    from src.rag.evaluation.dataset_loader import add_relevant_contexts
    from src.rag.query_pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline.build(verbose=False)
    except Exception as e:
        pytest.skip(f"RAGPipeline build failed: {e}")

    test_cases = [
        {"id": "test_1", "question": "What is FBA?", "ground_truth": "Answer"},
    ]

    captured = io.StringIO()
    sys.stdout = captured
    try:
        updated = add_relevant_contexts(test_cases, pipeline, k=3)
    finally:
        sys.stdout = sys.__stdout__

    assert "contexts" in updated[0]
    assert len(updated[0]["contexts"]) <= 3
    assert updated[0].get("contexts_source") == "auto_retrieved"
