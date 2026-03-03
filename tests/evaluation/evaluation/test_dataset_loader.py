"""
Unit tests for dataset_loader module.

Tests: _resolve_csv_path, load_fqa_dataset, validate_dataset.
All tests use mocks - no real CSV files except for load_fqa_dataset with tempfile.
"""

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag.evaluation.dataset_loader import (
    _resolve_csv_path,
    load_fqa_dataset,
    validate_dataset,
)


# --- _resolve_csv_path ---


def test_resolve_csv_path_absolute_unchanged():
    """Absolute path -> returns resolved path unchanged (relative to root)."""
    abs_path = "/tmp/absolute.csv"
    result = _resolve_csv_path(abs_path, project_root=PROJECT_ROOT)
    assert result == Path(abs_path).resolve()


def test_resolve_csv_path_relative_joins_with_root():
    """Relative path -> joins with project_root."""
    result = _resolve_csv_path("data/test.csv", project_root=PROJECT_ROOT)
    assert result == (PROJECT_ROOT / "data" / "test.csv").resolve()


def test_resolve_csv_path_relative_no_root_uses_default():
    """Relative path with project_root=None -> uses module PROJECT_ROOT."""
    result = _resolve_csv_path("data/test.csv", project_root=None)
    assert "data" in str(result)
    assert "test.csv" in str(result)


# --- load_fqa_dataset ---


def test_load_fqa_dataset_from_temp_csv():
    """Load from temp CSV with question, answer, category -> correct structure."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "category", "source"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow({
            "question": "What is FBA?",
            "answer": "Fulfillment by Amazon",
            "category": "shipping",
            "source": "Amazon",
        })
        writer.writerow({
            "question": "How to refund?",
            "answer": "Go to Orders",
            "category": "returns",
            "source": "Amazon",
        })
        path = f.name

    try:
        cases = load_fqa_dataset(path, project_root=Path("/"))
        assert len(cases) == 2
        assert cases[0]["id"] == "faq_001"
        assert cases[0]["question"] == "What is FBA?"
        assert cases[0]["ground_truth"] == "Fulfillment by Amazon"
        assert cases[0]["category"] == "shipping"
        assert cases[1]["id"] == "faq_002"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_fqa_dataset_limit():
    """limit=1 -> returns only first row."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"], extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"question": "Q1", "answer": "A1"})
        writer.writerow({"question": "Q2", "answer": "A2"})
        path = f.name

    try:
        cases = load_fqa_dataset(path, limit=1, project_root=Path("/"))
        assert len(cases) == 1
        assert cases[0]["question"] == "Q1"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_fqa_dataset_contexts_column():
    """CSV with contexts column -> parsed into list."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "contexts"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow({
            "question": "Q?",
            "answer": "A",
            "contexts": "ctx1 | ctx2",
        })
        path = f.name

    try:
        cases = load_fqa_dataset(path, project_root=Path("/"))
        assert len(cases) == 1
        assert cases[0]["contexts"] == ["ctx1", "ctx2"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_fqa_dataset_nonexistent_raises():
    """Non-existent file -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_fqa_dataset("/nonexistent/path/to/file.csv", project_root=Path("/"))


def test_load_fqa_dataset_empty_csv_warns():
    """Empty CSV (header only) -> returns [] with warning."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        f.write("question,answer\n")
        path = f.name

    try:
        with pytest.warns(UserWarning, match="No test cases"):
            cases = load_fqa_dataset(path, project_root=Path("/"))
        assert cases == []
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_fqa_dataset_ground_truth_fallback():
    """CSV with ground_truth column (no answer) -> uses ground_truth."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "ground_truth"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow({"question": "Q?", "ground_truth": "GT answer"})
        path = f.name

    try:
        cases = load_fqa_dataset(path, project_root=Path("/"))
        assert cases[0]["ground_truth"] == "GT answer"
    finally:
        Path(path).unlink(missing_ok=True)


# --- validate_dataset ---


def test_validate_dataset_valid():
    """Valid cases with question and ground_truth -> True."""
    cases = [
        {"id": "1", "question": "Q1", "ground_truth": "A1"},
        {"id": "2", "question": "Q2", "ground_truth": "A2"},
    ]
    assert validate_dataset(cases, warn_missing=False) is True


def test_validate_dataset_empty_returns_false():
    """Empty list -> False."""
    assert validate_dataset([], warn_missing=False) is False


def test_validate_dataset_missing_question_returns_false():
    """Case missing question -> False."""
    cases = [{"id": "1", "ground_truth": "A1"}]
    assert validate_dataset(cases, warn_missing=False) is False


def test_validate_dataset_missing_ground_truth_returns_false():
    """Case missing ground_truth -> False."""
    cases = [{"id": "1", "question": "Q1"}]
    assert validate_dataset(cases, warn_missing=False) is False


def test_validate_dataset_missing_contexts_returns_true():
    """Cases missing contexts -> returns True (warning only, not error)."""
    cases = [{"id": "1", "question": "Q1", "ground_truth": "A1"}]
    assert validate_dataset(cases, warn_missing=False) is True


def test_validate_dataset_empty_question_returns_false():
    """Case with empty question string -> False."""
    cases = [{"id": "1", "question": "   ", "ground_truth": "A1"}]
    assert validate_dataset(cases, warn_missing=False) is False
