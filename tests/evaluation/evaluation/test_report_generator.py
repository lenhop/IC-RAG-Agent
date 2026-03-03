"""
Unit tests for report_generator module.

Tests: _html_escape, _build_issue_list, generate_html_report.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.evaluation.report_generator import (
    _html_escape,
    _build_issue_list,
    generate_html_report,
)


# --- _html_escape ---


def test_html_escape_escapes_angle_brackets():
    """< and > -> escaped."""
    assert _html_escape("<script>") == "&lt;script&gt;"


def test_html_escape_escapes_quotes():
    """Quotes -> escaped."""
    assert "&quot;" in _html_escape('"test"')
    escaped = _html_escape("'test'")
    assert "&#x27;" in escaped or "&#39;" in escaped


def test_html_escape_empty_returns_empty():
    """Empty or None -> returns empty string."""
    assert _html_escape("") == ""
    assert _html_escape(None) == ""


def test_html_escape_safe_text_unchanged():
    """Safe text -> unchanged."""
    assert _html_escape("Hello World") == "Hello World"


# --- _build_issue_list ---


def test_build_issue_list_empty_inputs():
    """Empty retrieval and generation -> empty list."""
    assert _build_issue_list(None, None) == []
    assert _build_issue_list({}, {}) == []


def test_build_issue_list_retrieval_recall_zero():
    """Case with recall=0 -> retrieval issue (high severity)."""
    retrieval = {
        "per_case": [
            {"id": "faq_001", "recall": 0.0, "precision": 0.0, "mrr": 0.0},
        ],
    }
    issues = _build_issue_list(retrieval, None)
    assert len(issues) == 1
    assert issues[0]["type"] == "retrieval"
    assert issues[0]["id"] == "faq_001"
    assert "Recall@5 = 0.0" in issues[0]["message"]
    assert issues[0]["severity"] == "high"


def test_build_issue_list_retrieval_recall_low():
    """Case with recall < 0.4 -> medium severity."""
    retrieval = {
        "per_case": [
            {"id": "faq_002", "recall": 0.2, "precision": 0.2, "mrr": 0.2},
        ],
    }
    issues = _build_issue_list(retrieval, None)
    assert len(issues) == 1
    assert issues[0]["type"] == "retrieval"
    assert issues[0]["severity"] == "medium"


def test_build_issue_list_faithfulness_unfaithful():
    """Case with is_faithful=False -> faithfulness issue."""
    generation = {
        "per_case": [
            {
                "id": "faq_003",
                "is_faithful": False,
                "relevance_score": 5,
                "faithfulness_reasoning": "Hallucinated content",
            },
        ],
    }
    issues = _build_issue_list(None, generation)
    assert len(issues) == 1
    assert issues[0]["type"] == "faithfulness"
    assert "Unfaithful" in issues[0]["message"]
    assert issues[0]["severity"] == "high"


def test_build_issue_list_relevance_low():
    """Case with relevance_score < 4 -> relevance issue."""
    generation = {
        "per_case": [
            {
                "id": "faq_004",
                "is_faithful": True,
                "relevance_score": 2,
                "relevance_reasoning": "Incomplete",
            },
        ],
    }
    issues = _build_issue_list(None, generation)
    assert len(issues) == 1
    assert issues[0]["type"] == "relevance"
    assert "2/5" in issues[0]["message"]
    assert issues[0]["severity"] == "medium"


def test_build_issue_list_combined():
    """Retrieval + generation issues -> all included."""
    retrieval = {
        "per_case": [{"id": "faq_001", "recall": 0.0, "precision": 0.0, "mrr": 0.0}],
    }
    generation = {
        "per_case": [
            {
                "id": "faq_001",
                "is_faithful": False,
                "relevance_score": 2,
                "faithfulness_reasoning": "Bad",
                "relevance_reasoning": "Poor",
            },
        ],
    }
    issues = _build_issue_list(retrieval, generation)
    assert len(issues) >= 2
    types = {i["type"] for i in issues}
    assert "retrieval" in types
    assert "faithfulness" in types
    assert "relevance" in types


# --- generate_html_report ---


def test_generate_html_report_minimal():
    """Report with no results -> generates with Executive Summary."""
    out = Path("/tmp/report_src_test_minimal.html")
    path = generate_html_report(output_path=out)
    assert path == str(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "RAG Evaluation Report" in content
    assert "Issue List" in content


def test_generate_html_report_with_retrieval():
    """Report with retrieval results -> includes metrics table."""
    retrieval = {
        "avg_recall": 0.6,
        "avg_precision": 0.5,
        "avg_mrr": 0.4,
        "per_case": [
            {
                "id": "faq_001",
                "question": "Q1?",
                "recall": 0.8,
                "precision": 0.6,
                "mrr": 0.5,
                "min_distance": 0.3,
            },
            {
                "id": "faq_002",
                "question": "Q2?",
                "recall": 0.0,
                "precision": 0.0,
                "mrr": 0.0,
                "min_distance": 1.5,
            },
        ],
    }
    out = Path("/tmp/report_src_test_retrieval.html")
    generate_html_report(retrieval_results=retrieval, output_path=out)
    content = out.read_text(encoding="utf-8")
    assert "Retrieval Metrics" in content
    assert "faq_001" in content
    assert "faq_002" in content
    assert "0.60" in content or "0.6" in content
    assert "Recall@5 = 0.0" in content or "0.00" in content


def test_generate_html_report_with_generation():
    """Report with generation results -> includes generation table and issues."""
    generation = {
        "faithfulness_rate": 0.5,
        "faithful_count": 1,
        "total_count": 2,
        "avg_relevance_score": 3.0,
        "per_case": [
            {
                "id": "faq_001",
                "question": "Q1?",
                "is_faithful": True,
                "relevance_score": 4,
                "selected_mode": "hybrid",
                "answer": "A1",
                "faithfulness_reasoning": "OK",
                "relevance_reasoning": "Good",
            },
            {
                "id": "faq_002",
                "question": "Q2?",
                "is_faithful": False,
                "relevance_score": 2,
                "selected_mode": "documents",
                "answer": "A2",
                "faithfulness_reasoning": "Hallucinated",
                "relevance_reasoning": "Incomplete",
            },
        ],
    }
    out = Path("/tmp/report_src_test_generation.html")
    generate_html_report(generation_results=generation, output_path=out)
    content = out.read_text(encoding="utf-8")
    assert "Generation Metrics" in content
    assert "Unfaithful" in content or "Hallucinated" in content
    assert "2/5" in content or "Score 2" in content
    assert "faq_002" in content


def test_generate_html_report_with_config():
    """Report with config -> includes config section."""
    config = {"dataset": "test.csv", "limit": 10}
    out = Path("/tmp/report_src_test_config.html")
    generate_html_report(config=config, output_path=out)
    content = out.read_text(encoding="utf-8")
    assert "Configuration" in content
    assert "dataset" in content
    assert "test.csv" in content
