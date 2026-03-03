"""
Unit tests for evaluation report generator (Phase 2.2).
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generate_html_report_minimal():
    """Report generates with minimal (empty) results."""
    from src.rag.evaluation.report_generator import generate_html_report

    out = Path("/tmp/report_test_minimal.html")
    path = generate_html_report(output_path=out)

    assert path == str(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "RAG Evaluation Report" in content
    assert "Issue List" in content


def test_generate_html_report_with_retrieval():
    """Report includes retrieval metrics table."""
    from src.rag.evaluation.report_generator import generate_html_report

    retrieval = {
        "avg_recall": 0.6,
        "avg_precision": 0.5,
        "avg_mrr": 0.4,
        "per_case": [
            {"id": "faq_001", "question": "Q1?", "recall": 0.8, "precision": 0.6, "mrr": 0.5, "min_distance": 0.3},
            {"id": "faq_002", "question": "Q2?", "recall": 0.0, "precision": 0.0, "mrr": 0.0, "min_distance": 1.5},
        ],
    }

    out = Path("/tmp/report_test_retrieval.html")
    generate_html_report(retrieval_results=retrieval, output_path=out)

    content = out.read_text(encoding="utf-8")
    assert "Retrieval Metrics" in content
    assert "faq_001" in content
    assert "faq_002" in content
    assert "0.60" in content or "0.6" in content
    assert "Recall@5 = 0.0" in content


def test_generate_html_report_with_generation():
    """Report includes generation metrics and issue list."""
    from src.rag.evaluation.report_generator import generate_html_report

    generation = {
        "faithfulness_rate": 0.5,
        "faithful_count": 1,
        "total_count": 2,
        "avg_relevance_score": 3.0,
        "per_case": [
            {"id": "faq_001", "question": "Q1?", "is_faithful": True, "relevance_score": 4, "selected_mode": "hybrid",
             "answer": "A1", "faithfulness_reasoning": "OK", "relevance_reasoning": "Good"},
            {"id": "faq_002", "question": "Q2?", "is_faithful": False, "relevance_score": 2, "selected_mode": "documents",
             "answer": "A2", "faithfulness_reasoning": "Hallucinated", "relevance_reasoning": "Incomplete"},
        ],
    }

    out = Path("/tmp/report_test_generation.html")
    generate_html_report(generation_results=generation, output_path=out)

    content = out.read_text(encoding="utf-8")
    assert "Generation Metrics" in content
    assert "Unfaithful" in content or "Hallucinated" in content
    assert "Score 2/5" in content or "2/5" in content
    assert "faq_002" in content


def test_generate_html_report_with_umap():
    """Report embeds UMAP when path provided."""
    from src.rag.evaluation.report_generator import generate_html_report

    umap_file = PROJECT_ROOT / "tests" / "evaluation" / "umap_visualization.html"
    if not umap_file.exists():
        pytest.skip("umap_visualization.html not found (run UMAP test first)")

    out_dir = Path("/tmp/report_umap_test")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "report.html"
    # Copy umap to same dir for relative path to work
    import shutil
    dest_umap = out_dir / "umap_visualization.html"
    shutil.copy(umap_file, dest_umap)

    generate_html_report(umap_path=dest_umap, output_path=out)

    content = out.read_text(encoding="utf-8")
    assert "UMAP" in content
    assert "iframe" in content


def test_issue_list_in_report():
    """Report includes issues for retrieval, faithfulness, relevance failures."""
    from src.rag.evaluation.report_generator import generate_html_report

    retrieval = {
        "per_case": [
            {"id": "faq_001", "question": "Q1", "recall": 0.0, "precision": 0.0, "mrr": 0.0, "min_distance": 1.0},
            {"id": "faq_002", "question": "Q2", "recall": 0.3, "precision": 0.2, "mrr": 0.2, "min_distance": 0.8},
        ],
    }
    generation = {
        "per_case": [
            {"id": "faq_001", "question": "Q1", "is_faithful": False, "relevance_score": 5,
             "faithfulness_reasoning": "Hallucinated", "relevance_reasoning": "", "answer": "A", "selected_mode": "hybrid", "num_contexts": 3},
            {"id": "faq_002", "question": "Q2", "is_faithful": True, "relevance_score": 2,
             "faithfulness_reasoning": "", "relevance_reasoning": "Incomplete", "answer": "A", "selected_mode": "documents", "num_contexts": 2},
        ],
    }

    out = Path("/tmp/report_test_issues.html")
    generate_html_report(retrieval_results=retrieval, generation_results=generation, output_path=out)

    content = out.read_text(encoding="utf-8")
    assert "Recall@5 = 0.0" in content or "0.0" in content
    assert "Unfaithful" in content or "Hallucinated" in content
    assert "2/5" in content or "Score 2" in content
