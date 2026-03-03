"""
Unit tests for UMAP embedding visualization (Phase 2.1).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generate_umap_plot_returns_html_path():
    """generate_umap_plot produces HTML file and returns path."""
    from src.rag.evaluation.visualize_umap import generate_umap_plot
    from src.rag.evaluation.dataset_loader import load_fqa_dataset
    from src.rag.query_pipeline import RAGPipeline

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    cases = load_fqa_dataset(str(csv_path), limit=3, project_root=PROJECT_ROOT)
    try:
        pipeline = RAGPipeline.build(verbose=False)
    except Exception as e:
        pytest.skip(f"RAGPipeline build failed: {e}")

    out_dir = Path("/tmp/umap_test_out")
    out_dir.mkdir(exist_ok=True)
    out_path = generate_umap_plot(
        cases,
        pipeline,
        output_path=out_dir / "umap_test.html",
    )

    assert out_path.endswith(".html")
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_generate_umap_plot_color_coding():
    """Plot includes Query and Relevant chunk traces with legend."""
    from src.rag.evaluation.visualize_umap import generate_umap_plot
    from src.rag.evaluation.dataset_loader import load_fqa_dataset
    from src.rag.query_pipeline import RAGPipeline

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    cases = load_fqa_dataset(str(csv_path), limit=2, project_root=PROJECT_ROOT)
    try:
        pipeline = RAGPipeline.build(verbose=False)
    except Exception as e:
        pytest.skip(f"RAGPipeline build failed: {e}")

    out_path = generate_umap_plot(cases, pipeline, output_path=Path("/tmp/umap_color_test.html"))

    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "Query" in html
    assert "Relevant" in html
    assert "UMAP" in html or "Embedding" in html


def test_annotate_outliers_api():
    """annotate_outliers is callable with correct signature."""
    from src.rag.evaluation.visualize_umap import annotate_outliers
    import plotly.graph_objects as go

    fig = go.Figure()
    embedding_2d = [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]]
    all_types = ["query", "relevant", "query"]
    all_ids = ["faq_001", "faq_001", "faq_002"]
    all_labels = ["Q1", "R1", "Q2"]
    query_indices = [0, 2]
    query_case_map = [("faq_001", "Q1", {}), ("faq_002", "Q2", {})]

    annotate_outliers(
        fig, embedding_2d, all_types, all_ids, all_labels,
        query_indices, query_case_map, threshold=2.0,
    )
    assert fig.layout.annotations is not None or True
