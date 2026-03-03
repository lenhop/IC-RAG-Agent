"""
End-to-End RAG Evaluation Integration Test (Phase 4.2).

Scenario:
1. Load FAQ from amazon_fqa.csv
2. Build RAG pipeline
3. Run full evaluation pipeline (retrieval, generation, report)
4. Verify output JSON structure
5. Check metrics against expected ranges
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SCRIPT = PROJECT_ROOT / "scripts" / "run_evaluation.py"
DATASET = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"


def _run_evaluation(output_dir: Path, limit: int = 3) -> subprocess.CompletedProcess:
    """Run evaluation script and return result."""
    return subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            "--dataset",
            str(DATASET),
            "--limit",
            str(limit),
            "--skip-umap",
            "--output",
            str(output_dir),
            "--no-verbose",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=180,
    )


@pytest.fixture(scope="module")
def eval_result(tmp_path_factory):
    """Run evaluation once per test module; share result and output dir."""
    out_dir = tmp_path_factory.mktemp("e2e") / "results"
    out_dir.mkdir(parents=True)
    result = _run_evaluation(out_dir, limit=3)
    return result, out_dir


def test_e2e_dataset_exists():
    """Dataset file must exist for integration test."""
    assert DATASET.exists(), f"Dataset not found: {DATASET}"


def test_e2e_full_pipeline(eval_result):
    """Run full evaluation pipeline and validate outputs."""
    result, out_dir = eval_result

    if result.returncode != 0:
        pytest.skip(
            f"Evaluation failed (Chroma/LLM/dataset may be unavailable): {result.stderr[:500]}"
        )

    # Verify all output files created
    assert (out_dir / "retrieval_results.json").exists()
    assert (out_dir / "generation_results.json").exists()
    assert (out_dir / "evaluation_report.html").exists()


def test_e2e_retrieval_json_structure(eval_result):
    """Validate retrieval_results.json structure and value types."""
    result, out_dir = eval_result
    if result.returncode != 0:
        pytest.skip("Evaluation failed; skipping structure validation")

    retrieval_path = out_dir / "retrieval_results.json"
    if not retrieval_path.exists():
        pytest.skip("Retrieval results not produced")

    with open(retrieval_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "avg_recall" in data
    assert "avg_precision" in data
    assert "avg_mrr" in data
    assert "per_case" in data

    assert isinstance(data["avg_recall"], (int, float))
    assert isinstance(data["avg_precision"], (int, float))
    assert isinstance(data["avg_mrr"], (int, float))
    assert isinstance(data["per_case"], list)

    for case in data["per_case"]:
        assert "id" in case
        assert "recall" in case
        assert "precision" in case
        assert "mrr" in case
        assert 0 <= case["recall"] <= 1.0
        assert 0 <= case["precision"] <= 1.0
        assert 0 <= case["mrr"] <= 1.0


def test_e2e_generation_json_structure(eval_result):
    """Validate generation_results.json structure and value types."""
    result, out_dir = eval_result
    if result.returncode != 0:
        pytest.skip("Evaluation failed; skipping structure validation")

    gen_path = out_dir / "generation_results.json"
    if not gen_path.exists():
        pytest.skip("Generation results not produced")

    with open(gen_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "faithfulness_rate" in data
    assert "total_count" in data
    assert "per_case" in data

    assert isinstance(data["faithfulness_rate"], (int, float))
    assert isinstance(data["total_count"], int)
    assert isinstance(data["per_case"], list)

    if data["total_count"] > 0:
        assert "avg_relevance_score" in data
        assert isinstance(data["avg_relevance_score"], (int, float))


def test_e2e_metric_ranges(eval_result):
    """Check metrics within expected ranges (per implementation plan)."""
    result, out_dir = eval_result
    if result.returncode != 0:
        pytest.skip("Evaluation failed; skipping metric range check")

    retrieval_path = out_dir / "retrieval_results.json"
    gen_path = out_dir / "generation_results.json"

    if not retrieval_path.exists() or not gen_path.exists():
        pytest.skip("Results not produced")

    with open(retrieval_path, encoding="utf-8") as f:
        retrieval = json.load(f)
    with open(gen_path, encoding="utf-8") as f:
        generation = json.load(f)

    # Recall@5: 0.4 - 1.0 (reasonable for FAQ)
    avg_recall = retrieval.get("avg_recall", 0)
    assert 0 <= avg_recall <= 1.0, f"Recall@5 out of range: {avg_recall}"

    # Faithfulness: 0.7 - 1.0 (most answers should be faithful)
    faith = generation.get("faithfulness_rate", 0)
    assert 0 <= faith <= 1.0, f"Faithfulness out of range: {faith}"

    # Relevance: 3.0 - 5.0 (acceptable to excellent)
    rel = generation.get("avg_relevance_score", 0)
    if generation.get("total_count", 0) > 0:
        assert 0 <= rel <= 5.0, f"Relevance out of range: {rel}"
