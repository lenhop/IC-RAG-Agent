"""
Tests for tools.rewriting.run_rewrite_eval pipeline.
"""

from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

from tools.rewriting import run_rewrite_eval as ree


class _FakeEmbedder:
    """Simple deterministic embedder for pipeline tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            s = (text or "").strip()
            vectors.append([float(len(s)), float(sum(ord(ch) for ch in s) % 97 + 1)])
        return vectors


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "query"])
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_pipeline_skips_blank_and_writes_schema(tmp_path, monkeypatch):
    """Blank queries should be skipped and output schema should include metrics."""
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    _write_csv(
        input_csv,
        [
            {"id": "1", "query": "hello world"},
            {"id": "2", "query": "   "},
            {"id": "3", "query": "sales last week"},
        ],
    )

    def _fake_rewrite(**kwargs):
        query = kwargs["query"]
        return {
            "rewritten_query": f"rewritten: {query}",
            "rewrite_time_ms": "5",
            "rewrite_backend_used": "ollama",
            "rewrite_status": "ok",
            "rewrite_error": "",
        }

    monkeypatch.setattr(ree, "call_rewrite", _fake_rewrite)
    monkeypatch.setattr(ree, "load_embedder", lambda _model: _FakeEmbedder())

    args = Namespace(
        input=str(input_csv),
        output=str(output_csv),
        endpoint="http://127.0.0.1:8000/api/v1/rewrite",
        rewrite_backend="ollama",
        timeout=30,
        embed_model="models/all-MiniLM-L6-v2",
        too_similar_threshold=0.98,
        too_different_threshold=0.75,
    )
    stats = ree.run_pipeline(args)
    rows = _read_csv(output_csv)

    assert stats["rows"] == 3
    assert stats["rewrite_ok"] == 2
    assert stats["rewrite_skip"] == 1
    assert rows[1]["rewrite_status"] == "skipped"
    assert rows[1]["distance_flag"] == "skipped"

    expected_columns = {
        "rewritten_query",
        "rewrite_time_ms",
        "rewrite_backend_used",
        "rewrite_status",
        "rewrite_error",
        "cosine_similarity",
        "cosine_distance",
        "distance_flag",
    }
    assert expected_columns.issubset(set(rows[0].keys()))


def test_run_pipeline_error_row_sets_distance_error(tmp_path, monkeypatch):
    """Rewrite error rows should carry error distance flag and blank metrics."""
    input_csv = tmp_path / "input_err.csv"
    output_csv = tmp_path / "output_err.csv"
    _write_csv(
        input_csv,
        [
            {"id": "1", "query": "good query"},
            {"id": "2", "query": "bad query"},
        ],
    )

    def _fake_rewrite(**kwargs):
        query = kwargs["query"]
        if "bad" in query:
            return {
                "rewritten_query": "",
                "rewrite_time_ms": "",
                "rewrite_backend_used": "",
                "rewrite_status": "error",
                "rewrite_error": "mock rewrite failure",
            }
        return {
            "rewritten_query": "good rewritten query",
            "rewrite_time_ms": "7",
            "rewrite_backend_used": "ollama",
            "rewrite_status": "ok",
            "rewrite_error": "",
        }

    monkeypatch.setattr(ree, "call_rewrite", _fake_rewrite)
    monkeypatch.setattr(ree, "load_embedder", lambda _model: _FakeEmbedder())

    args = Namespace(
        input=str(input_csv),
        output=str(output_csv),
        endpoint="http://127.0.0.1:8000/api/v1/rewrite",
        rewrite_backend="ollama",
        timeout=30,
        embed_model="models/all-MiniLM-L6-v2",
        too_similar_threshold=0.98,
        too_different_threshold=0.75,
    )
    stats = ree.run_pipeline(args)
    rows = _read_csv(output_csv)

    assert stats["rewrite_error"] == 1
    assert rows[1]["rewrite_status"] == "error"
    assert rows[1]["distance_flag"] == "error"
    assert rows[1]["cosine_similarity"] == ""
    assert rows[1]["cosine_distance"] == ""


def test_classify_distance_thresholds():
    """Distance flags should follow configured threshold boundaries."""
    assert ree.classify_distance(0.99, 0.98, 0.75) == "too_similar"
    assert ree.classify_distance(0.75, 0.98, 0.75) == "too_different"
    assert ree.classify_distance(0.84, 0.98, 0.75) == "ok"

