"""
Basic tests for src.retrieval.vector_retrieval (VectorRetrieval).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.vector_retrieval import VectorCandidate, VectorRetrieval


def _dummy_store() -> VectorRetrieval:
    """Minimal config; empty-query path never opens Chroma."""
    return VectorRetrieval(
        chroma_path=Path("/nonexistent_chroma_for_unit_test"),
        collection_name="test_collection",
        top_k=3,
    )


def test_vector_retrieval_empty_query_returns_empty() -> None:
    """Empty query returns empty list without calling Chroma."""
    retriever = _dummy_store()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_vector_retrieval_constructor_holds_config() -> None:
    """Caller-supplied path and collection are stored."""
    p = Path("/tmp/chroma_x")
    r = VectorRetrieval(p, "my_collection", top_k=3, score_threshold=0.5, embed_backend="ollama")
    assert r._top_k == 3
    assert r._chroma_path == p
    assert r._collection_name == "my_collection"
    assert r._score_threshold == 0.5
