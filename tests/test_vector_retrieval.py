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


def test_vector_retrieval_empty_query_returns_empty() -> None:
    """Empty query returns empty list without calling Chroma."""
    retriever = VectorRetrieval()
    # Use a non-existent path so we don't need Chroma; retrieve() will only
    # call _ensure_client() when query is non-empty. For empty query it returns [].
    result = retriever.retrieve("")
    assert result == []
    result = retriever.retrieve("   ")
    assert result == []


def test_vector_retrieval_import_and_instantiate() -> None:
    """VectorRetrieval can be instantiated with default or custom path."""
    r = VectorRetrieval(top_k=3)
    assert r._top_k == 3
    assert "intent_registry" in str(r._chroma_path) or r._chroma_path.name == "intent_registry"
