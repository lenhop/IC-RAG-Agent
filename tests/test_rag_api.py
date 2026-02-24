"""
RAG API integration tests.

Requires the RAG API server to be running:
  ./scripts/run_rag_api.sh

Run tests:
  pytest tests/test_rag_api.py -v
  pytest tests/test_rag_api.py -v -s  # show print output
"""

import os
import sys
from pathlib import Path

import pytest
import requests

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Default base URL; override with RAG_API_URL env var
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def _api_available() -> bool:
    """Check if RAG API is reachable."""
    try:
        r = requests.get(f"{RAG_API_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def skip_if_unavailable():
    """Skip all tests in module if API is not running."""
    if not _api_available():
        pytest.skip(
            f"RAG API not available at {RAG_API_URL}. "
            "Start with: ./scripts/run_rag_api.sh"
        )


class TestHealthEndpoint:
    """GET /health - Health check."""

    def test_health_returns_200(self, skip_if_unavailable):
        """Health endpoint returns 200 OK."""
        r = requests.get(f"{RAG_API_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_health_response_schema(self, skip_if_unavailable):
        """Health response has status, pipeline_ready, chunks."""
        r = requests.get(f"{RAG_API_URL}/health", timeout=5)
        data = r.json()
        assert "status" in data
        assert "pipeline_ready" in data
        assert "chunks" in data
        assert data["status"] == "ok"
        assert data["pipeline_ready"] is True
        assert isinstance(data["chunks"], (int, type(None)))


class TestQueryEndpoint:
    """POST /query - RAG query."""

    def test_query_hybrid_mode(self, skip_if_unavailable):
        """Query with default hybrid mode returns answer and source."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"question": "What is this document about?", "mode": "hybrid"},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "source" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)

    def test_query_documents_mode(self, skip_if_unavailable):
        """Query with documents-only mode."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"question": "Summarize the main topic.", "mode": "documents"},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "source" in data
        # documents mode: either Document(s) or "No relevant documents found"
        assert "source" in data

    def test_query_general_mode(self, skip_if_unavailable):
        """Query with general-knowledge-only mode (no retrieval)."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"question": "What is 2 + 2?", "mode": "general"},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["source"] == "General Knowledge"
        assert data["sources"] == []

    def test_query_mode_default_is_hybrid(self, skip_if_unavailable):
        """Omitting mode defaults to hybrid."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"question": "What is machine learning?"},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "source" in data

    def test_query_empty_question_returns_422(self, skip_if_unavailable):
        """Empty question returns 422 Unprocessable Entity."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"question": "", "mode": "hybrid"},
            timeout=5,
        )
        assert r.status_code == 422

    def test_query_missing_question_returns_422(self, skip_if_unavailable):
        """Missing question returns 422."""
        r = requests.post(
            f"{RAG_API_URL}/query",
            json={"mode": "hybrid"},
            timeout=5,
        )
        assert r.status_code == 422
