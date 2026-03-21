"""Tests for ``src.rag`` workflow facade (amazon_business + legacy delegation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.deepseek_compose import DeepSeekRagFacade
from src.rag.workflow_service import ChromaRetrieveFacade, RagWorkflowFacade


class FakeDoc:
    """Minimal doc-like object for Chroma results."""

    def __init__(self, text: str, source: str = "a.pdf", page: int = 1) -> None:
        self.page_content = text
        self.metadata = {"source": source, "page": page}


class FakePipeline:
    """Stub pipeline with ``query`` for legacy modes."""

    def __init__(self) -> None:
        self.embedder = object()
        self.vector_store = object()

    def query(self, question: str, mode: str, verbose: bool = False):
        return f"legacy:{mode}", [FakeDoc("x")], mode


def test_legacy_delegates_to_pipeline():
    p = FakePipeline()
    ans, docs, mode = RagWorkflowFacade.run(p, "hello", "documents", verbose=False)
    assert ans == "legacy:documents"
    assert mode == "documents"
    assert len(docs) == 1


def test_amazon_business_dual_path_mocked(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-ci")

    def _fake_gate(cls, embedder, vector_store, question, **kwargs):
        return [FakeDoc("Chroma fact about fees.")], [0.1]

    def _fake_evidence(cls, question: str) -> str:
        return "- Bullet from model"

    def _fake_merge(cls, question: str, chroma_context: str, model_evidence: str) -> str:
        assert "Chroma fact" in chroma_context
        assert "Bullet" in model_evidence
        return "final merged answer"

    monkeypatch.setattr(
        ChromaRetrieveFacade,
        "retrieve_with_similarity_gate",
        classmethod(_fake_gate),
    )
    monkeypatch.setattr(
        DeepSeekRagFacade,
        "amazon_model_evidence",
        classmethod(_fake_evidence),
    )
    monkeypatch.setattr(
        DeepSeekRagFacade,
        "merge_chroma_and_model",
        classmethod(_fake_merge),
    )

    p = FakePipeline()
    ans, docs, mode = RagWorkflowFacade.run(p, "What are referral fees?", "amazon_business")
    assert mode == "amazon_business"
    assert ans == "final merged answer"
    assert len(docs) == 1


def test_amazon_business_requires_deepseek_key():
    p = FakePipeline()
    # Ensure key absent for this test
    import os

    old = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            RagWorkflowFacade.run(p, "q", "amazon_business")
    finally:
        if old is not None:
            os.environ["DEEPSEEK_API_KEY"] = old


def test_unknown_mode_raises():
    p = FakePipeline()
    with pytest.raises(ValueError, match="unsupported"):
        RagWorkflowFacade.run(p, "q", "invalid_mode")
