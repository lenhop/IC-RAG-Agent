"""
Tests for Agent RAG service: general vs amazon_business orchestration (mocked IO).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from src.agent.rag import merge_compose as merge_compose_module
from src.agent.rag.chroma_retrieve import ChromaRetriever
from src.agent.rag.deepseek_retrieve import DeepSeekRetrieveFacade
from src.agent.rag.merge_compose import MergeComposer
from src.agent.rag.service import RagQueryService


@pytest.fixture(autouse=True)
def _reset_rag_cache() -> Any:
    """Isolate lazy singletons between tests."""
    RagQueryService.reset_cache_for_testing()
    yield
    RagQueryService.reset_cache_for_testing()


def _fake_cfg(tmp_path: Path) -> Any:
    """Minimal RagRuntimeConfig for mocked Chroma."""
    from src.agent.rag.config import RagRuntimeConfig

    return RagRuntimeConfig(
        project_root=tmp_path,
        chroma_documents_path=tmp_path,
        chroma_collection_name="documents",
        embed_model="minilm",
        embed_extra={},
        similarity_threshold=0.7,
        chroma_top_k=3,
        chroma_query_prefetch=20,
    )


def test_merge_system_prompt_prefers_chroma() -> None:
    """Plan requirement: conflict resolution favors Chroma (Source A)."""
    assert "prefer Source A" in merge_compose_module._MERGE_SYSTEM


def test_general_uses_deepseek_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """general mode must not open Chroma."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-ci")
    monkeypatch.setenv("CHROMA_DOCUMENTS_PATH", str(tmp_path / "chroma"))
    # Avoid loading real sentence-transformers during lazy embedder init.
    monkeypatch.setattr(
        "src.agent.rag.service.create_embeddings",
        lambda **kwargs: MagicMock(),
    )
    called: List[str] = []

    def _fake_general(cls: Any, question: str, **kwargs: Any) -> str:
        called.append(question)
        return "general-answer"

    monkeypatch.setattr(
        DeepSeekRetrieveFacade,
        "general_answer",
        classmethod(_fake_general),
    )
    chroma_called: List[bool] = []

    def _boom(cls: Any) -> Any:
        chroma_called.append(True)
        raise AssertionError("Chroma should not be used in general mode")

    monkeypatch.setattr(RagQueryService, "_ensure_chroma_collection", classmethod(_boom))

    out = RagQueryService.run("What is RAG?", "general")
    assert out == {"answer": "general-answer", "sources": []}
    assert called == ["What is RAG?"]
    assert chroma_called == []


def test_amazon_business_dual_path_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """amazon_business runs Chroma retrieve, DeepSeek evidence, and merge."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-ci")
    cfg = _fake_cfg(tmp_path)

    def _fake_ensure(cls: Any) -> Tuple[Any, Any, Any]:
        return cfg, object(), object()

    monkeypatch.setattr(
        RagQueryService, "_ensure_chroma_collection", classmethod(_fake_ensure)
    )

    def _fake_retrieve(
        cls: Any,
        collection: Any,
        embedder: Any,
        question: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        assert question == "Amazon fees?"
        return [
            {
                "text": "Chroma says referral fee is X.",
                "metadata": {"source": "doc.pdf", "page": 1},
                "distance": 0.2,
                "similarity": 0.98,
            }
        ]

    monkeypatch.setattr(
        ChromaRetriever,
        "retrieve",
        classmethod(_fake_retrieve),
    )

    def _fake_evidence(cls: Any, question: str, **kwargs: Any) -> str:
        return "- Bullet from model knowledge"

    monkeypatch.setattr(
        DeepSeekRetrieveFacade,
        "evidence_for_query",
        classmethod(_fake_evidence),
    )

    merge_args: Dict[str, str] = {}

    def _fake_merge(
        cls: Any,
        question: str,
        chroma_blocks: str,
        deepseek_evidence: str,
        **kwargs: Any,
    ) -> str:
        merge_args["chroma"] = chroma_blocks
        merge_args["evidence"] = deepseek_evidence
        return "merged-final"

    monkeypatch.setattr(MergeComposer, "final_answer", classmethod(_fake_merge))

    out = RagQueryService.run("Amazon fees?", "amazon_business")
    assert out["answer"] == "merged-final"
    assert len(out["sources"]) == 1
    assert "Chroma says referral" in merge_args["chroma"]
    assert "Bullet from model" in merge_args["evidence"]


def test_unsupported_mode_returns_error() -> None:
    """Unknown mode yields gateway-style error dict."""
    out = RagQueryService.run("q", "unknown_mode")
    assert "error" in out
    assert out["error_type"] == "ValueError"
