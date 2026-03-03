"""
Unit tests for retrieval_metrics module.

Tests: calculate_recall_at_k, calculate_precision_at_k, calculate_mrr,
_get_relevant_contexts, RetrievalEvaluator.evaluate_batch (mocked).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.rag.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    _get_relevant_contexts,
    RetrievalEvaluator,
)


def _doc(content: str) -> Document:
    """Create a Document with given page_content."""
    return Document(page_content=content, metadata={})


# --- calculate_recall_at_k ---


def test_recall_empty_contexts_returns_zero():
    """Empty relevant_contexts -> returns 0.0."""
    docs = [_doc("chunk A")]
    assert calculate_recall_at_k(docs, [], k=5) == 0.0


def test_recall_all_relevant_returns_one():
    """All relevant found in top-k -> returns 1.0."""
    docs = [_doc("chunk A"), _doc("chunk B"), _doc("chunk C")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 1.0


def test_recall_partial_match_correct_fraction():
    """Partial match -> correct fraction."""
    docs = [_doc("chunk A"), _doc("irrelevant"), _doc("chunk C")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 0.5


def test_recall_k_larger_than_docs_handles_gracefully():
    """k larger than retrieved docs -> handles gracefully."""
    docs = [_doc("chunk A"), _doc("chunk B")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_recall_at_k(docs, contexts, k=10) == 1.0


def test_recall_case_insensitive():
    """Case-insensitive matching."""
    docs = [_doc("CHUNK A content")]
    contexts = ["chunk a"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 1.0


# --- calculate_precision_at_k ---


def test_precision_k_zero_returns_zero():
    """k=0 -> returns 0.0."""
    docs = [_doc("chunk A")]
    contexts = ["chunk A"]
    assert calculate_precision_at_k(docs, contexts, k=0) == 0.0


def test_precision_all_relevant_returns_one():
    """All top-k relevant -> returns 1.0."""
    docs = [_doc("chunk A"), _doc("chunk B")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_precision_at_k(docs, contexts, k=2) == 1.0


def test_precision_none_relevant_returns_zero():
    """No relevant in top-k -> returns 0.0."""
    docs = [_doc("irrelevant 1"), _doc("irrelevant 2")]
    contexts = ["chunk A"]
    assert calculate_precision_at_k(docs, contexts, k=2) == 0.0


def test_precision_mixed_correct_fraction():
    """Mixed results -> correct fraction."""
    docs = [
        _doc("chunk A"),
        _doc("irrelevant"),
        _doc("chunk B"),
        _doc("irrelevant"),
        _doc("irrelevant"),
    ]
    contexts = ["chunk A", "chunk B"]
    assert calculate_precision_at_k(docs, contexts, k=5) == 0.4


# --- calculate_mrr ---


def test_mrr_empty_contexts_returns_zero():
    """Empty relevant_contexts -> returns 0.0."""
    docs = [_doc("chunk A")]
    assert calculate_mrr(docs, []) == 0.0


def test_mrr_first_doc_relevant_returns_one():
    """First doc relevant -> returns 1.0."""
    docs = [_doc("chunk A"), _doc("irrelevant")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 1.0


def test_mrr_second_doc_relevant_returns_half():
    """Second doc relevant -> returns 0.5."""
    docs = [_doc("irrelevant"), _doc("chunk A")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 0.5


def test_mrr_no_relevant_returns_zero():
    """No relevant docs -> returns 0.0."""
    docs = [_doc("irrelevant 1"), _doc("irrelevant 2")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 0.0


# --- _get_relevant_contexts ---


def test_get_contexts_from_list():
    """Case with contexts list -> returns it."""
    case = {"contexts": ["ctx1", "ctx2"], "ground_truth": "answer"}
    assert _get_relevant_contexts(case) == ["ctx1", "ctx2"]


def test_get_contexts_from_string_wraps_in_list():
    """Case with contexts string -> wraps in list."""
    case = {"contexts": "single_ctx", "ground_truth": "answer"}
    assert _get_relevant_contexts(case) == ["single_ctx"]


def test_get_contexts_fallback_ground_truth():
    """Case without contexts, with ground_truth -> returns ground_truth[:100]."""
    case = {"ground_truth": "This is the full answer text for the question"}
    result = _get_relevant_contexts(case)
    assert result == ["This is the full answer text for the question"[:100]]


def test_get_contexts_neither_returns_empty():
    """Case with neither -> returns []."""
    case = {}
    assert _get_relevant_contexts(case) == []


def test_get_contexts_no_fallback_when_disabled():
    """When use_ground_truth_fallback=False and no contexts -> returns []."""
    case = {"ground_truth": "answer"}
    assert _get_relevant_contexts(case, use_ground_truth_fallback=False) == []


# --- RetrievalEvaluator.evaluate_batch (mocked) ---


@patch("ai_toolkit.chroma.chroma_to_documents")
@patch("ai_toolkit.chroma.query_collection")
@patch("ai_toolkit.chroma.get_chroma_collection")
def test_retrieval_evaluator_evaluate_batch(
    mock_get_collection,
    mock_query,
    mock_chroma_to_docs,
):
    """evaluate_batch with mocked pipeline computes avg_recall, avg_precision, avg_mrr."""
    mock_collection = MagicMock()
    mock_get_collection.return_value = mock_collection

    mock_query.return_value = {
        "documents": [["doc1 content", "doc2 content"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.5, 0.8]],
    }

    def chroma_to_docs(ids, documents, metadatas):
        return [
            Document(page_content=documents[0], metadata={}),
            Document(page_content=documents[1], metadata={}),
        ]

    mock_chroma_to_docs.side_effect = chroma_to_docs

    mock_pipeline = MagicMock()
    mock_pipeline.embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    test_cases = [
        {"id": "faq_001", "question": "Q1?", "contexts": ["doc1 content"], "ground_truth": "A1"},
        {"id": "faq_002", "question": "Q2?", "contexts": ["missing"], "ground_truth": "A2"},
    ]

    evaluator = RetrievalEvaluator()
    result = evaluator.evaluate_batch(mock_pipeline, test_cases, k=5, verbose=False)

    assert "avg_recall" in result
    assert "avg_precision" in result
    assert "avg_mrr" in result
    assert "per_case" in result
    assert len(result["per_case"]) == 2

    for case in result["per_case"]:
        assert "id" in case
        assert "recall" in case
        assert "precision" in case
        assert "mrr" in case
        assert "min_distance" in case
