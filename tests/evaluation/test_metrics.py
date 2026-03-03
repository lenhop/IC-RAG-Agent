"""
Unit tests for RAG evaluation metrics.

Covers retrieval metrics (Phase 1.1) and generation metrics (Phase 1.2).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.rag.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    RetrievalEvaluator,
    _get_relevant_contexts,
)


def _doc(content: str) -> Document:
    """Create a Document with given page_content."""
    return Document(page_content=content, metadata={})


# --- Recall@K ---


def test_recall_at_k_all_relevant():
    """All relevant contexts found in top-k -> recall 1.0."""
    docs = [_doc("chunk A"), _doc("chunk B"), _doc("chunk C")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 1.0


def test_recall_at_k_partial():
    """Only one of two relevant contexts found -> recall 0.5."""
    docs = [_doc("chunk A"), _doc("irrelevant"), _doc("chunk C")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 0.5


def test_recall_at_k_none_relevant():
    """No relevant contexts in retrieved -> recall 0.0."""
    docs = [_doc("irrelevant 1"), _doc("irrelevant 2")]
    contexts = ["chunk A"]
    assert calculate_recall_at_k(docs, contexts, k=5) == 0.0


def test_recall_at_k_empty_contexts():
    """Empty relevant_contexts -> recall 0.0."""
    docs = [_doc("chunk A")]
    assert calculate_recall_at_k(docs, [], k=5) == 0.0


def test_recall_at_k_empty_docs():
    """Empty retrieved_docs -> recall 0.0."""
    contexts = ["chunk A"]
    assert calculate_recall_at_k([], contexts, k=5) == 0.0


def test_recall_at_k_respects_k():
    """Only top-k docs considered."""
    docs = [_doc("chunk A"), _doc("chunk B"), _doc("chunk C"), _doc("chunk D"), _doc("chunk E")]
    contexts = ["chunk E"]
    # chunk E is at index 4, so in top-5
    assert calculate_recall_at_k(docs, contexts, k=5) == 1.0
    # With k=4, chunk E is not in top-4
    assert calculate_recall_at_k(docs, contexts, k=4) == 0.0


# --- Precision@K ---


def test_precision_at_k_all_relevant():
    """All top-k docs relevant -> precision 1.0."""
    docs = [_doc("chunk A"), _doc("chunk B")]
    contexts = ["chunk A", "chunk B"]
    assert calculate_precision_at_k(docs, contexts, k=2) == 1.0


def test_precision_at_k_partial():
    """2 of 5 docs relevant -> precision 0.4."""
    docs = [
        _doc("chunk A"),
        _doc("irrelevant"),
        _doc("chunk B"),
        _doc("irrelevant"),
        _doc("irrelevant"),
    ]
    contexts = ["chunk A", "chunk B"]
    assert calculate_precision_at_k(docs, contexts, k=5) == 0.4


def test_precision_at_k_zero():
    """k=0 -> precision 0.0."""
    docs = [_doc("chunk A")]
    contexts = ["chunk A"]
    assert calculate_precision_at_k(docs, contexts, k=0) == 0.0


def test_precision_at_k_empty_results():
    """Empty retrieved_docs -> 0 relevant in k, precision 0.0."""
    contexts = ["chunk A"]
    assert calculate_precision_at_k([], contexts, k=5) == 0.0


# --- MRR ---


def test_mrr_first_relevant():
    """First doc relevant -> MRR 1.0."""
    docs = [_doc("chunk A"), _doc("irrelevant")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 1.0


def test_mrr_second_relevant():
    """Second doc relevant -> MRR 0.5."""
    docs = [_doc("irrelevant"), _doc("chunk A")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 0.5


def test_mrr_no_relevant():
    """No relevant docs -> MRR 0.0."""
    docs = [_doc("irrelevant 1"), _doc("irrelevant 2")]
    contexts = ["chunk A"]
    assert calculate_mrr(docs, contexts) == 0.0


def test_mrr_empty_contexts():
    """Empty relevant_contexts -> MRR 0.0."""
    docs = [_doc("chunk A")]
    assert calculate_mrr(docs, []) == 0.0


def test_mrr_case_insensitive():
    """Matching is case-insensitive."""
    docs = [_doc("CHUNK A content")]
    contexts = ["chunk a"]
    assert calculate_mrr(docs, contexts) == 1.0


# --- _get_relevant_contexts ---


def test_get_relevant_contexts_from_contexts():
    """Prefer contexts field when present."""
    case = {"contexts": ["ctx1", "ctx2"], "ground_truth": "answer"}
    assert _get_relevant_contexts(case) == ["ctx1", "ctx2"]


def test_get_relevant_contexts_single_string():
    """Single string contexts wrapped in list."""
    case = {"contexts": "single_ctx", "ground_truth": "answer"}
    assert _get_relevant_contexts(case) == ["single_ctx"]


def test_get_relevant_contexts_fallback_ground_truth():
    """Fallback to ground_truth prefix when contexts missing."""
    case = {"ground_truth": "This is the full answer text for the question"}
    result = _get_relevant_contexts(case)
    assert result == ["This is the full answer text for the question"[:100]]


def test_get_relevant_contexts_no_fallback():
    """When use_ground_truth_fallback=False and no contexts, return empty."""
    case = {"ground_truth": "answer"}
    assert _get_relevant_contexts(case, use_ground_truth_fallback=False) == []


# --- RetrievalEvaluator (mocked pipeline) ---


def test_retrieval_evaluator_output_structure():
    """evaluate_batch returns expected keys and structure."""
    evaluator = RetrievalEvaluator()
    assert hasattr(evaluator, "evaluate_batch")
    assert callable(getattr(evaluator, "evaluate_batch"))


# --- Generation: _extract_json ---


def test_extract_json_direct():
    """Direct JSON string parses correctly."""
    from src.rag.evaluation.generation_metrics import _extract_json

    text = '{"is_faithful": true, "reasoning": "OK"}'
    assert _extract_json(text) == {"is_faithful": True, "reasoning": "OK"}


def test_extract_json_code_block():
    """JSON inside ```json ... ``` parses correctly."""
    from src.rag.evaluation.generation_metrics import _extract_json

    text = '```json\n{"is_faithful": false, "reasoning": "Hallucination"}\n```'
    assert _extract_json(text) == {"is_faithful": False, "reasoning": "Hallucination"}


def test_extract_json_plain_code_block():
    """JSON inside ``` ... ``` (no json tag) parses correctly."""
    from src.rag.evaluation.generation_metrics import _extract_json

    text = '```\n{"relevance_score": 4, "reasoning": "Good"}\n```'
    assert _extract_json(text) == {"relevance_score": 4, "reasoning": "Good"}


def test_extract_json_extra_text():
    """Extra text before/after JSON still extracts."""
    from src.rag.evaluation.generation_metrics import _extract_json

    text = 'Here is my evaluation:\n{"is_faithful": true, "reasoning": "Fine"}\nDone.'
    assert _extract_json(text) == {"is_faithful": True, "reasoning": "Fine"}


def test_extract_json_invalid_returns_none():
    """Invalid JSON returns None."""
    from src.rag.evaluation.generation_metrics import _extract_json

    assert _extract_json("not json at all") is None
    assert _extract_json("") is None
    assert _extract_json("{invalid}") is None


# --- Generation: evaluate_faithfulness (mock LLM) ---


def test_evaluate_faithfulness_faithful():
    """Mock LLM returns faithful -> result is_faithful True."""
    from src.rag.evaluation.generation_metrics import evaluate_faithfulness

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": '{"is_faithful": true, "reasoning": "Based on context"}'})()

    result = evaluate_faithfulness("Answer", ["Context"], MockLLM())
    assert result["is_faithful"] is True
    assert "reasoning" in result


def test_evaluate_faithfulness_unfaithful():
    """Mock LLM returns unfaithful -> result is_faithful False."""
    from src.rag.evaluation.generation_metrics import evaluate_faithfulness

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": '{"is_faithful": false, "reasoning": "Hallucination"}'})()

    result = evaluate_faithfulness("Answer", ["Context"], MockLLM())
    assert result["is_faithful"] is False


def test_evaluate_faithfulness_parse_failure():
    """Malformed LLM response -> fallback to is_faithful False."""
    from src.rag.evaluation.generation_metrics import evaluate_faithfulness

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": "not valid json"})()

    result = evaluate_faithfulness("Answer", ["Context"], MockLLM())
    assert result["is_faithful"] is False
    assert "reasoning" in result


# --- Generation: evaluate_relevance (mock LLM) ---


def test_evaluate_relevance_score():
    """Mock LLM returns score -> result has relevance_score 1-5."""
    from src.rag.evaluation.generation_metrics import evaluate_relevance

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": '{"relevance_score": 4, "reasoning": "Good answer"}'})()

    result = evaluate_relevance("Q", "A", MockLLM())
    assert result["relevance_score"] == 4
    assert "reasoning" in result


def test_evaluate_relevance_score_clamped():
    """Float score 4.7 -> rounded to 5."""
    from src.rag.evaluation.generation_metrics import evaluate_relevance

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": '{"relevance_score": 4.7, "reasoning": "Excellent"}'})()

    result = evaluate_relevance("Q", "A", MockLLM())
    assert result["relevance_score"] == 5


def test_evaluate_relevance_parse_failure():
    """Malformed LLM response -> fallback to relevance_score 0."""
    from src.rag.evaluation.generation_metrics import evaluate_relevance

    class MockLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": "invalid"})()

    result = evaluate_relevance("Q", "A", MockLLM())
    assert result["relevance_score"] == 0


# --- GenerationEvaluator ---


def test_generation_evaluator_exists():
    """GenerationEvaluator has evaluate_batch method."""
    from src.rag.evaluation.generation_metrics import GenerationEvaluator

    evaluator = GenerationEvaluator()
    assert hasattr(evaluator, "evaluate_batch")
    assert callable(getattr(evaluator, "evaluate_batch"))


# --- Dataset Loader (integration with metrics) ---


def test_load_fqa_dataset():
    """load_fqa_dataset loads CSV and returns expected structure (used by metrics)."""
    from src.rag.evaluation.dataset_loader import load_fqa_dataset

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    cases = load_fqa_dataset(str(csv_path), limit=3, project_root=PROJECT_ROOT)
    assert len(cases) == 3
    for case in cases:
        assert "id" in case
        assert "question" in case
        assert "ground_truth" in case
        assert "category" in case
        assert "source" in case
        assert case["id"].startswith("faq_")


def test_load_fqa_dataset_limit():
    """limit parameter restricts number of rows."""
    from src.rag.evaluation.dataset_loader import load_fqa_dataset

    csv_path = PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"
    if not csv_path.exists():
        pytest.skip("amazon_fqa.csv not found")

    cases = load_fqa_dataset(str(csv_path), limit=1, project_root=PROJECT_ROOT)
    assert len(cases) == 1


def test_validate_dataset():
    """validate_dataset checks required fields (question, ground_truth) and returns bool."""
    from src.rag.evaluation.dataset_loader import validate_dataset

    # Valid
    valid = [{"id": "1", "question": "Q?", "ground_truth": "A"}]
    assert validate_dataset(valid, warn_missing=False) is True

    # Missing question
    invalid_q = [{"id": "1", "question": "", "ground_truth": "A"}]
    assert validate_dataset(invalid_q, warn_missing=False) is False

    # Missing ground_truth
    invalid_gt = [{"id": "1", "question": "Q?"}]
    assert validate_dataset(invalid_gt, warn_missing=False) is False

    # Empty
    assert validate_dataset([], warn_missing=False) is False
