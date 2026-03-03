"""
Unit tests for generation_metrics module.

Tests: _extract_json, evaluate_faithfulness, evaluate_relevance (mocked LLM).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.evaluation.generation_metrics import (
    _extract_json,
    evaluate_faithfulness,
    evaluate_relevance,
)


# --- _extract_json ---


def test_extract_json_direct_parse():
    """Valid JSON string -> parses directly."""
    text = '{"is_faithful": true, "reasoning": "OK"}'
    result = _extract_json(text)
    assert result == {"is_faithful": True, "reasoning": "OK"}


def test_extract_json_from_code_block():
    """JSON inside ```json ... ``` -> extracts and parses."""
    text = '```json\n{"relevance_score": 4, "reasoning": "Good"}\n```'
    result = _extract_json(text)
    assert result == {"relevance_score": 4, "reasoning": "Good"}


def test_extract_json_from_plain_code_block():
    """JSON inside ``` ... ``` (no json tag) -> extracts."""
    text = '```\n{"key": "value"}\n```'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_from_braces():
    """JSON as outermost {...} in text -> extracts."""
    text = 'Here is the result: {"score": 5}'
    result = _extract_json(text)
    assert result == {"score": 5}


def test_extract_json_empty_returns_none():
    """Empty or whitespace -> returns None."""
    assert _extract_json("") is None
    assert _extract_json("   ") is None


def test_extract_json_invalid_returns_none():
    """Invalid JSON -> returns None."""
    assert _extract_json("not json at all") is None
    assert _extract_json("{invalid}") is None


# --- evaluate_faithfulness (mocked) ---


def test_evaluate_faithfulness_faithful():
    """LLM returns faithful -> is_faithful True."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"is_faithful": true, "reasoning": "Based on context"}'
    )

    result = evaluate_faithfulness(
        answer="The answer is X",
        contexts=["Context says X"],
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["is_faithful"] is True
    assert "reasoning" in result


def test_evaluate_faithfulness_unfaithful():
    """LLM returns unfaithful -> is_faithful False."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"is_faithful": false, "reasoning": "Hallucinated"}'
    )

    result = evaluate_faithfulness(
        answer="Made up answer",
        contexts=["Different context"],
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["is_faithful"] is False
    assert "Hallucinated" in result["reasoning"]


def test_evaluate_faithfulness_parse_failure_returns_unfaithful():
    """Unparseable LLM response -> is_faithful False with reasoning."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Not valid JSON")

    result = evaluate_faithfulness(
        answer="A",
        contexts=["C"],
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["is_faithful"] is False
    assert "Failed to parse" in result["reasoning"]


def test_evaluate_faithfulness_empty_contexts():
    """Empty contexts -> still invokes LLM (no crash)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"is_faithful": true, "reasoning": "No context"}'
    )

    result = evaluate_faithfulness(
        answer="Answer",
        contexts=[],
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["is_faithful"] is True


# --- evaluate_relevance (mocked) ---


def test_evaluate_relevance_returns_score():
    """LLM returns score 1-5 -> relevance_score in range."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"relevance_score": 4, "reasoning": "Good answer"}'
    )

    result = evaluate_relevance(
        question="Q?",
        answer="A",
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["relevance_score"] == 4
    assert "reasoning" in result


def test_evaluate_relevance_clamps_to_1_5():
    """Score outside 1-5 -> clamped to 1-5."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"relevance_score": 10, "reasoning": "Great"}'
    )

    result = evaluate_relevance(
        question="Q?",
        answer="A",
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["relevance_score"] == 5


def test_evaluate_relevance_parse_failure_returns_zero():
    """Unparseable response -> relevance_score 0."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Not JSON")

    result = evaluate_relevance(
        question="Q?",
        answer="A",
        judge_llm=mock_llm,
        verbose=False,
    )
    assert result["relevance_score"] == 0
    assert "Failed to parse" in result["reasoning"]
