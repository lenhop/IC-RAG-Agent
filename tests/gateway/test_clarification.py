"""
Tests for gateway clarification module (check_ambiguity).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.gateway.route_llm.clarification.clarification import check_ambiguity


def test_check_ambiguity_empty_query_returns_no_clarification():
    """Empty query should not require clarification."""
    result = check_ambiguity("")
    assert result == {"needs_clarification": False}
    result = check_ambiguity("   ")
    assert result == {"needs_clarification": False}


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_ollama_clear_returns_no_clarification(mock_ollama):
    """When LLM returns needs_clarification false, proceed normally."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    result = check_ambiguity("what are my sales last month?", backend="ollama")
    assert result == {"needs_clarification": False}
    mock_ollama.assert_called_once_with("what are my sales last month?")


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_ollama_ambiguous_returns_clarification(mock_ollama):
    """When LLM returns needs_clarification true, return question."""
    mock_ollama.return_value = (
        '{"needs_clarification": true, '
        '"clarification_question": "Please provide your order ID."}'
    )
    result = check_ambiguity("what about my order?", backend="ollama")
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Please provide your order ID."
    mock_ollama.assert_called_once_with("what about my order?")


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_ollama_empty_response_returns_no_clarification(mock_ollama):
    """When LLM returns empty, fall back to no clarification."""
    mock_ollama.return_value = ""
    result = check_ambiguity("test query", backend="ollama")
    assert result == {"needs_clarification": False}


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_ollama_invalid_json_returns_no_clarification(mock_ollama):
    """When LLM returns invalid JSON, fall back to no clarification."""
    mock_ollama.return_value = "not valid json"
    result = check_ambiguity("test query", backend="ollama")
    assert result == {"needs_clarification": False}


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_needs_true_but_empty_question_returns_no_clarification(mock_ollama):
    """When needs_clarification true but question empty, treat as no clarification."""
    mock_ollama.return_value = '{"needs_clarification": true, "clarification_question": ""}'
    result = check_ambiguity("test", backend="ollama")
    assert result == {"needs_clarification": False}


def test_check_ambiguity_show_me_the_fees_returns_clarification():
    """Ambiguous query 'Show me the fees' should trigger clarification (heuristic or LLM)."""
    result = check_ambiguity("Show me the fees", backend="ollama")
    assert result["needs_clarification"] is True
    assert "fees" in result["clarification_question"].lower()


def test_check_ambiguity_what_is_the_fee_returns_clarification():
    """Ambiguous query 'what is the fee ?' should trigger clarification (heuristic)."""
    result = check_ambiguity("what is the fee ?", backend="ollama")
    assert result["needs_clarification"] is True
    assert "fee" in result["clarification_question"].lower()


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_what_is_fee_with_context_llm_says_clear_trusts_llm(mock_ollama):
    """When heuristic matches but LLM (with context) returns false, trust LLM."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    ctx = "Turn 1: User asked about sales. Answer: Here are October sales."
    result = check_ambiguity("what is the fee ?", backend="ollama", conversation_context=ctx)
    assert result["needs_clarification"] is False


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_what_is_fee_with_context_llm_fails_returns_clarification(mock_ollama):
    """When heuristic matches and LLM fails (empty), return clarification."""
    mock_ollama.return_value = ""
    ctx = "Turn 1: User asked about sales."
    result = check_ambiguity("what is the fee ?", backend="ollama", conversation_context=ctx)
    assert result["needs_clarification"] is True
    assert "fee" in result["clarification_question"].lower()


def test_check_ambiguity_documentation_requirements_returns_no_clarification():
    """Documentation/policy/requirements questions should NOT trigger clarification."""
    result = check_ambiguity(
        "what are Amazon's product compliance and safety documentation requirements"
    )
    assert result["needs_clarification"] is False


@patch("src.gateway.route_llm.clarification.clarification._call_clarification_ollama")
def test_check_ambiguity_sales_with_yyyymmdd_returns_no_clarification(mock_ollama):
    """Sales query with YYYYMMDD date: heuristic skips (has date), LLM returns false."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    result = check_ambiguity("how are the sales for 20250101?", backend="ollama")
    assert result["needs_clarification"] is False
