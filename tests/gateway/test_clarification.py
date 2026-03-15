"""
Tests for gateway clarification module (check_ambiguity).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.gateway.route_llm.clarification.clarification import check_ambiguity

_OLLAMA_ENV = {
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "OLLAMA_GENERATE_MODEL": "qwen3:1.7b",
    "OLLAMA_REQUEST_TIMEOUT": "30",
    "OLLAMA_EMBED_MODEL": "all-minilm:latest",
}


def test_check_ambiguity_empty_query_returns_no_clarification():
    """Empty query should not require clarification."""
    result = check_ambiguity("")
    assert result["needs_clarification"] is False
    assert result.get("clarification_backend") is None
    result = check_ambiguity("   ")
    assert result["needs_clarification"] is False


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_ollama_clear_returns_no_clarification(mock_ollama):
    """When LLM returns needs_clarification false, proceed normally."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    result = check_ambiguity("what are my sales last month?")
    assert result["needs_clarification"] is False
    assert result.get("clarification_backend") == "ollama"
    mock_ollama.assert_called_once_with("what are my sales last month?", None)


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_ollama_ambiguous_returns_clarification(mock_ollama):
    """When LLM returns needs_clarification true, return question."""
    mock_ollama.return_value = (
        '{"needs_clarification": true, '
        '"clarification_question": "Please provide your order ID."}'
    )
    result = check_ambiguity("what about my order?")
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Please provide your order ID."
    mock_ollama.assert_called_once_with("what about my order?", None)


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_ollama_empty_response_raises(mock_ollama):
    """When LLM returns empty, raise ValueError."""
    mock_ollama.return_value = ""
    with pytest.raises(ValueError, match="Ollama clarification failed"):
        check_ambiguity("test query")


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_ollama_invalid_json_returns_no_clarification(mock_ollama):
    """When LLM returns invalid JSON, fall back to no clarification."""
    mock_ollama.return_value = "not valid json"
    result = check_ambiguity("test query")
    assert result["needs_clarification"] is False


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_generate_question")
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_needs_true_but_empty_question_uses_fallback(mock_ollama, mock_gen):
    """When needs_clarification true but question empty, use fallback to generate question."""
    mock_ollama.return_value = '{"needs_clarification": true, "clarification_question": ""}'
    mock_gen.return_value = "Which fees do you mean?"
    result = check_ambiguity("test")
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Which fees do you mean?"
    mock_gen.assert_called_once()


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_needs_true_empty_question_fallback_fails_uses_generic(mock_ollama):
    """When needs_clarification true, question empty, and fallback fails, use generic message."""
    mock_ollama.return_value = '{"needs_clarification": true, "clarification_question": ""}'
    with patch(
        "src.gateway.route_llm.clarification.clarification._ClarificationLLM.generate_question",
        return_value=None,
    ):
        result = check_ambiguity("test")
    assert result["needs_clarification"] is True
    assert "details" in result["clarification_question"].lower()


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_show_me_the_fees_returns_clarification(mock_ollama):
    """Ambiguous query 'Show me the fees' should trigger clarification (LLM)."""
    mock_ollama.return_value = (
        '{"needs_clarification": true, '
        '"clarification_question": "Which fees do you mean? FBA, storage, or referral?"}'
    )
    result = check_ambiguity("Show me the fees")
    assert result["needs_clarification"] is True
    assert "fees" in result["clarification_question"].lower()


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_what_is_the_fee_returns_clarification(mock_ollama):
    """Ambiguous query 'what is the fee ?' should trigger clarification (LLM)."""
    mock_ollama.return_value = (
        '{"needs_clarification": true, '
        '"clarification_question": "Which fee type do you mean?"}'
    )
    result = check_ambiguity("what is the fee ?")
    assert result["needs_clarification"] is True
    assert "fee" in result["clarification_question"].lower()


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_what_is_fee_with_context_llm_says_clear_trusts_llm(mock_ollama):
    """When LLM (with context) returns false, trust LLM."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    ctx = "Turn 1: User asked about sales. Answer: Here are October sales."
    result = check_ambiguity("what is the fee ?", conversation_context=ctx)
    assert result["needs_clarification"] is False


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_llm_empty_response_raises(mock_ollama):
    """When LLM returns empty (failure), raise ValueError."""
    mock_ollama.return_value = ""
    ctx = "Turn 1: User asked about sales."
    with pytest.raises(ValueError, match="Ollama clarification failed"):
        check_ambiguity("what is the fee ?", conversation_context=ctx)


def test_check_ambiguity_documentation_requirements_returns_no_clarification():
    """Documentation/policy/requirements questions should NOT trigger clarification."""
    result = check_ambiguity(
        "what are Amazon's product compliance and safety documentation requirements"
    )
    assert result["needs_clarification"] is False


def test_check_ambiguity_missing_backend_raises():
    """When GATEWAY_CLARIFICATION_BACKEND is not set, check_ambiguity raises ValueError."""
    with patch.dict("os.environ", {"GATEWAY_CLARIFICATION_BACKEND": ""}, clear=False):
        with pytest.raises(ValueError, match="GATEWAY_CLARIFICATION_BACKEND must be set"):
            check_ambiguity("what about my order?")


def test_check_ambiguity_missing_ollama_base_url_raises():
    """When backend=ollama but OLLAMA_BASE_URL is not set, raises ValueError."""
    env = {
        "GATEWAY_CLARIFICATION_BACKEND": "ollama",
        "OLLAMA_BASE_URL": "",
        "OLLAMA_GENERATE_MODEL": "qwen3:1.7b",
        "OLLAMA_REQUEST_TIMEOUT": "30",
        "OLLAMA_EMBED_MODEL": "all-minilm:latest",
    }
    with patch.dict("os.environ", env, clear=False):
        with pytest.raises(ValueError, match="OLLAMA_BASE_URL is not set"):
            check_ambiguity("what about my order?")


@patch.dict("os.environ", {**_OLLAMA_ENV, "GATEWAY_CLARIFICATION_BACKEND": "ollama"}, clear=False)
@patch("src.gateway.route_llm.clarification.clarification._ClarificationLLM._call_ollama_check_ambiguity")
def test_check_ambiguity_sales_with_yyyymmdd_returns_no_clarification(mock_ollama):
    """Sales query with YYYYMMDD date: LLM returns false (has date)."""
    mock_ollama.return_value = '{"needs_clarification": false}'
    result = check_ambiguity("how are the sales for 20250101?")
    assert result["needs_clarification"] is False
