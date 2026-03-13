"""
Unit tests for gateway Route LLM (route_llm.py).

Covers: route_with_llm, _parse_route_json, Ollama/DeepSeek backends with mocked
HTTP and OpenAI client. All tests are deterministic (no real LLM calls).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from src.gateway.route_llm import (
    ALLOWED_WORKFLOWS,
    SAFE_DEFAULT_CONFIDENCE,
    SAFE_DEFAULT_WORKFLOW,
    _parse_route_json,
    route_with_llm,
)


# ---------------------------------------------------------------------------
# _parse_route_json (and thus valid/invalid output handling)
# ---------------------------------------------------------------------------


def test_parse_route_json_valid_single_line():
    """Valid single-line JSON with workflow and confidence returns (workflow, confidence)."""
    raw = '{"workflow": "uds", "confidence": 0.92}'
    result = _parse_route_json(raw)
    assert result == ("uds", 0.92)


def test_parse_route_json_valid_key_order_reversed():
    """JSON with confidence first, workflow second is accepted."""
    raw = '{"confidence": 0.5, "workflow": "sp_api"}'
    result = _parse_route_json(raw)
    assert result == ("sp_api", 0.5)


def test_parse_route_json_all_workflows():
    """Each allowed workflow label parses correctly."""
    for wf in ALLOWED_WORKFLOWS:
        raw = json.dumps({"workflow": wf, "confidence": 0.8})
        result = _parse_route_json(raw)
        assert result == (wf, 0.8)


def test_parse_route_json_confidence_clamped():
    """Confidence above 1 or below 0 is clamped to [0, 1]."""
    raw = '{"workflow": "general", "confidence": 1.5}'
    result = _parse_route_json(raw)
    assert result == ("general", 1.0)

    raw2 = '{"workflow": "general", "confidence": -0.2}'
    result2 = _parse_route_json(raw2)
    assert result2 == ("general", 0.0)


def test_parse_route_json_invalid_workflow_returns_none():
    """Invalid workflow label yields None (caller will use safe default)."""
    raw = '{"workflow": "invalid_workflow", "confidence": 0.9}'
    result = _parse_route_json(raw)
    assert result is None


def test_parse_route_json_malformed_json_returns_none():
    """Malformed JSON or non-dict returns None."""
    assert _parse_route_json("not json at all") is None
    assert _parse_route_json("") is None
    assert _parse_route_json('["array"]') is None
    assert _parse_route_json('{"workflow": "general"}') is not None  # missing confidence -> uses default


def test_parse_route_json_embedded_in_text():
    """JSON object embedded in surrounding text can be parsed (first {...} or full)."""
    raw = 'Here is the result: {"workflow": "amazon_docs", "confidence": 0.88}'
    result = _parse_route_json(raw)
    assert result == ("amazon_docs", 0.88)


# ---------------------------------------------------------------------------
# route_with_llm: backend "none" / empty / unknown
# ---------------------------------------------------------------------------


def test_route_with_llm_backend_none_returns_safe_default():
    """Backend 'none' or empty returns (general, 0.0) without calling any API."""
    assert route_with_llm("any query", "none") == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
    assert route_with_llm("any query", "") == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
    assert route_with_llm("any query", "  ") == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


def test_route_with_llm_unknown_backend_returns_safe_default():
    """Unknown backend name returns safe default and does not raise."""
    result = route_with_llm("hello", "openai")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


def test_route_with_llm_empty_query_returns_safe_default():
    """Empty or whitespace-only query with any backend returns safe default."""
    with patch("src.gateway.route_llm._route_with_ollama") as mock_ollama:
        result = route_with_llm("", "ollama")
        assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
        mock_ollama.assert_not_called()
    result2 = route_with_llm("   ", "ollama")
    assert result2 == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


# ---------------------------------------------------------------------------
# route_with_llm: Ollama backend (mocked HTTP)
# ---------------------------------------------------------------------------


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_success_returns_workflow_and_confidence(mock_post):
    """Ollama returns 200 with valid JSON in 'response' -> (workflow, confidence)."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "response": '{"workflow": "uds", "confidence": 0.91}',
    }
    result = route_with_llm("what were my sales?", "ollama")
    assert result == ("uds", 0.91)
    mock_post.assert_called_once()


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_http_error_returns_safe_default(mock_post):
    """Ollama returns 500 -> safe default (no exception)."""
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    mock_post.return_value.json.side_effect = ValueError("no json")
    result = route_with_llm("test query", "ollama")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_timeout_returns_safe_default(mock_post):
    """Ollama request timeout -> safe default (no exception)."""
    mock_post.side_effect = requests.Timeout("timeout")
    result = route_with_llm("test query", "ollama")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_connection_error_returns_safe_default(mock_post):
    """Ollama connection error -> safe default (no exception)."""
    mock_post.side_effect = requests.ConnectionError("connection refused")
    result = route_with_llm("test query", "ollama")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_invalid_json_in_response_returns_safe_default(mock_post):
    """Ollama returns 200 but 'response' is not valid workflow JSON -> safe default."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"response": "not valid json"}
    result = route_with_llm("test query", "ollama")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


@patch("src.gateway.route_llm.requests.post")
def test_route_with_ollama_invalid_workflow_in_response_returns_safe_default(mock_post):
    """Ollama returns valid JSON but workflow label invalid -> safe default."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "response": '{"workflow": "unknown_wf", "confidence": 0.99}',
    }
    result = route_with_llm("test query", "ollama")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


# ---------------------------------------------------------------------------
# route_with_llm: DeepSeek backend (mocked OpenAI client)
# ---------------------------------------------------------------------------


@patch("src.gateway.route_llm._route_with_deepseek", return_value=("sp_api", 0.85))
def test_route_with_deepseek_success_returns_workflow_and_confidence(mock_deepseek):
    """DeepSeek backend returns (workflow, confidence) from _route_with_deepseek."""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        result = route_with_llm("create shipment for order", "deepseek")
    assert result == ("sp_api", 0.85)
    mock_deepseek.assert_called_once_with("create shipment for order")


def test_route_with_deepseek_no_api_key_returns_safe_default():
    """Missing DEEPSEEK_API_KEY -> safe default without calling API."""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
        result = route_with_llm("query", "deepseek")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)


@patch("src.gateway.route_llm._route_with_deepseek", side_effect=Exception("API error"))
def test_route_with_deepseek_api_exception_returns_safe_default(mock_deepseek):
    """DeepSeek API raises -> safe default (no exception)."""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        result = route_with_llm("query", "deepseek")
    assert result == (SAFE_DEFAULT_WORKFLOW, SAFE_DEFAULT_CONFIDENCE)
    mock_deepseek.assert_called_once()


# ---------------------------------------------------------------------------
# Backend case insensitivity
# ---------------------------------------------------------------------------


@patch("src.gateway.route_llm.requests.post")
def test_route_with_llm_backend_case_insensitive(mock_post):
    """Backend name is normalized to lowercase (Ollama)."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "response": '{"workflow": "ic_docs", "confidence": 0.9}',
    }
    result = route_with_llm("framework doc?", "OLLAMA")
    assert result == ("ic_docs", 0.9)
    mock_post.assert_called_once()
