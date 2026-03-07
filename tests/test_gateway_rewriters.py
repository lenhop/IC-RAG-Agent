"""
Tests for gateway rewriters (rewrite_with_ollama, rewrite_with_deepseek).

Uses unittest.mock to patch HTTP and API calls. Verifies success paths and
fallback behavior on connection error and timeout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.gateway.rewriters import (
    REWRITE_PROMPT,
    rewrite_with_ollama,
    rewrite_with_deepseek,
)


# ---------------------------------------------------------------------------
# rewrite_with_ollama
# ---------------------------------------------------------------------------


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_success(mock_post):
    """rewrite_with_ollama returns rewritten text when HTTP 200 with response field."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "  sales revenue October 2024  "}

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("What were my sales in October 2024?")

    assert result == "sales revenue October 2024"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert "model" in call_kwargs["json"]
    assert "prompt" in call_kwargs["json"]
    assert REWRITE_PROMPT in call_kwargs["json"]["prompt"]
    assert "What were my sales in October 2024?" in call_kwargs["json"]["prompt"]
    assert call_kwargs["json"]["stream"] is False


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_connection_error_returns_original(mock_post):
    """On ConnectionError, rewrite_with_ollama returns original query."""
    mock_post.side_effect = requests.ConnectionError("Connection refused")

    result = rewrite_with_ollama("my original query")

    assert result == "my original query"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_timeout_returns_original(mock_post):
    """On Timeout, rewrite_with_ollama returns original query."""
    mock_post.side_effect = requests.Timeout("Request timed out")

    result = rewrite_with_ollama("query with timeout")

    assert result == "query with timeout"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_http_error_returns_original(mock_post):
    """On HTTP non-200, rewrite_with_ollama returns original query."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": "Internal server error"}
    mock_resp.text = "Internal server error"

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("my query")

    assert result == "my query"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_empty_response_returns_original(mock_post):
    """When response field is empty, returns original query."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": ""}

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("original")

    assert result == "original"
    mock_post.assert_called_once()


def test_rewrite_with_ollama_empty_query_returns_as_is():
    """Empty or whitespace-only query returns unchanged."""
    assert rewrite_with_ollama("") == ""
    assert rewrite_with_ollama("   ") == "   "


# ---------------------------------------------------------------------------
# rewrite_with_deepseek
# ---------------------------------------------------------------------------


@patch("openai.OpenAI")
@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key-123"})
def test_rewrite_with_deepseek_success(mock_openai_class):
    """rewrite_with_deepseek returns rewritten text when API succeeds."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "  refined search query for knowledge base  "
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    result = rewrite_with_deepseek("What is the profit margin?")

    assert result == "refined search query for knowledge base"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"]
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert REWRITE_PROMPT in call_kwargs["messages"][0]["content"]
    assert call_kwargs["messages"][1]["content"] == "What is the profit margin?"


@patch("openai.OpenAI")
@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"})
def test_rewrite_with_deepseek_api_error_returns_original(mock_openai_class):
    """On API exception, rewrite_with_deepseek returns original query."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API rate limit")

    result = rewrite_with_deepseek("my question")

    assert result == "my question"
    mock_client.chat.completions.create.assert_called_once()


@patch.dict("os.environ", {}, clear=True)
def test_rewrite_with_deepseek_no_api_key_returns_original():
    """When DEEPSEEK_API_KEY not set, returns original query."""
    result = rewrite_with_deepseek("query without key")

    assert result == "query without key"


def test_rewrite_with_deepseek_empty_query_returns_as_is():
    """Empty or whitespace-only query returns unchanged."""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "key"}):
        assert rewrite_with_deepseek("") == ""
        assert rewrite_with_deepseek("   ") == "   "
