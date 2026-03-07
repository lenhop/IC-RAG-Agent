"""
Tests for Gradio UI (src/client/gradio_ui.py).

Tests chat handler logic with mocked GatewayClient.
Skips if gradio is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if gradio not installed
pytest.importorskip("gradio")

from src.client.gradio_ui import _chat_handler, create_demo


# ---------------------------------------------------------------------------
# Chat handler logic
# ---------------------------------------------------------------------------


def test_chat_handler_returns_answer():
    """Chat handler returns answer when GatewayClient returns success."""
    mock_client = MagicMock()
    mock_client.query_sync.return_value = {"answer": "The answer is 42."}

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("What is the answer?", [], "uds", True, "ollama", "sess-1")

    assert result == "The answer is 42."
    mock_client.query_sync.assert_called_once_with(
        query="What is the answer?",
        workflow="uds",
        rewrite_enable=True,
        rewrite_backend="ollama",
        session_id="sess-1",
    )


def test_chat_handler_returns_error_on_gateway_error():
    """Chat handler returns error message when GatewayClient returns error dict."""
    mock_client = MagicMock()
    mock_client.query_sync.return_value = {
        "error": "Connection refused",
        "error_type": "ConnectionError",
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", False, "ollama", None)

    assert result == "Error: Connection refused"


def test_chat_handler_empty_message():
    """Chat handler returns prompt when message is empty."""
    with patch("src.client.gradio_ui.GatewayClient") as mock_class:
        result = _chat_handler("", [], "auto", True, "ollama", "s1")
        result_ws = _chat_handler("   ", [], "auto", True, "ollama", "s1")

    mock_class.assert_not_called()
    assert result == "Please enter a question."
    assert result_ws == "Please enter a question."


def test_chat_handler_empty_answer():
    """Chat handler returns fallback when answer is empty."""
    mock_client = MagicMock()
    mock_client.query_sync.return_value = {"answer": ""}

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", True, "ollama", None)

    assert result == "No response from gateway."


# ---------------------------------------------------------------------------
# Gradio import / create_demo
# ---------------------------------------------------------------------------


def test_create_demo_returns_blocks():
    """create_demo returns a Gradio Blocks instance."""
    demo = create_demo()
    import gradio as gr
    assert demo is not None
    assert isinstance(demo, gr.Blocks)
