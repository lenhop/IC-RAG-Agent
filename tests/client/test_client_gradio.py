"""
Tests for Gradio UI (src/client/gradio_ui.py).

Tests chat handler logic with mocked GatewayClient.
Skips if gradio is not installed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if gradio not installed
pytest.importorskip("gradio")

from src.client.gradio_ui import _chat_handler, create_demo


def _collect_chat_outputs(result):
    """Normalize chat handler output (string or generator) into list of messages."""
    if isinstance(result, str):
        return [result]
    return list(result)


# ---------------------------------------------------------------------------
# Chat handler logic
# ---------------------------------------------------------------------------


def test_chat_handler_returns_answer():
    """Chat handler returns answer when GatewayClient returns success."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "What is the answer?",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 12,
    }
    mock_client.query_sync.return_value = {
        "answer": "The answer is 42.",
        "routing_confidence": 0.9,
        "debug": {"route_source": "heuristic"},
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("What is the answer?", [], "uds", True, "ollama", "sess-1")
        outputs = _collect_chat_outputs(result)

        assert len(outputs) == 2
        assert "Normalize: Completed" in outputs[0]
    assert "The answer is 42." in outputs[1]
    assert "Routed Input: `What is the answer?`" in outputs[1]
    mock_client.rewrite_sync.assert_called_once_with(
        query="What is the answer?",
        rewrite_enable=True,
        rewrite_backend="ollama",
        session_id="sess-1",
    )
    mock_client.query_sync.assert_called_once_with(
        query="What is the answer?",
        workflow="uds",
        rewrite_enable=False,
        rewrite_backend=None,
        session_id="sess-1",
    )


def test_chat_handler_returns_error_on_gateway_error():
    """Chat handler returns error message when GatewayClient returns error dict."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "hello",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 5,
    }
    mock_client.query_sync.return_value = {
        "error": "Connection refused",
        "error_type": "ConnectionError",
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", False, "ollama", None)
        outputs = _collect_chat_outputs(result)

    assert outputs == ["Error: Connection refused"]


def test_chat_handler_empty_message():
    """Chat handler returns prompt when message is empty."""
    with patch("src.client.gradio_ui.GatewayClient") as mock_class:
        result = _chat_handler("", [], "auto", True, "ollama", "s1")
        result_ws = _chat_handler("   ", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)
        outputs_ws = _collect_chat_outputs(result_ws)

    mock_class.assert_not_called()
    assert outputs[-1] == "Please enter a question."
    assert outputs_ws[-1] == "Please enter a question."


def test_chat_handler_empty_answer():
    """Chat handler returns fallback when answer is empty."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "hello",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 4,
    }
    mock_client.query_sync.return_value = {"answer": ""}

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", True, "ollama", None)
        outputs = _collect_chat_outputs(result)

    assert outputs[-1] == "No response from gateway."


def test_chat_handler_appends_trace_when_debug_present():
    """Chat handler appends rewrite/route trace when debug payload exists."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "total sales october 2025",
        "rewrite_time_ms": 18,
        "rewrite_backend": "ollama",
    }
    mock_client.query_sync.return_value = {
        "answer": "The answer is 42.",
        "routing_confidence": 0.85,
        "debug": {
            "route_source": "heuristic",
        },
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("What is total sales in October 2025?", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)

    final_output = outputs[-1]
    assert "The answer is 42." in final_output
    assert "Trace" in final_output
    assert "Routed Input: `total sales october 2025`" in final_output
    assert "Rewrite: `ollama` in `18 ms`" in final_output
    assert "Route Source: `heuristic` | Confidence: `0.85`" in final_output


def test_chat_handler_rewrite_fails_falls_back_to_original_query():
    """When rewrite endpoint fails, chat falls back to original query."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {"error": "rewrite backend unavailable"}
    mock_client.query_sync.return_value = {
        "answer": "Fallback answer",
        "routing_confidence": 0.7,
        "debug": {"route_source": "heuristic"},
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)

    assert "Rewrite failed; continuing with original query." in outputs[0]
    assert "Fallback answer" in outputs[-1]
    mock_client.query_sync.assert_called_once_with(
        query="hello",
        workflow="auto",
        rewrite_enable=False,
        rewrite_backend=None,
        session_id="s1",
    )


def test_chat_handler_does_not_treat_error_none_as_failure():
    """Chat handler should not render Error when backend includes error=None."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "hello",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 6,
    }
    mock_client.query_sync.return_value = {
        "answer": "Success response",
        "error": None,
        "routing_confidence": 1.0,
        "debug": {"route_source": "manual"},
    }

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)

    assert "Success response" in outputs[-1]


def test_chat_handler_streams_progress_while_waiting_for_query():
    """Chat handler streams progress updates before final answer on slow query."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "hello",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 6,
    }

    def slow_query_sync(**kwargs):
        """Simulate slow backend query to verify progress streaming."""
        time.sleep(1.2)
        return {
            "answer": "Delayed success",
            "routing_confidence": 0.9,
            "debug": {"route_source": "heuristic"},
        }

    mock_client.query_sync.side_effect = slow_query_sync

    with patch("src.client.gradio_ui.GatewayClient", return_value=mock_client):
        result = _chat_handler("hello", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)

    assert any("Processing routed query..." in item for item in outputs)
    assert "Delayed success" in outputs[-1]


def test_chat_handler_clarification_required_displays_question_and_stores_pending():
    """When gateway returns clarification_required, display question and store pending_query."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "what about my order?",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 5,
    }
    mock_client.query_sync.return_value = {
        "clarification_required": True,
        "clarification_question": "Please provide your order ID.",
        "pending_query": "what about my order?",
        "answer": "Please provide your order ID.",
    }

    with (
        patch("src.client.gradio_ui.GatewayClient", return_value=mock_client),
        patch("src.client.gradio_ui._set_pending_query") as mock_set_pending,
    ):
        result = _chat_handler("what about my order?", [], "auto", True, "ollama", "sess-1")
        outputs = _collect_chat_outputs(result)

    assert any("Clarification needed" in item for item in outputs)
    assert any("Please provide your order ID" in item for item in outputs)
    mock_set_pending.assert_called_once_with("sess-1", "what about my order?")


def test_chat_handler_merge_pending_query_on_followup():
    """When pending query exists, merge with user message before sending."""
    mock_client = MagicMock()
    merged_query = "what about my order? 112-9876543-12"
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": merged_query,
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 2,
    }
    mock_client.query_sync.return_value = {
        "answer": "Order 112-9876543-12 status: shipped.",
        "routing_confidence": 0.95,
        "debug": {"route_source": "heuristic"},
    }

    with (
        patch("src.client.gradio_ui.GatewayClient", return_value=mock_client),
        patch("src.client.gradio_ui._get_pending_query", return_value="what about my order?"),
    ):
        result = _chat_handler("112-9876543-12", [], "auto", True, "ollama", "sess-1")
        outputs = _collect_chat_outputs(result)

    # Merged query sent to rewrite_sync and then to query_sync
    mock_client.rewrite_sync.assert_called_once_with(
        query=merged_query,
        rewrite_enable=True,
        rewrite_backend="ollama",
        session_id="sess-1",
    )
    mock_client.query_sync.assert_called_once_with(
        query=merged_query,
        workflow="auto",
        rewrite_enable=False,
        rewrite_backend=None,
        session_id="sess-1",
    )
    assert "Order 112-9876543-12 status" in outputs[-1]


def test_chat_handler_rewrite_only_mode_skips_query_sync():
    """Rewrite-only UI mode should stop after rewrite and not call query endpoint."""
    mock_client = MagicMock()
    mock_client.rewrite_sync.return_value = {
        "rewritten_query": "rewritten only",
        "rewrite_backend": "ollama",
        "rewrite_time_ms": 3,
    }

    with (
        patch("src.client.gradio_ui.GatewayClient", return_value=mock_client),
        patch("src.client.gradio_ui.REWRITE_ONLY_TEST_MODE", True),
    ):
        result = _chat_handler("original query", [], "auto", True, "ollama", "s1")
        outputs = _collect_chat_outputs(result)

    assert any("Rewrite-only test mode" in item for item in outputs)
    mock_client.query_sync.assert_not_called()
    # Ensure session_id is passed so chat turns are saved to Redis
    mock_client.rewrite_sync.assert_called_once_with(
        query="original query",
        rewrite_enable=True,
        rewrite_backend="ollama",
        session_id="s1",
    )


# ---------------------------------------------------------------------------
# Gradio import / create_demo
# ---------------------------------------------------------------------------


def test_create_demo_returns_blocks():
    """create_demo returns a Gradio Blocks instance."""
    demo = create_demo()
    import gradio as gr
    assert demo is not None
    assert isinstance(demo, gr.Blocks)
