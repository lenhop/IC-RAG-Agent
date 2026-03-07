"""
Unit tests for gateway services layer (call_general, call_ic_docs, etc.).

Covers IC docs skip when IC_DOCS_ENABLED is false: no RAG HTTP call, friendly message returned.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.gateway.services import (
    IC_DOCS_NOT_READY_MESSAGE,
    call_ic_docs,
)


@patch("src.gateway.services._ic_docs_enabled", return_value=False)
def test_call_ic_docs_disabled_returns_friendly_message_no_http(mock_enabled):
    """When IC docs is disabled, call_ic_docs returns friendly message and does not call RAG."""
    with patch("src.gateway.services._http_post") as mock_post:
        result = call_ic_docs("any query", None)
    assert result["answer"] == IC_DOCS_NOT_READY_MESSAGE
    assert result["sources"] == []
    assert "error" not in result
    mock_post.assert_not_called()
    mock_enabled.assert_called()


@patch("src.gateway.services._ic_docs_enabled", return_value=True)
@patch("src.gateway.services._http_post", return_value={"answer": "from RAG", "sources": [{"title": "doc"}]})
def test_call_ic_docs_enabled_calls_rag(mock_post, mock_enabled):
    """When IC docs is enabled, call_ic_docs calls RAG and returns backend response."""
    result = call_ic_docs("IC docs query", "sess-1")
    assert result["answer"] == "from RAG"
    assert result["sources"] == [{"title": "doc"}]
    mock_post.assert_called_once()
    url, payload = mock_post.call_args[0]
    assert "/query" in url
    assert payload.get("mode") == "documents"
    assert "IC docs:" in payload.get("question", "")
