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
    call_sp_api,
    call_uds,
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


@patch(
    "src.gateway.services._http_post",
    return_value={
        "status": "completed",
        "response": {
            "summary": "Total sales in October 2025 were $12,345.",
            "sources": [{"type": "table", "name": "amz_order"}],
        },
    },
)
def test_call_uds_completed_maps_summary(mock_post):
    """call_uds maps completed UDS response summary into gateway answer."""
    result = call_uds("Total sales in October 2025", None)
    assert result["answer"] == "Total sales in October 2025 were $12,345."
    assert result["sources"] == [{"type": "table", "name": "amz_order"}]
    mock_post.assert_called_once()


@patch(
    "src.gateway.services._http_post",
    return_value={
        "status": "failed",
        "error": "Database connection timeout",
    },
)
def test_call_uds_failed_status_propagates_error(mock_post):
    """call_uds returns error when UDS backend reports failed status."""
    result = call_uds("Total sales in October 2025", None)
    assert result["error"] == "Database connection timeout"
    assert result["error_type"] == "UDSQueryFailed"
    mock_post.assert_called_once()


@patch(
    "src.gateway.services._http_post",
    return_value={
        "status": "completed",
        "response": {},
    },
)
def test_call_uds_empty_response_uses_fallback_message(mock_post):
    """call_uds avoids returning empty answer when UDS response is empty."""
    result = call_uds("Total sales in October 2025", None)
    assert "returned no summary" in result["answer"]
    assert result["sources"] == []
    mock_post.assert_called_once()


@patch(
    "src.gateway.services._http_post",
    return_value={
        "query_id": "q-1",
        "status": "completed",
        "query": "Total sales in October 2025",
        "intent": "sales",
        "response": {
            "summary": "Sales data retrieved successfully.",
            "sources": [],
        },
        "error": None,
    },
)
def test_call_uds_error_null_is_not_treated_as_failure(mock_post):
    """A null error field from UDS should still be treated as successful response."""
    result = call_uds("Total sales in October 2025", None)
    assert result["answer"] == "Sales data retrieved successfully."
    assert result["sources"] == []
    assert "error" not in result
    mock_post.assert_called_once()


@patch("src.gateway.services._http_post", return_value={"response": "ok"})
def test_call_sp_api_omits_none_session_id(mock_post):
    """call_sp_api should not send session_id when it is None."""
    result = call_sp_api("fee query", None)
    assert result["answer"] == "ok"
    assert result["sources"] == []
    mock_post.assert_called_once()
    _url, payload = mock_post.call_args[0]
    assert payload["query"] == "fee query"
    assert "session_id" not in payload
