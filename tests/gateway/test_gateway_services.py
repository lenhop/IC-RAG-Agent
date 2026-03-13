"""
Unit tests for gateway services layer (call_general, call_amazon_docs, call_ic_docs, etc.).

Covers:
- call_general happy path, backend error, and ConfigError when RAG_API_URL is empty.
- call_amazon_docs happy path, backend error, and ConfigError when RAG_API_URL is empty.
- call_ic_docs skip when IC_DOCS_ENABLED is false: no RAG HTTP call, friendly message returned.
- call_ic_docs enabled via env values (1, yes).
- call_sp_api / call_uds ConfigError when backend URL is empty.
- call_uds completed, failed, empty, and null-error scenarios.
- call_sp_api session_id omission.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.gateway.dispatcher.services import (
    IC_DOCS_NOT_READY_MESSAGE,
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)


@patch("src.gateway.dispatcher.services._ic_docs_enabled", return_value=False)
def test_call_ic_docs_disabled_returns_friendly_message_no_http(mock_enabled):
    """When IC docs is disabled, call_ic_docs returns friendly message and does not call RAG."""
    with patch("src.gateway.dispatcher.services._http_post") as mock_post:
        result = call_ic_docs("any query", None)
    assert result["answer"] == IC_DOCS_NOT_READY_MESSAGE
    assert result["sources"] == []
    assert "error" not in result
    mock_post.assert_not_called()
    mock_enabled.assert_called()


@patch("src.gateway.dispatcher.services._ic_docs_enabled", return_value=True)
@patch("src.gateway.dispatcher.services._http_post", return_value={"answer": "from RAG", "sources": [{"title": "doc"}]})
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
    "src.gateway.dispatcher.services._http_post",
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
    "src.gateway.dispatcher.services._http_post",
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
    "src.gateway.dispatcher.services._http_post",
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
    "src.gateway.dispatcher.services._http_post",
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


@patch("src.gateway.dispatcher.services._http_post", return_value={"response": "ok"})
def test_call_sp_api_omits_none_session_id(mock_post):
    """call_sp_api should not send session_id when it is None."""
    result = call_sp_api("fee query", None)
    assert result["answer"] == "ok"
    assert result["sources"] == []
    mock_post.assert_called_once()
    _url, payload = mock_post.call_args[0]
    assert payload["query"] == "fee query"
    assert "session_id" not in payload


# ---------------------------------------------------------------------------
# Task 2.1: Unit tests for call_general and call_amazon_docs
# ---------------------------------------------------------------------------


@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"answer": "General knowledge answer", "sources": [{"title": "wiki"}]},
)
def test_call_general_success(mock_post):
    """Happy path: call_general returns answer and sources from RAG in general mode."""
    result = call_general("what is machine learning", "sess-1")
    assert result["answer"] == "General knowledge answer"
    assert result["sources"] == [{"title": "wiki"}]
    assert "error" not in result
    mock_post.assert_called_once()
    url, payload = mock_post.call_args[0]
    assert "/query" in url
    assert payload["mode"] == "general"
    assert payload["question"] == "what is machine learning"


@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"error": "Connection refused", "error_type": "ConnectionError"},
)
def test_call_general_backend_error_propagates(mock_post):
    """Backend error from RAG is propagated through call_general result."""
    result = call_general("test query", None)
    assert "error" in result
    assert result["error"] == "Connection refused"
    assert result["error_type"] == "ConnectionError"
    mock_post.assert_called_once()


@patch("src.gateway.dispatcher.services.RAG_API_URL", "")
def test_call_general_rag_url_empty_returns_config_error():
    """When RAG_API_URL is empty, call_general returns ConfigError without HTTP call."""
    with patch("src.gateway.dispatcher.services._http_post") as mock_post:
        result = call_general("some query", None)
    assert result["error"] == "RAG_API_URL not configured"
    assert result["error_type"] == "ConfigError"
    mock_post.assert_not_called()


@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"answer": "FBA storage fee is $0.87 per cubic foot", "sources": [{"doc": "fba-fees"}]},
)
def test_call_amazon_docs_success(mock_post):
    """Happy path: call_amazon_docs returns answer from RAG in documents mode with Amazon bias."""
    result = call_amazon_docs("what is FBA storage fee", "sess-2")
    assert result["answer"] == "FBA storage fee is $0.87 per cubic foot"
    assert result["sources"] == [{"doc": "fba-fees"}]
    assert "error" not in result
    mock_post.assert_called_once()
    url, payload = mock_post.call_args[0]
    assert "/query" in url
    assert payload["mode"] == "documents"
    assert "Amazon docs:" in payload["question"]


@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"error": "Timeout after 120s", "error_type": "Timeout"},
)
def test_call_amazon_docs_backend_error_propagates(mock_post):
    """Backend error from RAG is propagated through call_amazon_docs result."""
    result = call_amazon_docs("FBA fee query", None)
    assert "error" in result
    assert result["error"] == "Timeout after 120s"
    assert result["error_type"] == "Timeout"
    mock_post.assert_called_once()


@patch("src.gateway.dispatcher.services.RAG_API_URL", "")
def test_call_amazon_docs_rag_url_empty_returns_config_error():
    """When RAG_API_URL is empty, call_amazon_docs returns ConfigError without HTTP call."""
    with patch("src.gateway.dispatcher.services._http_post") as mock_post:
        result = call_amazon_docs("some query", None)
    assert result["error"] == "RAG_API_URL not configured"
    assert result["error_type"] == "ConfigError"
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2.2: ConfigError tests for call_sp_api and call_uds
# ---------------------------------------------------------------------------


@patch("src.gateway.dispatcher.services.SP_API_URL", "")
def test_call_sp_api_url_empty_returns_config_error():
    """When SP_API_URL is empty, call_sp_api returns ConfigError without HTTP call."""
    with patch("src.gateway.dispatcher.services._http_post") as mock_post:
        result = call_sp_api("check inventory", None)
    assert result["error"] == "SP_API_URL not configured"
    assert result["error_type"] == "ConfigError"
    mock_post.assert_not_called()


@patch("src.gateway.dispatcher.services.UDS_API_URL", "")
def test_call_uds_url_empty_returns_config_error():
    """When UDS_API_URL is empty, call_uds returns ConfigError without HTTP call."""
    with patch("src.gateway.dispatcher.services._http_post") as mock_post:
        result = call_uds("show me sales", None)
    assert result["error"] == "UDS_API_URL not configured"
    assert result["error_type"] == "ConfigError"
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2.3: IC docs enabled via env values (1, yes)
# ---------------------------------------------------------------------------


@patch("src.gateway.dispatcher.services.os.getenv", return_value="1")
@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"answer": "IC doc content", "sources": [{"title": "ic-doc"}]},
)
def test_call_ic_docs_enabled_when_env_is_one(mock_post, mock_getenv):
    """When IC_DOCS_ENABLED=1, call_ic_docs should call RAG backend."""
    result = call_ic_docs("IC docs query", None)
    assert result["answer"] == "IC doc content"
    assert result["sources"] == [{"title": "ic-doc"}]
    mock_post.assert_called_once()
    url, payload = mock_post.call_args[0]
    assert "/query" in url
    assert payload["mode"] == "documents"


@patch("src.gateway.dispatcher.services.os.getenv", return_value="yes")
@patch(
    "src.gateway.dispatcher.services._http_post",
    return_value={"answer": "IC doc content 2", "sources": []},
)
def test_call_ic_docs_enabled_when_env_is_yes(mock_post, mock_getenv):
    """When IC_DOCS_ENABLED=yes, call_ic_docs should call RAG backend."""
    result = call_ic_docs("another IC query", "sess-3")
    assert result["answer"] == "IC doc content 2"
    assert result["sources"] == []
    mock_post.assert_called_once()
