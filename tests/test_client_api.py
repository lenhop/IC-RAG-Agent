"""
Tests for GatewayClient (src/client/api_client.py).

Covers query_sync: 200 success, ConnectionError, Timeout, 5xx, mock mode, payload.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.client.api_client import GatewayClient, VALID_WORKFLOWS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """GatewayClient with non-empty base_url to trigger real HTTP (mocked)."""
    return GatewayClient(base_url="http://test-gateway:8000", timeout=30)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------


def test_mock_mode_empty_base_url():
    """Mock mode: empty base_url returns simulated response without HTTP call."""
    client = GatewayClient(base_url="")
    with patch("src.client.api_client.requests.post") as mock_post:
        result = client.query_sync(
            query="hello",
            workflow="uds",
            rewrite_enable=True,
            session_id="sess-123",
        )
    mock_post.assert_not_called()
    assert "answer" in result
    assert "[Mock]" in result["answer"]
    assert "hello" in result["answer"]
    assert result["workflow"] == "uds"
    assert result.get("routing_confidence") == 1.0


def test_mock_mode_gateway_mock_env(monkeypatch):
    """Mock mode: GATEWAY_MOCK=true returns simulated response without HTTP call."""
    monkeypatch.setenv("GATEWAY_MOCK", "true")
    monkeypatch.setenv("GATEWAY_API_URL", "http://localhost:8000")
    client = GatewayClient(base_url="http://localhost:8000")
    with patch("src.client.api_client.requests.post") as mock_post:
        result = client.query_sync("test query", workflow="auto")
    mock_post.assert_not_called()
    assert "answer" in result
    assert "[Mock]" in result["answer"]
    assert "test query" in result["answer"]


# ---------------------------------------------------------------------------
# Success (200)
# ---------------------------------------------------------------------------


def test_query_sync_200_returns_answer(client):
    """200 response returns answer dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "The answer is 42.",
        "workflow": "uds",
        "sources": [],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.client.api_client.requests.post", return_value=mock_resp) as mock_post:
        result = client.query_sync(
            query="What is the answer?",
            workflow="uds",
            rewrite_enable=False,
            session_id="s1",
        )

    assert result["answer"] == "The answer is 42."
    assert result["workflow"] == "uds"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["query"] == "What is the answer?"
    assert call_kwargs["json"]["workflow"] == "uds"
    assert call_kwargs["json"]["rewrite_enable"] is False
    assert call_kwargs["json"]["session_id"] == "s1"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_query_sync_connection_error(client):
    """ConnectionError returns error dict with error_type ConnectionError."""
    with patch(
        "src.client.api_client.requests.post",
        side_effect=requests.ConnectionError("Connection refused"),
    ):
        result = client.query_sync("hello", workflow="auto")

    assert "error" in result
    assert result["error_type"] == "ConnectionError"
    assert "Cannot connect" in result["error"]


def test_query_sync_timeout(client):
    """Timeout returns error dict with error_type Timeout."""
    with patch(
        "src.client.api_client.requests.post",
        side_effect=requests.Timeout("Timed out"),
    ):
        result = client.query_sync("hello", workflow="auto")

    assert "error" in result
    assert result["error_type"] == "Timeout"
    assert "timed out" in result["error"].lower()


def test_query_sync_5xx_returns_error(client):
    """5xx response (raise_for_status) returns error dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_resp.json.return_value = {"detail": "Internal server error"}

    with patch("src.client.api_client.requests.post", return_value=mock_resp):
        result = client.query_sync("hello", workflow="auto")

    assert "error" in result
    assert result["error_type"] == "RequestException"


def test_query_sync_503_returns_error(client):
    """503 response returns error dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")

    with patch("src.client.api_client.requests.post", return_value=mock_resp):
        result = client.query_sync("hello", workflow="auto")

    assert "error" in result
    assert result["error_type"] == "RequestException"


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def test_query_sync_payload_has_required_fields(client):
    """Request payload contains query, workflow, rewrite_enable, session_id."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"answer": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.client.api_client.requests.post", return_value=mock_resp) as mock_post:
        client.query_sync(
            query="my query",
            workflow="sp_api",
            rewrite_enable=True,
            session_id="uuid-123",
        )

    call_kwargs = mock_post.call_args[1]
    payload = call_kwargs["json"]
    assert payload["query"] == "my query"
    assert payload["workflow"] == "sp_api"
    assert payload["rewrite_enable"] is True
    assert payload["session_id"] == "uuid-123"


def test_query_sync_payload_includes_rewrite_backend_when_enabled(client):
    """When rewrite_enable=True and rewrite_backend provided, payload includes rewrite_backend."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"answer": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.client.api_client.requests.post", return_value=mock_resp) as mock_post:
        client.query_sync(
            query="q",
            workflow="auto",
            rewrite_enable=True,
            rewrite_backend="deepseek",
            session_id=None,
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["rewrite_backend"] == "deepseek"


def test_query_sync_payload_session_id_none(client):
    """Request payload allows session_id=None."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"answer": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.client.api_client.requests.post", return_value=mock_resp) as mock_post:
        client.query_sync("q", workflow="auto", session_id=None)

    payload = mock_post.call_args[1]["json"]
    assert payload["session_id"] is None


# ---------------------------------------------------------------------------
# Workflow normalization
# ---------------------------------------------------------------------------


def test_query_sync_invalid_workflow_defaults_to_auto(client):
    """Invalid workflow value defaults to auto in payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"answer": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.client.api_client.requests.post", return_value=mock_resp) as mock_post:
        client.query_sync("q", workflow="invalid_workflow")

    payload = mock_post.call_args[1]["json"]
    assert payload["workflow"] == "auto"


def test_valid_workflows_constant():
    """VALID_WORKFLOWS contains expected values."""
    assert "auto" in VALID_WORKFLOWS
    assert "general" in VALID_WORKFLOWS
    assert "amazon_docs" in VALID_WORKFLOWS
    assert "ic_docs" in VALID_WORKFLOWS
    assert "sp_api" in VALID_WORKFLOWS
    assert "uds" in VALID_WORKFLOWS
