"""
Tests for gateway API request validation (schema enforcement via Pydantic + FastAPI).

Covers:
- Missing required 'query' field returns HTTP 422.
- Empty 'query' string is accepted by schema (no min_length constraint).
- Invalid 'workflow' string is accepted by schema (no enum constraint);
  gateway falls back to general workflow at dispatch time.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.gateway.api_and_auth.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Task 2.5: API schema validation tests
# ---------------------------------------------------------------------------


def test_query_missing_query_returns_422():
    """Missing 'query' field in request body should return HTTP 422 validation error."""
    payload = {
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data
    # Pydantic error should reference the 'query' field
    errors = data["detail"]
    field_names = [e.get("loc", [])[-1] for e in errors if isinstance(e, dict)]
    assert "query" in field_names


def test_query_empty_body_returns_422():
    """Completely empty JSON body should return HTTP 422 (query is required)."""
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data


def test_query_no_json_body_returns_422():
    """Request with no body at all should return HTTP 422."""
    resp = client.post("/api/v1/query")
    assert resp.status_code == 422


@patch("src.gateway.api_and_auth.api.call_general", return_value={"answer": "fallback answer", "sources": []})
@patch(
    "src.gateway.api_and_auth.api.route_workflow",
    return_value=("general", 0.5, "heuristic", None, None),
)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("", None, 0, 0))
def test_query_empty_query_accepted_by_schema(mock_rewrite, mock_route, mock_call):
    """Empty 'query' string is accepted (no min_length); gateway processes normally."""
    payload = {
        "query": "",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    # Schema allows empty string; request should not be rejected at validation layer
    assert resp.status_code == 200


@patch("src.gateway.api_and_auth.api.call_general", return_value={"answer": "fallback answer", "sources": []})
@patch(
    "src.gateway.api_and_auth.api.route_workflow",
    return_value=("general", 0.5, "heuristic", None, None),
)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("test query", None, 0, 0))
def test_query_invalid_workflow_falls_back_to_general(mock_rewrite, mock_route, mock_call):
    """Invalid 'workflow' value is accepted by schema; gateway falls back to general at dispatch."""
    payload = {
        "query": "test query",
        "workflow": "nonexistent_workflow",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    # No enum constraint → schema accepts; gateway dispatches to general fallback
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "fallback answer"


def test_rewrite_missing_query_returns_422():
    """Missing 'query' field on /api/v1/rewrite should return HTTP 422."""
    payload = {"workflow": "auto", "rewrite_enable": True}
    resp = client.post("/api/v1/rewrite", json=payload)
    assert resp.status_code == 422
