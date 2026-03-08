"""Tests for FastAPI endpoints."""
import os
os.environ["SP_API_TEST_MODE"] = "true"

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.sp_api.fast_api import app, get_agent, get_memory, _agent, _memory


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.query = lambda q, s: "Mock response"
    agent.list_tools = lambda: [{"name": "product_catalog", "description": "Look up product"}]
    agent._memory = MagicMock()
    agent._memory.get_history = lambda s, last_n=10: []
    return agent


@pytest.fixture
def mock_memory():
    m = MagicMock()
    m.get_history = lambda s, last_n=10: []
    m.clear_session = lambda s: None
    return m


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


def test_tools_endpoint_when_agent_available():
    with TestClient(app) as client:
        resp = client.get("/api/v1/seller/tools")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_query_endpoint_structure():
    with TestClient(app) as client:
        resp = client.post("/api/v1/seller/query", json={"query": "test", "session_id": "s1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "session_id" in data
        assert "iterations" in data
        assert "tools_used" in data


def test_error_handler_returns_structured_json():
    with TestClient(app) as client:
        resp = client.get("/api/v1/seller/session/nonexistent")
        assert resp.status_code in (200, 503)


def test_query_stream_yields_sse_chunks():
    with TestClient(app) as client:
        resp = client.post("/api/v1/seller/query/stream", json={"query": "test", "session_id": "s1"})
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert any("data:" in line for line in lines)
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 1
        import json
        last_data = json.loads(data_lines[-1].replace("data:", "").strip())
        assert last_data.get("type") == "final"
        assert "response" in last_data
