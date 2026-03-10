"""
Tests for gateway short-term memory (Redis-backed session history).

Unit tests: GatewayConversationMemory with mocked Redis.
Integration tests: save_turn called when query succeeds with session_id.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.gateway.api import app
from src.gateway.memory import GatewayConversationMemory


# ---------------------------------------------------------------------------
# Unit tests: GatewayConversationMemory
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """Redis mock with decode_responses=True behavior (returns str)."""
    r = MagicMock()
    r.rpush.return_value = 1
    r.expire.return_value = True
    r.ltrim.return_value = True
    r.lrange.return_value = []
    r.delete.return_value = 1
    return r


@pytest.fixture
def memory(mock_redis):
    """GatewayConversationMemory with mocked Redis."""
    return GatewayConversationMemory(mock_redis)


def test_save_turn_appends_turn(memory, mock_redis):
    """save_turn should RPUSH JSON payload to user key and call EXPIRE."""
    memory.save_turn(
        "sess-1",
        "What are my sales?",
        "Your sales are $100.",
        "uds",
        user_id="user-123",
    )
    mock_redis.rpush.assert_called_once()
    key, payload = mock_redis.rpush.call_args[0]
    assert key == "gateway:user:user-123:history"
    data = json.loads(payload)
    assert data["query"] == "What are my sales?"
    assert data["answer"] == "Your sales are $100."
    assert data["workflow"] == "uds"
    assert data["user_id"] == "user-123"
    assert data["session_id"] == "sess-1"
    assert "timestamp" in data
    mock_redis.expire.assert_called_once_with(key, 86400)
    mock_redis.ltrim.assert_called_once_with(key, -50, -1)


def test_save_turn_skips_empty_user_id(memory, mock_redis):
    """save_turn should not call Redis when user_id is empty or absent."""
    memory.save_turn("sess-1", "q", "a", "general", user_id="")
    memory.save_turn("sess-1", "q", "a", "general", user_id="  ")
    memory.save_turn("sess-1", "q", "a", "general")
    mock_redis.rpush.assert_not_called()


def test_get_history_returns_last_n(memory, mock_redis):
    """get_history should LRANGE and parse JSON."""
    raw = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
        '{"query": "q2", "answer": "a2", "workflow": "general", "timestamp": "2024-01-01T00:01:00Z"}',
    ]
    mock_redis.lrange.return_value = raw
    history = memory.get_history("sess-1", last_n=10)
    assert len(history) == 2
    assert history[0]["query"] == "q1"
    assert history[1]["query"] == "q2"
    mock_redis.lrange.assert_called_once_with(
        "gateway:session:sess-1:history", -10, -1
    )


def test_get_history_empty_session_returns_empty(memory, mock_redis):
    """get_history should return [] for empty session_id."""
    assert memory.get_history("", last_n=10) == []
    assert memory.get_history("  ", last_n=10) == []
    mock_redis.lrange.assert_not_called()


def test_clear_session_deletes_key(memory, mock_redis):
    """clear_session should DELETE the session key."""
    memory.clear_session("sess-1")
    mock_redis.delete.assert_called_once_with("gateway:session:sess-1:history")


def test_clear_session_skips_empty(memory, mock_redis):
    """clear_session should not call Redis when session_id is empty."""
    memory.clear_session("")
    memory.clear_session("  ")
    mock_redis.delete.assert_not_called()


def test_get_history_by_user_returns_last_n(memory, mock_redis):
    """get_history_by_user should LRANGE user key and parse JSON."""
    raw = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z", "user_id": "u1", "session_id": "s1"}',
        '{"query": "q2", "answer": "a2", "workflow": "general", "timestamp": "2024-01-01T00:01:00Z", "user_id": "u1", "session_id": "s1"}',
    ]
    mock_redis.lrange.return_value = raw
    history = memory.get_history_by_user("user-1", last_n=10)
    assert len(history) == 2
    assert history[0]["query"] == "q1"
    assert history[1]["query"] == "q2"
    mock_redis.lrange.assert_called_once_with(
        "gateway:user:user-1:history", -10, -1
    )


def test_get_history_by_user_empty_user_returns_empty(memory, mock_redis):
    """get_history_by_user should return [] for empty user_id."""
    assert memory.get_history_by_user("", last_n=10) == []
    assert memory.get_history_by_user("  ", last_n=10) == []
    mock_redis.lrange.assert_not_called()


def test_clear_user_history_deletes_key(memory, mock_redis):
    """clear_user_history should DELETE the user key."""
    memory.clear_user_history("user-1")
    mock_redis.delete.assert_called_once_with("gateway:user:user-1:history")


def test_clear_user_history_skips_empty(memory, mock_redis):
    """clear_user_history should not call Redis when user_id is empty."""
    memory.clear_user_history("")
    memory.clear_user_history("  ")
    mock_redis.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests: save_turn called on successful query (requires user_id)
# ---------------------------------------------------------------------------


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
@patch("src.gateway.api.build_execution_plan")
def test_query_success_saves_turn_when_user_id_present(
    mock_build_plan, mock_rewrite, mock_route, mock_call, mock_clar_enabled
):
    """When query succeeds with user_id and non-empty answer, save_turn is called."""
    from src.gateway.schemas import RewritePlan, TaskGroup, TaskItem

    mock_plan = RewritePlan(
        task_groups=[
            TaskGroup(
                group_id="g1",
                tasks=[TaskItem(task_id="t1", query="rewritten query", workflow="general")],
            )
        ],
    )
    mock_build_plan.return_value = mock_plan

    mock_memory = MagicMock()
    with patch("src.gateway.api.gateway_memory", mock_memory):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/query",
            json={
                "query": "What are my sales?",
                "workflow": "auto",
                "rewrite_enable": True,
                "session_id": "sess-integration-1",
                "user_id": "user-integration-1",
                "stream": False,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general answer"
    assert data["workflow"] == "general"
    mock_memory.save_turn.assert_called_once_with(
        "sess-integration-1",
        "What are my sales?",
        "general answer",
        "general",
        user_id="user-integration-1",
    )


@patch("src.gateway.api.call_general", return_value={"answer": "clarification reply", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("clarification", 0.0, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten", None, 0, 0))
@patch("src.gateway.api.build_execution_plan")
@patch("src.gateway.api.check_ambiguity")
@patch("src.gateway.api._clarification_enabled", return_value=True)
def test_query_clarification_triggers_save_turn(
    mock_clar_enabled, mock_check, mock_build_plan, mock_rewrite, mock_route, mock_call
):
    """Clarification responses should trigger save_turn when user_id present."""
    mock_check.return_value = {
        "needs_clarification": True,
        "clarification_question": "Which fees do you mean?",
    }

    mock_memory = MagicMock()
    with patch("src.gateway.api.gateway_memory", mock_memory):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/query",
            json={
                "query": "Show fees",
                "workflow": "auto",
                "rewrite_enable": True,
                "session_id": "sess-1",
                "user_id": "user-1",
                "stream": False,
            },
        )
    assert resp.status_code == 200
    mock_memory.save_turn.assert_called_once_with(
        "sess-1",
        "Show fees",
        "Which fees do you mean?",
        "clarification",
        user_id="user-1",
    )


@patch("src.gateway.api._is_rewrite_only_mode", return_value=True)
@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten query for retrieval", None, 0, 0))
@patch("src.gateway.api.build_execution_plan")
def test_query_rewrite_only_triggers_save_turn(
    mock_build_plan, mock_rewrite, mock_clar_enabled, mock_rewrite_only
):
    """Rewrite-only response should trigger save_turn when user_id present."""
    from src.gateway.schemas import RewritePlan, TaskGroup, TaskItem

    mock_plan = RewritePlan(
        task_groups=[
            TaskGroup(
                group_id="g1",
                tasks=[TaskItem(task_id="t1", query="rewritten query for retrieval", workflow="rewrite_only")],
            )
        ],
    )
    mock_build_plan.return_value = mock_plan

    mock_memory = MagicMock()
    with patch("src.gateway.api.gateway_memory", mock_memory):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/query",
            json={
                "query": "What about last month?",
                "workflow": "auto",
                "rewrite_enable": True,
                "session_id": "sess-rewrite-only",
                "user_id": "user-rewrite-only",
                "stream": False,
            },
        )
    assert resp.status_code == 200
    mock_memory.save_turn.assert_called_once_with(
        "sess-rewrite-only",
        "What about last month?",
        "rewritten query for retrieval",
        "rewrite_only",
        user_id="user-rewrite-only",
    )


@patch("src.gateway.api._clarification_enabled", return_value=False)
@patch("src.gateway.api.call_general", return_value={"answer": "ok", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value=("rewritten", None, 0, 0))
@patch("src.gateway.api.build_execution_plan")
def test_query_without_memory_does_not_fail(
    mock_build_plan, mock_rewrite, mock_route, mock_call, mock_clar_enabled
):
    """When gateway_memory is None, query still succeeds (no save)."""
    from src.gateway.schemas import RewritePlan, TaskGroup, TaskItem

    mock_plan = RewritePlan(
        task_groups=[
            TaskGroup(
                group_id="g1",
                tasks=[TaskItem(task_id="t1", query="rewritten", workflow="general")],
            )
        ],
    )
    mock_build_plan.return_value = mock_plan

    client = TestClient(app)
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "test",
            "workflow": "auto",
            "rewrite_enable": True,
            "session_id": "sess-1",
            "stream": False,
        },
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Session history API tests
# ---------------------------------------------------------------------------


def test_get_session_history_when_memory_disabled():
    """GET /api/v1/session/{id} returns error when memory disabled."""
    with patch("src.gateway.api.gateway_memory", None):
        client = TestClient(app)
        resp = client.get("/api/v1/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["history"] == []
    assert "error" in data


def test_delete_session_when_memory_disabled():
    """DELETE /api/v1/session/{id} returns cleared=False when memory disabled."""
    with patch("src.gateway.api.gateway_memory", None):
        client = TestClient(app)
        resp = client.delete("/api/v1/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["cleared"] is False
    assert "error" in data


def test_get_session_history_with_memory(mock_redis):
    """GET /api/v1/session/{id} returns history when memory enabled."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    mem = GatewayConversationMemory(mock_redis)
    with patch("src.gateway.api.gateway_memory", mem):
        client = TestClient(app)
        resp = client.get("/api/v1/session/sess-1?last_n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert len(data["history"]) == 1
    assert data["history"][0]["query"] == "q1"


def test_delete_session_with_memory(mock_redis):
    """DELETE /api/v1/session/{id} clears session when memory enabled."""
    mem = GatewayConversationMemory(mock_redis)
    with patch("src.gateway.api.gateway_memory", mem):
        client = TestClient(app)
        resp = client.delete("/api/v1/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["cleared"] is True
    mock_redis.delete.assert_called_once_with("gateway:session:sess-1:history")


# ---------------------------------------------------------------------------
# User history API tests
# ---------------------------------------------------------------------------


@patch("src.gateway.api.verify_token", return_value={"sub": "user-123"})
def test_get_user_history_returns_history_when_authenticated(mock_verify, mock_redis):
    """GET /api/v1/user/history returns history when JWT valid and memory enabled."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    mem = GatewayConversationMemory(mock_redis)
    with patch("src.gateway.api.gateway_memory", mem):
        client = TestClient(app)
        resp = client.get(
            "/api/v1/user/history?last_n=5",
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert len(data["history"]) == 1
    assert data["history"][0]["query"] == "q1"
    assert data["history"][0]["answer"] == "a1"
    mock_redis.lrange.assert_called_once_with("gateway:user:user-123:history", -5, -1)


def test_get_user_history_returns_401_when_no_auth():
    """GET /api/v1/user/history returns 401 when Authorization header missing."""
    client = TestClient(app)
    resp = client.get("/api/v1/user/history")
    assert resp.status_code == 401


@patch("src.gateway.api.verify_token", return_value=None)
def test_get_user_history_returns_401_when_token_invalid(mock_verify):
    """GET /api/v1/user/history returns 401 when token invalid or expired."""
    client = TestClient(app)
    resp = client.get(
        "/api/v1/user/history",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert resp.status_code == 401
