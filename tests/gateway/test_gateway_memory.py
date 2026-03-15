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

from src.gateway.api_and_auth.api import app
from src.memory.short_term import GatewayConversationMemory
from src.memory.short_term import MemoryEvent


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
    """save_turn should write both user and session keys and set TTL/LTRIM."""
    memory.save_turn(
        "sess-1",
        "What are my sales?",
        "Your sales are $100.",
        "uds",
        user_id="user-123",
    )
    assert mock_redis.rpush.call_count == 2
    first_key, payload = mock_redis.rpush.call_args_list[0][0]
    assert first_key == "gateway:user:user-123:history"
    data = json.loads(payload)
    assert data["query"] == "What are my sales?"
    assert data["answer"] == "Your sales are $100."
    assert data["workflow"] == "uds"
    assert data["user_id"] == "user-123"
    assert data["session_id"] == "sess-1"
    assert "timestamp" in data
    mock_redis.expire.assert_any_call("gateway:user:user-123:history", 86400)
    mock_redis.expire.assert_any_call("gateway:session:sess-1:history", 86400)
    mock_redis.ltrim.assert_any_call("gateway:user:user-123:history", -50, -1)
    mock_redis.ltrim.assert_any_call("gateway:session:sess-1:history", -50, -1)


def test_save_turn_skips_empty_user_id(memory, mock_redis):
    """save_turn should not call Redis when user_id is empty or absent."""
    memory.save_turn("sess-1", "q", "a", "general", user_id="")
    memory.save_turn("sess-1", "q", "a", "general", user_id="  ")
    memory.save_turn("sess-1", "q", "a", "general")
    mock_redis.rpush.assert_not_called()


def test_get_history_by_session_returns_last_n(memory, mock_redis):
    """get_history_by_session should LRANGE and parse JSON."""
    raw = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
        '{"query": "q2", "answer": "a2", "workflow": "general", "timestamp": "2024-01-01T00:01:00Z"}',
    ]
    mock_redis.lrange.return_value = raw
    history = memory.get_history_by_session("sess-1", last_n=10)
    assert len(history) == 2
    assert history[0]["query"] == "q1"
    assert history[1]["query"] == "q2"
    mock_redis.lrange.assert_called_once_with(
        "gateway:session:sess-1:history", -10, -1
    )


def test_get_history_by_session_empty_session_returns_empty(memory, mock_redis):
    """get_history_by_session should return [] for empty session_id."""
    assert memory.get_history_by_session("", last_n=10) == []
    assert memory.get_history_by_session("  ", last_n=10) == []
    mock_redis.lrange.assert_not_called()


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


def test_no_clear_methods_exist_on_gateway_memory():
    """Conversation history cannot be deleted; clear_session and clear_user_history must not exist."""
    mem = GatewayConversationMemory(MagicMock())
    assert not hasattr(mem, "clear_session"), "clear_session must be removed; history is immutable"
    assert not hasattr(mem, "clear_user_history"), "clear_user_history must be removed; history is immutable"


def test_get_history_by_session_read_only_does_not_call_delete(memory, mock_redis):
    """get_history_by_session is read-only; it must not call Redis delete."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    result = memory.get_history_by_session("sess-1", last_n=10)
    assert len(result) == 1
    assert result[0]["query"] == "q1"
    mock_redis.delete.assert_not_called()


def test_append_event_writes_user_and_session_keys(memory, mock_redis):
    """append_event should store v1 event payload in user/session lists."""
    event = MemoryEvent(
        user_id="user-1",
        session_id="sess-1",
        request_id="req-1",
        event_type="user_query",
        event_content='{"query":"hello"}',
        status="ok",
    )
    memory.append_event(event)
    assert mock_redis.rpush.call_count == 2
    user_call = mock_redis.rpush.call_args_list[0][0]
    session_call = mock_redis.rpush.call_args_list[1][0]
    assert user_call[0] == "gateway:user:user-1:history"
    assert session_call[0] == "gateway:session:sess-1:history"
    user_payload = json.loads(user_call[1])
    assert user_payload["event_type"] == "user_query"
    assert user_payload["request_id"] == "req-1"


# ---------------------------------------------------------------------------
# Integration tests: save_turn called on successful query (requires user_id)
# ---------------------------------------------------------------------------


@patch("src.gateway.api_and_auth.api._clarification_enabled", return_value=False)
@patch("src.gateway.api_and_auth.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api_and_auth.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("rewritten query", None, 0, 0))
@patch("src.gateway.api_and_auth.api.build_execution_plan")
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
    with patch("src.gateway.api_and_auth.api.gateway_memory", mock_memory):
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


@patch("src.gateway.api_and_auth.api.call_general", return_value={"answer": "clarification reply", "sources": []})
@patch(
    "src.gateway.api_and_auth.api.route_workflow",
    return_value=("clarification", 0.0, "manual", None, None),
)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("rewritten", None, 0, 0))
@patch("src.gateway.api_and_auth.api.build_execution_plan")
@patch("src.gateway.api_and_auth.api.check_ambiguity")
@patch("src.gateway.api_and_auth.api._clarification_enabled", return_value=True)
def test_query_clarification_triggers_save_turn(
    mock_clar_enabled, mock_check, mock_build_plan, mock_rewrite, mock_route, mock_call
):
    """Clarification responses should trigger save_turn when user_id present."""
    mock_check.return_value = {
        "needs_clarification": True,
        "clarification_question": "Which fees do you mean?",
    }

    mock_memory = MagicMock()
    with patch("src.gateway.api_and_auth.api.gateway_memory", mock_memory):
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


@patch("src.gateway.api_and_auth.api._is_rewrite_only_mode", return_value=True)
@patch("src.gateway.api_and_auth.api._clarification_enabled", return_value=False)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("rewritten query for retrieval", None, 0, 0))
@patch("src.gateway.api_and_auth.api.build_execution_plan")
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
    with patch("src.gateway.api_and_auth.api.gateway_memory", mock_memory):
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
        "(Rewrite-only; no execution.)",
        "rewrite_only",
        user_id="user-rewrite-only",
    )


@patch("src.gateway.api_and_auth.api._clarification_enabled", return_value=False)
@patch("src.gateway.api_and_auth.api.call_general", return_value={"answer": "ok", "sources": []})
@patch(
    "src.gateway.api_and_auth.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api_and_auth.api.rewrite_query", return_value=("rewritten", None, 0, 0))
@patch("src.gateway.api_and_auth.api.build_execution_plan")
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
    with patch("src.gateway.api_and_auth.api.gateway_memory", None):
        client = TestClient(app)
        resp = client.get("/api/v1/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["history"] == []
    assert "error" in data


def test_get_session_history_with_memory(mock_redis):
    """GET /api/v1/session/{id} returns history when memory enabled."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    mem = GatewayConversationMemory(mock_redis)
    with patch("src.gateway.api_and_auth.api.gateway_memory", mem):
        client = TestClient(app)
        resp = client.get("/api/v1/session/sess-1?last_n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert len(data["history"]) == 1
    assert data["history"][0]["query"] == "q1"


