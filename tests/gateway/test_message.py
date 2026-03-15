"""
Tests for gateway message module (SessionHistoryHandler, UserHistoryHandler, MemoryEventWriter).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.gateway.api_and_auth.message import (
    ContextHistoryHelper,
    MemoryEventWriter,
    SessionHistoryHandler,
    TurnSummaryPersistence,
    UserHistoryHandler,
)
from src.gateway.memory.short_term import GatewayConversationMemory


@pytest.fixture
def mock_redis():
    """Redis mock with decode_responses=True behavior."""
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


# ---------------------------------------------------------------------------
# SessionHistoryHandler
# ---------------------------------------------------------------------------


def test_session_history_handler_get_history_memory_disabled():
    """SessionHistoryHandler.get_history returns error when memory disabled."""
    result = SessionHistoryHandler.get_history(None, "sess-1", last_n=10)
    assert result["session_id"] == "sess-1"
    assert result["history"] == []
    assert "error" in result


def test_session_history_handler_get_history_with_memory(memory, mock_redis):
    """SessionHistoryHandler.get_history delegates to memory.get_history."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    result = SessionHistoryHandler.get_history(memory, "sess-1", last_n=5)
    assert result["session_id"] == "sess-1"
    assert len(result["history"]) == 1
    assert result["history"][0]["query"] == "q1"
    assert "error" not in result


def test_session_history_handler_clear_memory_disabled():
    """SessionHistoryHandler.clear returns cleared=False when memory disabled."""
    result = SessionHistoryHandler.clear(None, "sess-1")
    assert result["session_id"] == "sess-1"
    assert result["cleared"] is False
    assert "error" in result


def test_session_history_handler_clear_with_memory(memory, mock_redis):
    """SessionHistoryHandler.clear calls memory.clear_session."""
    result = SessionHistoryHandler.clear(memory, "sess-1")
    assert result["session_id"] == "sess-1"
    assert result["cleared"] is True
    mock_redis.delete.assert_called_once_with("gateway:session:sess-1:history")


# ---------------------------------------------------------------------------
# UserHistoryHandler
# ---------------------------------------------------------------------------


def test_user_history_handler_get_history_memory_disabled():
    """UserHistoryHandler.get_history returns error when memory disabled."""
    result = UserHistoryHandler.get_history(None, "user-1", last_n=5)
    assert result["history"] == []
    assert "error" in result


def test_user_history_handler_get_history_v0_format(memory, mock_redis):
    """UserHistoryHandler.get_history normalizes v0 turn format."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    result = UserHistoryHandler.get_history(memory, "user-1", last_n=5)
    assert len(result["history"]) == 1
    assert result["history"][0]["query"] == "q1"
    assert result["history"][0]["answer"] == "a1"
    assert result["history"][0]["workflow"] == "uds"
    assert "error" not in result


def test_user_history_handler_get_history_v1_turn_summary_format(memory, mock_redis):
    """UserHistoryHandler.get_history normalizes v1 turn_summary format."""
    mock_redis.lrange.return_value = [
        json.dumps({
            "event_type": "turn_summary",
            "event_content": json.dumps({"query": "q1", "answer": "a1", "workflow": "uds"}),
            "ts_utc": "2024-01-01T00:00:00Z",
        }),
    ]
    result = UserHistoryHandler.get_history(memory, "user-1", last_n=5)
    assert len(result["history"]) == 1
    assert result["history"][0]["query"] == "q1"
    assert result["history"][0]["answer"] == "a1"
    assert result["history"][0]["workflow"] == "uds"
    assert result["history"][0]["timestamp"] == "2024-01-01T00:00:00Z"


def test_user_history_handler_normalize_raw_history_v1_and_v0_mixed():
    """_normalize_raw_history handles mixed v1 and v0 records."""
    raw = [
        {
            "event_type": "turn_summary",
            "event_content": json.dumps({"query": "q1", "answer": "a1", "workflow": "w1"}),
            "ts_utc": "t1",
        },
        {"query": "q2", "answer": "a2", "workflow": "w2", "timestamp": "t2"},
    ]
    result = UserHistoryHandler._normalize_raw_history(raw)
    assert len(result) == 2
    assert result[0]["query"] == "q1" and result[0]["answer"] == "a1" and result[0]["timestamp"] == "t1"
    assert result[1]["query"] == "q2" and result[1]["answer"] == "a2" and result[1]["timestamp"] == "t2"


def test_user_history_handler_clamps_last_n(memory, mock_redis):
    """UserHistoryHandler.get_history clamps last_n to [1, 50]."""
    mock_redis.lrange.return_value = []
    UserHistoryHandler.get_history(memory, "user-1", last_n=0)
    mock_redis.lrange.assert_called_once()
    # LRANGE key start end: last_n=1 -> lrange(key, -1, -1)
    call_args = mock_redis.lrange.call_args[0]
    assert call_args[1] == -1 and call_args[2] == -1
    mock_redis.reset_mock()
    UserHistoryHandler.get_history(memory, "user-1", last_n=100)
    call_args = mock_redis.lrange.call_args[0]
    assert call_args[1] == -50 and call_args[2] == -1


# ---------------------------------------------------------------------------
# MemoryEventWriter
# ---------------------------------------------------------------------------


def test_memory_event_writer_append_event_skips_when_memory_none():
    """MemoryEventWriter.append_event does nothing when memory is None."""
    MemoryEventWriter.append_event(
        None,
        user_id="user-1",
        session_id="sess-1",
        request_id="req-1",
        event_type="user_query",
        event_content={"query": "q"},
    )
    # No exception, no Redis call (cannot assert on mock since we pass None)


def test_memory_event_writer_append_event_skips_when_user_id_empty(memory):
    """MemoryEventWriter.append_event does nothing when user_id is empty."""
    MemoryEventWriter.append_event(
        memory,
        user_id="",
        session_id="sess-1",
        request_id="req-1",
        event_type="user_query",
        event_content={"query": "q"},
    )
    memory._redis.rpush.assert_not_called()


def test_memory_event_writer_append_event_calls_append_event(memory, mock_redis):
    """MemoryEventWriter.append_event calls memory.append_event when valid."""
    MemoryEventWriter.append_event(
        memory,
        user_id="user-1",
        session_id="sess-1",
        request_id="req-1",
        event_type="user_query",
        event_content={"query": "q", "workflow": "general"},
    )
    mock_redis.rpush.assert_called()
    # append_event may push to both user and session keys; check user key was used
    calls = mock_redis.rpush.call_args_list
    keys_used = [c[0][0] for c in calls]
    assert "gateway:user:user-1:history" in keys_used
    payload = next(c[0][1] for c in calls if c[0][0] == "gateway:user:user-1:history")
    data = json.loads(payload)
    assert data["event_type"] == "user_query"
    assert data["user_id"] == "user-1"
    assert "query" in json.loads(data["event_content"])


# ---------------------------------------------------------------------------
# TurnSummaryPersistence
# ---------------------------------------------------------------------------


def test_turn_summary_persistence_skips_when_memory_none():
    """TurnSummaryPersistence.persist_turn does nothing when memory is None."""
    TurnSummaryPersistence.persist_turn(
        None,
        user_id="user-1",
        session_id="sess-1",
        request_id="req-1",
        query="q",
        answer="a",
        workflow="general",
    )
    # No exception


def test_turn_summary_persistence_skips_when_user_id_empty(memory):
    """TurnSummaryPersistence.persist_turn does nothing when user_id is empty."""
    TurnSummaryPersistence.persist_turn(
        memory,
        user_id="",
        session_id="sess-1",
        request_id="req-1",
        query="q",
        answer="a",
        workflow="general",
    )
    memory._redis.rpush.assert_not_called()


def test_turn_summary_persistence_calls_append_and_save_turn(memory, mock_redis):
    """TurnSummaryPersistence.persist_turn appends turn_summary and calls save_turn."""
    TurnSummaryPersistence.persist_turn(
        memory,
        user_id="user-1",
        session_id="sess-1",
        request_id="req-1",
        query="the query",
        answer="the answer",
        workflow="general",
    )
    # append_event (turn_summary) and save_turn both push to Redis
    assert mock_redis.rpush.call_count >= 1
    # At least one payload should contain turn_summary or turn dict with query/answer
    all_payloads = []
    for call in mock_redis.rpush.call_args_list:
        key, payload = call[0][0], call[0][1]
        all_payloads.append((key, json.loads(payload) if isinstance(payload, str) else payload))
    user_key = "gateway:user:user-1:history"
    user_payloads = [p for k, p in all_payloads if k == user_key]
    assert len(user_payloads) >= 1
    # Either event_content with turn_summary or direct query/answer in turn
    has_turn = any(
        p.get("event_type") == "turn_summary" or ("query" in p and "answer" in p and "workflow" in p)
        for p in user_payloads
    )
    assert has_turn


# ---------------------------------------------------------------------------
# ContextHistoryHelper
# ---------------------------------------------------------------------------


def test_context_history_helper_get_raw_memory_none():
    """ContextHistoryHelper.get_raw returns [] when memory is None."""
    result = ContextHistoryHelper.get_raw(
        None,
        session_id="s1",
        user_id="u1",
        last_n=5,
    )
    assert result == []


def test_context_history_helper_get_raw_prefers_user_id(memory, mock_redis):
    """ContextHistoryHelper.get_raw uses get_history_by_user when user_id present."""
    mock_redis.lrange.return_value = ['{"query": "q1", "answer": "a1"}']
    result = ContextHistoryHelper.get_raw(
        memory,
        session_id="s1",
        user_id="u1",
        last_n=5,
    )
    assert len(result) == 1
    assert result[0]["query"] == "q1"
    mock_redis.lrange.assert_called()
    assert "gateway:user:u1:history" in str(mock_redis.lrange.call_args[0])


def test_context_history_helper_get_raw_falls_back_to_session(memory, mock_redis):
    """ContextHistoryHelper.get_raw uses get_history when user_id empty and session_id set."""
    mock_redis.lrange.return_value = []
    result = ContextHistoryHelper.get_raw(
        memory,
        session_id="s1",
        user_id=None,
        last_n=5,
    )
    mock_redis.lrange.assert_called_once()
    assert mock_redis.lrange.call_args[0][0] == "gateway:session:s1:history"


def test_context_history_helper_get_raw_empty_ids_returns_empty(memory):
    """ContextHistoryHelper.get_raw returns [] when both user_id and session_id empty."""
    result = ContextHistoryHelper.get_raw(
        memory,
        session_id="",
        user_id="",
        last_n=5,
    )
    assert result == []
