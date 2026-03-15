"""
Tests for gateway message module (ConversationHistoryHandler, MemoryEventWriter).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.gateway.api_and_auth.message import (
    ConversationHistoryHandler,
    MemoryEventWriter,
    TurnSummaryPersistence,
)
from src.memory.short_term import GatewayConversationMemory


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
# ConversationHistoryHandler (session)
# ---------------------------------------------------------------------------


def test_session_history_handler_get_history_memory_disabled():
    """ConversationHistoryHandler.get_session_history returns error when memory disabled."""
    result = ConversationHistoryHandler.get_session_history(None, "sess-1", last_n=10)
    assert result["session_id"] == "sess-1"
    assert result["history"] == []
    assert "error" in result


def test_session_history_handler_get_history_with_memory(memory, mock_redis):
    """ConversationHistoryHandler.get_session_history delegates to memory.get_history_by_session."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    result = ConversationHistoryHandler.get_session_history(memory, "sess-1", last_n=5)
    assert result["session_id"] == "sess-1"
    assert len(result["history"]) == 1
    assert result["history"][0]["query"] == "q1"
    assert "error" not in result


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
