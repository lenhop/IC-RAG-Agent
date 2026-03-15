"""
Unit tests for gateway short-term memory (src.memory.short_term).

Tests get_history_by_session, get_history_by_user, and enforces that
conversation history is read-only (no clear_session / clear_user_history).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.memory.short_term import GatewayConversationMemory, MemoryEvent


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


def test_get_history_by_session_returns_last_n(memory, mock_redis):
    """get_history_by_session should LRANGE session key and parse JSON."""
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


def test_no_clear_methods_exist():
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


def test_get_history_by_user_read_only_does_not_call_delete(memory, mock_redis):
    """get_history_by_user is read-only; it must not call Redis delete."""
    mock_redis.lrange.return_value = [
        '{"query": "q1", "answer": "a1", "workflow": "uds", "timestamp": "2024-01-01T00:00:00Z"}',
    ]
    result = memory.get_history_by_user("user-1", last_n=10)
    assert len(result) == 1
    mock_redis.delete.assert_not_called()
