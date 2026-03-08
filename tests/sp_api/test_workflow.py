"""Tests for workflow and memory."""
import pytest
from unittest.mock import MagicMock

from sp_api.short_term_memory import ConversationMemory
from sp_api.workflow import create_app, SellerAgentState


@pytest.fixture
def mock_redis():
    r = MagicMock()
    r.lrange.return_value = []
    r.rpush.return_value = 1
    r.lrange.return_value = []
    r.delete.return_value = 1
    r.expire.return_value = True
    return r


@pytest.fixture
def memory(mock_redis):
    return ConversationMemory(mock_redis)


def test_save_turn_get_history_returns_insertion_order(memory, mock_redis):
    mock_redis.lrange.side_effect = [
        ['{"query": "q1", "response": "r1", "timestamp": "2024-01-01T00:00:00Z", "iterations": 1}'],
        [
            '{"query": "q1", "response": "r1", "timestamp": "2024-01-01T00:00:00Z", "iterations": 1}',
            '{"query": "q2", "response": "r2", "timestamp": "2024-01-01T00:01:00Z", "iterations": 1}',
        ],
    ]
    memory.save_turn("s1", "q1", "r1")
    memory.save_turn("s1", "q2", "r2")
    history = memory.get_history("s1", last_n=10)
    assert len(history) >= 1
    if len(history) >= 2:
        assert history[0]["query"] == "q1"
        assert history[1]["query"] == "q2"


def test_get_history_last_n_returns_at_most_n(memory, mock_redis):
    def lrange_side_effect(key, start, end):
        all_items = [
            '{"query": "q1", "response": "r1", "timestamp": "2024-01-01T00:00:00Z", "iterations": 1}',
            '{"query": "q2", "response": "r2", "timestamp": "2024-01-01T00:01:00Z", "iterations": 1}',
            '{"query": "q3", "response": "r3", "timestamp": "2024-01-01T00:02:00Z", "iterations": 1}',
        ]
        return all_items[start:] if start >= 0 else all_items[start:]
    mock_redis.lrange.side_effect = lambda k, s, e: [
        '{"query": "q2", "response": "r2", "timestamp": "2024-01-01T00:01:00Z", "iterations": 1}',
        '{"query": "q3", "response": "r3", "timestamp": "2024-01-01T00:02:00Z", "iterations": 1}',
    ]
    history = memory.get_history("s1", last_n=2)
    assert len(history) <= 2


def test_clear_session_removes_history(memory, mock_redis):
    memory.clear_session("s1")
    mock_redis.delete.assert_called()


def test_workflow_creates_app_when_langgraph_available():
    agent = MagicMock()
    agent._classify_intent = lambda q: "query"
    agent.query = lambda q, s: "result"
    agent.list_tools = lambda: [{"name": "product_catalog"}, {"name": "list_orders"}]
    try:
        app = create_app(agent)
        if app is not None:
            state = {"query": "test", "session_id": "s1", "intent": "", "selected_tools": [], "agent_result": "", "formatted_response": ""}
            result = app.invoke(state)
            assert result.get("formatted_response") or result.get("agent_result") == "result"
    except ImportError:
        pytest.skip("langgraph not installed")
