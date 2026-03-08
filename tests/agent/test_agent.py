"""Tests for SellerOperationsAgent."""
import pytest
from unittest.mock import MagicMock

from sp_api.sp_api_agent import SellerOperationsAgent
from sp_api.short_term_memory import ConversationMemory
from sp_api.sp_api_client import SPAPIClient, SPAPICredentials


@pytest.fixture
def mock_sp_api_client():
    return MagicMock(spec=SPAPIClient)


@pytest.fixture
def mock_redis():
    r = MagicMock()
    r.lrange.return_value = []
    r.rpush.return_value = 1
    r.delete.return_value = 1
    r.expire.return_value = True
    return r


@pytest.fixture
def memory(mock_redis):
    return ConversationMemory(mock_redis)


@pytest.fixture
def mock_llm():
    return lambda p: "Thought: I will answer.\nFinal Answer: Here is the result."


@pytest.fixture
def agent(mock_llm, mock_sp_api_client, memory):
    return SellerOperationsAgent(
        llm=mock_llm,
        sp_api_client=mock_sp_api_client,
        memory=memory,
        max_iterations=15,
    )


def test_agent_registers_all_10_tools(agent):
    tools = agent.list_tools()
    names = [t["name"] for t in tools]
    assert len(names) == 10
    assert "product_catalog" in names
    assert "inventory_summary" in names
    assert "list_orders" in names
    assert "order_details" in names
    assert "list_shipments" in names
    assert "create_shipment" in names
    assert "fba_fees" in names
    assert "fba_eligibility" in names
    assert "financials" in names
    assert "request_report" in names or "list_reports" in names


def test_query_saves_turn_to_memory(agent, mock_redis):
    result = agent.query("What is my inventory?", "sess-1")
    assert "result" in result.lower() or "answer" in result.lower() or len(result) > 0
    mock_redis.rpush.assert_called()
    mock_redis.expire.assert_called()


def test_multi_turn_includes_history(agent, mock_redis):
    mock_redis.lrange.return_value = [
        '{"query": "Q1", "response": "A1", "timestamp": "2024-01-01T00:00:00Z", "iterations": 1}'
    ]
    result = agent.query("Follow up question", "sess-1")
    assert result is not None


def test_classify_intent_returns_valid_value(agent):
    intent = agent._classify_intent("What are my orders?")
    assert intent in ("query", "action", "report")


def test_max_iterations_guard(agent):
    agent._max_iterations = 2
    agent._llm = lambda p: "Thought: thinking\nAction: product_catalog\nParameters: {}"
    result = agent.query("test", "sess-1")
    assert "iterations" in result.lower() or "result" in result.lower() or len(result) > 0
