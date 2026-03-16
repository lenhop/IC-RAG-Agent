"""
Tests for gateway message module (ConversationHistoryHandler, MemoryEventWriter).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.gateway.message import (
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


def test_merge_context_strings_both_empty_returns_none():
    """ConversationHistoryHandler.merge_context_strings returns None when both empty."""
    assert ConversationHistoryHandler.merge_context_strings(None, None) is None
    assert ConversationHistoryHandler.merge_context_strings("", "") is None
    assert ConversationHistoryHandler.merge_context_strings("  ", None) is None


def test_merge_context_strings_one_empty_returns_other():
    """ConversationHistoryHandler.merge_context_strings returns non-empty when one is empty."""
    ctx = (
        "## Historical Conversation\n\n"
        "### Turn1: happened at 2026-03-14 08:59:20 UTC\n\n"
        "- **user_query:** q\n"
        "- **query_clarification:** \u2014\n"
        "- **query_rewriting:** q\n"
        "- **answer:** a"
    )
    assert ConversationHistoryHandler.merge_context_strings(ctx, None) == ctx
    assert ConversationHistoryHandler.merge_context_strings(None, ctx) == ctx
    assert ConversationHistoryHandler.merge_context_strings("", ctx) == ctx


def test_merge_context_strings_merges_and_deduplicates():
    """ConversationHistoryHandler.merge_context_strings merges markdown blocks and deduplicates."""
    pre = (
        "## Historical Conversation\n\n"
        "### Turn1: happened at \u2014 UTC\n\n"
        "- **user_query:** first\n"
        "- **query_clarification:** \u2014\n"
        "- **query_rewriting:** first\n"
        "- **answer:** a1\n"
    )
    mem = (
        "## Historical Conversation\n\n"
        "### Turn1: happened at \u2014 UTC\n\n"
        "- **user_query:** second\n"
        "- **answer:** a2\n\n"
        "---\n\n"
        "### Turn2: happened at \u2014 UTC\n\n"
        "- **user_query:** first\n"
        "- **answer:** a1\n"
    )
    out = ConversationHistoryHandler.merge_context_strings(pre, mem)
    assert out is not None
    assert "## Historical Conversation" in out
    assert "first" in out
    assert "second" in out
    assert "a1" in out
    assert "a2" in out
    assert out.count("### Turn") == 2
    assert "### Turn1:" in out
    assert "### Turn2:" in out


def test_count_context_rounds_empty_returns_zero():
    """ConversationHistoryHandler.count_context_rounds returns 0 for empty/whitespace."""
    assert ConversationHistoryHandler.count_context_rounds(None) == 0
    assert ConversationHistoryHandler.count_context_rounds("") == 0
    assert ConversationHistoryHandler.count_context_rounds("  \n  ") == 0


def test_count_context_rounds_counts_turn_headers():
    """ConversationHistoryHandler.count_context_rounds counts ### TurnN: in markdown."""
    ctx = (
        "## Historical Conversation\n\n"
        "### Turn1: happened at 2026-03-14 08:59:20 UTC\n\n"
        "- **user_query:** q1\n- **answer:** a1\n\n"
        "### Turn2: happened at 2026-03-14 09:00:00 UTC\n\n"
        "- **user_query:** q2\n- **answer:** a2\n"
    )
    assert ConversationHistoryHandler.count_context_rounds(ctx) == 2
    assert ConversationHistoryHandler.count_context_rounds("Single line") == 0


def test_format_history_for_llm_markdown_mixed_events_excludes_intent_classification():
    """format_history_for_llm_markdown: headers, event lines, no intent_classification."""
    rid1 = "req-001"
    rid2 = "req-002"
    history = [
        {
            "ts_utc": "2026-03-14T08:59:20.399782+00:00",
            "request_id": rid1,
            "event_type": "user_query",
            "event_content": json.dumps({"query": "what is FBA fee?", "workflow": "auto"}),
        },
        {
            "ts_utc": "2026-03-14T08:59:20.400000+00:00",
            "request_id": rid1,
            "event_type": "query_rewriting",
            "event_content": json.dumps({"original_query": "what is FBA fee?", "rewritten_query": "what is FBA fee?"}),
        },
        {
            "ts_utc": "2026-03-14T08:59:20.451284+00:00",
            "request_id": rid1,
            "event_type": "intent_classification",
            "event_content": json.dumps({"rewritten_query": "what is FBA fee?", "intents": ["fba_fee"]}),
        },
        {
            "ts_utc": "2026-03-14T08:59:20.451284+00:00",
            "request_id": rid1,
            "event_type": "turn_summary",
            "event_content": json.dumps({"query": "what is FBA fee?", "answer": "Rewrite-only; no execution.", "workflow": "rewrite_only"}),
        },
        {
            "ts_utc": "2026-03-14T08:59:36.839748+00:00",
            "request_id": rid2,
            "event_type": "turn_summary",
            "event_content": json.dumps({"query": "what is RAG?", "answer": "RAG is retrieval-augmented generation.", "workflow": "general"}),
        },
    ]
    out = ConversationHistoryHandler.format_history_for_llm_markdown(history)
    assert "Turn1:" in out
    assert "Turn2:" in out
    assert "happened at" in out
    assert "UTC" in out
    assert "user_query:" in out
    assert "what is FBA fee?" in out
    assert "Rewrite-only; no execution." in out
    assert "what is RAG?" in out
    assert "RAG is retrieval-augmented generation." in out
    assert "intent_classification" not in out
    assert "fba_fee" not in out
    assert "Historical Conversation" in out


def test_format_history_for_llm_markdown_with_sample_session_data():
    """format_history_for_llm_markdown with user-provided Redis-style session list."""
    history = [
        {"ts_utc": "2026-03-14T08:59:20.399782+00:00", "user_id": "u1", "session_id": "s1", "request_id": "712677f1-5071-40ba-8024-668826bc9140", "event_type": "turn_summary", "event_content": '{"query": "what is the weather today, and what is Amazon FBA fee?", "answer": "what is the weather today, and what is Amazon FBA fee?", "workflow": "rewrite_only"}', "status": "ok", "note": "v1 summary event"},
        {"query": "what is the weather today, and what is Amazon FBA fee?", "answer": "(Rewrite-only; no execution.)", "workflow": "rewrite_only", "timestamp": "2026-03-14T08:59:20.451284Z", "user_id": "u1", "session_id": "s1"},
        {"ts_utc": "2026-03-14T08:59:36.299039+00:00", "user_id": "u1", "session_id": "s1", "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21", "event_type": "user_query", "event_content": '{"query": "what is RAG and LLM ?", "workflow": "auto"}', "status": "ok", "note": ""},
        {"ts_utc": "2026-03-14T08:59:36.789564+00:00", "user_id": "u1", "session_id": "s1", "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21", "event_type": "llm_answer", "event_content": '{"answer": "", "workflow": "error", "error": "DeepSeek clarification failed"}', "status": "failed", "note": "RuntimeError"},
        {"ts_utc": "2026-03-14T08:59:36.839748+00:00", "user_id": "u1", "session_id": "s1", "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21", "event_type": "turn_summary", "event_content": '{"query": "what is RAG and LLM ?", "answer": "DeepSeek clarification failed: Error code: 400", "workflow": "error"}', "status": "ok", "note": "v1 summary event"},
        {"query": "what is RAG and LLM ?", "answer": "DeepSeek clarification failed: Error code: 400", "workflow": "error", "timestamp": "2026-03-14T08:59:36.896273Z", "user_id": "u1", "session_id": "s1"},
    ]
    out = ConversationHistoryHandler.format_history_for_llm_markdown(history)
    assert "Turn1:" in out and "Turn2:" in out
    assert "happened at" in out and "UTC" in out
    assert "what is the weather today" in out
    assert "(Rewrite-only; no execution.)" in out
    assert "what is RAG and LLM ?" in out
    assert "DeepSeek clarification failed" in out
    assert "Historical Conversation" in out


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
