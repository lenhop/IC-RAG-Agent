#!/usr/bin/env python3
"""
Print session history as Markdown (for LLM context format).

Usage:
  python scripts/format_session_history_markdown.py [session_id]

If session_id is omitted, uses 7cf0dfac-d842-448b-ac24-0ba253b71290.
If Redis is unavailable, runs with built-in sample data and prints the demo output.
"""

from __future__ import annotations

import os
import sys

# Add project root for imports when run as script
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from src.gateway.message import ConversationHistoryHandler, get_gateway_memory

DEFAULT_SESSION_ID = "7cf0dfac-d842-448b-ac24-0ba253b71290"

# Sample data (same structure as user-provided Redis log) when Redis is unavailable
SAMPLE_HISTORY = [
    {
        "ts_utc": "2026-03-14T08:59:20.399782+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "712677f1-5071-40ba-8024-668826bc9140",
        "event_type": "turn_summary",
        "event_content": '{"query": "what is the weather today, and what is Amazon FBA fee?", "answer": "what is the weather today, and what is Amazon FBA fee?", "workflow": "rewrite_only"}',
        "status": "ok",
        "note": "v1 summary event",
    },
    {
        "query": "what is the weather today, and what is Amazon FBA fee?",
        "answer": "(Rewrite-only; no execution.)",
        "workflow": "rewrite_only",
        "timestamp": "2026-03-14T08:59:20.451284Z",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
    },
    {
        "ts_utc": "2026-03-14T08:59:36.299039+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21",
        "event_type": "user_query",
        "event_content": '{"query": "what is RAG and LLM ?", "workflow": "auto"}',
        "status": "ok",
        "note": "",
    },
    {
        "ts_utc": "2026-03-14T08:59:36.789564+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21",
        "event_type": "llm_answer",
        "event_content": '{"answer": "", "workflow": "error", "error": "DeepSeek clarification failed: Error code: 400"}',
        "status": "failed",
        "note": "RuntimeError",
    },
    {
        "ts_utc": "2026-03-14T08:59:36.839748+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "6d143051-86f1-4bea-9ef3-6b5b20b57a21",
        "event_type": "turn_summary",
        "event_content": '{"query": "what is RAG and LLM ?", "answer": "DeepSeek clarification failed: Error code: 400 - Model Not Exist", "workflow": "error"}',
        "status": "ok",
        "note": "v1 summary event",
    },
    {
        "query": "what is RAG and LLM ?",
        "answer": "DeepSeek clarification failed: Error code: 400 - Model Not Exist",
        "workflow": "error",
        "timestamp": "2026-03-14T08:59:36.896273Z",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
    },
    {
        "ts_utc": "2026-03-14T09:00:14.173315+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "265db3f2-9ef0-423e-96c1-c36d6dec8459",
        "event_type": "user_query",
        "event_content": '{"query": "what is RAG and LLM ?", "workflow": "auto"}',
        "status": "ok",
        "note": "",
    },
    {
        "ts_utc": "2026-03-14T09:00:14.663845+00:00",
        "user_id": "0145953e-b2a8-4e87-9eaa-fe4ad975cfe4",
        "session_id": "7cf0dfac-d842-448b-ac24-0ba253b71290",
        "request_id": "265db3f2-9ef0-423e-96c1-c36d6dec8459",
        "event_type": "turn_summary",
        "event_content": '{"query": "what is RAG and LLM ?", "answer": "DeepSeek clarification failed: Error code: 400", "workflow": "error"}',
        "status": "ok",
        "note": "v1 summary event",
    },
]


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SESSION_ID
    memory = get_gateway_memory()
    history: list = []
    if memory:
        res = ConversationHistoryHandler.get_session_history(memory, session_id, last_n=100)
        history = res.get("history") or []
    if not history:
        print("# Using built-in sample data (Redis unavailable or empty session)\n", file=sys.stderr)
        history = SAMPLE_HISTORY
    out = ConversationHistoryHandler.format_history_for_llm_markdown(history)
    print(out)


if __name__ == "__main__":
    main()
