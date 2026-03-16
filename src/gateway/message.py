"""
Uniform gateway memory interface.

This module is the single gateway-facing API for short-term memory. Gateway code
(api.py, router, etc.) must use only this module for memory operations; do not
import or call short_term directly. Storage (Redis/ClickHouse) is an implementation
detail behind this interface.

Public entry points:
  - get_gateway_memory(): obtain the singleton memory instance (or None)
  - ConversationHistoryHandler: session history retrieval for API endpoints
  - MemoryEventWriter: append memory events to Redis (+ optional ClickHouse)
  - TurnSummaryPersistence: append turn_summary event and save_turn in one call

All handler/writer methods accept memory as first parameter (from get_gateway_memory()).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.memory.short_term import GatewayConversationMemory, MemoryEvent

logger = logging.getLogger(__name__)

# Cached gateway memory instance; initialized once by get_gateway_memory().
_gateway_memory: Optional[GatewayConversationMemory] = None


def get_gateway_memory() -> Optional[GatewayConversationMemory]:
    """
    Return the singleton gateway short-term memory instance (Redis-backed).

    Initializes on first call; returns None if Redis is unreachable.
    Gateway code should obtain memory via this function and pass it to
    ConversationHistoryHandler,
    MemoryEventWriter, TurnSummaryPersistence.
    """
    global _gateway_memory
    if _gateway_memory is not None:
        return _gateway_memory
    try:
        import redis
        redis_url = os.getenv("GATEWAY_REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        _gateway_memory = GatewayConversationMemory(_redis_client)
        logger.info(
            "Gateway memory initialized (Redis: %s)",
            redis_url.split("@")[-1] if "@" in redis_url else redis_url,
        )
        return _gateway_memory
    except Exception as exc:
        logger.warning("Gateway memory disabled (Redis unreachable): %s", exc)
        return None


class ConversationHistoryHandler:
    """
    Session-only history operations for API and rewrite context.

    Provides session history retrieval for API endpoints and LLM context.
    History is retrieved only by session_id (no user-scoped retrieval).
    """

    @classmethod
    def get_session_history(
        cls,
        memory: Optional[GatewayConversationMemory],
        session_id: str,
        last_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Return last N turns for a session.

        Args:
            memory: Gateway memory instance (Redis-backed). None when disabled.
            session_id: Session identifier.
            last_n: Max turns to return (capped at 50).

        Returns:
            Dict with session_id, history list, and optional error key.
        """
        if not memory:
            logger.debug("Session history skipped: gateway memory disabled")
            return {"session_id": session_id, "history": [], "error": "Gateway memory disabled"}
        history = memory.get_history_by_session(session_id, last_n=min(last_n, 50))
        return {"session_id": session_id, "history": history}

    @staticmethod
    def _parse_context_turns_markdown(block: str) -> List[Tuple[str, str]]:
        """
        Parse markdown-formatted context into (user_query, answer) pairs per turn.

        Expects ## Historical Conversation, ### TurnN: ..., - **user_query:** ...,
        - **answer:** ... (and optional query_clarification, query_rewriting).
        """
        turns: List[Tuple[str, str]] = []
        text = (block or "").strip()
        if not text:
            return turns
        parts = re.split(r"\n### Turn\d+:", text, flags=re.IGNORECASE)
        uq_pattern = re.compile(r"^-\s*\*\*user_query:\*\*\s*(.*)$", re.IGNORECASE)
        ans_pattern = re.compile(r"^-\s*\*\*answer:\*\*\s*(.*)$", re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            query_val = ""
            answer_val = ""
            for line in part.splitlines():
                line = (line or "").strip()
                m = uq_pattern.match(line)
                if m:
                    query_val = (m.group(1) or "").strip()
                    if query_val == "\u2014" or query_val == "-":
                        query_val = ""
                    continue
                m = ans_pattern.match(line)
                if m:
                    answer_val = (m.group(1) or "").strip()
                    if answer_val == "\u2014" or answer_val == "-":
                        answer_val = ""
            if query_val or answer_val:
                turns.append((query_val or "", answer_val or ""))
        return turns

    @classmethod
    def merge_context_strings(
        cls,
        preloaded_context: Optional[str],
        memory_context: Optional[str],
    ) -> Optional[str]:
        """
        Merge two markdown-formatted context strings (deduplicate by query/answer).

        Parses markdown, merges, and outputs markdown. Used by router to merge
        clarification context with memory-loaded context.
        """
        preloaded = (preloaded_context or "").strip()
        memory = (memory_context or "").strip()
        if not preloaded and not memory:
            return None
        if not preloaded:
            return memory
        if not memory:
            return preloaded
        merged_turns: List[Tuple[str, str]] = []
        seen_keys: set[str] = set()
        for block in (preloaded, memory):
            for query, answer in cls._parse_context_turns_markdown(block):
                key = f"{query.lower()}::{answer.lower()}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_turns.append((query, answer))
        if not merged_turns:
            return None
        lines_out: List[str] = ["## Historical Conversation", ""]
        for idx, (q, a) in enumerate(merged_turns, start=1):
            uq = (q or "").strip() or "\u2014"
            qc = "\u2014"
            qr = (q or "").strip() or "\u2014"
            ans = (a or "").strip() or "\u2014"
            lines_out.append(f"### Turn{idx}: happened at \u2014 UTC")
            lines_out.append("")
            lines_out.append(f"- **user_query:** {uq}")
            lines_out.append(f"- **query_clarification:** {qc}")
            lines_out.append(f"- **query_rewriting:** {qr}")
            lines_out.append(f"- **answer:** {ans}")
            lines_out.append("")
            if idx < len(merged_turns):
                lines_out.append("---")
                lines_out.append("")
        return "\n".join(lines_out).strip()

    @staticmethod
    def count_context_rounds(context: Optional[str]) -> int:
        """Count turns in markdown-formatted context (### TurnN: lines)."""
        text = (context or "").strip()
        if not text:
            return 0
        return len(re.findall(r"^### Turn\d+:", text, re.MULTILINE | re.IGNORECASE))

    @staticmethod
    def _normalize_ts_utc(ts_str: Optional[str]) -> str:
        """Normalize timestamp to 'YYYY-MM-DD HH:MM:SS UTC' for Markdown header."""
        if not ts_str or not str(ts_str).strip():
            return ""
        raw = str(ts_str).strip().replace("Z", "+00:00")
        try:
            if "+" in raw or raw.endswith("00:00"):
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception as exc:
            logger.debug("_normalize_ts_utc parse failed: %s", exc)
            return raw[:19].replace("T", " ") + " UTC" if len(raw) >= 19 else raw + " UTC"

    @staticmethod
    def _parse_event_content(raw: Any) -> Dict[str, Any]:
        """Parse event_content (JSON string or dict) to a dict. Returns {} on failure."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw) if raw.strip() else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _group_history_into_turns(cls, history: list) -> List[Dict[str, Any]]:
        """
        Group raw history into turns; extract user_query, query_clarification,
        query_rewriting, answer. Exclude intent_classification.
        """
        turns: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        current_rid: Optional[str] = None

        def flush_current() -> None:
            nonlocal current, current_rid
            if current and (current.get("events", {}).get("user_query") or current.get("events", {}).get("answer")):
                turns.append(current)
            current = None
            current_rid = None

        def ensure_turn(ts_utc: Optional[str] = None) -> Dict[str, Any]:
            nonlocal current
            if current is None:
                current = {"ts_utc": ts_utc or "", "events": {"user_query": None, "query_clarification": None, "query_rewriting": None, "answer": None}}
            if ts_utc and not current.get("ts_utc"):
                current["ts_utc"] = ts_utc
            return current

        for item in history:
            if not isinstance(item, dict):
                continue
            if "event_type" in item:
                rid = (item.get("request_id") or "").strip()
                ts_utc = cls._normalize_ts_utc(item.get("ts_utc"))
                if rid and current_rid is not None and rid != current_rid:
                    flush_current()
                current_rid = rid
                turn = ensure_turn(ts_utc)
                ev = turn["events"]
                content = cls._parse_event_content(item.get("event_content"))
                etype = (item.get("event_type") or "").strip()
                if etype == "user_query":
                    ev["user_query"] = content.get("query") or ev.get("user_query")
                elif etype == "query_clarification":
                    ev["query_clarification"] = content.get("clarification_question") or content.get("query") or ev.get("query_clarification")
                elif etype == "query_rewriting":
                    ev["query_rewriting"] = content.get("rewritten_query") or content.get("original_query") or ev.get("query_rewriting")
                elif etype == "intent_classification":
                    pass
                elif etype == "llm_answer":
                    ev["answer"] = content.get("answer") or ev.get("answer")
                elif etype == "turn_summary":
                    ev["answer"] = content.get("answer") or ev.get("answer")
                    if not ev.get("user_query"):
                        ev["user_query"] = content.get("query")
                continue
            if "query" in item and "answer" in item:
                flush_current()
                ts = cls._normalize_ts_utc(item.get("timestamp"))
                turns.append({
                    "ts_utc": ts,
                    "events": {
                        "user_query": item.get("query"),
                        "query_clarification": None,
                        "query_rewriting": item.get("query"),
                        "answer": item.get("answer"),
                    },
                })
                current_rid = None
                current = None
        flush_current()
        return turns

    @classmethod
    def format_history_for_llm_markdown(cls, history: list) -> str:
        """
        Format session history as Markdown for LLM context: per-turn headers with
        timestamp, event-type breakdown (user_query, query_clarification, query_rewriting,
        answer). intent_classification is excluded.
        """
        turns = cls._group_history_into_turns(history)
        if not turns:
            return ""
        lines: List[str] = ["## Historical Conversation", ""]
        for idx, turn in enumerate(turns, start=1):
            ts = (turn.get("ts_utc") or "").strip() or "—"
            lines.append(f"### Turn{idx}: happened at {ts}")
            lines.append("")
            ev = turn.get("events") or {}
            uq = (ev.get("user_query") or "").strip() or "—"
            qc = (ev.get("query_clarification") or "").strip() or "—"
            qr = (ev.get("query_rewriting") or "").strip() or "—"
            ans = (ev.get("answer") or "").strip() or "—"
            lines.append(f"- **user_query:** {uq}")
            lines.append(f"- **query_clarification:** {qc}")
            lines.append(f"- **query_rewriting:** {qr}")
            lines.append(f"- **answer:** {ans}")
            lines.append("")
            if idx < len(turns):
                lines.append("---")
                lines.append("")
        return "\n".join(lines).strip()


class MemoryEventWriter:
    """
    Memory event append operations.

    Best-effort write to Redis (+ optional ClickHouse). Never raises.
    """

    @classmethod
    def append_event(
        cls,
        memory: Optional[GatewayConversationMemory],
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        request_id: str,
        event_type: str,
        event_content: Any,
        status: str = "ok",
        note: str = "",
    ) -> None:
        """
        Append one memory event. No-op when memory disabled or user_id empty.

        Args:
            memory: Gateway memory instance. None when disabled.
            user_id: User identifier (required for write).
            session_id: Session identifier.
            request_id: Request identifier.
            event_type: Event type (e.g. user_query, turn_summary).
            event_content: Content dict or JSON string.
            status: Event status (ok, failed, skipped).
            note: Optional note.
        """
        if not memory or not user_id or not str(user_id).strip():
            return
        try:
            if isinstance(event_content, str):
                content = event_content
            else:
                content = json.dumps(event_content, ensure_ascii=False)
            memory.append_event(
                MemoryEvent(
                    user_id=str(user_id).strip(),
                    session_id=(session_id or "").strip(),
                    request_id=request_id,
                    event_type=event_type,  # type: ignore[arg-type]
                    event_content=content,
                    status=status,  # type: ignore[arg-type]
                    note=note,
                )
            )
            logger.debug("Memory event appended: event_type=%s request_id=%s", event_type, request_id)
        except Exception as exc:
            logger.warning("Gateway memory append_event failed (non-fatal): %s", exc)


class TurnSummaryPersistence:
    """
    Persist a completed turn: append turn_summary event and save_turn.

    Single place for the try/except so api.py does not repeat the block.
    """

    @classmethod
    def persist_turn(
        cls,
        memory: Optional[GatewayConversationMemory],
        *,
        user_id: Optional[str],
        session_id: str,
        request_id: str,
        query: str,
        answer: str,
        workflow: str,
    ) -> None:
        """
        Append turn_summary event and save_turn. No-op when memory or user_id missing.
        """
        if not memory or not user_id or not str(user_id).strip():
            return
        try:
            MemoryEventWriter.append_event(
                memory,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                event_type="turn_summary",
                event_content={"query": query, "answer": answer, "workflow": workflow},
                status="ok",
                note="v1 summary event",
            )
            memory.save_turn(
                session_id or "",
                query,
                answer,
                workflow,
                user_id=str(user_id).strip(),
            )
        except Exception as exc:
            logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
