"""
Unified rewrite implementation: one LLM call returns JSON with rewrite display + intent clauses.

Merges former plain-text rewrite, intent_split, and combined_rewrite_split into a single prompt
(`rewriting/rewrite_prompt.md`). Public entry points for the facade: `run_unified_rewrite`,
`split_intents` (standalone), `_RewriteRouter`, `RouterEnvConfig`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...message import ConversationHistoryHandler
from ...prompt_loader import load_prompt
from ...schemas import QueryRequest
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from src.llm.chat_backend_policy import resolve_chat_backend
from src.logger import get_logger_facade
from src.retrieval.query_process import QueryProcessor, normalize_query

logger = logging.getLogger(__name__)

_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None

# System instruction for JSON-only output (DeepSeek).
_JSON_SYSTEM_PROMPT = (
    "You rewrite and split user queries. Output ONLY valid JSON with an intents array. "
    "No markdown fences or explanation."
)


@dataclass(frozen=True)
class UnifiedRewriteResult:
    """Outcome of the unified rewrite+split stage (single LLM)."""

    rewritten_display: str
    intents: List[str]
    memory_rounds_used: int
    memory_text_length: int


class RouterEnvConfig:
    """Env-backed settings for rewrite memory depth and default backend."""

    @staticmethod
    def get_memory_rounds() -> int:
        """GATEWAY_REWRITE_MEMORY_ROUNDS (default 3)."""
        try:
            return max(1, int(os.getenv("GATEWAY_REWRITE_MEMORY_ROUNDS", "3")))
        except (ValueError, TypeError):
            logger.warning("GATEWAY_REWRITE_MEMORY_ROUNDS invalid; using 3")
            return 3

    @staticmethod
    def get_rewrite_backend(request_backend: Optional[str]) -> str:
        """Prefer request rewrite_backend, else env chain (see chat_backend_policy)."""
        try:
            return resolve_chat_backend(
                "rewrite",
                request_override=request_backend,
            )
        except Exception as exc:
            logger.warning("get_rewrite_backend failed: %s; using deepseek", exc)
            return "deepseek"


class JsonRewriteParser:
    """Parse and normalize JSON output from the unified rewrite LLM."""

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Remove optional ```json ... ``` wrapper."""
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return raw

    @staticmethod
    def parse_json_object(raw: str) -> Optional[dict]:
        """Parse a JSON object; tolerate leading/trailing noise."""
        try:
            return json.loads(raw.strip())
        except ValueError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except ValueError:
                    return None
            return None

    @staticmethod
    def dedupe_intents(intents: List[object]) -> List[str]:
        """Case-insensitive dedupe while preserving first-seen casing."""
        seen: set[str] = set()
        result: List[str] = []
        for item in intents:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            lowered = cleaned.lower()
            if cleaned and lowered not in seen:
                seen.add(lowered)
                result.append(cleaned)
        return result

    @classmethod
    def display_and_intents_from_parsed(cls, parsed: dict) -> Tuple[str, List[str]]:
        """
        Extract rewritten_display and intents from parsed JSON.

        Raises:
            ValueError: If intents missing or empty after dedupe.
        """
        intents_raw = parsed.get("intents")
        if not isinstance(intents_raw, list) or not intents_raw:
            raise ValueError("missing or empty intents")
        intents = cls.dedupe_intents(intents_raw)
        if not intents:
            raise ValueError("dedupe produced empty intents")
        display_val = parsed.get("rewritten_display")
        if isinstance(display_val, str) and display_val.strip():
            rewritten_display = display_val.strip()
        else:
            rewritten_display = "; ".join(intents) if len(intents) > 1 else intents[0]
        return rewritten_display, intents


def render_unified_prompt(history: str, query: str) -> str:
    """Fill {history} and {query} in rewriting/rewrite_prompt.md."""
    template = load_prompt("rewriting/rewrite_prompt")
    history_text = (history or "").strip() or "(no conversation history)"
    return (
        template.replace("{history}", history_text).replace("{query}", (query or "").strip())
    )


def call_unified_rewrite_llm(
    rendered_prompt: str,
    backend: str,
    model: Optional[str] = None,
) -> str:
    """
    Invoke Ollama or DeepSeek for unified JSON rewrite+split.

    Returns:
        Raw model text (expected JSON).

    Raises:
        RuntimeError: On DeepSeek failure.
        Exception: Propagated from Ollama if generate fails critically.
    """
    if backend == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
        try:
            out = DeepSeekChat().complete(
                _JSON_SYSTEM_PROMPT,
                rendered_prompt,
                model_override=model,
                max_tokens=768,
            )
            return out if out else ""
        except Exception as exc:
            logger.error("DeepSeek unified rewrite failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek unified rewrite failed: {exc}") from exc

    if backend != "ollama":
        logger.error("Unknown rewrite backend %s; falling back to ollama", backend)
    return OllamaClient().generate(rendered_prompt, model=model, empty_fallback="")


def _fallback_result(
    normalized_query: str,
    memory_rounds_used: int,
    memory_text_length: int,
) -> UnifiedRewriteResult:
    """When LLM fails or JSON is invalid: single intent = normalized query."""
    stripped = (normalized_query or "").strip()
    if not stripped:
        return UnifiedRewriteResult("", [], memory_rounds_used, memory_text_length)
    return UnifiedRewriteResult(stripped, [stripped], memory_rounds_used, memory_text_length)


class _RewriteRouter:
    """Build merged session context and run unified rewrite; optional workflow routing."""

    @classmethod
    def build_merged_context_for_rewrite(
        cls,
        request: QueryRequest,
        gateway_memory: Optional[Any] = None,
        conversation_context: Optional[str] = None,
    ) -> Tuple[str, Optional[str], int, int]:
        """
        Normalize query and merge clarification + Redis history (no LLM).

        Returns:
            (normalized_query, merged_context, memory_rounds_used, memory_text_length).

        Raises:
            ValueError: If session_id is missing (required for memory-backed rewrite).
        """
        normalized = normalize_query(request.query or "")
        if not normalized or not normalized.strip():
            return ("", None, 0, 0)

        memory_rounds_used = 0
        memory_text_length = 0
        memory_context: Optional[str] = None
        last_n = RouterEnvConfig.get_memory_rounds()
        sid = (request.session_id or "").strip()
        if not sid:
            logger.error("build_merged_context_for_rewrite: session_id is required")
            raise ValueError("session_id is required for rewrite_query")

        res = ConversationHistoryHandler.get_session_history(
            gateway_memory, sid, last_n=last_n
        )
        history = res.get("history", [])
        if history:
            memory_context = ConversationHistoryHandler.format_history_for_llm_markdown(history)
            memory_rounds_used = len(history)
            logger.debug(
                "build_merged_context_for_rewrite: loaded %d memory rounds",
                memory_rounds_used,
            )

        merged = ConversationHistoryHandler.merge_context_strings(
            conversation_context, memory_context
        )
        if merged and merged.strip():
            memory_text_length = len(merged)
            if memory_rounds_used <= 0:
                memory_rounds_used = ConversationHistoryHandler.count_context_rounds(merged)

        return (normalized, merged, memory_rounds_used, memory_text_length)

    @classmethod
    def run_unified_rewrite(
        cls,
        request: QueryRequest,
        gateway_memory: Optional[Any] = None,
        conversation_context: Optional[str] = None,
    ) -> UnifiedRewriteResult:
        """
        Full unified rewrite+split: merged context + one JSON LLM call.

        On LLM/parse failure, returns fallback single-intent result (non-fatal).
        """
        normalized = normalize_query(request.query or "")
        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="router_rewrite_start",
                    stage="router_rewrite",
                    message="unified rewrite started",
                    status="started",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    workflow=request.workflow,
                    query_raw=request.query or "",
                    query_rewritten=normalized,
                )
            except Exception:
                pass

        if not normalized or not normalized.strip():
            logger.debug("run_unified_rewrite: empty query, early exit")
            return UnifiedRewriteResult("", [], 0, 0)

        try:
            normalized2, merged_context, memory_rounds_used, memory_text_length = (
                cls.build_merged_context_for_rewrite(
                    request,
                    gateway_memory=gateway_memory,
                    conversation_context=conversation_context,
                )
            )
        except ValueError:
            raise

        backend = RouterEnvConfig.get_rewrite_backend(request.rewrite_backend)
        logger.info("run_unified_rewrite: backend=%s session=%s", backend, request.session_id)

        history_text = (merged_context or "").strip() or "(no conversation history)"
        rendered = render_unified_prompt(history_text, normalized2)

        try:
            text = call_unified_rewrite_llm(rendered, backend)
        except Exception as exc:
            logger.warning("Unified rewrite LLM failed; fallback to normalized: %s", exc)
            fb = _fallback_result(normalized2, memory_rounds_used, memory_text_length)
            cls._log_done(request, fb, backend, memory_rounds_used, memory_text_length)
            return fb

        if not text or not text.strip():
            logger.warning("Unified rewrite empty output; fallback to normalized")
            fb = _fallback_result(normalized2, memory_rounds_used, memory_text_length)
            cls._log_done(request, fb, backend, memory_rounds_used, memory_text_length)
            return fb

        raw = JsonRewriteParser.strip_markdown_fences(text)
        parsed = JsonRewriteParser.parse_json_object(raw)
        if parsed is None:
            logger.warning("Unified rewrite JSON parse failed; fallback to normalized")
            fb = _fallback_result(normalized2, memory_rounds_used, memory_text_length)
            cls._log_done(request, fb, backend, memory_rounds_used, memory_text_length)
            return fb

        try:
            display, intents = JsonRewriteParser.display_and_intents_from_parsed(parsed)
        except ValueError as exc:
            logger.warning("Unified rewrite invalid intents: %s; fallback", exc)
            fb = _fallback_result(normalized2, memory_rounds_used, memory_text_length)
            cls._log_done(request, fb, backend, memory_rounds_used, memory_text_length)
            return fb

        display_norm = QueryProcessor.normalize(display)
        result = UnifiedRewriteResult(
            display_norm,
            intents,
            memory_rounds_used,
            memory_text_length,
        )
        cls._log_done(request, result, backend, memory_rounds_used, memory_text_length)
        logger.debug("run_unified_rewrite: done, intents=%d", len(intents))
        return result

    @classmethod
    def _log_done(
        cls,
        request: QueryRequest,
        result: UnifiedRewriteResult,
        backend: str,
        memory_rounds_used: int,
        memory_text_length: int,
    ) -> None:
        if not _gateway_logger:
            return
        try:
            _gateway_logger.log_runtime(
                event_name="router_rewrite_done",
                stage="router_rewrite",
                message="unified rewrite completed",
                status="success",
                session_id=request.session_id,
                user_id=request.user_id,
                workflow=request.workflow,
                query_raw=request.query or "",
                query_rewritten=result.rewritten_display,
                intent_list=result.intents,
                latency_ms=None,
                metadata={
                    "backend": backend,
                    "memory_rounds_used": memory_rounds_used,
                    "memory_text_length": memory_text_length,
                    "intent_count": len(result.intents),
                },
            )
        except Exception:
            pass

    @classmethod
    def route_workflow(
        cls,
        query: str,
        request: QueryRequest,
    ) -> tuple[str, float, str, str | None, float | None]:
        """LLM workflow selection (unchanged from legacy rewriters)."""
        try:
            backend = resolve_chat_backend("route")
        except Exception as exc:
            logger.warning("route_workflow backend resolve failed: %s; using deepseek", exc)
            backend = "deepseek"

        from src.gateway import route_llm as route_pkg

        llm_wf, llm_conf = route_pkg.route_with_llm(query or "", backend)
        logger.debug("route_workflow: llm workflow=%s conf=%.2f", llm_wf, llm_conf)
        return llm_wf, llm_conf, "llm", backend, llm_conf


def split_intents(
    query: str,
    conversation_context: Optional[str] = None,
    request_backend: Optional[str] = None,
) -> List[str]:
    """
    Split a query string into intents using the same unified prompt and backend policy.

    Used by dispatcher when no full QueryRequest context is available (e.g. planner fallback).
    On failure returns a single-element list with the trimmed query.
    """
    if not query or not query.strip():
        return []

    backend = RouterEnvConfig.get_rewrite_backend(request_backend)
    history_text = (conversation_context or "").strip() or "(no conversation history)"
    rendered = render_unified_prompt(history_text, query.strip())

    try:
        text = call_unified_rewrite_llm(rendered, backend)
    except Exception as exc:
        logger.warning("split_intents LLM failed: %s; using single clause", exc)
        return [query.strip()]

    if not text or not text.strip():
        return [query.strip()]

    raw = JsonRewriteParser.strip_markdown_fences(text)
    parsed = JsonRewriteParser.parse_json_object(raw)
    if parsed is None:
        return [query.strip()]

    try:
        _, intents = JsonRewriteParser.display_and_intents_from_parsed(parsed)
    except ValueError:
        return [query.strip()]

    try:
        _log_split_standalone(query, intents)
    except Exception:
        pass
    return intents


def _log_split_standalone(query: str, result: List[str]) -> None:
    if not _gateway_logger:
        return
    _gateway_logger.log_runtime(
        event_name="intent_split_completed",
        stage="rewrite",
        message="split_intents completed",
        status="success",
        workflow="rewriting",
        query_raw=query,
        intent_list=result,
        metadata={"intent_count": len(result)},
    )


# Backward-compatible facade for code/tests expecting IntentSplitMethod.
class IntentSplitMethod:
    """Delegates to split_intents; exposes static helpers for tests."""

    @classmethod
    def split(cls, query: str, conversation_context: Optional[str] = None) -> List[str]:
        return split_intents(query, conversation_context)

    _strip_markdown_fences = staticmethod(JsonRewriteParser.strip_markdown_fences)
    _dedupe_intents = staticmethod(JsonRewriteParser.dedupe_intents)


class RewriteSplitMethod:
    """Parse helper for tests; unified path uses run_unified_rewrite in production."""

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[dict]:
        """Same as combined_rewrite_split tests expect."""
        return JsonRewriteParser.parse_json_object(raw)


__all__ = [
    "IntentSplitMethod",
    "JsonRewriteParser",
    "RewriteSplitMethod",
    "RouterEnvConfig",
    "UnifiedRewriteResult",
    "_RewriteRouter",
    "call_unified_rewrite_llm",
    "render_unified_prompt",
    "split_intents",
]
