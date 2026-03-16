"""
Route LLM clarification: detect ambiguous queries before rewriting.

Clarification workflow (方案一 统一 LLM):
  1. Skip: empty query, documentation/policy questions
  2. Single LLM: call clarification_detect_ambiguity prompt to decide needs_clarification
     - Covers: referent not in history, multiple referents, semantically vague,
       conflict with history, missing identifiers
     - General knowledge (FBA, ASIN, compliance) -> no clarification
  3. Fallback: when LLM returns needs_clarification=true but empty question,
     call clarification_generate_question prompt to generate one

Backend and all config from env (no defaults); missing values raise ValueError.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ...message import ConversationHistoryHandler
from ...prompt_loader import load_prompt
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient

logger = logging.getLogger(__name__)


class ClarificationEnvValidator:
    """
    Validate and resolve clarification env parameters from os.environ.

    All methods raise ValueError if the env var is missing or invalid.
    No defaults; every required param must be explicitly set.
    """

    @staticmethod
    def get_backend() -> str:
        """Read and validate GATEWAY_CLARIFICATION_BACKEND. Must be 'ollama' or 'deepseek'."""
        value = (os.getenv("GATEWAY_CLARIFICATION_BACKEND") or "").strip().lower()
        if not value:
            logger.error("GATEWAY_CLARIFICATION_BACKEND is not set")
            raise ValueError("GATEWAY_CLARIFICATION_BACKEND must be set")
        if value not in ("ollama", "deepseek"):
            logger.error("GATEWAY_CLARIFICATION_BACKEND unknown: %s", value)
            raise ValueError(f"Unknown backend {value}; must be 'ollama' or 'deepseek'")
        return value


class QueryAndResponseProcessor:
    """
    Utilities for query preparation and response parsing.

    - build_user_input: build LLM input from query and optional conversation context
    - strip_markdown_fences: remove ``` code block markers from LLM output
    - extract_first_json_object: extract first {...} JSON object from text
    """

    @staticmethod
    def build_user_input(query: str, conversation_context: Optional[str] = None) -> str:
        """
        Build LLM input with strict structural separation.

        The history block is marked as reference-only. The current query block
        is explicitly marked as the only query to analyze/clarify.
        """
        history_block = (
            conversation_context.strip()
            if conversation_context and conversation_context.strip()
            else "(none)"
        )
        current_query = (query or "").strip()
        return (
            "# CONVERSATION HISTORY (ONLY for pronoun reference)\n"
            f"{history_block}\n\n"
            "# CURRENT USER QUERY (ONLY THIS ONE MATTERS)\n"
            f"{current_query}\n"
        )

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Remove ``` code block markers from LLM output."""
        raw = (text or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```") and lines[-1].strip() == "```":
                    return "\n".join(lines[1:-1]).strip()
        return raw

    @staticmethod
    def extract_first_json_object(text: str) -> Optional[str]:
        """Extract first complete {...} JSON object from text."""
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None


class _ClarificationLLM:
    """
    Unified LLM caller for clarification: ambiguity detection and question generation.
    Supports DeepSeek (remote) and Ollama (local) backends.
    """
    CLARIFICATION_PROMPT = load_prompt("clarification/clarification_detect_ambiguity")
    GENERATE_QUESTION_PROMPT = load_prompt("clarification/clarification_generate_question")

    @classmethod
    def call(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Call LLM for ambiguity detection. Dispatches to DeepSeek or Ollama.
        Backend is read from GATEWAY_CLARIFICATION_BACKEND env (required).
        Returns (raw_response_text, used_backend). used_backend is None when no response.
        """
        effective = ClarificationEnvValidator.get_backend()
        logger.info("Clarification check: backend=%s query_len=%d", effective, len(query or ""))
        if effective == "deepseek":
            text = cls._call_deepseek_check_ambiguity(query, conversation_context)
            if text and text.strip():
                logger.debug("DeepSeek check_ambiguity returned %d chars", len(text))
                return text, "deepseek"
            logger.error("DeepSeek clarification returned empty response")
            raise ValueError("DeepSeek clarification failed")
        elif effective == "ollama":
            text = cls._call_ollama_check_ambiguity(query, conversation_context)
            if text and text.strip():
                logger.debug("Ollama check_ambiguity returned %d chars", len(text))
                return text, "ollama"
            logger.error("Ollama clarification returned empty response")
            raise ValueError("Ollama clarification failed")
        else:
            logger.error("Unknown backend: %s", effective)
            raise ValueError(f"Unknown backend {effective}; must be 'ollama' or 'deepseek'")

    @classmethod
    def _call_ollama_check_ambiguity(cls,
        query: str, conversation_context: Optional[str] = None
    ) -> str:
        """Call Ollama for ambiguity detection via OllamaClient (OLLAMA_* env only)."""
        logger.debug("Ollama check_ambiguity via OllamaClient")
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        prompt = f"{cls.CLARIFICATION_PROMPT}{user_input}"
        return OllamaClient().generate(prompt, empty_fallback="")

    @classmethod
    def _call_deepseek_check_ambiguity(cls,
        query: str, conversation_context: Optional[str] = None
    ) -> str:
        """Call DeepSeek for ambiguity detection via unified DeepSeekChat."""
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        logger.debug("DeepSeek check_ambiguity via DeepSeekChat (stage=clarification)")
        try:
            return DeepSeekChat().complete(
                cls.CLARIFICATION_PROMPT,
                user_input,
            )
        except Exception as exc:
            logger.error("DeepSeek clarification failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek clarification failed: {exc}") from exc

    @classmethod
    def generate_question(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate clarification question when ambiguity LLM returns empty.
        Backend is read from GATEWAY_CLARIFICATION_BACKEND env (required).
        Dispatches to DeepSeek or Ollama. Returns question string or None.
        """
        effective = ClarificationEnvValidator.get_backend()
        logger.info("Generate clarification question: backend=%s", effective)
        if effective == "deepseek":
            text = cls._call_deepseek_generate_question(query, conversation_context)
            if text:
                logger.debug("DeepSeek generate_question returned %d chars", len(text))
                return text
            else :
                logger.warning("DeepSeek generate_question returned empty, falling back to Ollama")
                raise ValueError("DeepSeek generate_question failed")
        elif effective == "ollama":
            text = cls._call_ollama_generate_question(query, conversation_context)
            if text:
                logger.debug("Ollama generate_question returned %d chars", len(text))
                return text
            else:
                logger.error("Ollama generate_question returned empty")
                raise ValueError("Ollama generate_question failed")
        else:
            logger.error("Unknown backend: %s", effective)
            raise ValueError(f"Unknown backend {effective}; must be 'ollama' or 'deepseek'")

    @classmethod
    def _call_ollama_generate_question(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> Optional[str]:
        """Call Ollama to generate clarification question via OllamaClient (OLLAMA_* env only)."""
        logger.debug("Ollama generate_question via OllamaClient")
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        prompt = f"{cls.GENERATE_QUESTION_PROMPT}{user_input}"
        text = OllamaClient().generate(prompt, empty_fallback="")
        return cls._parse_clarification_question(text)

    @classmethod
    def _call_deepseek_generate_question(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> Optional[str]:
        """Call DeepSeek to generate clarification question via DeepSeekChat."""
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        try:
            text = DeepSeekChat().complete(
                cls.GENERATE_QUESTION_PROMPT,
                user_input,
            )
            return cls._parse_clarification_question(text)
        except Exception as exc:
            logger.error("DeepSeek generate-question failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek generate-question failed: {exc}") from exc

    @classmethod
    def _parse_clarification_question(cls, text: Optional[str]) -> Optional[str]:
        """Parse clarification_question from LLM JSON output."""
        if not text or not text.strip():
            logger.debug("_parse_clarification_question: empty input")
            return None
        raw = QueryAndResponseProcessor.strip_markdown_fences(text)
        candidate = QueryAndResponseProcessor.extract_first_json_object(raw)
        if not candidate:
            logger.warning("_parse_clarification_question: no JSON object found in response")
            return None
        try:
            parsed = json.loads(candidate)
            q = parsed.get("clarification_question")
            if isinstance(q, str) and q.strip():
                return q.strip()
            logger.warning("_parse_clarification_question: missing or empty clarification_question")
        except (ValueError, TypeError) as exc:
            logger.warning("_parse_clarification_question: JSON parse failed: %s", exc)
        return None


def check_ambiguity(
    query: str,
    conversation_context: Optional[str] = None,
) -> dict:
    """
    [入口] 澄清流程主入口。判断用户查询是否需澄清。

    Flow (方案一 统一 LLM):
      1. 空查询/文档类 -> 直接返回 needs_clarification=False
      2. 调用 LLM (clarification_detect_ambiguity) 做歧义检测
      3. 若 needs_clarification=true 且 question 为空，用 clarification_generate_question 生成

    Args:
        query: Raw user query text.
        conversation_context: Optional formatted conversation history (last 3 rounds)
            from Redis. When provided, the LLM considers prior turns so it won't
            ask for info the user already supplied in recent conversation.

    Returns:
        {"needs_clarification": True, "clarification_question": "..."} when ambiguous,
        or {"needs_clarification": False} when clear. On LLM failure, returns
        {"needs_clarification": False} to allow normal flow to proceed.

    Raises:
        ValueError: If required env vars are not set. Ollama backend needs
            OLLAMA_BASE_URL, OLLAMA_GENERATE_MODEL, OLLAMA_REQUEST_TIMEOUT,
            OLLAMA_EMBED_MODEL. DeepSeek needs DEEPSEEK_API_KEY (and base URL).
        RuntimeError: On LLM/network failures (connection, timeout, HTTP error,
            invalid response, missing openai client).
    """
    # [Step 1] 空查询直接返回
    if not query or not query.strip():
        logger.debug("check_ambiguity: empty query, skip")
        return {"needs_clarification": False, "clarification_backend": None}

    logger.info("check_ambiguity: calling LLM for query_len=%d", len(query.strip()))
    text, used_backend = _ClarificationLLM.call(query.strip(), conversation_context)

    if not text or not text.strip():
        logger.warning("check_ambiguity: LLM returned empty, backend=%s", used_backend)
        return {"needs_clarification": False, "clarification_backend": used_backend}

    # [Step 2] 解析 LLM 返回的 JSON
    raw = QueryAndResponseProcessor.strip_markdown_fences(text)
    parsed = None
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.debug("check_ambiguity: direct JSON parse failed, extracting object")
        candidate = QueryAndResponseProcessor.extract_first_json_object(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except ValueError:
                parsed = None

    # --- Annotated: Parsing & Validating LLM Ambiguity Output ---
    # Check that the response from the LLM is a dict; return default if not.
    if not isinstance(parsed, dict):
        logger.warning("check_ambiguity: invalid or non-dict LLM response, backend=%s", used_backend)
        # Could not parse valid JSON object, treat as not needing clarification.
        return {"needs_clarification": False, "clarification_backend": used_backend}

    # Extract whether clarification is needed (must be truthy). If not present or false, skip clarification.
    needs = parsed.get("needs_clarification")
    if not needs:
        logger.debug("check_ambiguity: needs_clarification=false, backend=%s", used_backend)
        return {"needs_clarification": False, "clarification_backend": used_backend}

    # Try to get the clarification question. If missing or blank, call model to generate one.
    question = parsed.get("clarification_question")
    if not isinstance(question, str) or not question.strip():
        logger.info("check_ambiguity: needs_clarification=true but empty question, generating")
        question = _ClarificationLLM.generate_question(query.strip(), conversation_context)
    # If still blank, fallback to a generic prompt.
    if not question or not question.strip():
        logger.warning("check_ambiguity: generate_question returned empty, using generic")
        question = "Could you please provide more details?"

    # All checks passed; log and return structured response for frontend/services.
    logger.info("check_ambiguity: needs_clarification=true backend=%s", used_backend)
    return {
        "needs_clarification": True,
        "clarification_question": question.strip(),
        "clarification_backend": used_backend,
    }


# Default number of conversation rounds to load for clarification context.
_CLARIFICATION_MEMORY_ROUNDS = 3


class ClarificationService:
    """
    Service that runs clarification check using session history from message.py.

    Loads history via ConversationHistoryHandler.get_session_history and
    format_history_for_llm_markdown, then passes formatted context to the ambiguity checker.
    """

    @staticmethod
    def is_enabled() -> bool:
        """Return True when clarification is enabled via env."""
        value = (os.getenv("GATEWAY_CLARIFICATION_ENABLED") or "").strip().lower()
        return value in ("1", "true", "yes", "on")

    @staticmethod
    def check(
        query: str,
        memory: Any,
        user_id: Optional[str],
        session_id: Optional[str],
        ambiguity_checker: Callable[..., dict],
    ) -> "ClarificationCheckResult":
        """
        Run clarification: load session history from message.py, format, then call checker.

        Returns ClarificationCheckResult with conversation_context set from
        message.py so api.py and dispatcher receive the same context.
        """
        conversation_context: Optional[str] = None
        sid = (session_id or "").strip()
        if sid and memory:
            try:
                n = int(os.getenv("GATEWAY_CLARIFICATION_MEMORY_ROUNDS", str(_CLARIFICATION_MEMORY_ROUNDS)))
            except (TypeError, ValueError):
                n = _CLARIFICATION_MEMORY_ROUNDS
            last_n = min(max(n, 1), 50)
            res = ConversationHistoryHandler.get_session_history(memory, sid, last_n=last_n)
            history = res.get("history") or []
            if history:
                conversation_context = ConversationHistoryHandler.format_history_for_llm_markdown(history)
        raw = ambiguity_checker(query, conversation_context=conversation_context)
        needs = bool(raw.get("needs_clarification"))
        question = raw.get("clarification_question")
        if isinstance(question, str):
            question = question.strip() or None
        backend = raw.get("clarification_backend") or "unknown"
        return ClarificationCheckResult(
            needs_clarification=needs,
            clarification_question=question,
            backend=backend,
            conversation_context=conversation_context,
        )


@dataclass
class ClarificationCheckResult:
    """Normalized clarification check output used by API endpoints."""

    needs_clarification: bool
    clarification_question: Optional[str]
    backend: str
    conversation_context: Optional[str]


