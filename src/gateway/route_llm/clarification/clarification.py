"""
Route LLM clarification: detect ambiguous queries before rewriting.

Clarification workflow (unified single-prompt):
  1. Skip: empty query
  2. Single LLM call with clarification_prompt (detect + generate in one prompt)
     - Part 1: checklist-based ambiguity detection
     - Part 2: if ambiguous, generate one short clarification question
  3. Fallback: if LLM returns needs_clarification=true but empty question,
     use a generic fallback question

Backend for the clarification LLM call is resolved via ``chat_backend_policy`` (global default
``deepseek`` when unset). Other clarification settings may still require env as documented.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from ...keyword_regular_match import ClarificationLayer12Gate
from ...message import ConversationHistoryHandler
from ...prompt_loader import load_prompt
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from src.llm.chat_backend_policy import resolve_chat_backend

logger = logging.getLogger(__name__)


class ClarificationEnvValidator:
    """
    Validate and resolve clarification env parameters from os.environ.

    ``get_backend`` uses unified chat backend policy (no longer requires
    ``GATEWAY_CLARIFICATION_BACKEND`` to be set). Other helpers may still raise when invalid.
    """

    @staticmethod
    def is_enabled() -> bool:
        """Return True when clarification is enabled via GATEWAY_CLARIFICATION_ENABLED env."""
        value = (os.getenv("GATEWAY_CLARIFICATION_ENABLED") or "").strip().lower()
        return value in ("1", "true", "yes", "on")

    @staticmethod
    def get_backend() -> str:
        """
        Resolve clarification chat backend (ollama or deepseek).

        Precedence: GATEWAY_CLARIFICATION_BACKEND, then GATEWAY_CHAT_LLM_BACKEND, then deepseek.
        """
        try:
            return resolve_chat_backend("clarification")
        except Exception as exc:
            logger.warning("get_backend clarification resolve failed: %s; using deepseek", exc)
            return "deepseek"


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
    Unified LLM caller for clarification (single prompt: detect + generate).
    Supports DeepSeek (remote) and Ollama (local) backends.
    """
    CLARIFICATION_PROMPT = load_prompt("clarification/clarification_prompt")

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
        """Call Ollama for clarification via OllamaClient (OLLAMA_* env only)."""
        logger.debug("Ollama clarification via OllamaClient")
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        prompt = f"{cls.CLARIFICATION_PROMPT}{user_input}"
        return OllamaClient().generate(prompt, empty_fallback="")

    @classmethod
    def _call_deepseek_check_ambiguity(cls,
        query: str, conversation_context: Optional[str] = None
    ) -> str:
        """Call DeepSeek for clarification via unified DeepSeekChat."""
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        logger.debug("DeepSeek clarification via DeepSeekChat")
        try:
            return DeepSeekChat().complete(
                cls.CLARIFICATION_PROMPT,
                user_input,
            )
        except Exception as exc:
            logger.error("DeepSeek clarification failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek clarification failed: {exc}") from exc


def check_ambiguity(
    query: str,
    conversation_context: Optional[str] = None,
) -> dict:
    """
    [入口] 澄清流程主入口。判断用户查询是否需澄清。

    Flow:
      0. ``GATEWAY_CLARIFICATION_ENABLED=false`` -> 返回 ``clarification_path=disabled``，不调 LLM
      1. 空查询 -> 直接返回 needs_clarification=False
      2. ``ClarificationLayer12Gate``（``keyword_regular_match``）: L1/L2 规则可跳过 L3 时直接返回
      3. 否则调用 LLM (clarification_prompt) 歧义检测 + 澄清问题
      4. 若 needs_clarification=true 且 question 为空，用 generic fallback

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
    # [Step 0] 澄清总开关关闭：不加载规则、不调 LLM
    if not ClarificationEnvValidator.is_enabled():
        logger.debug("check_ambiguity: GATEWAY_CLARIFICATION_ENABLED=false")
        return {
            "needs_clarification": False,
            "clarification_backend": None,
            "clarification_path": "disabled",
        }

    # [Step 1] 空查询直接返回
    if not query or not query.strip():
        logger.debug("check_ambiguity: empty query, skip")
        return {"needs_clarification": False, "clarification_backend": None}

    qstrip = query.strip()
    has_hist = bool(conversation_context and conversation_context.strip())

    # [Step 1b] L1+L2 规则门（keyword_regular_match.ClarificationLayer12Gate）→ 可跳过 L3 LLM
    fast_path_result = ClarificationLayer12Gate.try_resolve_without_l3(
        qstrip,
        has_conversation_history=has_hist,
    )
    if fast_path_result is not None:
        return fast_path_result

    logger.info("check_ambiguity: calling LLM for query_len=%d", len(qstrip))
    text, used_backend = _ClarificationLLM.call(qstrip, conversation_context)

    if not text or not text.strip():
        logger.warning("check_ambiguity: LLM returned empty, backend=%s", used_backend)
        return {
            "needs_clarification": False,
            "clarification_backend": used_backend,
            "clarification_path": "l3_llm",
        }

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
        return {
            "needs_clarification": False,
            "clarification_backend": used_backend,
            "clarification_path": "l3_llm",
        }

    # Extract whether clarification is needed (must be truthy). If not present or false, skip clarification.
    needs = parsed.get("needs_clarification")
    if not needs:
        logger.debug("check_ambiguity: needs_clarification=false, backend=%s", used_backend)
        return {
            "needs_clarification": False,
            "clarification_backend": used_backend,
            "clarification_path": "l3_llm",
        }

    # Try to get the clarification question. If missing or blank, use generic fallback.
    question = parsed.get("clarification_question")
    if not isinstance(question, str) or not question.strip():
        logger.warning("check_ambiguity: needs_clarification=true but empty question, using generic")
        question = "Could you please provide more details?"

    # All checks passed; log and return structured response for frontend/services.
    logger.info("check_ambiguity: needs_clarification=true backend=%s", used_backend)
    return {
        "needs_clarification": True,
        "clarification_question": question.strip(),
        "clarification_backend": used_backend,
        "clarification_path": "l3_llm",
    }





