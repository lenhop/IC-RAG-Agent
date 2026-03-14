"""
Route LLM clarification: detect ambiguous queries before rewriting.

Clarification workflow (方案一 统一 LLM):
  1. Skip: empty query, documentation/policy questions
  2. Single LLM: call clarification_detect_ambiguity.txt to decide needs_clarification
     - Covers: referent not in history, multiple referents, semantically vague,
       conflict with history, missing identifiers
     - General knowledge (FBA, ASIN, compliance) -> no clarification
  3. Fallback: when LLM returns needs_clarification=true but empty question,
     call clarification_generate_question.txt to generate one

Backend and all config from env (no defaults); missing values raise ValueError.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

from ...prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class ClarificationEnvValidator:
    """
    Validate and resolve clarification env parameters from os.environ.

    All methods raise ValueError if the env var is missing or invalid.
    No defaults; every required param must be explicitly set.
    """

    @staticmethod
    def get_timeout() -> int:
        """Read and validate GATEWAY_CLARIFICATION_TIMEOUT. Must be positive integer."""
        raw = (os.getenv("GATEWAY_CLARIFICATION_TIMEOUT") or "").strip()
        if not raw:
            logger.error("GATEWAY_CLARIFICATION_TIMEOUT is not set")
            raise ValueError("GATEWAY_CLARIFICATION_TIMEOUT must be set")
        try:
            val = int(raw)
            if val <= 0:
                logger.error("GATEWAY_CLARIFICATION_TIMEOUT must be positive, got %s", raw)
                raise ValueError("GATEWAY_CLARIFICATION_TIMEOUT must be positive")
            return val
        except ValueError as e:
            if "invalid literal" in str(e).lower():
                logger.error("GATEWAY_CLARIFICATION_TIMEOUT invalid integer: %s", raw, exc_info=True)
                raise ValueError("GATEWAY_CLARIFICATION_TIMEOUT must be a valid integer") from e
            raise

    @staticmethod
    def get_ollama_url() -> str:
        """Read and validate GATEWAY_CLARIFICATION_OLLAMA_URL."""
        value = (os.getenv("GATEWAY_CLARIFICATION_OLLAMA_URL") or "").strip()
        if not value:
            logger.error("GATEWAY_CLARIFICATION_OLLAMA_URL is not set")
            raise ValueError(
                "GATEWAY_CLARIFICATION_OLLAMA_URL must be set (e.g. http://localhost:11434/api/generate)"
            )
        return value

    @staticmethod
    def get_model() -> str:
        """Read and validate GATEWAY_CLARIFICATION_MODEL."""
        value = (os.getenv("GATEWAY_CLARIFICATION_MODEL") or "").strip()
        if not value:
            logger.error("GATEWAY_CLARIFICATION_MODEL is not set")
            raise ValueError("GATEWAY_CLARIFICATION_MODEL must be set")
        return value

    @staticmethod
    def get_deepseek_base_url() -> str:
        """Read and validate GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL."""
        value = (os.getenv("GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL") or "").strip()
        if not value:
            logger.error("GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL is not set")
            raise ValueError(
                "GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL must be set (e.g. https://api.deepseek.com)"
            )
        return value.rstrip("/")

    @staticmethod
    def get_deepseek_api_key() -> str:
        """Read and validate DEEPSEEK_API_KEY."""
        value = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not value:
            logger.error("DEEPSEEK_API_KEY is not set for DeepSeek backend")
            raise ValueError("DEEPSEEK_API_KEY must be set for DeepSeek backend")
        return value

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


def _is_concrete_documentation_query(query: str) -> bool:
    """
    [Step 1a] 文档/政策类问题跳过澄清。
    Return True for documentation, policy, compliance, requirements questions.
    These are self-contained conceptual questions and do NOT need clarification.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = [
        r"documentation\s+requirements",
        r"product\s+compliance",
        r"safety\s+documentation",
        r"policy\s+on",
        r"what\s+are\s+.*\s+requirements",
        r"what\s+does\s+amazon",
        r"guidelines",
        r"business\s+rules",
        r"compliance\s+and\s+safety",
        r"requirements\s+for",
    ]
    return any(re.search(p, q) for p in patterns)


def _is_out_of_scope_query(query: str) -> bool:
    """
    Return True when query is clearly outside Amazon seller gateway scope.

    This prevents unnecessary clarification on unrelated topics such as weather,
    jokes, and generic chit-chat.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = [
        r"\bweather\b",
        r"\btemperature\b",
        r"\brain\b",
        r"\bforecast\b",
        r"\bjoke\b",
        r"\bwho are you\b",
        r"\bwhat time is it\b",
        r"\bnews\b",
        r"\bmovie\b",
        r"\bmusic\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _should_use_conversation_history(query: str) -> bool:
    """
    Return True only when query likely depends on previous turns.

    We avoid injecting unrelated history for self-contained queries, which can
    cause the model to ask clarification questions about old topics.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    # Pronouns and continuation cues typically require context.
    markers = [
        r"\bit\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthose\b",
        r"\bthese\b",
        r"\bthe product\b",
        r"\bthe order\b",
        r"\bthe fee\b",
        r"\bthe issue\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bcontinue\b",
        r"\band\b",
        r"\balso\b",
    ]
    return any(re.search(p, q) for p in markers)


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
        """Call Ollama for ambiguity detection."""
        url = ClarificationEnvValidator.get_ollama_url()
        mdl = ClarificationEnvValidator.get_model()
        timeout = ClarificationEnvValidator.get_timeout()
        logger.debug("Ollama check_ambiguity: url=%s model=%s timeout=%s", url, mdl, timeout)
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        payload = {
            "model": mdl,
            "prompt": f"{cls.CLARIFICATION_PROMPT}{user_input}",
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
            logger.error("Ollama clarification check failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama clarification check failed: {exc}") from exc
        if resp.status_code != 200:
            logger.error("Ollama clarification HTTP %s", resp.status_code)
            raise RuntimeError(f"Ollama clarification HTTP {resp.status_code}")
        try:
            data = resp.json()
            return (data.get("response") or "").strip()
        except (ValueError, TypeError) as exc:
            logger.error("Ollama clarification invalid response: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama clarification invalid response: {exc}") from exc

    @classmethod
    def _call_deepseek_check_ambiguity(cls,
        query: str, conversation_context: Optional[str] = None
    ) -> str:
        """Call DeepSeek for ambiguity detection."""
        api_key = ClarificationEnvValidator.get_deepseek_api_key()
        mdl = ClarificationEnvValidator.get_model()
        timeout = ClarificationEnvValidator.get_timeout()
        base_url = ClarificationEnvValidator.get_deepseek_base_url()
        logger.debug("DeepSeek check_ambiguity: base_url=%s model=%s timeout=%s", base_url, mdl, timeout)
        try:
            from openai import OpenAI
        except ImportError as exc:
            logger.error("openai client not installed; cannot call DeepSeek clarification: %s", exc)
            raise RuntimeError("openai client not installed; cannot call DeepSeek clarification") from exc
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        messages = [
            {"role": "system", "content": cls.CLARIFICATION_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            response = client.chat.completions.create(
                model=mdl,
                messages=messages,
                max_tokens=256,
                temperature=0.3,
            )
            choice = response.choices[0] if response.choices else None
            if choice and choice.message and choice.message.content:
                return (choice.message.content or "").strip()
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
        """Call Ollama to generate clarification question."""
        url = ClarificationEnvValidator.get_ollama_url()
        mdl = ClarificationEnvValidator.get_model()
        timeout = ClarificationEnvValidator.get_timeout()
        logger.debug("Ollama generate_question: url=%s model=%s", url, mdl)
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        payload = {
            "model": mdl,
            "prompt": f"{cls.GENERATE_QUESTION_PROMPT}{user_input}",
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
            logger.error("Ollama generate-question failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama generate-question failed: {exc}") from exc
        if resp.status_code != 200:
            logger.error("Ollama generate-question HTTP %s", resp.status_code)
            raise RuntimeError(f"Ollama generate-question HTTP {resp.status_code}")
        try:
            data = resp.json()
            text = (data.get("response") or "").strip()
        except (ValueError, TypeError) as exc:
            logger.error("Ollama generate-question invalid response: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama generate-question invalid response: {exc}") from exc
        return cls._parse_clarification_question(text)

    @classmethod
    def _call_deepseek_generate_question(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> Optional[str]:
        """Call DeepSeek to generate clarification question."""
        api_key = ClarificationEnvValidator.get_deepseek_api_key()
        mdl = ClarificationEnvValidator.get_model()
        timeout = ClarificationEnvValidator.get_timeout()
        base_url = ClarificationEnvValidator.get_deepseek_base_url()
        logger.debug("DeepSeek generate_question: base_url=%s model=%s", base_url, mdl)
        try:
            from openai import OpenAI
        except ImportError as exc:
            logger.error("openai client not installed; cannot call DeepSeek generate-question: %s", exc)
            raise RuntimeError(
                "openai client not installed; cannot call DeepSeek generate-question"
            ) from exc
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        user_input = QueryAndResponseProcessor.build_user_input(query, conversation_context)
        messages = [
            {"role": "system", "content": cls.GENERATE_QUESTION_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            response = client.chat.completions.create(
                model=mdl,
                messages=messages,
                max_tokens=256,
                temperature=0.3,
            )
            choice = response.choices[0] if response.choices else None
            if choice and choice.message and choice.message.content:
                text = (choice.message.content or "").strip()
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
      2. 调用 LLM (clarification_detect_ambiguity.txt) 做歧义检测
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
        ValueError: If required env vars are not set (GATEWAY_CLARIFICATION_BACKEND,
            GATEWAY_CLARIFICATION_TIMEOUT, GATEWAY_CLARIFICATION_MODEL,
            GATEWAY_CLARIFICATION_OLLAMA_URL, and when backend=deepseek:
            GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY).
        RuntimeError: On LLM/network failures (connection, timeout, HTTP error,
            invalid response, missing openai client).
    """
    # [Step 1] 空查询直接返回
    if not query or not query.strip():
        logger.debug("check_ambiguity: empty query, skip")
        return {"needs_clarification": False, "clarification_backend": None}

    # [Step 1a] 文档/政策/合规类问题跳过澄清
    if _is_concrete_documentation_query(query):
        logger.debug("check_ambiguity: documentation query, skip")
        return {"needs_clarification": False, "clarification_backend": None}

    # [Step 1b] 非业务范围问题跳过澄清
    if _is_out_of_scope_query(query):
        logger.debug("check_ambiguity: out-of-scope query, skip")
        return {"needs_clarification": False, "clarification_backend": None}

    logger.info("check_ambiguity: calling LLM for query_len=%d", len(query.strip()))
    text, used_backend = _ClarificationLLM.call(query.strip(), conversation_context)

    if not text or not text.strip():
        logger.warning("check_ambiguity: LLM returned empty, backend=%s", used_backend)
        return {"needs_clarification": False, "clarification_backend": used_backend}

    # [Step 3] 解析 LLM 返回的 JSON
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


@dataclass
class ClarificationCheckResult:
    """Normalized clarification check output used by API endpoints."""

    needs_clarification: bool
    clarification_question: Optional[str]
    backend: str
    conversation_context: Optional[str]


class ClarificationService:
    """Encapsulates clarification config, context loading, and ambiguity check."""

    def is_enabled(self) -> bool:
        """Return True when clarification check is enabled."""
        value = os.getenv("GATEWAY_CLARIFICATION_ENABLED", "true").strip().lower()
        return value not in ("0", "false", "no", "off")

    def resolve_backend(self) -> str:
        """Resolve and validate clarification backend from env."""
        return ClarificationEnvValidator.get_backend()

    def resolve_memory_rounds(self) -> int:
        """Resolve rounds used for clarification context (hard cap: 3)."""
        try:
            configured = int(os.getenv("GATEWAY_CLARIFICATION_MEMORY_ROUNDS", "3"))
            return min(3, max(1, configured))
        except (ValueError, TypeError):
            return 3

    def get_conversation_context(
        self,
        memory: Any,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[str]:
        """Build clarification context from short-term memory."""
        if not memory:
            return None

        clarification_rounds = self.resolve_memory_rounds()
        default_rounds = 3
        history: list[dict[str, Any]] = []

        # Always fetch the larger window; trim later if no clarification found.
        if user_id and str(user_id).strip():
            history = memory.get_history_by_user(str(user_id).strip(), last_n=clarification_rounds)
        elif session_id and str(session_id).strip():
            history = memory.get_history(str(session_id).strip(), last_n=clarification_rounds)

        if not history:
            return None

        has_clarification = any(
            (turn.get("workflow") or "").strip().lower() == "clarification"
            for turn in history
        )
        effective_rounds = clarification_rounds if has_clarification else default_rounds
        if len(history) > effective_rounds:
            history = history[-effective_rounds:]

        lines: list[str] = []
        for idx, turn in enumerate(history, start=1):
            query = (turn.get("query") or "").strip()
            answer = (turn.get("answer") or "").strip()
            if not query:
                continue
            lines.append(f'Turn {idx}: User asked "{query}" -> Answer: "{answer}"')
        return "\n".join(lines) if lines else None

    def check(
        self,
        query: str,
        memory: Any,
        user_id: Optional[str],
        session_id: Optional[str],
        ambiguity_checker: Callable[..., dict[str, Any]] = check_ambiguity,
    ) -> ClarificationCheckResult:
        """Run end-to-end clarification check with unified output."""
        backend = self.resolve_backend()
        context = None
        if _should_use_conversation_history(query):
            context = self.get_conversation_context(memory, user_id, session_id)
        ambiguity_result = ambiguity_checker(query, conversation_context=context)

        question = ambiguity_result.get("clarification_question")
        if not isinstance(question, str):
            question = None

        return ClarificationCheckResult(
            needs_clarification=bool(ambiguity_result.get("needs_clarification")),
            clarification_question=question.strip() if question else None,
            backend=backend,
            conversation_context=context,
        )
