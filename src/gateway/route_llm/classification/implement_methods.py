"""
Intent classification implementation methods.

Architecture:
  IntentSplitMethod — LLM-based multi-intent split (moved from classification._IntentSplitter).
  ClassificationImplementMethod — keyword then vector then LLM (serial short-circuit per framework).

Internal (not exported):
  Used by classification.py public API (split_intents, classify_intent, classify_intents_batch).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from ...prompt_loader import load_prompt
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from src.retrieval.keyword_retrieval import KeywordRetrieval
from src.retrieval.vector_retrieval import VectorRetrieval

from .classification import IntentResult, _LLMIntentDetector

logger = logging.getLogger(__name__)

_split_gateway_logger = None
try:
    from src.logger import get_logger_facade

    _split_gateway_logger = get_logger_facade()
except Exception:
    _split_gateway_logger = None


def _intent_split_backend() -> str:
    """LLM backend for intent split (ollama or deepseek), from env."""
    return (os.getenv("GATEWAY_INTENT_SPLIT_BACKEND") or "ollama").strip().lower()


def _use_keyword() -> bool:
    """Whether keyword classification is enabled (default: True)."""
    val = (os.getenv("GATEWAY_INTENT_USE_KEYWORD") or "true").strip().lower()
    return val in ("true", "1", "yes")


def _use_vector() -> bool:
    """Whether vector classification is enabled (default: True)."""
    val = (os.getenv("GATEWAY_INTENT_USE_VECTOR") or "true").strip().lower()
    return val in ("true", "1", "yes")


def _vector_threshold() -> float:
    """Minimum similarity score for vector match (default: 0.7)."""
    try:
        return float(os.getenv("GATEWAY_VECTOR_INTENT_THRESHOLD") or "0.7")
    except ValueError:
        return 0.7


class IntentSplitMethod:
    """
    Multi-intent splitter: uses LLM only; on failure returns a single-element list
    with the original query (no heuristic splitting).
    """

    @classmethod
    def split(cls, query: str, conversation_context: Optional[str] = None) -> List[str]:
        """
        Split rewritten user query into independent intent clauses.

        Args:
            query: Rewritten query text.
            conversation_context: Optional history for prompt {history} placeholder.

        Returns:
            List of intent strings; on LLM failure or invalid JSON, [query.strip()].
        """
        if not query or not query.strip():
            return []

        prompt_template = load_prompt("classification/intent_split_query")
        history_text = (conversation_context or "").strip() or "(no conversation history)"
        prompt = (
            prompt_template.replace("{history}", history_text).replace(
                "{rewritten_query}", query.strip()
            )
        )
        text = ""

        if _intent_split_backend() == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            try:
                text = DeepSeekChat().complete(
                    system_prompt="You are an intent splitter. Output ONLY valid JSON.",
                    user_content=prompt,
                    max_tokens=512,
                )
            except Exception as exc:
                logger.warning(
                    "Intent split DeepSeek failed: %s; returning original query", exc
                )
                return [query.strip()]
        else:
            try:
                text = OllamaClient().generate(prompt, empty_fallback="")
            except Exception as exc:
                logger.warning(
                    "Intent split LLM call failed: %s; returning original query", exc
                )
                return [query.strip()]

        if not text:
            return [query.strip()]

        raw = cls._strip_markdown_fences(text)
        parsed = cls._parse_json_response(raw)
        if parsed is None:
            return [query.strip()]

        intents = parsed.get("intents")
        if not isinstance(intents, list) or not intents:
            return [query.strip()]

        result = cls._dedupe_intents(intents)
        if not result:
            return [query.strip()]

        cls._log_split_result(query, result)
        return result

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return raw

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, object]]:
        try:
            return json.loads(raw)
        except ValueError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except ValueError:
                    logger.warning("Intent split JSON parse failed; heuristic fallback")
                    return None
            logger.warning("Intent split no JSON found; heuristic fallback")
            return None

    @staticmethod
    def _dedupe_intents(intents: List[object]) -> List[str]:
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

    @staticmethod
    def _log_split_result(query: str, result: List[str]) -> None:
        if not _split_gateway_logger:
            return
        try:
            _split_gateway_logger.log_runtime(
                event_name="intent_split_completed",
                stage="intent_classification",
                message="split_intents completed",
                status="success",
                workflow="intent_classification",
                query_raw=query,
                intent_list=result,
                metadata={"intent_count": len(result)},
            )
        except Exception:
            pass


class ClassificationImplementMethod:
    """
    Unified intent classification: keyword, then vector, then LLM (serial short-circuit).

    If keyword matches, vector and LLM are skipped. If vector matches, LLM is skipped.
    If all miss through LLM, workflow general (generic knowledge).
    """

    def keyword_classification_method(
        self, query: str, context: Optional[str]
    ) -> Optional[IntentResult]:
        """
        First layer: keyword/regex match. Calls KeywordRetrieval.detect_parallel().
        """
        if not _use_keyword():
            return None
        if not query or not query.strip():
            return None
        result = KeywordRetrieval().detect_parallel(query.strip())
        if result is None:
            return None
        return IntentResult(
            intent_name=result.intent_name,
            workflow=result.workflow,
            confidence=result.confidence,
            source="keyword",
        )

    def vector_classification_method(
        self, query: str, context: Optional[str]
    ) -> Optional[IntentResult]:
        """
        Second layer: Chroma vector similarity. Calls VectorRetrieval.retrieve().
        Returns IntentResult if top candidate score >= threshold.
        """
        if not _use_vector():
            return None
        if not query or not query.strip():
            return None
        try:
            candidates = VectorRetrieval().retrieve(
                query.strip(),
                top_k=1,
                score_threshold=_vector_threshold(),
            )
        except Exception:
            return None
        if not candidates:
            return None
        top = candidates[0]
        return IntentResult(
            intent_name=top.intent or "unknown",
            workflow=top.workflow or "general",
            confidence="medium",
            source="vector",
        )

    def llm_classification_method(
        self, query: str, context: Optional[str]
    ) -> IntentResult:
        """
        Third layer: Prompt + LLM. Reuses _LLMIntentDetector.detect().
        Returns general when sp_api/uds/amazon_docs all miss.
        """
        return _LLMIntentDetector.detect(query, conversation_context=context)

    def detect(
        self, query: str, conversation_context: Optional[str] = None
    ) -> IntentResult:
        """
        Serial short-circuit: keyword -> vector -> LLM.

        Stops at first hit; LLM only runs if keyword and vector do not match.
        """
        if not query or not query.strip():
            return IntentResult(
                intent_name="general",
                workflow="general",
                confidence="low",
                source="fallback",
            )

        stripped = query.strip()

        if _use_keyword():
            kw = self.keyword_classification_method(stripped, conversation_context)
            if kw is not None:
                return kw

        if _use_vector():
            vec = self.vector_classification_method(stripped, conversation_context)
            if vec is not None:
                return vec

        return self.llm_classification_method(stripped, conversation_context)
