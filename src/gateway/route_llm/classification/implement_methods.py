"""
Intent classification implementation methods.

Architecture:
  IntentResult — classification result dataclass (re-exported via classification facade).
  ClassificationIntentVectorStore — cached VectorRetrieval + parallel LLM intent detection (layer 3).
  IntentSplitMethod — LLM multi-intent split.
  ClassificationImplementMethod — keyword then vector then LLM (serial short-circuit).

This module does not import classification.py (avoids circular dependency); classification
imports from here and re-exports public types and API wrappers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ...prompt_loader import load_prompt
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from src.retrieval.keyword_retrieval import KeywordRetrieval, LoadKeywordRule
from src.retrieval.vector_retrieval import VectorRetrieval

logger = logging.getLogger(__name__)

_split_gateway_logger = None
try:
    from src.logger import get_logger_facade

    _split_gateway_logger = get_logger_facade()
except Exception:
    _split_gateway_logger = None


# ---------------------------------------------------------------------------
# Intent result model (LLM parallel detection lives on ClassificationIntentVectorStore)
# ---------------------------------------------------------------------------

_amazon_examples_cache: str | None = None


def _load_amazon_intent_examples() -> str:
    """Load amazon_intents.csv beside this package; return markdown bullet list for prompts."""
    global _amazon_examples_cache
    if _amazon_examples_cache is not None:
        return _amazon_examples_cache

    csv_path = Path(__file__).resolve().parent / "amazon_intents.csv"
    try:
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        keywords = [line.strip() for line in lines[1:] if line.strip()]
        _amazon_examples_cache = "\n".join(f"- {kw}" for kw in keywords)
    except Exception as exc:
        logger.warning("Failed to load amazon_intents.csv: %s; using empty examples", exc)
        _amazon_examples_cache = "(no examples available)"

    return _amazon_examples_cache


@dataclass
class IntentResult:
    """Intent classification result (keyword / vector / LLM layers)."""

    intent_name: str
    workflow: str
    confidence: str = "medium"
    source: str = "llm"
    required_fields: List[str] = field(default_factory=list)
    clarification_template: str = ""
    intent_elapsed_ms: Optional[int] = None
    step_timings: Optional[List[Dict[str, Any]]] = None


def _intent_detect_backend() -> str:
    """LLM backend for parallel intent detection steps (ollama or deepseek)."""
    return (os.getenv("GATEWAY_INTENT_DETECT_BACKEND") or "ollama").strip().lower()


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


# ---------------------------------------------------------------------------
# Classification keyword rules — load classification_data once via LoadKeywordRule
# ---------------------------------------------------------------------------

_CLASSIFICATION_DATA_DIR = Path(__file__).resolve().parent / "classification_data"
# Project root: .../src/gateway/route_llm/classification/implement_methods.py -> parents[4]
_IC_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_INTENT_REGISTRY_CHROMA = _IC_PROJECT_ROOT / "data" / "chroma_db" / "intent_registry"


class ClassificationIntentVectorStore:
    """
    Intent-registry vector layer and parallel LLM intent detection (classification layer 3).

    Caches VectorRetrieval for Chroma (path/collection/embed from env) and exposes
    llm_detect() for sp_api / uds / amazon_docs parallel prompts (first match wins).
    Naming reflects historical vector focus; LLM detection is co-located to avoid a
    separate singleton for gateway wiring.
    """

    _vector_retrieval: ClassVar[Optional[VectorRetrieval]] = None

    # Parallel LLM steps (first affirmative JSON wins); merged from former _LLMIntentDetector.
    _LLM_DETECTION_STEPS: ClassVar[List[Dict[str, str]]] = [
        {"prompt_name": "classification/sp_api_prompts", "workflow": "sp_api"},
        {"prompt_name": "classification/uds_prompts", "workflow": "uds"},
        {"prompt_name": "classification/amazon_prompts", "workflow": "amazon_docs"},
    ]

    @classmethod
    def _resolve_chroma_path(cls) -> Path:
        """Intent-registry Chroma directory from env or project default."""
        raw = (os.getenv("CHROMA_INTENT_REGISTRY_PATH") or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return _DEFAULT_INTENT_REGISTRY_CHROMA.resolve()

    @classmethod
    def _resolve_collection_name(cls) -> str:
        return (os.getenv("CHROMA_INTENT_REGISTRY_COLLECTION") or "intent_registry").strip()

    @classmethod
    def _resolve_embed_backend(cls) -> str:
        return (os.getenv("INTENT_REGISTRY_EMBED_BACKEND") or "ollama").strip().lower()

    @classmethod
    def _ensure_retriever(cls) -> None:
        if cls._vector_retrieval is not None:
            return
        cls._vector_retrieval = VectorRetrieval(
            cls._resolve_chroma_path(),
            cls._resolve_collection_name(),
            top_k=5,
            score_threshold=0.0,
            embed_backend=cls._resolve_embed_backend(),
        )

    @classmethod
    def get_vector_retrieval(cls) -> VectorRetrieval:
        """Shared VectorRetrieval for classification vector layer."""
        cls._ensure_retriever()
        if cls._vector_retrieval is None:
            raise RuntimeError("ClassificationIntentVectorStore failed to build VectorRetrieval")
        return cls._vector_retrieval

    @classmethod
    def llm_detect(
        cls, query: str, conversation_context: Optional[str] = None
    ) -> IntentResult:
        """
        Parallel prompt + LLM detector: sp_api, uds, amazon_docs; first Yes wins, else general.

        Args:
            query: User query text.
            conversation_context: Optional history for prompt {history} placeholder.

        Returns:
            IntentResult with source llm or fallback general on empty query.
        """
        if not query or not query.strip():
            return IntentResult(
                intent_name="general",
                workflow="general",
                confidence="low",
                source="fallback",
            )

        stripped = query.strip()
        step_timings: List[Dict[str, Any]] = []
        step_results: List[Tuple[Optional[IntentResult], int]] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    cls._run_detection_step,
                    query=stripped,
                    prompt_name=step["prompt_name"],
                    target_workflow=step["workflow"],
                    conversation_context=conversation_context,
                )
                for step in cls._LLM_DETECTION_STEPS
            ]
            for step, future in zip(cls._LLM_DETECTION_STEPS, futures):
                try:
                    result, step_ms = future.result()
                    step_timings.append(
                        {
                            "step": step["prompt_name"],
                            "workflow": step["workflow"],
                            "ms": step_ms,
                        }
                    )
                    step_results.append((result, step_ms))
                except Exception as exc:
                    logger.warning(
                        "Intent detection step '%s' failed: %s; treating as None",
                        step["prompt_name"],
                        exc,
                    )
                    step_timings.append(
                        {
                            "step": step["prompt_name"],
                            "workflow": step["workflow"],
                            "ms": 0,
                        }
                    )
                    step_results.append((None, 0))

        total_elapsed_ms = max((r[1] for r in step_results), default=0)
        for result, _ in step_results:
            if result is not None:
                result.intent_elapsed_ms = total_elapsed_ms
                result.step_timings = step_timings
                cls._log_detection_result(stripped, result)
                return result

        general_result = IntentResult(
            intent_name="general",
            workflow="general",
            confidence="low",
            source="llm",
            intent_elapsed_ms=total_elapsed_ms,
            step_timings=step_timings if step_timings else None,
        )
        cls._log_detection_result(stripped, general_result)
        return general_result

    @classmethod
    def _run_detection_step(
        cls,
        query: str,
        prompt_name: str,
        target_workflow: str,
        conversation_context: Optional[str] = None,
    ) -> Tuple[Optional[IntentResult], int]:
        """Run one detection prompt; returns (IntentResult or None, elapsed ms)."""
        prompt_template = load_prompt(prompt_name)
        history_text = (conversation_context or "").strip() or "(no conversation history)"
        prompt_text = prompt_template.replace("{history}", history_text).replace(
            "{query}", query
        )
        if "{examples}" in prompt_text:
            prompt_text = prompt_text.replace(
                "{examples}", _load_amazon_intent_examples()
            )

        step_start = time.perf_counter()
        response_text = cls._call_llm(prompt_text)
        step_elapsed_ms = int((time.perf_counter() - step_start) * 1000)
        query_preview = (query[:50] + "...") if len(query) > 50 else query
        logger.info(
            "[Perf] intent_classification step %s (workflow=%s) query=%r: %d ms",
            prompt_name,
            target_workflow,
            query_preview,
            step_elapsed_ms,
        )
        if not response_text:
            return None, step_elapsed_ms

        parsed = cls._parse_llm_response(response_text)
        if parsed is None:
            return None, step_elapsed_ms

        match_value = parsed.get("match")
        if match_value is None:
            result_str = str(parsed.get("result", "")).strip().lower()
            match_value = result_str in ("yes", "true", "1")
        if not match_value:
            return None, step_elapsed_ms

        intent_name = target_workflow
        confidence = parsed.get("confidence", "medium") or "medium"

        return (
            IntentResult(
                intent_name=intent_name,
                workflow=target_workflow,
                confidence=confidence,
                source="llm",
            ),
            step_elapsed_ms,
        )

    @classmethod
    def _call_llm(cls, prompt: str) -> str:
        """Route to DeepSeek or Ollama based on GATEWAY_INTENT_DETECT_BACKEND / keys."""
        backend = _intent_detect_backend()

        if backend == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            try:
                return DeepSeekChat().complete(
                    system_prompt="You are an intent classifier. Output ONLY valid JSON.",
                    user_content=prompt,
                    max_tokens=256,
                )
            except Exception as exc:
                logger.warning("Intent detect DeepSeek failed: %s; trying Ollama", exc)

        try:
            return OllamaClient().generate(prompt, empty_fallback="")
        except Exception as exc:
            logger.warning("Intent detect Ollama failed: %s", exc)
            return ""

    @staticmethod
    def _parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output; strip markdown fences if present."""
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                raw = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(raw)
        except ValueError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except ValueError:
                    pass
            logger.debug("Intent detect JSON parse failed: %s", raw[:200])
            return None

    @staticmethod
    def _log_detection_result(query: str, result: IntentResult) -> None:
        """Best-effort structured log for resolved intent (no raise on failure)."""
        if not _split_gateway_logger:
            return
        try:
            _split_gateway_logger.log_runtime(
                event_name="intent_classification_resolved",
                stage="intent_classification",
                message="classify_intent resolved",
                status="success",
                workflow=result.workflow,
                query_raw=query,
                metadata={
                    "intent_name": result.intent_name,
                    "source": result.source,
                    "confidence": result.confidence,
                },
            )
        except Exception:
            pass


class ClassificationKeywordRuleStore:
    """
    Loads intent keyword rules from classification_data/ exactly once (class-level cache).

    Internal to the classification package; wires CSV/YAML into KeywordRetrieval for
    ClassificationImplementMethod.keyword_classification_method.
    """

    _dict_data: ClassVar[Optional[List[Tuple[str, str]]]] = None
    _for_loop_data: ClassVar[Optional[List[Tuple[str, str, str]]]] = None
    _regex_data: ClassVar[Optional[List[Tuple[re.Pattern, str, str]]]] = None
    _keyword_retrieval: ClassVar[Optional[KeywordRetrieval]] = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Load three rule sets from disk once and build KeywordRetrieval singleton."""
        if cls._keyword_retrieval is not None:
            return
        data_dir = _CLASSIFICATION_DATA_DIR
        try:
            cls._dict_data = LoadKeywordRule.load_dict_sentences_csv(data_dir)
            cls._for_loop_data = LoadKeywordRule.load_frequent_variable_yml(data_dir)
            cls._regex_data = LoadKeywordRule.load_regular_rules_csv(data_dir)
        except Exception as exc:
            logger.warning("ClassificationKeywordRuleStore load failed: %s", exc)
            cls._dict_data = []
            cls._for_loop_data = []
            cls._regex_data = []
        cls._keyword_retrieval = KeywordRetrieval(
            list(cls._dict_data or []),
            list(cls._for_loop_data or []),
            list(cls._regex_data or []),
        )

    @classmethod
    def get_keyword_retrieval(cls) -> KeywordRetrieval:
        """Return shared KeywordRetrieval backed by class-cached rule rows."""
        cls._ensure_loaded()
        if cls._keyword_retrieval is None:
            raise RuntimeError("ClassificationKeywordRuleStore failed to build KeywordRetrieval")
        return cls._keyword_retrieval


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
        First layer: keyword match (dict → for-loop phrases → regex). Calls KeywordRetrieval.match().
        """
        if not _use_keyword():
            return None
        if not query or not query.strip():
            return None
        result = ClassificationKeywordRuleStore.get_keyword_retrieval().match(
            query.strip()
        )
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
        Second layer: Chroma vector similarity via ClassificationIntentVectorStore.
        Returns IntentResult if top candidate score >= threshold.
        """
        if not _use_vector():
            return None
        if not query or not query.strip():
            return None
        try:
            candidates = ClassificationIntentVectorStore.get_vector_retrieval().retrieve(
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
        Third layer: Prompt + LLM via ClassificationIntentVectorStore.llm_detect().
        Returns general when sp_api/uds/amazon_docs all miss.
        """
        return ClassificationIntentVectorStore.llm_detect(
            query, conversation_context=context
        )

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
