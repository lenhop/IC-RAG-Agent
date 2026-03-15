"""
Route LLM (Planning): query rewriters.

LLM-based query rewriting for the gateway. Supports:
- Ollama (local): HTTP POST to Ollama /api/generate
- DeepSeek (remote): OpenAI-compatible chat completions API

Refactored per code_development_refactor_by_workflow.md:
- Grouped by workflow into classes
- @classmethod for consistency
- Raise on exception (no silent null return)
- Comprehensive logging
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from pydantic import ValidationError

from ...schemas import RewritePlan, TaskGroup, TaskItem
from ...prompt_loader import load_prompt
from src.logger import get_logger_facade
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient

logger = logging.getLogger(__name__)

_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None

VALID_WORKFLOWS = {"general", "amazon_docs", "ic_docs", "sp_api", "uds"}
VALID_MERGE_STRATEGIES = {"none", "concat", "compare", "synthesize"}


class RewriteEnvValidator:
    """
    Validate and resolve rewrite env parameters from os.environ.

    Ollama rewrite uses OLLAMA_BASE_URL, OLLAMA_GENERATE_MODEL,
    OLLAMA_REQUEST_TIMEOUT, OLLAMA_EMBED_MODEL (via OllamaClient).
    """

    @staticmethod
    def get_deepseek_model() -> str:
        """Read GATEWAY_REWRITE_DEEPSEEK_MODEL from env."""
        return (os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL") or "deepseek-chat").strip()

    @staticmethod
    def get_deepseek_base_url() -> str:
        """Read DeepSeek base URL."""
        return (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")

    @staticmethod
    def get_deepseek_api_key() -> Optional[str]:
        """Read DEEPSEEK_API_KEY from env."""
        value = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        return value if value else None

    @staticmethod
    def get_backend() -> str:
        """Read GATEWAY_REWRITE_BACKEND from env (default ollama)."""
        return (os.getenv("GATEWAY_REWRITE_BACKEND") or "ollama").strip().lower()


class RewriteResponseProcessor:
    """
    Utilities for rewrite LLM response parsing and post-processing.

    - strip_markdown_fences: remove ``` code block markers
    - extract_first_json_object: extract first {...} JSON from text
    - strip_echoed_context: detect echoed context, return fallback
    - collapse_to_single_sentence: enforce single-line output
    - apply_normalization_fixes: typos, filler, lowercase
    - enforce_rewrite_responsibility: validate and fallback if non-compliant
    """

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Remove optional markdown code fences from model output."""
        raw = (text or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return raw

    @staticmethod
    def extract_first_json_object(text: str) -> Optional[str]:
        """Extract the first balanced JSON object from text."""
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

    @staticmethod
    def strip_echoed_context(response: str, fallback_query: str) -> str:
        """Detect when LLM echoed context; return fallback if so."""
        if not response or not response.strip():
            return fallback_query
        r_lower = response.strip().lower()
        echo_patterns = (
            "normalize: completed",
            "integrate short-term memory",
            "rewrite backend",
            "rewrite time",
            "intent classification",
            "rewrite-only test mode",
            "user:",
            "assistant:",
        )
        if any(p in r_lower for p in echo_patterns):
            logger.debug("Rewrite output contains echoed context; using fallback query")
            return fallback_query
        return response.strip()

    @staticmethod
    def collapse_to_single_sentence(text: str) -> str:
        """Enforce rewrite output as single line."""
        if not text or not text.strip():
            return text or ""
        line = re.sub(r"\s*\n+\s*", " ", text.strip())
        line = re.sub(r"\s+", " ", line).strip()
        return line if line else text.strip()

    @staticmethod
    def apply_normalization_fixes(text: str) -> str:
        """Apply typos, filler removal, lowercase."""
        if not text or not text.strip():
            return text or ""
        s = text.strip()
        s = re.sub(r"\?+", "?", s)
        s = re.sub(r"!+", "!", s)
        replacements = [
            (r"\bwat\b", "what"),
            (r"\binvetory\b", "inventory"),
            (r"\binvnetory\b", "inventory"),
            (r"\btehm\b", "them"),
            (r"\bpls\b", "please"),
            (r"\bthx\b", "thanks"),
            (r"\bu\b", "you"),
            (r"\bur\b", "your"),
        ]
        for pattern, repl in replacements:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*hey\s*,?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*(thx|thanks)\s*[.!?]*\s*$", "", s, flags=re.IGNORECASE).strip()
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else text.strip()

    @staticmethod
    def is_rewrite_responsibility_compliant(text: str) -> bool:
        """Check output against Rewriting_Responsibility constraints."""
        candidate = (text or "").strip()
        if not candidate:
            return False
        lower_text = candidate.lower()
        if "```" in candidate or candidate.startswith("{") or candidate.startswith("["):
            return False
        if '"intents"' in lower_text or '"task_groups"' in lower_text or '"workflow"' in lower_text:
            return False
        if "user:" in lower_text or "assistant:" in lower_text:
            return False
        if re.search(r"\b1\.\s+\S+.*\b2\.\s+\S+", candidate):
            return False
        if re.search(r"(?:^|\s)-\s+\S+.*(?:\s-\s+\S+)", candidate):
            return False
        return True

    @classmethod
    def enforce_rewrite_responsibility(cls, rewritten_text: str, fallback_query: str) -> str:
        """Enforce rewrite-stage responsibilities with fallback."""
        collapsed = cls.collapse_to_single_sentence(rewritten_text)
        if cls.is_rewrite_responsibility_compliant(collapsed):
            return collapsed
        logger.warning(
            "Rewrite output violates rewriting responsibility; fallback to normalized original query"
        )
        return cls.collapse_to_single_sentence(fallback_query)


class RewritePlanParser:
    """
    Parse LLM planner output into RewritePlan.

    Handles full planner JSON, intents-only format, and fallback.
    """

    @classmethod
    def parse(cls, text: str, fallback_query: str) -> Optional[RewritePlan]:
        """
        Parse LLM planner output into RewritePlan with robust fallback.

        Returns None only when neither planner output nor fallback query is usable.
        """
        raw = RewriteResponseProcessor.strip_markdown_fences(text)
        if not raw:
            return cls._normalize_plan(RewritePlan(task_groups=[]), fallback_query)

        parsed_data = None
        try:
            parsed_data = json.loads(raw)
        except ValueError:
            candidate = RewriteResponseProcessor.extract_first_json_object(raw)
            if candidate:
                try:
                    parsed_data = json.loads(candidate)
                except ValueError:
                    parsed_data = None

        if not isinstance(parsed_data, dict):
            logger.warning("Planner rewrite output is not valid JSON; falling back to single-task plan")
            return cls._normalize_plan(RewritePlan(task_groups=[]), fallback_query)

        intents_raw = parsed_data.get("intents")
        if isinstance(intents_raw, list) and not parsed_data.get("task_groups"):
            normalized_intents = []
            seen = set()
            for item in intents_raw:
                if isinstance(item, str):
                    s = (item or "").strip()
                    if s and s.lower() not in seen:
                        seen.add(s.lower())
                        normalized_intents.append(s)
            if normalized_intents:
                plan = RewritePlan(
                    plan_type="hybrid" if len(normalized_intents) > 1 else "single_domain",
                    merge_strategy="concat",
                    task_groups=[],
                    extracted_intents=normalized_intents,
                )
                return cls._normalize_plan(plan, fallback_query)

        try:
            plan = RewritePlan.model_validate(parsed_data)
        except ValidationError as exc:
            logger.warning("Planner rewrite output failed schema validation: %s", exc)
            return cls._normalize_plan(RewritePlan(task_groups=[]), fallback_query)

        return cls._normalize_plan(plan, fallback_query)

    @classmethod
    def _normalize_plan(cls, plan: RewritePlan, fallback_query: str) -> Optional[RewritePlan]:
        """Normalize planner output: enforce workflows, dedupe, max tasks."""
        dedupe_keys = set()
        normalized_groups = []
        total_tasks = 0
        max_tasks = int(os.getenv("GATEWAY_REWRITE_PLANNER_MAX_TASKS", "8"))

        for group_idx, group in enumerate(plan.task_groups, start=1):
            normalized_tasks = []
            for task in group.tasks:
                workflow = (task.workflow or "").strip().lower()
                query = (task.query or "").strip()
                if workflow not in VALID_WORKFLOWS or not query:
                    continue
                dedupe_key = f"{workflow}::{query.lower()}"
                if dedupe_key in dedupe_keys:
                    continue
                dedupe_keys.add(dedupe_key)
                total_tasks += 1
                if total_tasks > max_tasks:
                    break
                normalized_tasks.append(
                    TaskItem(
                        task_id=(task.task_id or f"t{total_tasks}").strip() or f"t{total_tasks}",
                        workflow=workflow,
                        query=query,
                        depends_on=[d for d in (task.depends_on or []) if isinstance(d, str) and d.strip()],
                        reason=(task.reason or "").strip() or None,
                    )
                )
            if normalized_tasks:
                normalized_groups.append(
                    TaskGroup(
                        group_id=(group.group_id or f"g{group_idx}").strip() or f"g{group_idx}",
                        parallel=bool(group.parallel),
                        tasks=normalized_tasks,
                    )
                )
            if total_tasks >= max_tasks:
                break

        if not normalized_groups and not plan.extracted_intents:
            fallback = (fallback_query or "").strip()
            if not fallback:
                return None
            normalized_groups = [
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[
                        TaskItem(
                            task_id="t1",
                            workflow="general",
                            query=fallback,
                            depends_on=[],
                            reason="fallback_single_task",
                        )
                    ],
                )
            ]

        merge_strategy = (plan.merge_strategy or "concat").strip().lower()
        if merge_strategy not in VALID_MERGE_STRATEGIES:
            merge_strategy = "concat"
        plan_type = (plan.plan_type or "single_domain").strip().lower()
        if plan_type not in {"single_domain", "hybrid"}:
            plan_type = (
                "single_domain"
                if len({t.workflow for g in normalized_groups for t in g.tasks}) == 1
                else "hybrid"
            )

        return RewritePlan(
            plan_type=plan_type,
            merge_strategy=merge_strategy,
            task_groups=normalized_groups,
            extracted_intents=plan.extracted_intents,
        )


class _RewriteLLM:
    """
    LLM-based query rewriting. Supports Ollama and DeepSeek backends.

    Main entry: rewrite_with_context. On failure raises RuntimeError per rule 8.
    """

    REWRITE_PROMPT = load_prompt("rewriting/rewrite_query_clean")

    @classmethod
    def rewrite_with_context(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
        backend: str = "ollama",
        model: Optional[str] = None,
    ) -> str:
        """
        Rewrite query with optional conversation context.

        Raises RuntimeError on LLM/network failure (no silent fallback).
        """
        if not query or not query.strip():
            return query

        effective_backend = (backend or "").strip().lower() or RewriteEnvValidator.get_backend()
        logger.info("Rewrite with context: backend=%s query_len=%d", effective_backend, len(query.strip()))

        if conversation_context and conversation_context.strip():
            prompt_prefix = (
                f"Conversation context (recent turns):\n{conversation_context.strip()}\n\n"
                f"Current query to rewrite: {query.strip()}\n\n"
                "CRITICAL: Output ONLY the rewritten query. Do NOT repeat the context, "
                "'user:', 'assistant:', or any trace (Normalize, Rewrite Backend, etc.)."
            )
        else:
            prompt_prefix = query.strip()

        raw: str
        if effective_backend == "ollama":
            raw = cls._call_ollama(prompt_prefix, query.strip(), model)
        elif effective_backend == "deepseek":
            raw = cls._call_deepseek(prompt_prefix, query.strip(), model)
        else:
            logger.error("rewrite_with_context: unknown backend %s", effective_backend)
            raise ValueError(f"Unknown rewrite backend {effective_backend}; must be 'ollama' or 'deepseek'")

        cleaned = RewriteResponseProcessor.strip_echoed_context(raw, query.strip())
        out = RewriteResponseProcessor.enforce_rewrite_responsibility(cleaned, query.strip())
        normalized = RewriteResponseProcessor.apply_normalization_fixes(out)

        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="rewriter_with_context_done",
                    stage="rewriter",
                    message="rewrite_with_context completed",
                    status="success",
                    workflow="rewrite",
                    query_raw=query.strip(),
                    query_rewritten=normalized,
                    metadata={"backend": effective_backend},
                )
            except Exception:
                pass

        logger.debug("Rewrite completed: %d chars", len(normalized))
        return normalized

    @classmethod
    def _call_ollama(
        cls,
        prompt_content: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """Call Ollama via OllamaClient (OLLAMA_* env only)."""
        logger.debug("Ollama rewrite via OllamaClient")
        prompt = f"{cls.REWRITE_PROMPT}\n\nUser question: {prompt_content}"
        return OllamaClient().generate(
            prompt,
            model=model,
            empty_fallback=fallback_query,
        )

    @classmethod
    def _call_deepseek(
        cls,
        prompt_content: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """Call DeepSeek via unified DeepSeekChat; raises RuntimeError on failure."""
        if not RewriteEnvValidator.get_deepseek_api_key():
            logger.error("DEEPSEEK_API_KEY not set; cannot call DeepSeek rewrite")
            raise ValueError("DEEPSEEK_API_KEY must be set for DeepSeek rewrite")
        logger.debug("DeepSeek rewrite via DeepSeekChat (stage=rewrite)")
        try:
            out = DeepSeekChat().complete(
                cls.REWRITE_PROMPT,
                prompt_content,
                model_override=model,
            )
            return out if out else fallback_query
        except Exception as exc:
            logger.error("DeepSeek rewrite failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek rewrite failed: {exc}") from exc


# --- Backward-compatible public API (delegates to classes) ---

REWRITE_PROMPT = _RewriteLLM.REWRITE_PROMPT


def parse_rewrite_plan_text(text: str, fallback_query: str) -> Optional[RewritePlan]:
    """Parse LLM planner output into RewritePlan. Backward-compatible wrapper."""
    return RewritePlanParser.parse(text, fallback_query)


def rewrite_with_context(
    query: str,
    conversation_context: Optional[str] = None,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> str:
    """Rewrite query with optional context. Backward-compatible wrapper."""
    try:
        return _RewriteLLM.rewrite_with_context(
            query,
            conversation_context=conversation_context,
            backend=backend,
            model=model,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("rewrite_with_context failed; returning original query: %s", exc)
        return (query or "").strip() or ""


def rewrite_with_ollama(query: str, model: Optional[str] = None) -> str:
    """Rewrite query using Ollama. Backward-compatible wrapper."""
    if not query or not query.strip():
        return query
    prompt = f"{REWRITE_PROMPT}\n\nUser question: {query.strip()}"
    try:
        return _RewriteLLM._call_ollama(prompt, query.strip(), model)
    except (RuntimeError, ValueError):
        logger.warning("rewrite_with_ollama failed; returning original query")
        return query.strip()


def rewrite_with_deepseek(query: str, model: Optional[str] = None) -> str:
    """Rewrite query using DeepSeek. Backward-compatible wrapper."""
    if not query or not query.strip():
        return query
    prompt = query.strip()
    try:
        return _RewriteLLM._call_deepseek(prompt, query.strip(), model)
    except (RuntimeError, ValueError):
        logger.warning("rewrite_with_deepseek failed; returning original query")
        return query.strip()


# Backward compatibility for tests
_enforce_rewrite_responsibility = RewriteResponseProcessor.enforce_rewrite_responsibility
_strip_echoed_context_from_rewrite = RewriteResponseProcessor.strip_echoed_context

__all__ = [
    "REWRITE_PROMPT",
    "parse_rewrite_plan_text",
    "rewrite_with_context",
    "rewrite_with_ollama",
    "rewrite_with_deepseek",
]
