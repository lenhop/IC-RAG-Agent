"""
Route LLM (Planning): query rewriters.

LLM-based query rewriting for the gateway. Supports:
- Ollama (local): HTTP POST to Ollama /api/generate
- DeepSeek (remote): OpenAI-compatible chat completions API

On failure: returns original query, logs error, does not raise.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests
from pydantic import ValidationError

from .schemas import RewritePlan, TaskGroup, TaskItem
from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# Prompts loaded from src/prompts/*.txt (cached after first access)
REWRITE_PROMPT = load_prompt("query_rewriting/rewrite_query_clean")

# Environment defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10
VALID_WORKFLOWS = {"general", "amazon_docs", "ic_docs", "sp_api", "uds"}
VALID_MERGE_STRATEGIES = {"none", "concat", "compare", "synthesize"}


def _get_rewrite_timeout() -> int:
    """Read GATEWAY_REWRITE_TIMEOUT from env (default 10s)."""
    try:
        return int(os.getenv("GATEWAY_REWRITE_TIMEOUT", str(DEFAULT_REWRITE_TIMEOUT)))
    except ValueError:
        return DEFAULT_REWRITE_TIMEOUT










def _strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            # Remove first/last fence if present; keep middle content.
            if lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object from text.

    This tolerates accidental leading/trailing text from LLM responses.
    """
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
                return text[start:idx + 1]
    return None


def _normalize_plan(plan: RewritePlan, fallback_query: str) -> Optional[RewritePlan]:
    """
    Normalize planner output:
    - enforce allowed workflow names
    - trim/skip blank task queries
    - deduplicate tasks by (workflow, query)
    - enforce max task guard
    """
    dedupe_keys = set()
    normalized_groups = []
    total_tasks = 0
    max_tasks = int(os.getenv("GATEWAY_REWRITE_PLANNER_MAX_TASKS", "8"))

    for group_idx, group in enumerate(plan.task_groups, start=1):
        normalized_tasks = []
        for task_idx, task in enumerate(group.tasks, start=1):
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

    if not normalized_groups:
        # When we have extracted_intents, leave task_groups empty so caller
        # (build_execution_plan) uses _build_plan_from_extracted_intents with proper routing.
        if plan.extracted_intents:
            pass
        else:
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
        plan_type = "single_domain" if len({t.workflow for g in normalized_groups for t in g.tasks}) == 1 else "hybrid"

    return RewritePlan(
        plan_type=plan_type,
        merge_strategy=merge_strategy,
        task_groups=normalized_groups,
        extracted_intents=plan.extracted_intents,
    )


def parse_rewrite_plan_text(text: str, fallback_query: str) -> Optional[RewritePlan]:
    """
    Parse LLM planner output into RewritePlan with robust fallback.

    Handles:
    - Full planner JSON: task_groups, extracted_intents, etc.
    - Phase 1 intents-only: {"intents": ["...", "..."]} -> RewritePlan(extracted_intents=..., task_groups=[]).

    Returns None only when neither planner output nor fallback query is usable.
    """
    raw = _strip_markdown_fences(text)
    if not raw:
        return _normalize_plan(RewritePlan(task_groups=[]), fallback_query)

    parsed_data = None
    try:
        parsed_data = json.loads(raw)
    except ValueError:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                parsed_data = json.loads(candidate)
            except ValueError:
                parsed_data = None

    if not isinstance(parsed_data, dict):
        logger.warning("Planner rewrite output is not valid JSON; falling back to single-task plan.")
        return _normalize_plan(RewritePlan(task_groups=[]), fallback_query)

    # Phase 1 intents-only format: {"intents": [...]} without task_groups.
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
            return _normalize_plan(plan, fallback_query)

    try:
        plan = RewritePlan.model_validate(parsed_data)
    except ValidationError as exc:
        logger.warning("Planner rewrite output failed schema validation: %s", exc)
        return _normalize_plan(RewritePlan(task_groups=[]), fallback_query)

    return _normalize_plan(plan, fallback_query)


def rewrite_with_ollama(query: str, model: Optional[str] = None) -> str:
    """
    Rewrite query using Ollama local LLM via HTTP API.

    Calls GATEWAY_REWRITE_OLLAMA_URL (default http://localhost:11434/api/generate)
    with GATEWAY_REWRITE_OLLAMA_MODEL (default qwen2.5:1.5b).
    Timeout: GATEWAY_REWRITE_TIMEOUT (default 10s).

    Args:
        query: User question to rewrite.
        model: Optional model override; uses env default if None.

    Returns:
        Rewritten query string, or original query on failure.
    """
    if not query or not query.strip():
        return query

    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = model or os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_rewrite_timeout()

    payload = {
        "model": mdl,
        "prompt": f"{REWRITE_PROMPT}\n\nUser question: {query.strip()}",
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        logger.warning("Ollama rewrite connection failed (%s): %s", url, exc)
        return query.strip()
    except requests.Timeout as exc:
        logger.warning("Ollama rewrite timed out (%s, %ds): %s", url, timeout, exc)
        return query.strip()
    except requests.RequestException as exc:
        logger.warning("Ollama rewrite request error (%s): %s", url, exc)
        return query.strip()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        logger.warning("Ollama rewrite HTTP %s (%s): %s", resp.status_code, url, detail)
        return query.strip()

    try:
        data = resp.json()
        rewritten = data.get("response", "").strip()
        if rewritten:
            return rewritten
        logger.warning("Ollama rewrite returned empty response; using original query")
        return query.strip()
    except (ValueError, TypeError) as exc:
        logger.warning("Ollama rewrite invalid JSON (%s): %s", url, exc)
        return query.strip()


def _strip_echoed_context_from_rewrite(response: str, fallback_query: str) -> str:
    """
    Detect when LLM echoed conversation context and return clean query.
    When response contains trace patterns (Normalize, Rewrite Backend, etc.) or
    user/assistant labels, the LLM echoed instead of outputting only the rewrite.
    Return fallback to avoid doubling the trace in memory and display.
    """
    if not response or not response.strip():
        return fallback_query
    r = response.strip()
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
    r_lower = r.lower()
    if any(p in r_lower for p in echo_patterns):
        logger.debug("Rewrite output contains echoed context; using fallback query")
        return fallback_query
    return r


def rewrite_with_context(
    query: str,
    conversation_context: Optional[str] = None,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> str:
    """
    Rewrite query with optional conversation context for retrieval optimization.

    When conversation_context is provided, prepends it to the prompt so the LLM
    can produce a retrieval-optimized query using recent turns.

    Args:
        query: Current user query to rewrite.
        conversation_context: Optional formatted history (e.g. "Turn 1: Q: ... A: ...").
        backend: "ollama" or "deepseek".
        model: Optional model override.

    Returns:
        Rewritten query string, or original query on failure.
    """
    if not query or not query.strip():
        return query

    effective_backend = (backend or "").strip().lower() or os.getenv(
        "GATEWAY_REWRITE_BACKEND", "ollama"
    ).strip().lower()

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
        raw = _rewrite_with_ollama_prompt(prompt_prefix, query.strip(), model)
    elif effective_backend == "deepseek":
        raw = _rewrite_with_deepseek_prompt(prompt_prefix, query.strip(), model)
    else:
        logger.warning("rewrite_with_context: unknown backend %s; returning original", effective_backend)
        return query.strip()
    return _strip_echoed_context_from_rewrite(raw, query.strip())


def _rewrite_with_ollama_prompt(
    prompt_content: str, fallback_query: str, model: Optional[str] = None
) -> str:
    """Call Ollama with custom prompt content; return rewritten text or fallback."""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = model or os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_rewrite_timeout()
    payload = {
        "model": mdl,
        "prompt": f"{REWRITE_PROMPT}\n\nUser question: {prompt_content}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama rewrite connection failed (%s): %s", url, exc)
        return fallback_query
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        logger.warning("Ollama rewrite HTTP %s (%s): %s", resp.status_code, url, detail)
        return fallback_query
    try:
        data = resp.json()
        rewritten = (data.get("response") or "").strip()
        return rewritten if rewritten else fallback_query
    except (ValueError, TypeError) as exc:
        logger.warning("Ollama rewrite invalid JSON (%s): %s", url, exc)
        return fallback_query


def _rewrite_with_deepseek_prompt(
    prompt_content: str, fallback_query: str, model: Optional[str] = None
) -> str:
    """Call DeepSeek with custom prompt content; return rewritten text or fallback."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek rewrite")
        return fallback_query
    mdl = model or os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    timeout = _get_rewrite_timeout()
    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("openai client not installed; cannot call DeepSeek rewrite: %s", exc)
        return fallback_query
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": prompt_content},
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
            rewritten = (choice.message.content or "").strip()
            return rewritten if rewritten else fallback_query
    except Exception as exc:
        logger.warning("DeepSeek rewrite failed: %s", exc)
    return fallback_query


def rewrite_with_deepseek(query: str, model: Optional[str] = None) -> str:
    """
    Rewrite query using DeepSeek API (OpenAI-compatible client).

    Uses DEEPSEEK_API_KEY and GATEWAY_REWRITE_DEEPSEEK_MODEL from env.
    Timeout: GATEWAY_REWRITE_TIMEOUT (default 10s).

    Args:
        query: User question to rewrite.
        model: Optional model override; uses env default if None.

    Returns:
        Rewritten query string, or original query on failure.
    """
    if not query or not query.strip():
        return query

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek rewrite")
        return query.strip()

    mdl = model or os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    timeout = _get_rewrite_timeout()

    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("openai client not installed; cannot call DeepSeek rewrite: %s", exc)
        return query.strip()

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": query.strip()},
    ]

    try:
        response = client.chat.completions.create(
            model=mdl,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("DeepSeek rewrite failed: %s", exc)
        return query.strip()

    try:
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            rewritten = choice.message.content.strip()
            if rewritten:
                return rewritten
        logger.warning("DeepSeek rewrite returned empty response; using original query")
        return query.strip()
    except (IndexError, AttributeError, TypeError) as exc:
        logger.warning("DeepSeek rewrite unexpected response format: %s", exc)
        return query.strip()


__all__ = [
    "REWRITE_PROMPT",
    "parse_rewrite_plan_text",
    "rewrite_with_context",
    "rewrite_with_ollama",
    "rewrite_with_deepseek",
]
