"""
Route LLM (Planning): query rewriters.

LLM-based query rewriting for the gateway. Supports:
- Ollama (local): HTTP POST to Ollama /api/generate
- DeepSeek (remote): OpenAI-compatible chat completions API

Planner mode (GATEWAY_REWRITE_PLANNER_ENABLED): outputs JSON task decomposition
for hybrid queries. Otherwise: simple rewrite.

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

logger = logging.getLogger(__name__)

# Shared prompt for both backends (per PLAN.md)
REWRITE_PROMPT = (
    "Rewrite this user question into a clearer search query for our knowledge base. "
    "Preserve all dates, numbers, filters, and entity names. "
    "Do not answer the question; only output the rewritten query. "
    "Output a single line, no explanation."
)

# Phase 1: Intent classification only. Lists distinct sub-questions; no task building.
# Used when GATEWAY_REWRITE_PLANNER_ENABLED to avoid LLM merging intents.
INTENT_CLASSIFICATION_PROMPT = (
    "List each distinct sub-question from the user query. Output JSON only: {\"intents\": [\"...\", \"...\"]}. "
    "CRITICAL: Do NOT merge multiple questions into one. Each sub-question must be a separate item. "
    "Preserve ASINs, order IDs, dates exactly. Do NOT split dates that contain commas "
    "(e.g. keep \"September 1st, 2026\" or \"January 1, 2025\" as one intent, not two). "
    "Example 1: \"what is FBA get order 123 which table\" -> "
    "{\"intents\": [\"what is FBA\", \"get order 123\", \"which table stores referral fee data\"]}. "
    "Example 2: \"what is FBA vs FBM, get order 112-123, which table stores fee, show storage trend\" -> "
    "{\"intents\": [\"what is the difference between FBA and FBM\", \"get order status for 112-123\", "
    "\"which ClickHouse table stores referral fee breakdown\", \"show me month over month trend of storage fees\"]}"
)

# Planner-style rewrite prompt for hybrid task decomposition.
# Enabled via GATEWAY_REWRITE_PLANNER_ENABLED to avoid changing default behavior.
# When two-phase flow is used, this is bypassed in favor of INTENT_CLASSIFICATION_PROMPT.
REWRITE_PLANNER_PROMPT = (
    "You are a query rewriting and task planning engine for a multi-agent gateway. "
    "Your task is to split a complex user query into executable subtasks. "
    "Allowed workflows: general, amazon_docs, ic_docs, sp_api, uds. "
    "Preserve all entities, dates, metrics, ASINs, and constraints from user text. "
    "Do not answer the user question. "
    "Output JSON only. No markdown, no extra text. "
    "Process: Step 1) List each distinct sub-question in extracted_intents (one per item). "
    "Step 2) Create exactly one task per extracted intent in task_groups. "
    "JSON schema: "
    "{"
    '"extracted_intents":["sub-question 1","sub-question 2",...],'
    '"plan_type":"single_domain|hybrid",'
    '"merge_strategy":"none|concat|compare|synthesize",'
    '"task_groups":[{'
    '"group_id":"g1",'
    '"parallel":true,'
    '"tasks":[{'
    '"task_id":"t1",'
    '"workflow":"general|amazon_docs|ic_docs|sp_api|uds",'
    '"query":"agent-ready query (must match one extracted_intents item)",'
    '"depends_on":["task_id_optional"],'
    '"reason":"short routing reason optional"'
    "}]"
    "}]"
    "} "
    "Routing policy constraints: "
    "A) amazon_docs for Amazon business rules/policies/requirements/fee definitions; "
    "B) uds should be preferred first for analytical/historical questions that can be answered from warehouse snapshots; "
    "C) sp_api must be used only when the requested data is real-time/current-state or can only be retrieved via live SP-API endpoints; "
    "D) uds is historical (daily loaded) and should handle trend/aggregate/by-period/table analytics; "
    "E) sp_api must not be used for policy/business-rule explanation tasks. "
    "Constraints: "
    "1) extracted_intents must list every distinct sub-question point by point; do not merge. "
    "2) Each task query must match exactly one extracted_intents item. "
    "3) keep task_groups ordered by dependency stage; "
    "4) each task must have non-empty query; "
    "5) use hybrid when more than one workflow is required; "
    "6) default merge_strategy to concat unless explicit compare/synthesize intent. "
    "Example: user asks 'what does FBA removal policy say check ASIN B07 active listings which table stores fee data' -> "
    "extracted_intents: [\"what does Amazon FBA removal policy say\", \"check if ASIN B07 has any active listings\", \"which table stores referral fee data\"]."
)

# Environment defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10
DEFAULT_MAX_PLANNER_TASKS = 8
VALID_WORKFLOWS = {"general", "amazon_docs", "ic_docs", "sp_api", "uds"}
VALID_MERGE_STRATEGIES = {"none", "concat", "compare", "synthesize"}


def _get_rewrite_timeout() -> int:
    """Read GATEWAY_REWRITE_TIMEOUT from env (default 10s)."""
    try:
        return int(os.getenv("GATEWAY_REWRITE_TIMEOUT", str(DEFAULT_REWRITE_TIMEOUT)))
    except ValueError:
        return DEFAULT_REWRITE_TIMEOUT


def planner_rewrite_enabled() -> bool:
    """Return True when planner-style rewrite prompt is enabled."""
    value = os.getenv("GATEWAY_REWRITE_PLANNER_ENABLED", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _get_rewrite_prompt() -> str:
    """Resolve active rewrite prompt based on environment toggle."""
    if planner_rewrite_enabled():
        return REWRITE_PLANNER_PROMPT
    return REWRITE_PROMPT


def _planner_max_tasks() -> int:
    """Read max planner tasks guard from env with a safe default."""
    try:
        return max(1, int(os.getenv("GATEWAY_REWRITE_PLANNER_MAX_TASKS", str(DEFAULT_MAX_PLANNER_TASKS))))
    except ValueError:
        return DEFAULT_MAX_PLANNER_TASKS


def rewrite_intents_only(query: str, backend: Optional[str] = None) -> Optional[dict]:
    """
    Phase 1 intent classification: call LLM to list distinct sub-questions only.

    Returns {"intents": ["...", "..."]} on success, None on failure.
    Uses same backends as rewrite (ollama/deepseek) and env for URL/model.
    """
    if not query or not query.strip():
        return None

    effective_backend = (backend or "").strip().lower() or os.getenv(
        "GATEWAY_REWRITE_BACKEND", "ollama"
    ).strip().lower()

    if effective_backend == "ollama":
        text = _call_intents_ollama(query.strip())
    elif effective_backend == "deepseek":
        text = _call_intents_deepseek(query.strip())
    else:
        logger.warning("rewrite_intents_only: unknown backend %s; skipping", effective_backend)
        return None

    if not text or not text.strip():
        return None

    raw = _strip_markdown_fences(text)
    parsed = None
    try:
        parsed = json.loads(raw)
    except ValueError:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except ValueError:
                parsed = None

    if not isinstance(parsed, dict):
        return None

    # Accept "intents", "extracted_intents", or extract from task_groups.
    intents = parsed.get("intents") or parsed.get("extracted_intents")
    if not isinstance(intents, list) or not intents:
        # Fallback: collect queries from task_groups if LLM returned full planner format.
        groups = parsed.get("task_groups") or []
        intents = []
        for g in groups:
            if isinstance(g, dict):
                for t in (g.get("tasks") or []):
                    if isinstance(t, dict) and isinstance(t.get("query"), str):
                        intents.append((t.get("query") or "").strip())
    if not isinstance(intents, list) or not intents:
        return None

    # Normalize: filter empty, strip, dedupe
    seen = set()
    normalized = []
    for item in intents:
        if not isinstance(item, str):
            continue
        s = (item or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(s)

    if not normalized:
        return None

    return {"intents": normalized}


def _call_intents_ollama(query: str) -> str:
    """Call Ollama with intent classification prompt; return raw response text."""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_rewrite_timeout()
    payload = {
        "model": mdl,
        "prompt": f"{INTENT_CLASSIFICATION_PROMPT}\n\nUser question: {query}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama intent classification failed: %s", exc)
        return ""
    if resp.status_code != 200:
        logger.warning("Ollama intent classification HTTP %s", resp.status_code)
        return ""
    try:
        data = resp.json()
        return (data.get("response") or "").strip()
    except (ValueError, TypeError):
        return ""


def _call_intents_deepseek(query: str) -> str:
    """Call DeepSeek with intent classification prompt; return raw response text."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek intent classification")
        return ""
    mdl = os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    timeout = _get_rewrite_timeout()
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai client not installed; cannot call DeepSeek intent classification")
        return ""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
    messages = [
        {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
        {"role": "user", "content": query},
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
        logger.warning("DeepSeek intent classification failed: %s", exc)
    return ""


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
    max_tasks = _planner_max_tasks()

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
        "prompt": f"{_get_rewrite_prompt()}\n\nUser question: {query.strip()}",
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
        {"role": "system", "content": _get_rewrite_prompt()},
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
    "INTENT_CLASSIFICATION_PROMPT",
    "REWRITE_PROMPT",
    "REWRITE_PLANNER_PROMPT",
    "planner_rewrite_enabled",
    "parse_rewrite_plan_text",
    "rewrite_intents_only",
    "rewrite_with_ollama",
    "rewrite_with_deepseek",
]
