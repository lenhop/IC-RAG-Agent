"""
View-layer helpers for gateway API endpoints.

IntentDetailsBuilder: intent + workflow details for /rewrite UI preview.
PlanHelper: execution plan helpers (extract query, derive workflow, merge answers).
DebugTraceBuilder: debug trace dict for UI clients.

Merged from: intent_rewrite.py + plan_helper.py + debug_trace.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import QueryRequest, RewritePlan, TaskExecutionResult
from .config import GatewayConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentDetailsBuilder — intent and workflow details for /rewrite endpoint
# ---------------------------------------------------------------------------


class IntentDetailsBuilder:
    """
    Build intent list, intent_details (per-intent workflow), and workflows list.

    Used by the rewrite endpoint to show UI preview. Uses LLM prompt-based
    classification (classification module) for all intent detection.
    """

    @classmethod
    def build_intent_details(
        cls, rewritten_query: str
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]], List[str]]:
        """
        Build (intents, intent_details, workflows) from rewritten query.

        Runs split_intents; classifies each intent via classify_intent().
        Falls back to "general" workflow when classification is unavailable.
        """
        intents: Optional[List[str]] = None
        intent_details: List[Dict[str, str]] = []
        workflows: List[str] = []
        if not (rewritten_query or "").strip():
            return intents, intent_details, workflows
        try:
            from ..route_llm.classification import (
                classify_intent,
                split_intents,
            )
            intents = split_intents(rewritten_query)
            if intents:
                for intent in intents:
                    q = (intent or "").strip()
                    if not q:
                        continue
                    result = classify_intent(q)
                    wf = result.workflow if result else "general"
                    intent_details.append(
                        {
                            "intent": q,
                            "workflow": wf,
                        }
                    )
                    if wf and wf not in workflows:
                        workflows.append(wf)
            if not workflows and (intents or []):
                q = (intents[0] if intents else rewritten_query or "").strip()
                if q:
                    result = classify_intent(q)
                    wf = result.workflow if result else "general"
                    if wf:
                        workflows = [wf]
        except Exception as exc:
            logger.warning(
                "Intent split or classification failed (rewrite response): %s", exc
            )
            # Fallback: single "general" intent for the whole query
            q = (rewritten_query or "").strip()
            if q:
                intent_details.append(
                    {
                        "intent": q,
                        "workflow": "general",
                    }
                )
                workflows = ["general"]
        return intents, intent_details, workflows


# ---------------------------------------------------------------------------
# PlanHelper — execution plan and route metadata helpers
# ---------------------------------------------------------------------------


class PlanHelper:
    """
    Helpers for execution plan and route metadata.

    Pure functions; no gateway state. All class methods.
    """

    @classmethod
    def extract_route_input_query(cls, plan: RewritePlan, fallback_query: str) -> str:
        """Build route-input summary query from plan task queries."""
        task_queries = [
            task.query
            for group in plan.task_groups
            for task in group.tasks
            if (task.query or "").strip()
        ]
        if not task_queries:
            return (fallback_query or "").strip()
        if len(task_queries) == 1:
            return task_queries[0].strip()
        return " | ".join(q.strip() for q in task_queries if q.strip())

    @classmethod
    def derive_workflow(
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult], fallback: str
    ) -> str:
        """Derive top-level workflow label from plan/result context."""
        workflows = {task.workflow for group in plan.task_groups for task in group.tasks}
        if len(workflows) > 1:
            return "hybrid"
        if len(workflows) == 1:
            return next(iter(workflows))
        return fallback

    @classmethod
    def merge_task_answers(
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult]
    ) -> str:
        """Build deterministic merged answer from successful task outputs."""
        completed = [
            r for r in task_results
            if r.status == "completed" and (r.answer or "").strip()
        ]
        if not completed:
            return ""
        if len(completed) == 1:
            return completed[0].answer.strip()
        if plan.merge_strategy == "none":
            return completed[0].answer.strip()
        if plan.merge_strategy in ("compare", "synthesize"):
            merged_lines = ["Combined results:"]
            for result in completed:
                merged_lines.append(f"- [{result.workflow}] {result.answer.strip()}")
            return "\n".join(merged_lines)
        return "\n".join(
            f"- [{r.workflow}] {r.answer.strip()}" for r in completed
        )


# ---------------------------------------------------------------------------
# DebugTraceBuilder — observability trace for UI clients
# ---------------------------------------------------------------------------


class DebugTraceBuilder:
    """
    Build optional observability trace returned to UI clients.
    """

    @classmethod
    def build_debug_trace(
        cls,
        original_query: str,
        rewritten_query: str,
        rewrite_time_ms: int,
        request: QueryRequest,
        route_input_query: str,
        route_source: str,
        route_backend: str | None,
        route_llm_confidence: float | None,
    ) -> Dict[str, Any]:
        """Build debug trace dict with rewrite and route metadata."""
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_enabled": bool(request.rewrite_enable),
            "rewrite_backend": GatewayConfig.resolve_rewrite_backend(request),
            "rewrite_time_ms": rewrite_time_ms,
            "route_input_query": route_input_query,
            "route_source": route_source,
            "route_backend": route_backend,
            "route_llm_confidence": route_llm_confidence,
        }
