"""
View-layer pure helpers for gateway API endpoints.

Builders (stateless helper classes):
  IntentDetailsBuilder: intent + workflow details for /rewrite UI preview.
  PlanHelper: execution plan helpers (extract query, derive workflow, merge answers).
  DebugTraceBuilder: debug trace dict for UI clients.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..dispatcher.summary.merge import ResultAggregator
from ..schemas import RewritePlan, TaskExecutionResult
from .config import GatewayConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentDetailsBuilder — intent and workflow details for /rewrite endpoint
# ---------------------------------------------------------------------------


class IntentDetailsBuilder:
    """
    Build intent list, intent_details (per-intent workflow + timing), and workflows list.

    Used by the rewrite endpoint to show UI preview. Uses classify_intents_batch
    for unified classification with per-intent and per-step timing.
    """

    @classmethod
    def build_intent_details(
        cls,
        rewritten_query: str,
        intents_override: Optional[List[str]] = None,
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]], List[str]]:
        """
        Build (intents, intent_details, workflows) from rewritten query.

        If intents_override is set, classifies those strings. Otherwise treats the rewritten
        string as a single sub-query (callers should pass intents from unified rewrite).
        classifies via classify_intents_batch (includes
        intent_elapsed_ms and step_timings for UI/log). Falls back to "general"
        when classification fails.
        """
        intents: Optional[List[str]] = None
        intent_details: List[Dict[str, Any]] = []
        workflows: List[str] = []
        if not (rewritten_query or "").strip() and not (intents_override or []):
            return intents, intent_details, workflows
        try:
            from ..route_llm.classification import classify_intents_batch
            if intents_override is not None:
                intents = [s.strip() for s in intents_override if (s or "").strip()]
            else:
                rq = (rewritten_query or "").strip()
                intents = [rq] if rq else []
            if intents:
                batch_results = classify_intents_batch(intents)
                for r in batch_results:
                    q = (r.get("query") or "").strip()
                    if not q:
                        continue
                    wf = (r.get("workflow") or "general").strip() or "general"
                    detail: Dict[str, Any] = {"intent": q, "workflow": wf}
                    if r.get("intent_elapsed_ms") is not None:
                        detail["intent_elapsed_ms"] = r["intent_elapsed_ms"]
                    if r.get("step_timings"):
                        detail["step_timings"] = r["step_timings"]
                    intent_details.append(detail)
                    if wf and wf not in workflows:
                        workflows.append(wf)
            if not workflows and (intents or []):
                q = (intents[0] if intents else rewritten_query or "").strip()
                if q:
                    batch_results = classify_intents_batch([q])
                    if batch_results:
                        wf = (batch_results[0].get("workflow") or "general").strip() or "general"
                        if wf:
                            workflows = [wf]
        except Exception as exc:
            logger.warning("Intent split or classification failed (rewrite response): %s", exc)
            q = (rewritten_query or "").strip()
            if q:
                intent_details.append({"intent": q, "workflow": "general"})
                workflows = ["general"]
        return intents, intent_details, workflows


# ---------------------------------------------------------------------------
# PlanHelper — execution plan and route metadata helpers
# ---------------------------------------------------------------------------


class PlanHelper:
    """Helpers for execution plan and route metadata. Pure class methods."""

    @classmethod
    def extract_route_input_query(cls, plan: RewritePlan, fallback_query: str) -> str:
        """Build route-input summary query from plan task queries."""
        task_queries = [
            task.query for group in plan.task_groups for task in group.tasks
            if (task.query or "").strip()
        ]
        if not task_queries:
            return (fallback_query or "").strip()
        if len(task_queries) == 1:
            return task_queries[0].strip()
        return " | ".join(q.strip() for q in task_queries if q.strip())

    @classmethod
    def derive_workflow(
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult], fallback: str,
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
        cls, plan: RewritePlan, task_results: List[TaskExecutionResult],
    ) -> str:
        """
        Merge task outputs into one answer (rule merge by default).

        When ``GATEWAY_SUMMARY_LLM_ENABLED`` and ``DEEPSEEK_API_KEY`` are set,
        multi-task merges use DeepSeek with fallback to rule merge on failure.
        """
        return ResultAggregator.merge(plan, task_results)


# ---------------------------------------------------------------------------
# DebugTraceBuilder — observability trace for UI clients
# ---------------------------------------------------------------------------


class DebugTraceBuilder:
    """Build optional observability trace returned to UI clients."""

    @classmethod
    def build(
        cls, *,
        original_query: str, rewritten_query: str, rewrite_time_ms: int,
        request: Any, route_input_query: str,
        route_source: str = "unknown",
        route_backend: str | None = None,
        route_llm_confidence: float | None = None,
    ) -> Dict[str, Any]:
        """Build debug trace dict with rewrite and route metadata."""
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_backend": GatewayConfig.resolve_rewrite_backend(request),
            "rewrite_time_ms": rewrite_time_ms,
            "route_input_query": route_input_query,
            "route_source": route_source,
            "route_backend": route_backend,
            "route_llm_confidence": route_llm_confidence,
        }

    # Backward-compatible alias
    build_debug_trace = build
