"""
Dispatcher (Project Manager): builds execution plans from intents and orchestrates execution.

Receives intents from Route LLM (Decision Maker), maps them to workflows via
LLM intent classification (classification module), builds task_groups, and
applies plan correction. The api module uses this to obtain RewritePlan before
executing tasks.

Phase 5: Per-intent required field validation — after classification, validates that
each sub-query contains the fields required by its intent (order_id, asin_or_sku, etc.).
Returns a clarification question if any are missing.

Phase 6: LLM prompt-based intent classification — uses sequential prompt detection
(SP-API → UDS → Amazon Business → General) to classify each sub-query.

Planning logic lives in ``planning/``; execution in ``execution/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import QueryRequest, RewritePlan
from .execution import DispatcherExecutor
from .planning import (
    FallbackPlanBuilder,
    IntentClassificationFlags,
    PlanBuilder,
    PlanPipeline,
    PlanPostProcessor,
)

__all__ = [
    "build_execution_plan",
    "DispatcherExecutor",
    "_correct_plan_workflows",
    "_intent_classification_enabled",
]


def _intent_classification_enabled() -> bool:
    """Return True when gateway LLM intent classification is enabled."""
    return IntentClassificationFlags.vector_intent_enabled()


# Backward-compatible aliases (tests and legacy imports patch these names on this module).
_build_plan_from_extracted_intents = PlanBuilder.build_from_extracted_intents
_build_multi_task_plan_from_query = FallbackPlanBuilder.build_multi_task_plan_from_query
_build_single_task_plan = FallbackPlanBuilder.build_single_task_plan
_expand_merged_tasks = PlanPostProcessor.expand_merged_tasks
_correct_plan_workflows = PlanPostProcessor.correct_plan_workflows


def build_execution_plan(
    request: QueryRequest,
    rewritten_query: str,
    intents: Optional[List[str]] = None,
    conversation_context: Optional[str] = None,
    classified_intents: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[RewritePlan, Optional[str]]:
    """
    Build a validated execution plan for query orchestration.

    Phase 6: Routes each intent via LLM prompt-based classification. Falls back
    to \"general\" workflow when classification is unavailable.

    Phase 5: After classification, validates required fields per intent. Returns a
    clarification question string when any sub-query is missing required fields.

    Args:
        request: Parsed QueryRequest.
        rewritten_query: Optimized retrieval query from Route LLM.
        intents: Optional list of sub-queries from intent classification.
        conversation_context: Optional conversation history for classification context.
        classified_intents: Optional pre-classified intent metadata from caller.

    Returns:
        (RewritePlan, clarification_question) — clarification_question is None when
        all required fields are present.
    """
    return PlanPipeline.build_execution_plan(
        request,
        rewritten_query,
        intents=intents,
        conversation_context=conversation_context,
        classified_intents=classified_intents,
    )
