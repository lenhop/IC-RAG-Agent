"""
Dispatcher (Project Manager): builds execution plans from intents and orchestrates execution.

Receives intents from Route LLM (Decision Maker), maps them to workflows via heuristics
or vector intent classification (Phase 6), builds task_groups, and applies plan correction.
The api module uses this to obtain RewritePlan before executing tasks.

Phase 5: Per-intent required field validation — after classification, validates that
each sub-query contains the fields required by its intent (order_id, asin_or_sku, etc.).
Returns a clarification question if any are missing.

Phase 6: Vector intent classification — uses all-minilm + Chroma intent_registry to
classify each sub-query before falling back to keyword heuristics.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .routing_heuristics import (
    apply_docs_preference,
    normalize_query,
    route_workflow_heuristic,
    split_multi_intent_clauses,
)
from .schemas import QueryRequest, RewritePlan, TaskGroup, TaskItem
from src.logger import get_logger_facade

logger = logging.getLogger(__name__)
_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


def _vector_classification_enabled() -> bool:
    """Return True when gateway vector intent classification is enabled."""
    v = os.getenv("GATEWAY_VECTOR_INTENT_ENABLED", "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def _classify_query_to_workflow(
    query: str,
    conversation_context: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Classify a single query to (workflow, intent_name) using vector classification
    when enabled, falling back to keyword heuristics.

    Returns:
        (workflow, intent_name) — intent_name is empty string for heuristic path.
    """
    if _vector_classification_enabled():
        try:
            from .intent_classification.intent_classifier import classify_intent
            result = classify_intent(query, conversation_context=conversation_context)
            if result is not None:
                logger.debug(
                    "Vector intent: query='%s' -> intent=%s workflow=%s (dist=%.3f, conf=%s, src=%s)",
                    query[:60], result.intent_name, result.workflow,
                    result.distance, result.confidence, result.source,
                )
                return result.workflow, result.intent_name
        except Exception as exc:
            logger.warning("Vector intent classification failed, falling back to heuristic: %s", exc)

    # Heuristic fallback
    wf, _ = route_workflow_heuristic(query)
    wf = apply_docs_preference(query, wf)
    return wf, ""


def _build_plan_from_extracted_intents(
    intents: List[str],
    conversation_context: Optional[str] = None,
) -> Tuple[RewritePlan, List[Dict]]:
    """
    Build execution plan from extracted_intents.

    Phase 6: Routes each intent via vector classification (when enabled) or heuristic.
    Phase 5: Returns intent metadata list for required-field validation by caller.

    Returns:
        (RewritePlan, intents_with_meta) where intents_with_meta is a list of dicts
        with query, intent_name, required_fields, clarification_template.
    """
    tasks: List[TaskItem] = []
    intents_with_meta: List[Dict] = []

    for idx, intent in enumerate(intents, start=1):
        q = (intent or "").strip()
        if not q:
            continue

        wf, intent_name = _classify_query_to_workflow(q, conversation_context)

        # Collect metadata for Phase 5 field validation.
        required_fields: List[str] = []
        clarification_template: str = ""
        if intent_name:
            try:
                from .intent_classification.intent_registry import get_intent_metadata
                meta = get_intent_metadata()
                intent_info = meta.get(intent_name, {})
                required_fields = intent_info.get("required_fields") or []
                clarification_template = intent_info.get("clarification_template") or ""
            except Exception as exc:
                logger.debug("Could not load intent metadata for %s: %s", intent_name, exc)

        intents_with_meta.append({
            "query": q,
            "intent_name": intent_name,
            "required_fields": required_fields,
            "clarification_template": clarification_template,
        })

        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=wf,
                query=q,
                depends_on=[],
                reason=f"intent_classified:{intent_name}" if intent_name else "extracted_intents_heuristic",
            )
        )

    if not tasks:
        fallback_q = (intents[0] if intents else "unable to extract intents").strip() or "unable to extract intents"
        plan = RewritePlan(
            plan_type="single_domain",
            merge_strategy="concat",
            task_groups=[
                TaskGroup(
                    group_id="g1",
                    parallel=True,
                    tasks=[TaskItem(task_id="t1", workflow="general", query=fallback_q, depends_on=[], reason="empty_intents_fallback")],
                )
            ],
        )
        return plan, []

    plan = RewritePlan(
        plan_type="hybrid" if len({t.workflow for t in tasks}) > 1 else "single_domain",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )
    return plan, intents_with_meta


def _build_multi_task_plan_from_query(query: str) -> Optional[RewritePlan]:
    """
    Build a heuristic multi-task plan from comma/semicolon separated mixed query.
    Used as a robust fallback when planner JSON is unavailable.
    """
    clauses = split_multi_intent_clauses(query)
    if len(clauses) < 2:
        return None
    tasks: List[TaskItem] = []
    for idx, clause in enumerate(clauses, start=1):
        workflow, _ = route_workflow_heuristic(clause)
        workflow = apply_docs_preference(clause, workflow)
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=workflow,
                query=clause,
                depends_on=[],
                reason="heuristic_multi_intent_fallback",
            )
        )
    return RewritePlan(
        plan_type="hybrid",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )


def _build_single_task_plan(query: str, workflow: str) -> RewritePlan:
    """Build a single-task execution plan for legacy/fallback flow."""
    task_query = (query or "").strip()
    task_workflow = (workflow or "general").strip().lower() or "general"
    return RewritePlan(
        plan_type="single_domain",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(
                        task_id="t1",
                        workflow=task_workflow,
                        query=task_query,
                        depends_on=[],
                        reason="single_task_fallback",
                    )
                ],
            )
        ],
    )


def _expand_merged_tasks(plan: RewritePlan) -> RewritePlan:
    """
    Split any task whose query contains multiple distinct sub-queries into
    separate tasks. Ensures each sub-question is a separate plan item.
    """
    expanded_groups = []
    for group in plan.task_groups or []:
        expanded_tasks: List[TaskItem] = []
        next_id = 1
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            clauses = split_multi_intent_clauses(q)
            if len(clauses) < 2:
                expanded_tasks.append(task)
                next_id += 1
                continue
            for clause in clauses:
                wf, _ = route_workflow_heuristic(clause)
                wf = apply_docs_preference(clause, wf)
                expanded_tasks.append(
                    TaskItem(
                        task_id=f"t{next_id}",
                        workflow=wf,
                        query=clause,
                        depends_on=task.depends_on or [],
                        reason=(task.reason or "") + "_expanded" if task.reason else "merged_split",
                    )
                )
                next_id += 1
        if expanded_tasks:
            expanded_groups.append(
                TaskGroup(
                    group_id=group.group_id,
                    parallel=group.parallel,
                    tasks=expanded_tasks,
                )
            )
    if not expanded_groups:
        return plan
    return RewritePlan(
        plan_type=(
            "hybrid"
            if len({t.workflow for g in expanded_groups for t in g.tasks}) > 1
            else plan.plan_type
        ),
        merge_strategy=plan.merge_strategy,
        task_groups=expanded_groups,
    )


def _correct_plan_workflows(plan: RewritePlan) -> RewritePlan:
    """
    Apply heuristic-based workflow correction to fix LLM misclassifications.
    When the heuristic returns a different workflow with confidence >= 0.9,
    override the task's workflow to correct obvious routing errors.
    """
    HEURISTIC_OVERRIDE_CONF = 0.9
    for group in plan.task_groups or []:
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            h_wf, h_conf = route_workflow_heuristic(q)
            h_wf = apply_docs_preference(q, h_wf)
            current = (task.workflow or "").strip().lower()
            if h_wf != current and h_conf >= HEURISTIC_OVERRIDE_CONF:
                logger.debug(
                    "Plan correction: task %s workflow %s -> %s (heuristic conf=%.2f)",
                    task.task_id,
                    current,
                    h_wf,
                    h_conf,
                )
                task.workflow = h_wf
    return plan


def build_execution_plan(
    request: QueryRequest,
    rewritten_query: str,
    intents: Optional[List[str]] = None,
    conversation_context: Optional[str] = None,
) -> Tuple[RewritePlan, Optional[str]]:
    """
    Build a validated execution plan for query orchestration.

    Phase 6: Routes each intent via vector classification (when GATEWAY_VECTOR_INTENT_ENABLED=true)
    before falling back to keyword heuristics.

    Phase 5: After classification, validates required fields per intent. Returns a
    clarification question string when any sub-query is missing required fields.

    Args:
        request: Parsed QueryRequest.
        rewritten_query: Optimized retrieval query from Route LLM.
        intents: Optional list of sub-queries from intent classification.
        conversation_context: Optional conversation history for vector classification context.

    Returns:
        (RewritePlan, clarification_question) — clarification_question is None when
        all required fields are present.
    """
    from .router import route_workflow

    explicit = (request.workflow or "auto").strip().lower() or "auto"
    normalized_query = normalize_query(request.query or "")
    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="dispatcher_plan_start",
                stage="dispatcher",
                message="build_execution_plan started",
                status="started",
                session_id=request.session_id,
                user_id=request.user_id,
                workflow=explicit,
                query_raw=request.query or "",
                query_rewritten=rewritten_query,
                intent_list=intents or [],
            )
        except Exception:
            pass

    if explicit != "auto":
        task_query = (rewritten_query or normalized_query).strip() or normalized_query
        return _build_single_task_plan(task_query, explicit), None

    # Intent classification ran on optimized retrieval query; use intents when available.
    if intents and len(intents) > 0:
        plan, intents_with_meta = _build_plan_from_extracted_intents(intents, conversation_context)
        plan = _correct_plan_workflows(plan)

        # Phase 5: validate required fields per intent.
        clarification_question: Optional[str] = None
        if intents_with_meta:
            try:
                from .intent_classification.intent_validator import validate_intents
                clarification_question = validate_intents(intents_with_meta, conversation_context)
            except Exception as exc:
                logger.warning("Intent field validation failed (non-fatal): %s", exc)

        return plan, clarification_question

    multi_task_plan = _build_multi_task_plan_from_query(rewritten_query or normalized_query)
    if multi_task_plan is not None:
        return multi_task_plan, None

    workflow, _, _, _, _ = route_workflow(
        (rewritten_query or normalized_query).strip(), request
    )
    task_query = (rewritten_query or normalized_query).strip() or normalized_query
    final_plan = _build_single_task_plan(task_query, workflow)
    if _gateway_logger:
        try:
            _gateway_logger.log_runtime(
                event_name="dispatcher_plan_done",
                stage="dispatcher",
                message="build_execution_plan completed",
                status="success",
                session_id=request.session_id,
                user_id=request.user_id,
                workflow=workflow,
                query_raw=request.query or "",
                query_rewritten=rewritten_query,
                metadata={"plan_type": final_plan.plan_type},
            )
        except Exception:
            pass
    return final_plan, None


__all__ = [
    "build_execution_plan",
    "_correct_plan_workflows",
    "_vector_classification_enabled",
]
