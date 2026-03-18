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
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..route_llm.classification import (
    classify_intents_batch,
    split_intents,
    validate_intents,
)
from ..schemas import QueryRequest, RewritePlan, TaskExecutionResult, TaskGroup, TaskItem
from .services import (
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)
from src.logger import get_logger_facade

logger = logging.getLogger(__name__)
_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


def _intent_classification_enabled() -> bool:
    """Return True when gateway LLM intent classification is enabled."""
    v = os.getenv("GATEWAY_VECTOR_INTENT_ENABLED", "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def _normalize_query(text: str) -> str:
    """Trim and collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _build_plan_from_extracted_intents(
    intents: List[str],
    conversation_context: Optional[str] = None,
    classified_intents: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[RewritePlan, List[Dict]]:
    """
    Build execution plan from extracted intents.

    Delegates the intent loop to classify_intents_batch() in the classification
    public API (per flowchart: the loop belongs inside the Intent Classification
    boundary). Falls back to "general" workflow when classification fails.

    Returns:
        (RewritePlan, intents_with_meta) — intents_with_meta is the list used
        by Phase 5 required-field validation.
    """
    tasks: List[TaskItem] = []
    intents_with_meta: List[Dict] = []

    # ── Batch classification via public API（流程图中 loop 在 classification 内部） ──
    batch_results = classified_intents
    if batch_results is None:
        try:
            batch_results = classify_intents_batch(intents, conversation_context)
        except Exception as exc:
            logger.warning("Batch classification failed, falling back to general: %s", exc)
            batch_results = None

    if batch_results is not None:
        for idx, item in enumerate(batch_results, start=1):
            q = item["query"]
            wf = item["workflow"]
            intent_name = item["intent_name"]

            intents_with_meta.append({
                "query": q,
                "intent_name": intent_name,
                "required_fields": item.get("required_fields") or [],
                "clarification_template": item.get("clarification_template") or "",
            })
            tasks.append(
                TaskItem(
                    task_id=f"t{idx}",
                    workflow=wf,
                    query=q,
                    depends_on=[],
                    reason=f"intent_classified:{intent_name}" if intent_name else "classified_fallback",
                )
            )
    else:
        # Classification unavailable — assign "general" to each intent
        for idx, intent in enumerate(intents, start=1):
            q = (intent or "").strip()
            if not q:
                continue
            intents_with_meta.append({
                "query": q,
                "intent_name": "",
                "required_fields": [],
                "clarification_template": "",
            })
            tasks.append(
                TaskItem(
                    task_id=f"t{idx}",
                    workflow="general",
                    query=q,
                    depends_on=[],
                    reason="classification_unavailable_fallback",
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
    Build a multi-task plan by splitting and classifying a mixed query.
    Used as a robust fallback when planner JSON is unavailable.
    """
    clauses = split_intents(query)
    if len(clauses) < 2:
        return None

    batch_results: list[dict[str, Any]] | None = None
    try:
        batch_results = classify_intents_batch(clauses)
    except Exception as exc:
        logger.warning("Batch classification failed in multi-task fallback: %s", exc)

    tasks: List[TaskItem] = []
    for idx, clause in enumerate(clauses, start=1):
        workflow = "general"
        if batch_results and idx - 1 < len(batch_results):
            workflow = (batch_results[idx - 1].get("workflow") or "general").strip() or "general"
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=workflow,
                query=clause,
                depends_on=[],
                reason="multi_intent_fallback",
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
            clauses = split_intents(q)
            if len(clauses) < 2:
                expanded_tasks.append(task)
                next_id += 1
                continue

            batch_results: list[dict[str, Any]] | None = None
            try:
                batch_results = classify_intents_batch(clauses)
            except Exception as exc:
                logger.warning("Batch classification failed in merged-task expansion: %s", exc)

            for idx, clause in enumerate(clauses):
                workflow = "general"
                if batch_results and idx < len(batch_results):
                    workflow = (batch_results[idx].get("workflow") or "general").strip() or "general"
                expanded_tasks.append(
                    TaskItem(
                        task_id=f"t{next_id}",
                        workflow=workflow,
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
    Post-classification plan correction hook.

    Previously applied keyword-heuristic overrides. Now that LLM prompt-based
    classification is the single authority (with "general" as built-in fallback),
    this function is a no-op pass-through. Kept for interface stability.
    """
    return plan


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
    to "general" workflow when classification is unavailable.

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
    from ..route_llm.rewriting.rewriters import _RewriteRouter

    explicit = (request.workflow or "auto").strip().lower() or "auto"
    normalized_query = _normalize_query(request.query or "")
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
        plan, intents_with_meta = _build_plan_from_extracted_intents(
            intents,
            conversation_context,
            classified_intents=classified_intents,
        )
        plan = _correct_plan_workflows(plan)

        # Phase 5: validate required fields per intent.
        clarification_question: Optional[str] = None
        if intents_with_meta:
            try:
                clarification_question = validate_intents(intents_with_meta, conversation_context)
            except Exception as exc:
                logger.warning("Intent field validation failed (non-fatal): %s", exc)

        return plan, clarification_question

    multi_task_plan = _build_multi_task_plan_from_query(rewritten_query or normalized_query)
    if multi_task_plan is not None:
        return multi_task_plan, None

    workflow, _, _, _, _ = _RewriteRouter.route_workflow(
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


# ─────────────────────────────────────────────────────────────────────────────
# 执行层 — DispatcherExecutor
# ─────────────────────────────────────────────────────────────────────────────


class DispatcherExecutor:
    """
    Execute workflow backends and task plans.

    call_workflow_backend: invoke one worker (RAG/UDS/SP-API etc).
    execute_task / execute_task_group / execute_plan: run plan and return results.
    """

    @classmethod
    def call_workflow_backend(
        cls, workflow: str, query_text: str, session_id: str | None
    ) -> Dict[str, Any]:
        """Invoke one worker agent (RAG/UDS/SP-API) and return normalized dict."""
        workflow_lower = (workflow or "general").strip().lower()
        if workflow_lower == "general":
            return call_general(query_text, session_id)
        if workflow_lower == "amazon_docs":
            return call_amazon_docs(query_text, session_id)
        if workflow_lower == "ic_docs":
            return call_ic_docs(query_text, session_id)
        if workflow_lower == "sp_api":
            return call_sp_api(query_text, session_id)
        if workflow_lower == "uds":
            return call_uds(query_text, session_id)
        logger.warning("Unknown workflow '%s', falling back to general", workflow)
        return call_general(query_text, session_id)

    @classmethod
    def execute_task(cls, task: TaskItem, request: QueryRequest) -> TaskExecutionResult:
        """Execute a single task and return structured result."""
        started = time.perf_counter()
        task_query = (task.query or "").strip()
        if not task_query:
            return TaskExecutionResult(
                task_id=task.task_id,
                workflow=task.workflow,
                query=task_query,
                status="skipped",
                answer="",
                sources=[],
                error="Empty task query",
                duration_ms=0,
            )
        backend_result = cls.call_workflow_backend(
            task.workflow, task_query, request.session_id
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        error_msg = backend_result.get("error")
        if error_msg:
            return TaskExecutionResult(
                task_id=task.task_id,
                workflow=task.workflow,
                query=task_query,
                status="failed",
                answer="",
                sources=backend_result.get("sources", []),
                error=str(error_msg),
                duration_ms=duration_ms,
            )
        return TaskExecutionResult(
            task_id=task.task_id,
            workflow=task.workflow,
            query=task_query,
            status="completed",
            answer=str(backend_result.get("answer", "")),
            sources=backend_result.get("sources", []),
            error=None,
            duration_ms=duration_ms,
        )

    @classmethod
    def execute_task_group(
        cls, group: TaskGroup, request: QueryRequest
    ) -> List[TaskExecutionResult]:
        """Execute one task group; tasks in parallel when group.parallel is True."""
        if not group.tasks:
            return []
        if not group.parallel or len(group.tasks) == 1:
            return [cls.execute_task(task, request) for task in group.tasks]
        results: List[TaskExecutionResult] = []
        max_workers = min(4, len(group.tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(cls.execute_task, task, request): task.task_id
                for task in group.tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as exc:
                    task_id = future_map[future]
                    logger.exception(
                        "Task execution failed for task_id=%s: %s", task_id, exc
                    )
                    results.append(
                        TaskExecutionResult(
                            task_id=task_id,
                            workflow="general",
                            query="",
                            status="failed",
                            answer="",
                            sources=[],
                            error=str(exc),
                            duration_ms=0,
                        )
                    )
        by_task_id = {r.task_id: r for r in results}
        return [by_task_id[t.task_id] for t in group.tasks if t.task_id in by_task_id]

    @classmethod
    def execute_plan(
        cls, plan: RewritePlan, request: QueryRequest
    ) -> List[TaskExecutionResult]:
        """Execute full plan; groups sequential, tasks within group in parallel."""
        all_results: List[TaskExecutionResult] = []
        for group in plan.task_groups:
            all_results.extend(cls.execute_task_group(group, request))
        return all_results


__all__ = [
    "build_execution_plan",
    "DispatcherExecutor",
    "_correct_plan_workflows",
    "_intent_classification_enabled",
]
