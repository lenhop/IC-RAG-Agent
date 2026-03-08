"""
Gateway FastAPI application.

Exposes a minimal REST API for the unified query gateway:
- POST /api/v1/query: accept QueryRequest, return QueryResponse (stubbed routing)
- GET /health: basic health check endpoint

Routing and backend integration will be implemented in later tasks; this
module focuses on the HTTP contract and app wiring.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import build_execution_plan, route_workflow, rewrite_query
from .logging_utils import format_route_metadata
from .services import (
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)
from .schemas import (
    QueryRequest,
    QueryResponse,
    RewritePlan,
    RewriteResponse,
    TaskExecutionResult,
    TaskGroup,
    TaskItem,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IC-RAG Gateway API",
    description="Unified query gateway for IC-RAG-Agent.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Simple health check endpoint.

    Returns:
        JSON payload with status field.
    """
    return {"status": "ok"}


def _call_workflow_backend(workflow: str, query_text: str, session_id: str | None) -> Dict[str, Any]:
    """Dispatch one task to the target backend and return normalized dict."""
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


def _execute_task(task: TaskItem, request: QueryRequest) -> TaskExecutionResult:
    """Execute a single task and return structured task result."""
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

    backend_result = _call_workflow_backend(task.workflow, task_query, request.session_id)
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


def _execute_task_group(group: TaskGroup, request: QueryRequest) -> List[TaskExecutionResult]:
    """Execute one task group, parallelizing tasks when group.parallel is true."""
    if not group.tasks:
        return []
    if not group.parallel or len(group.tasks) == 1:
        return [_execute_task(task, request) for task in group.tasks]

    results: List[TaskExecutionResult] = []
    max_workers = min(4, len(group.tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_execute_task, task, request): task.task_id for task in group.tasks}
        for future in concurrent.futures.as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive fallback
                task_id = future_map[future]
                logger.exception("Task execution failed unexpectedly for task_id=%s: %s", task_id, exc)
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

    # Keep stable ordering aligned with task declaration order.
    by_task_id = {result.task_id: result for result in results}
    ordered = [by_task_id[task.task_id] for task in group.tasks if task.task_id in by_task_id]
    return ordered


def _execute_plan(plan: RewritePlan, request: QueryRequest) -> List[TaskExecutionResult]:
    """Execute plan groups sequentially and tasks within each group per group.parallel."""
    all_results: List[TaskExecutionResult] = []
    for group in plan.task_groups:
        all_results.extend(_execute_task_group(group, request))
    return all_results


def _extract_route_input_query(plan: RewritePlan, fallback_query: str) -> str:
    """Build route-input summary query from plan task queries."""
    task_queries = [task.query for group in plan.task_groups for task in group.tasks if (task.query or "").strip()]
    if not task_queries:
        return (fallback_query or "").strip()
    if len(task_queries) == 1:
        return task_queries[0].strip()
    return " | ".join(query.strip() for query in task_queries if query.strip())


def _derive_workflow(plan: RewritePlan, task_results: List[TaskExecutionResult], fallback: str) -> str:
    """Derive top-level workflow label from plan/result context."""
    workflows = {task.workflow for group in plan.task_groups for task in group.tasks}
    if len(workflows) > 1:
        return "hybrid"
    if len(workflows) == 1:
        return next(iter(workflows))
    return fallback


def _merge_task_answers(plan: RewritePlan, task_results: List[TaskExecutionResult]) -> str:
    """Build deterministic merged answer from successful task outputs."""
    completed = [result for result in task_results if result.status == "completed" and result.answer.strip()]
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
    return "\n".join(f"- [{result.workflow}] {result.answer.strip()}" for result in completed)


def _resolve_rewrite_backend(request: QueryRequest) -> str | None:
    """Resolve effective rewrite backend used by gateway."""
    if not request.rewrite_enable:
        return None
    backend = (request.rewrite_backend or "").strip().lower()
    if backend:
        return backend
    return os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower()


def _build_debug_trace(
    original_query: str,
    rewritten_query: str,
    rewrite_time_ms: int,
    request: QueryRequest,
    route_input_query: str,
    route_source: str,
    route_backend: str | None,
    route_llm_confidence: float | None,
) -> Dict[str, Any]:
    """Build optional observability trace returned to UI clients."""
    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "rewrite_enabled": bool(request.rewrite_enable),
        "rewrite_backend": _resolve_rewrite_backend(request),
        "rewrite_time_ms": rewrite_time_ms,
        "route_input_query": route_input_query,
        "route_source": route_source,
        "route_backend": route_backend,
        "route_llm_confidence": route_llm_confidence,
    }


def _is_rewrite_only_mode() -> bool:
    """Return True when gateway runs in rewrite-only quick test mode."""
    return os.getenv("GATEWAY_REWRITE_ONLY_MODE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@app.post(
    "/api/v1/rewrite",
    response_model=RewriteResponse,
    summary="Rewrite query only",
)
async def rewrite(request: QueryRequest) -> RewriteResponse:
    """
    Rewrite query and return rewrite metadata without routing/execution.

    This endpoint is used by UI clients that want to show rewritten query
    immediately before running full downstream workflow execution.
    """
    original_query = (request.query or "").strip()
    rewrite_started = time.perf_counter()
    rewritten_query = rewrite_query(request)
    rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)
    rewrite_plan = None
    if (request.workflow or "auto").strip().lower() == "auto":
        rewrite_plan = build_execution_plan(request, rewritten_query)
    return RewriteResponse(
        original_query=original_query,
        rewritten_query=rewritten_query,
        rewrite_enabled=bool(request.rewrite_enable),
        rewrite_backend=_resolve_rewrite_backend(request),
        rewrite_time_ms=rewrite_time_ms,
        plan=rewrite_plan,
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Submit unified gateway query",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Handle unified gateway query with rewriting and routing.

    Steps:
        1) Rewrite query (placeholder for future LLM).
        2) Route to a workflow with confidence score.
        3) Call services layer (currently stubbed) for execution.

    Args:
        request: Parsed QueryRequest from the client.

    Returns:
        QueryResponse with answer from the selected workflow.
    """
    request_id = str(uuid.uuid4())
    original_query = (request.query or "").strip()
    rewritten_query = original_query
    rewrite_time_ms = 0
    route_source = "unknown"
    route_backend = None
    route_llm_confidence = None
    routing_confidence = 0.0
    workflow = (request.workflow or "auto").strip().lower() or "auto"
    route_input_query = rewritten_query

    try:
        # Step 1: rewrite query
        rewrite_started = time.perf_counter()
        rewritten_query = rewrite_query(request)
        rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)
        execution_plan = build_execution_plan(request, rewritten_query)
        route_input_query = _extract_route_input_query(execution_plan, original_query)

        if _is_rewrite_only_mode():
            debug_trace = _build_debug_trace(
                original_query=original_query,
                rewritten_query=rewritten_query,
                rewrite_time_ms=rewrite_time_ms,
                request=request,
                route_input_query=route_input_query,
                route_source="rewrite_only",
                route_backend=None,
                route_llm_confidence=None,
            )
            return QueryResponse(
                answer=rewritten_query,
                workflow="rewrite_only",
                routing_confidence=1.0,
                sources=[],
                request_id=request_id,
                error=None,
                debug=debug_trace,
            )

        # Step 2: route workflow (returns metadata for logging)
        planned_task_count = sum(len(group.tasks) for group in execution_plan.task_groups)
        if planned_task_count > 1:
            route_source = "planner"
            routing_confidence = 1.0
            workflow = _derive_workflow(execution_plan, [], fallback=workflow)
            route_backend = (request.route_backend or "").strip().lower() or None
            route_llm_confidence = None
        else:
            (
                workflow,
                routing_confidence,
                route_source,
                route_backend,
                route_llm_confidence,
            ) = route_workflow(route_input_query, request)
            # Keep workflow aligned with planned task when single task exists.
            if execution_plan.task_groups and execution_plan.task_groups[0].tasks:
                execution_plan.task_groups[0].tasks[0].workflow = workflow

        # Step 3: execute planned tasks.
        task_results = _execute_plan(execution_plan, request)
        merged_answer = _merge_task_answers(execution_plan, task_results)
        workflow = _derive_workflow(execution_plan, task_results, fallback=workflow)
        aggregated_sources: List[Dict[str, Any]] = []
        for result in task_results:
            if result.status == "completed":
                aggregated_sources.extend(result.sources)
        failed_tasks = [result for result in task_results if result.status == "failed"]
        top_error = None
        if failed_tasks and not merged_answer:
            top_error = failed_tasks[0].error or "All planned tasks failed."
        elif failed_tasks:
            top_error = f"{len(failed_tasks)} planned task(s) failed."

        meta = format_route_metadata(route_source, route_backend, route_llm_confidence)
        debug_trace = _build_debug_trace(
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_time_ms=rewrite_time_ms,
            request=request,
            route_input_query=route_input_query,
            route_source=route_source,
            route_backend=route_backend,
            route_llm_confidence=route_llm_confidence,
        )

        if top_error and not merged_answer:
            logger.warning(
                "Planned execution failed for request %s (workflow=%s): %s %s",
                request_id,
                workflow,
                top_error,
                meta,
            )
            return QueryResponse(
                answer="",
                workflow=workflow,
                routing_confidence=routing_confidence,
                sources=aggregated_sources,
                request_id=request_id,
                error=top_error,
                debug=debug_trace,
                plan=execution_plan,
                task_results=task_results,
                merged_answer="",
            )

        response = QueryResponse(
            answer=merged_answer,
            workflow=workflow,
            routing_confidence=routing_confidence,
            sources=aggregated_sources,
            request_id=request_id,
            error=top_error,
            debug=debug_trace,
            plan=execution_plan,
            task_results=task_results,
            merged_answer=merged_answer,
        )
        # main success log includes routing metadata
        logger.info(
            "Gateway handled request %s with workflow=%s (confidence=%.2f) %s",
            request_id,
            workflow,
            routing_confidence,
            meta,
        )
        return response
    except Exception as exc:  # pragma: no cover - defensive fallback
        # Defensive error handling so the API still returns a valid schema.
        logger.exception("Gateway query handler failed: %s", exc)
        return QueryResponse(
            answer="",
            workflow=workflow,
            routing_confidence=routing_confidence,
            sources=[],
            request_id=request_id,
            error=str(exc),
            debug=_build_debug_trace(
                original_query=original_query,
                rewritten_query=rewritten_query,
                rewrite_time_ms=rewrite_time_ms,
                request=request,
                route_input_query=route_input_query,
                route_source=route_source,
                route_backend=route_backend,
                route_llm_confidence=route_llm_confidence,
            ),
        )


__all__ = ["app"]

