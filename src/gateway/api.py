"""
Gateway FastAPI application (Dispatcher / Supervisor Agent).

Exposes REST API for the unified query gateway:
- POST /api/v1/query: accept QueryRequest, return QueryResponse
- POST /api/v1/rewrite: rewrite-only endpoint (no execution)
- GET /health: health check

Architecture: Route LLM (clarification, rewriters, router) does 3 steps: Clarification,
Rewriting (normalize, memory merge, rewrite with context), Intent classification.
Dispatcher (build_execution_plan, services) builds the plan and executes tasks.
This module (Dispatcher) receives the plan, executes tasks in parallel within groups,
invokes worker agents (RAG, Amazon docs RAG, SP-API Agent, UDS Agent), and merges results.
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

from .clarification import check_ambiguity
from .dispatcher import build_execution_plan
from .memory import GatewayConversationMemory
from .router import route_workflow, rewrite_query
from .routing_heuristics import (
    apply_docs_preference,
    format_rewritten_query_bullets,
    route_workflow_heuristic,
)
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

# Gateway short-term memory (Redis-backed). None if Redis unreachable.
gateway_memory: GatewayConversationMemory | None = None
try:
    import redis
    redis_url = os.getenv("GATEWAY_REDIS_URL", "redis://localhost:6379/0")
    _redis_client = redis.from_url(redis_url, decode_responses=True)
    _redis_client.ping()
    gateway_memory = GatewayConversationMemory(_redis_client)
    logger.info("Gateway memory initialized (Redis: %s)", redis_url.split("@")[-1] if "@" in redis_url else redis_url)
except Exception as exc:
    logger.warning("Gateway memory disabled (Redis unreachable): %s", exc)
    gateway_memory = None

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


@app.get("/api/v1/session/{session_id}")
async def get_session_history(session_id: str, last_n: int = 10) -> dict[str, Any]:
    """
    Return last N turns for a session (for UI or debugging).

    Requires gateway memory (Redis) to be enabled.
    """
    if not gateway_memory:
        return {"session_id": session_id, "history": [], "error": "Gateway memory disabled"}
    history = gateway_memory.get_history(session_id, last_n=min(last_n, 50))
    return {"session_id": session_id, "history": history}


@app.delete("/api/v1/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, Any]:
    """
    Clear session history (mirror SP-API).

    Requires gateway memory (Redis) to be enabled.
    """
    if not gateway_memory:
        return {"session_id": session_id, "cleared": False, "error": "Gateway memory disabled"}
    gateway_memory.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


def _call_workflow_backend(workflow: str, query_text: str, session_id: str | None) -> Dict[str, Any]:
    """Dispatcher: invoke one worker agent (RAG/UDS/SP-API) and return normalized dict."""
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
    """Execute one task group; tasks run in parallel when group.parallel is True (Dispatcher)."""
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
    """Dispatcher: execute plan; groups sequential, tasks within group in parallel."""
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
    """
    Return True when gateway runs in Route LLM-only mode (truncate downstream).

    When set, /api/v1/query returns after Route LLM (clarification + rewrite + intents) + plan building;
    no worker execution. Use for quick testing of Route LLM without RAG/UDS/SP-API.
    Env: GATEWAY_REWRITE_ONLY_MODE or GATEWAY_ROUTE_ONLY_MODE.
    """
    v = (
        os.getenv("GATEWAY_REWRITE_ONLY_MODE", "") or os.getenv("GATEWAY_ROUTE_ONLY_MODE", "")
    ).strip().lower()
    return v in ("1", "true", "yes", "on")


def _clarification_enabled() -> bool:
    """Return True when clarification check is enabled (runs before rewrite). Required by default."""
    v = os.getenv("GATEWAY_CLARIFICATION_ENABLED", "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


@app.post(
    "/api/v1/rewrite",
    response_model=RewriteResponse,
    summary="Route LLM only (no execution)",
)
async def rewrite(request: QueryRequest) -> RewriteResponse:
    """
    Run Route LLM pipeline only: clarification, rewriting, intent classification.
    No execution plan building (Dispatcher's job) or downstream worker execution.

    Route LLM steps: 1) Clarification, 2) Rewriting (normalize, memory merge,
    rewrite with context), 3) Intent classification.
    """
    original_query = (request.query or "").strip()

    # Clarification (required): return early if query is ambiguous
    if _clarification_enabled():
        backend = _resolve_rewrite_backend(request)
        ambiguity_result = check_ambiguity(original_query, backend)
        if ambiguity_result.get("needs_clarification"):
            clarification_question = ambiguity_result.get("clarification_question", "")
            if gateway_memory and request.session_id:
                try:
                    gateway_memory.save_turn(
                        request.session_id,
                        original_query,
                        clarification_question,
                        "clarification",
                    )
                except Exception as exc:
                    logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
            return RewriteResponse(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_enabled=bool(request.rewrite_enable),
                rewrite_backend=_resolve_rewrite_backend(request),
                rewrite_time_ms=0,
                plan=None,
                clarification_required=True,
                clarification_question=clarification_question,
                pending_query=original_query,
            )

    rewrite_started = time.perf_counter()
    rewritten_query, intents, memory_rounds, memory_text_length = rewrite_query(
        request, gateway_memory=gateway_memory
    )
    rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)
    rewritten_query_length = len(rewritten_query or "")

    # Derive workflow names from intents for UI display (lightweight heuristic).
    workflows: List[str] = []
    if intents:
        seen: set[str] = set()
        for intent in intents:
            q = (intent or "").strip()
            if not q:
                continue
            wf, _ = route_workflow_heuristic(q)
            wf = apply_docs_preference(q, wf)
            if wf and wf not in seen:
                seen.add(wf)
                workflows.append(wf)
    else:
        q = (rewritten_query or "").strip()
        if q:
            wf, _ = route_workflow_heuristic(q)
            wf = apply_docs_preference(q, wf)
            if wf:
                workflows = [wf]

    if gateway_memory and request.session_id:
        try:
            gateway_memory.save_turn(
                request.session_id,
                original_query,
                rewritten_query,
                "rewrite",
            )
        except Exception as exc:
            logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
    # Enforce bullet-point display for long/multi-part queries (LLM often ignores format).
    rewritten_query_display = format_rewritten_query_bullets(
        rewritten_query, intents=intents, min_length=60
    )
    return RewriteResponse(
        original_query=original_query,
        rewritten_query=rewritten_query,
        rewrite_enabled=bool(request.rewrite_enable),
        rewrite_backend=_resolve_rewrite_backend(request),
        rewrite_time_ms=rewrite_time_ms,
        plan=None,
        memory_rounds=memory_rounds,
        memory_text_length=memory_text_length,
        rewritten_query_length=rewritten_query_length,
        workflows=workflows if workflows else None,
        rewritten_query_display=rewritten_query_display,
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Submit unified gateway query",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Handle unified gateway query: Route LLM (planning) + Dispatcher (execution).

    Steps:
        1) Route LLM: clarification, rewriting (normalize, memory merge, rewrite), intent classification.
        2) Dispatcher: build execution plan, execute tasks in parallel, invoke worker agents.
        3) Merge task results and return QueryResponse.

    Args:
        request: Parsed QueryRequest from the client.

    Returns:
        QueryResponse with answer, plan, task_results, merged_answer.
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
        # Clarification check (required, before rewrite): return early if query is ambiguous
        if _clarification_enabled():
            backend = _resolve_rewrite_backend(request)
            ambiguity_result = check_ambiguity(original_query, backend)
            if ambiguity_result.get("needs_clarification"):
                clarification_question = ambiguity_result.get("clarification_question", "")
                if gateway_memory and request.session_id:
                    try:
                        gateway_memory.save_turn(
                            request.session_id,
                            original_query,
                            clarification_question,
                            "clarification",
                        )
                    except Exception as exc:
                        logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
                return QueryResponse(
                    answer=clarification_question,
                    workflow="clarification",
                    routing_confidence=0.0,
                    sources=[],
                    request_id=request_id,
                    error=None,
                    debug={
                        "original_query": original_query,
                        "clarification_required": True,
                    },
                    clarification_required=True,
                    clarification_question=clarification_question,
                    pending_query=original_query,
                )

        # Route LLM: rewrite + intent classification -> rewritten_query, intents
        # Dispatcher: build_execution_plan -> execution_plan
        rewrite_started = time.perf_counter()
        rewritten_query, intents, _, _ = rewrite_query(
            request, gateway_memory=gateway_memory
        )
        rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)

        # Skip downstream when normalized query is empty.
        if not rewritten_query or not rewritten_query.strip():
            return QueryResponse(
                answer="Please provide a non-empty query.",
                workflow="general",
                routing_confidence=0.0,
                sources=[],
                request_id=request_id,
                error=None,
                debug={
                    "original_query": original_query,
                    "rewritten_query": "",
                    "clarification_required": False,
                },
            )

        execution_plan = build_execution_plan(request, rewritten_query, intents=intents)
        route_input_query = _extract_route_input_query(execution_plan, original_query)

        if _is_rewrite_only_mode():
            if gateway_memory and request.session_id:
                try:
                    gateway_memory.save_turn(
                        request.session_id,
                        original_query,
                        rewritten_query,
                        "rewrite_only",
                    )
                except Exception as exc:
                    logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
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
                plan=execution_plan,
                task_results=[],
                merged_answer="",
            )

        # Route metadata for logging (single-task path uses route_workflow)
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

        # Dispatcher (Execution): execute tasks in parallel, invoke worker agents
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
            if gateway_memory and request.session_id:
                try:
                    gateway_memory.save_turn(
                        request.session_id,
                        original_query,
                        top_error or "error",
                        workflow,
                    )
                except Exception as exc:
                    logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
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
        # Save turn to short-term memory (session history) when applicable
        if gateway_memory and request.session_id and merged_answer and workflow not in ("clarification", "rewrite_only"):
            try:
                gateway_memory.save_turn(
                    request.session_id,
                    original_query,
                    merged_answer,
                    workflow,
                )
            except Exception as exc:
                logger.warning("Gateway memory save_turn failed (non-fatal): %s", exc)
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
        # Persist error turn to Redis so chat box content is complete
        if gateway_memory and request.session_id:
            try:
                gateway_memory.save_turn(
                    request.session_id,
                    original_query,
                    str(exc),
                    "error",
                )
            except Exception as save_exc:
                logger.warning("Gateway memory save_turn failed (non-fatal): %s", save_exc)
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

