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

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..dispatcher.dispatcher import build_execution_plan
from ..logging_utils import format_route_metadata
from ..route_llm.clarification import clarification_enabled, load_clarification_context, check_ambiguity
from ..route_llm.rewriting.router import route_workflow, rewrite_query
from ..schemas import (
    QueryRequest,
    QueryResponse,
    RewritePlan,
    RewriteResponse,
    TaskExecutionResult,
)
from .auth import AuthGuard
from .auth_routes import router as auth_router
from .debug_trace import DebugTraceBuilder
from .dispatcher import DispatcherExecutor
from .gateway_config import GatewayConfig
from .gateway_logger import GatewayEventLogger
from .intent_rewrite import IntentDetailsBuilder
from ..message import (
    get_gateway_memory,
    ConversationHistoryHandler,
    MemoryEventWriter,
    TurnSummaryPersistence,
)
from .plan_helper import PlanHelper
from src.logger import get_logger_facade

logger = logging.getLogger(__name__)


async def _get_optional_user_if_required(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict | None:
    """FastAPI Depends: inject Authorization header and delegate to AuthGuard."""
    return await AuthGuard.get_optional_user(authorization)


# Gateway short-term memory (Redis-backed). None if Redis unreachable.
gateway_memory = get_gateway_memory()

# Unified logger facade (short-term Redis + long-term ClickHouse). Best effort.
gateway_logger = None
try:
    gateway_logger = get_logger_facade()
except Exception as exc:
    logger.warning("Gateway logger disabled (init failed): %s", exc)
    gateway_logger = None
if gateway_logger is not None:
    GatewayEventLogger.set_facade(gateway_logger)


# ---------------------------------------------------------------------------
# FastAPI app and route handlers (thin layer; logic in auth, message, dispatcher, etc.)
# ---------------------------------------------------------------------------

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

app.include_router(auth_router)


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
    return ConversationHistoryHandler.get_session_history(
        gateway_memory,
        session_id,
        min(last_n, 50),
    )


@app.post(
    "/api/v1/rewrite",
    response_model=RewriteResponse,
    summary="Route LLM only (no execution)",
)
async def rewrite(
    request: QueryRequest,
    _user: dict | None = Depends(_get_optional_user_if_required),
) -> RewriteResponse:
    """
    Run Route LLM pipeline only: clarification, rewriting, intent classification.
    No execution plan building (Dispatcher's job) or downstream worker execution.

    Route LLM steps: 1) Clarification, 2) Rewriting (normalize, memory merge,
    rewrite with context), 3) Intent classification.
    """
    request_id = str(uuid.uuid4())
    original_query = (request.query or "").strip()
    effective_user_id = AuthGuard.resolve_user_id(request, _user)
    clarification_context: str | None = None
    GatewayEventLogger.log_runtime(
        event_name="gateway_rewrite_request",
        stage="rewrite",
        message="rewrite endpoint request started",
        status="started",
        request_id=request_id,
        session_id=request.session_id,
        user_id=effective_user_id,
        workflow=request.workflow or "auto",
        query_raw=original_query,
    )

    # Clarification (required): return early if query is ambiguous
    clarification_status: str = "Skip"
    clarification_backend: str | None = None
    try:
        if GatewayConfig.clarification_enabled():
            clarification_context = load_clarification_context(gateway_memory, request.session_id)
            clarification_raw = check_ambiguity(original_query, conversation_context=clarification_context)
            clarification_backend = clarification_raw.get("clarification_backend")
            if clarification_raw.get("needs_clarification"):
                clarification_question = clarification_raw.get("clarification_question") or ""
                GatewayEventLogger.log_interaction(
                    event_name="gateway_rewrite_clarification",
                    status="success",
                    request_id=request_id,
                    session_id=request.session_id,
                    user_id=effective_user_id,
                    workflow="clarification",
                    query_raw=original_query,
                    clarification_question=clarification_question,
                    answer=clarification_question,
                    latency_ms=0,
                )
                # NOTE: /rewrite is a preview endpoint — do NOT save to Redis here.
                # The /query endpoint is responsible for all conversation logging.
                return RewriteResponse(
                    original_query=original_query,
                    rewritten_query=original_query,
                    rewrite_enabled=bool(request.rewrite_enable),
                    rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
                    rewrite_time_ms=0,
                    plan=None,
                    clarification_required=True,
                    clarification_question=clarification_question,
                    pending_query=original_query,
                    clarification_status="Required",
                    clarification_backend=clarification_backend,
                )

        # Ensure router receives user_id for memory merge (from JWT or request body)
        req_for_rewrite = request.model_copy(update={"user_id": effective_user_id}) if effective_user_id else request
        rewrite_started = time.perf_counter()
        rewritten_query, _, memory_rounds, memory_text_length = rewrite_query(
            req_for_rewrite, gateway_memory=gateway_memory, conversation_context=clarification_context
        )
        # Hard guard: rewrite stage must return a single line; intent splitting happens later.
        rewritten_query = re.sub(r"\s+", " ", (rewritten_query or "")).strip()
        rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)
        rewritten_query_length = len(rewritten_query or "")

        intents, intent_details, workflows = IntentDetailsBuilder.build_intent_details(
            rewritten_query
        )

        # NOTE: /rewrite is a preview endpoint — do NOT save to Redis here.
        # The /query endpoint is responsible for all conversation logging.

        # Rewrite stage outputs one sentence only; do not show as bullet list (splitting is intent-classification step).
        rewritten_query_display = None
        GatewayEventLogger.log_interaction(
            event_name="gateway_rewrite_preview",
            status="success",
            request_id=request_id,
            session_id=request.session_id,
            user_id=effective_user_id,
            workflow="rewrite_only_preview",
            query_raw=original_query,
            query_rewritten=rewritten_query,
            intent_list=intents or [],
            intent_details=intent_details or [],
            answer=rewritten_query,
            latency_ms=rewrite_time_ms,
        )
        if GatewayConfig.clarification_enabled():
            clarification_status = "Complete"
        return RewriteResponse(
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_enabled=bool(request.rewrite_enable),
            rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
            rewrite_time_ms=rewrite_time_ms,
            plan=None,
            clarification_status=clarification_status,
            clarification_backend=clarification_backend,
            memory_rounds=memory_rounds,
            memory_text_length=memory_text_length,
            rewritten_query_length=rewritten_query_length,
            intents=intents if intents else None,
            intent_details=intent_details if intent_details else None,
            workflows=workflows if workflows else None,
            rewritten_query_display=rewritten_query_display,
        )

    except Exception as exc:
        logger.exception("Rewrite endpoint failed: %s", exc)
        return RewriteResponse(
            original_query=original_query,
            rewritten_query=original_query,
            rewrite_enabled=bool(request.rewrite_enable),
            rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
            rewrite_time_ms=0,
            plan=None,
            error=str(exc),
        )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Submit unified gateway query",
)
async def query(
    request: QueryRequest,
    _user: dict | None = Depends(_get_optional_user_if_required),
) -> QueryResponse:
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
    effective_user_id = AuthGuard.resolve_user_id(request, _user)
    clarification_context: str | None = None
    rewritten_query = original_query
    rewrite_time_ms = 0
    route_source = "unknown"
    route_backend = None
    route_llm_confidence = None
    routing_confidence = 0.0
    workflow = (request.workflow or "auto").strip().lower() or "auto"
    route_input_query = rewritten_query
    GatewayEventLogger.log_runtime(
        event_name="gateway_query_request",
        stage="query",
        message="query endpoint request started",
        status="started",
        request_id=request_id,
        session_id=request.session_id,
        user_id=effective_user_id,
        workflow=request.workflow or "auto",
        query_raw=original_query,
    )
    MemoryEventWriter.append_event(
        gateway_memory,
        user_id=effective_user_id,
        session_id=request.session_id,
        request_id=request_id,
        event_type="user_query",
        event_content={"query": original_query, "workflow": request.workflow or "auto"},
        status="ok",
    )

    try:
        # Clarification check (required, before rewrite): return early if query is ambiguous
        if GatewayConfig.clarification_enabled():
            clarification_context = load_clarification_context(gateway_memory, request.session_id)
            clarification_raw = check_ambiguity(original_query, conversation_context=clarification_context)
            if clarification_raw.get("needs_clarification"):
                clarification_question = clarification_raw.get("clarification_question") or ""
                MemoryEventWriter.append_event(
                    gateway_memory,
                    user_id=effective_user_id,
                    session_id=request.session_id,
                    request_id=request_id,
                    event_type="query_clarification",
                    event_content={"query": original_query, "clarification_question": clarification_question},
                    status="ok",
                )
                GatewayEventLogger.log_interaction(
                    event_name="gateway_query_clarification",
                    status="success",
                    request_id=request_id,
                    session_id=request.session_id,
                    user_id=effective_user_id,
                    workflow="clarification",
                    query_raw=original_query,
                    clarification_question=clarification_question,
                    answer=clarification_question,
                    latency_ms=0,
                )
                if gateway_memory and effective_user_id:
                    TurnSummaryPersistence.persist_turn(
                        gateway_memory,
                        user_id=effective_user_id,
                        session_id=request.session_id or "",
                        request_id=request_id,
                        query=original_query,
                        answer=clarification_question,
                        workflow="clarification",
                    )
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
                    clarification_backend=clarification_raw.get("clarification_backend"),
                )

        # Route LLM: rewrite -> rewritten_query; then intent splitting when enabled.
        req_for_rewrite = request.model_copy(update={"user_id": effective_user_id}) if effective_user_id else request
        rewrite_started = time.perf_counter()
        rewritten_query, _, _, _ = rewrite_query(
            req_for_rewrite, gateway_memory=gateway_memory, conversation_context=clarification_context
        )
        rewritten_query = re.sub(r"\s+", " ", (rewritten_query or "")).strip()
        rewrite_time_ms = int((time.perf_counter() - rewrite_started) * 1000)
        MemoryEventWriter.append_event(
            gateway_memory,
            user_id=effective_user_id,
            session_id=request.session_id,
            request_id=request_id,
            event_type="query_rewriting",
            event_content={"original_query": original_query, "rewritten_query": rewritten_query},
            status="ok",
        )

        # Intent classification (plan: split then dual retrieval per clause).
        # When enabled, split rewritten query into sub-questions and pass to dispatcher.
        intents = None
        if rewritten_query and rewritten_query.strip():
            if os.getenv("GATEWAY_INTENT_CLASSIFICATION_ENABLED", "").lower() in ("1", "true", "yes", "on"):
                try:
                    from ..route_llm.classification import split_intents
                    intents = split_intents(rewritten_query)
                except Exception as exc:
                    logger.warning("Intent split failed (non-fatal): %s", exc)
        MemoryEventWriter.append_event(
            gateway_memory,
            user_id=effective_user_id,
            session_id=request.session_id,
            request_id=request_id,
            event_type="intent_classification",
            event_content={"rewritten_query": rewritten_query, "intents": intents or []},
            status="ok",
        )

        # Skip downstream when normalized query is empty.
        if not rewritten_query or not rewritten_query.strip():
            empty_answer = "Please provide a non-empty query."
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="llm_answer",
                event_content={"answer": empty_answer, "workflow": "general"},
                status="failed",
                note="empty rewritten query",
            )
            if gateway_memory and effective_user_id:
                TurnSummaryPersistence.persist_turn(
                    gateway_memory,
                    user_id=effective_user_id,
                    session_id=request.session_id or "",
                    request_id=request_id,
                    query=original_query,
                    answer=empty_answer,
                    workflow="general",
                )
            return QueryResponse(
                answer=empty_answer,
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

        plan_result = build_execution_plan(
            request, rewritten_query, intents=intents, conversation_context=clarification_context
        )
        if isinstance(plan_result, tuple):
            execution_plan, field_clarification = plan_result
        else:
            execution_plan, field_clarification = plan_result, None

        # Phase 5: per-intent required field validation — return clarification if fields missing.
        if field_clarification:
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="query_clarification",
                event_content={"query": original_query, "clarification_question": field_clarification},
                status="ok",
                note="missing required fields",
            )
            if gateway_memory and effective_user_id:
                TurnSummaryPersistence.persist_turn(
                    gateway_memory,
                    user_id=effective_user_id,
                    session_id=request.session_id or "",
                    request_id=request_id,
                    query=original_query,
                    answer=field_clarification,
                    workflow="clarification",
                )
            return QueryResponse(
                answer=field_clarification,
                workflow="clarification",
                routing_confidence=0.0,
                sources=[],
                request_id=request_id,
                error=None,
                debug={"original_query": original_query, "clarification_required": True},
                clarification_required=True,
                clarification_question=field_clarification,
                pending_query=original_query,
            )

        route_input_query = PlanHelper.extract_route_input_query(execution_plan, original_query)

        if GatewayConfig.is_rewrite_only_mode():
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="llm_answer",
                event_content={"answer": rewritten_query, "workflow": "rewrite_only"},
                status="ok",
                note="rewrite-only mode",
            )
            if gateway_memory and effective_user_id:
                TurnSummaryPersistence.persist_turn(
                    gateway_memory,
                    user_id=effective_user_id,
                    session_id=request.session_id or "",
                    request_id=request_id,
                    query=original_query,
                    answer=rewritten_query,
                    workflow="rewrite_only",
                )
            debug_trace = DebugTraceBuilder.build_debug_trace(
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
            workflow = PlanHelper.derive_workflow(execution_plan, [], fallback=workflow)
            route_backend = None
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
        task_results = DispatcherExecutor.execute_plan(execution_plan, request)
        merged_answer = PlanHelper.merge_task_answers(execution_plan, task_results)
        workflow = PlanHelper.derive_workflow(execution_plan, task_results, fallback=workflow)
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
        debug_trace = DebugTraceBuilder.build_debug_trace(
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
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="llm_answer",
                event_content={"answer": "", "workflow": workflow, "error": top_error},
                status="failed",
                note="planned tasks failed",
            )
            GatewayEventLogger.log_error(
                event_name="gateway_query_failed",
                request_id=request_id,
                session_id=request.session_id,
                user_id=effective_user_id,
                workflow=workflow,
                query_raw=original_query,
                query_rewritten=rewritten_query,
                error_type="PlannedTaskFailure",
                error_message=top_error or "All planned tasks failed.",
                metadata={"failed_tasks": [t.task_id for t in failed_tasks]},
            )
            if gateway_memory and effective_user_id:
                TurnSummaryPersistence.persist_turn(
                    gateway_memory,
                    user_id=effective_user_id,
                    session_id=request.session_id or "",
                    request_id=request_id,
                    query=original_query,
                    answer=top_error or "error",
                    workflow=workflow,
                )
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
        # Save turn to short-term memory (user history) when applicable
        if gateway_memory and effective_user_id and merged_answer and workflow not in ("clarification", "rewrite_only"):
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="llm_answer",
                event_content={"answer": merged_answer, "workflow": workflow},
                status="ok",
            )
            TurnSummaryPersistence.persist_turn(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id or "",
                request_id=request_id,
                query=original_query,
                answer=merged_answer,
                workflow=workflow,
            )
        # main success log includes routing metadata
        logger.info(
            "Gateway handled request %s with workflow=%s (confidence=%.2f) %s",
            request_id,
            workflow,
            routing_confidence,
            meta,
        )
        GatewayEventLogger.log_interaction(
            event_name="gateway_query_success",
            status="success",
            request_id=request_id,
            session_id=request.session_id,
            user_id=effective_user_id,
            workflow=workflow,
            query_raw=original_query,
            query_rewritten=rewritten_query,
            answer=merged_answer,
            latency_ms=rewrite_time_ms,
        )
        return response
    except Exception as exc:  # pragma: no cover - defensive fallback
        # Defensive error handling so the API still returns a valid schema.
        logger.exception("Gateway query handler failed: %s", exc)
        GatewayEventLogger.log_error(
            event_name="gateway_query_exception",
            request_id=request_id,
            session_id=request.session_id,
            user_id=effective_user_id,
            workflow=workflow,
            query_raw=original_query,
            query_rewritten=rewritten_query,
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"route_source": route_source},
        )
        # Persist error turn to Redis so chat box content is complete
        if gateway_memory and effective_user_id:
            MemoryEventWriter.append_event(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id,
                request_id=request_id,
                event_type="llm_answer",
                event_content={"answer": "", "workflow": "error", "error": str(exc)},
                status="failed",
                note=type(exc).__name__,
            )
            TurnSummaryPersistence.persist_turn(
                gateway_memory,
                user_id=effective_user_id,
                session_id=request.session_id or "",
                request_id=request_id,
                query=original_query,
                answer=str(exc),
                workflow="error",
            )
        return QueryResponse(
            answer="",
            workflow=workflow,
            routing_confidence=routing_confidence,
            sources=[],
            request_id=request_id,
            error=str(exc),
            debug=DebugTraceBuilder.build_debug_trace(
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

