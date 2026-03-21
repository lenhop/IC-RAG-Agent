"""
Gateway FastAPI application — routes + pipeline orchestration.

Exposes REST API for the unified query gateway:
- POST /api/v1/query   → QueryPipeline.run()
- POST /api/v1/rewrite → RewritePipeline.run()
- GET  /health         → health check
- GET  /api/v1/session → conversation history

Pipeline classes (RewritePipeline, QueryPipeline) orchestrate the full
request lifecycle; pure helpers live in view_helpers.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from ..dispatcher import build_execution_plan, DispatcherExecutor
from ..route_llm.clarification import check_ambiguity
from ..route_llm.rewriting.rewriters import rewrite_and_route
from ..schemas import (
    QueryRequest,
    QueryResponse,
    RewriteResponse,
)
from ..message import (
    get_gateway_memory,
    ConversationHistoryHandler,
    MemoryEventWriter,
    TurnSummaryPersistence,
)
from .auth import AuthGuard, router as auth_router
from .config import GatewayConfig, GatewayEventLogger
from .view_helpers import (
    DebugTraceBuilder,
    IntentDetailsBuilder,
    PlanHelper,
)
from src.logger import format_route_metadata, get_logger_facade
from src.retrieval.query_process import QueryProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gateway singletons (module-level init)
# ---------------------------------------------------------------------------

async def _get_optional_user(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict | None:
    """FastAPI Depends: inject Authorization header and delegate to AuthGuard."""
    return await AuthGuard.get_optional_user(authorization)


# Gateway short-term memory (Redis-backed). None if Redis unreachable.
gateway_memory = get_gateway_memory()

# Unified logger facade (short-term Redis + long-term ClickHouse). Best effort.
_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception as exc:
    logger.warning("Gateway logger disabled (init failed): %s", exc)
if _gateway_logger is not None:
    GatewayEventLogger.set_facade(_gateway_logger)


# ---------------------------------------------------------------------------
# FastAPI app
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


# ---------------------------------------------------------------------------
# Shared internal helpers for pipelines
# ---------------------------------------------------------------------------

_CLARIFICATION_MEMORY_ROUNDS = 3


def _load_clarification_context(memory: Any, session_id: str | None) -> str | None:
    """Load last-N conversation history from Redis, formatted as markdown."""
    sid = (session_id or "").strip()
    if not sid or not memory:
        return None
    try:
        n = int(os.getenv("GATEWAY_CLARIFICATION_MEMORY_ROUNDS", str(_CLARIFICATION_MEMORY_ROUNDS)))
    except (TypeError, ValueError):
        n = _CLARIFICATION_MEMORY_ROUNDS
    last_n = min(max(n, 1), 50)
    res = ConversationHistoryHandler.get_session_history(memory, sid, last_n=last_n)
    history = res.get("history") or []
    return ConversationHistoryHandler.format_history_for_llm_markdown(history) if history else None


def _run_clarification(
    original_query: str, session_id: str | None, memory: Any,
) -> tuple[str | None, dict | None]:
    """Return (context, raw) — raw is non-None only when clarification is needed."""
    if not GatewayConfig.clarification_enabled():
        return None, None
    context = _load_clarification_context(memory, session_id)
    raw = check_ambiguity(original_query, conversation_context=context)
    if raw.get("needs_clarification"):
        return context, raw
    return context, None


def _log_and_persist(
    memory: Any, *,
    event_type: str, content: dict,
    status: str = "ok", note: str | None = None,
    user_id: str | None, session_id: str | None, request_id: str,
    original_query: str, answer: str, workflow: str,
    persist: bool = True,
) -> None:
    """Unified MemoryEventWriter.append_event + optional TurnSummaryPersistence.persist_turn."""
    MemoryEventWriter.append_event(
        memory, user_id=user_id, session_id=session_id, request_id=request_id,
        event_type=event_type, event_content=content, status=status, note=note,
    )
    if persist and memory and user_id:
        TurnSummaryPersistence.persist_turn(
            memory, user_id=user_id, session_id=session_id or "",
            request_id=request_id, query=original_query, answer=answer, workflow=workflow,
        )


def _classify_intents_batch_safe(
    intents: list[str] | None,
    conversation_context: str | None,
) -> list[dict[str, Any]] | None:
    """Run classify_intents_batch; never raises."""
    if not intents:
        return None
    try:
        from ..route_llm.classification import classify_intents_batch

        return classify_intents_batch(intents, conversation_context)
    except Exception as exc:
        logger.warning("Intent batch classification failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# RewritePipeline — /api/v1/rewrite orchestration
# ---------------------------------------------------------------------------


class RewritePipeline:
    """
    Orchestrates the /rewrite preview endpoint:
      1) Clarification check (early return if ambiguous)
      2) Rewrite (normalize, memory merge, rewrite with context)
      3) Intent classification preview
    """

    @classmethod
    def run(cls, request: QueryRequest, user_payload: dict | None, memory: Any) -> RewriteResponse:
        request_id = str(uuid.uuid4())
        original_query = (request.query or "").strip()
        uid = AuthGuard.resolve_user_id(request, user_payload)

        GatewayEventLogger.log_runtime(
            event_name="gateway_rewrite_request", stage="rewrite",
            message="rewrite endpoint request started", status="started",
            request_id=request_id, session_id=request.session_id,
            user_id=uid, workflow=request.workflow or "auto", query_raw=original_query,
        )

        clarification_status: str = "Skip"
        clarification_backend: str | None = None
        try:
            # 1. Clarification
            clar_start = time.perf_counter()
            ctx, clar_raw = _run_clarification(original_query, request.session_id, memory)
            clar_elapsed_ms = int((time.perf_counter() - clar_start) * 1000)
            logger.info("[Perf] rewrite endpoint clarification: %d ms", clar_elapsed_ms)
            GatewayEventLogger.log_runtime(
                event_name="gateway_query_stage",
                stage="clarification",
                message="clarification completed",
                status="success",
                request_id=request_id,
                session_id=request.session_id,
                user_id=uid,
                workflow=request.workflow or "auto",
                latency_ms=clar_elapsed_ms,
            )
            if clar_raw:
                q = clar_raw.get("clarification_question") or ""
                clarification_backend = clar_raw.get("clarification_backend")
                GatewayEventLogger.log_interaction(
                    event_name="gateway_rewrite_clarification", status="success",
                    request_id=request_id, session_id=request.session_id,
                    user_id=uid, workflow="clarification", query_raw=original_query,
                    clarification_question=q, answer=q, latency_ms=0,
                )
                return RewriteResponse(
                    original_query=original_query, rewritten_query=original_query,
                    rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
                    rewrite_time_ms=0, plan=None,
                    clarification_required=True, clarification_question=q,
                    pending_query=original_query, clarification_status="Required",
                    clarification_backend=clarification_backend,
                    clarification_time_ms=clar_elapsed_ms,
                )

            # 2. Unified rewrite (JSON: display + intents) + 3. Intent preview
            req = request.model_copy(update={"user_id": uid}) if uid else request
            started = time.perf_counter()
            rr = rewrite_and_route(
                req,
                gateway_memory=memory,
                conversation_context=ctx,
                enable_routing=False,
            )
            rewritten_query = QueryProcessor.normalize(rr[0])
            memory_rounds = rr[2]
            memory_text_length = rr[3]
            intents_override = list(rr[9])
            rewrite_time_ms = int((time.perf_counter() - started) * 1000)
            logger.info("[Perf] rewrite endpoint rewrite: %d ms", rewrite_time_ms)

            intent_start = time.perf_counter()
            intents, intent_details, workflows = IntentDetailsBuilder.build_intent_details(
                rewritten_query, intents_override=intents_override
            )
            classification_time_ms = int((time.perf_counter() - intent_start) * 1000)
            logger.info("[Perf] rewrite endpoint intent_classification: %d ms", classification_time_ms)

            GatewayEventLogger.log_interaction(
                event_name="gateway_rewrite_preview", status="success",
                request_id=request_id, session_id=request.session_id,
                user_id=uid, workflow="rewrite_only_preview",
                query_raw=original_query, query_rewritten=rewritten_query,
                intent_list=intents or [], intent_details=intent_details or [],
                answer=rewritten_query, latency_ms=rewrite_time_ms,
            )
            if GatewayConfig.clarification_enabled():
                clarification_status = "Complete"

            return RewriteResponse(
                original_query=original_query, rewritten_query=rewritten_query,
                rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
                rewrite_time_ms=rewrite_time_ms, plan=None,
                clarification_status=clarification_status,
                clarification_backend=clarification_backend,
                memory_rounds=memory_rounds, memory_text_length=memory_text_length,
                rewritten_query_length=len(rewritten_query or ""),
                intents=intents or None, intent_details=intent_details or None,
                workflows=workflows or None, rewritten_query_display=None,
                clarification_time_ms=clar_elapsed_ms,
                classification_time_ms=classification_time_ms,
            )
        except Exception as exc:
            logger.exception("Rewrite endpoint failed: %s", exc)
            return RewriteResponse(
                original_query=original_query, rewritten_query=original_query,
                rewrite_backend=GatewayConfig.resolve_rewrite_backend(request),
                rewrite_time_ms=0, plan=None, error=str(exc),
            )


# ---------------------------------------------------------------------------
# QueryPipeline — /api/v1/query orchestration
# ---------------------------------------------------------------------------


class QueryPipeline:
    """
    Orchestrates the /query endpoint:
      1) Clarification     → early return if ambiguous
      2) Unified rewrite   → one LLM (JSON: display + intents) + memory merge
      3) Batch classify    → classify_intents_batch(intents)
      4) Build plan        → execution plan + field validation
      5) Route             → determine workflow / route metadata
      6) Execute           → Dispatcher runs worker agents
      7) Merge + respond   → aggregate results, persist, return
    """

    @classmethod
    def run(cls, request: QueryRequest, user_payload: dict | None, memory: Any) -> QueryResponse:
        request_id = str(uuid.uuid4())
        original_query = (request.query or "").strip()
        uid = AuthGuard.resolve_user_id(request, user_payload)
        rewritten_query = original_query
        rewrite_time_ms = 0
        route_source = "unknown"
        route_backend: str | None = None
        route_llm_confidence: float | None = None
        routing_confidence = 0.0
        workflow = (request.workflow or "auto").strip().lower() or "auto"
        route_input_query = rewritten_query

        # Shared kwargs for _log_and_persist
        _lp: dict = dict(
            user_id=uid, session_id=request.session_id,
            request_id=request_id, original_query=original_query,
        )

        GatewayEventLogger.log_runtime(
            event_name="gateway_query_request", stage="query",
            message="query endpoint request started", status="started",
            request_id=request_id, session_id=request.session_id,
            user_id=uid, workflow=request.workflow or "auto", query_raw=original_query,
        )
        MemoryEventWriter.append_event(
            memory, user_id=uid, session_id=request.session_id,
            request_id=request_id, event_type="user_query",
            event_content={"query": original_query, "workflow": request.workflow or "auto"},
            status="ok",
        )

        try:
            total_start = time.perf_counter()

            # 1. Clarification
            clar_start = time.perf_counter()
            ctx, clar_raw = _run_clarification(original_query, request.session_id, memory)
            clar_elapsed_ms = int((time.perf_counter() - clar_start) * 1000)
            logger.info("[Perf] query endpoint clarification: %d ms", clar_elapsed_ms)
            if clar_raw:
                q = clar_raw.get("clarification_question") or ""
                _log_and_persist(
                    memory,
                    event_type="query_clarification",
                    content={"query": original_query, "clarification_question": q},
                    answer=q,
                    workflow="clarification",
                    **_lp,
                )
                GatewayEventLogger.log_interaction(
                    event_name="gateway_query_clarification",
                    status="success",
                    request_id=request_id,
                    session_id=request.session_id,
                    user_id=uid,
                    workflow="clarification",
                    query_raw=original_query,
                    clarification_question=q,
                    answer=q,
                    latency_ms=0,
                )
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    "[Perf] Total pipeline use time (clarification early return): %.3fs",
                    total_elapsed,
                )
                return QueryResponse(
                    answer=q,
                    workflow="clarification",
                    routing_confidence=0.0,
                    sources=[],
                    request_id=request_id,
                    error=None,
                    debug={"original_query": original_query, "clarification_required": True},
                    clarification_required=True,
                    clarification_question=q,
                    pending_query=original_query,
                    clarification_backend=clar_raw.get("clarification_backend"),
                )

            # 2. Unified rewrite + 3. Batch intent classification
            rewrite_start = time.perf_counter()
            req_qp = request.model_copy(update={"user_id": uid}) if uid else request
            classified_intents: list[dict[str, Any]] | None = None

            rr = rewrite_and_route(
                req_qp,
                gateway_memory=memory,
                conversation_context=ctx,
                enable_routing=False,
            )
            rewritten_query = QueryProcessor.normalize(rr[0])
            memory_rounds_q = rr[2]
            memory_text_length_q = rr[3]
            intents: list[str] | None = list(rr[9]) if rr[9] else None
            rewrite_time_ms = int((time.perf_counter() - rewrite_start) * 1000)
            MemoryEventWriter.append_event(
                memory,
                user_id=uid,
                session_id=request.session_id,
                request_id=request_id,
                event_type="query_rewriting",
                event_content={
                    "original_query": original_query,
                    "rewritten_query": rewritten_query,
                },
                status="ok",
            )
            GatewayEventLogger.log_runtime(
                event_name="gateway_query_stage",
                stage="rewrite",
                message="unified rewrite completed",
                status="success",
                request_id=request_id,
                session_id=request.session_id,
                user_id=uid,
                workflow=request.workflow or "auto",
                latency_ms=rewrite_time_ms,
            )
            logger.info("[Perf] query endpoint rewrite: %d ms", rewrite_time_ms)

            intent_start = time.perf_counter()
            classified_intents = _classify_intents_batch_safe(intents, ctx)
            intent_elapsed_ms = int((time.perf_counter() - intent_start) * 1000)

            MemoryEventWriter.append_event(
                memory,
                user_id=uid,
                session_id=request.session_id,
                request_id=request_id,
                event_type="intent_classification",
                event_content={
                    "rewritten_query": rewritten_query,
                    "intents": intents or [],
                    "classified_intents": classified_intents or [],
                },
                status="ok",
            )
            GatewayEventLogger.log_runtime(
                event_name="gateway_query_stage",
                stage="intent_classification",
                message="intent classification completed",
                status="success",
                request_id=request_id,
                session_id=request.session_id,
                user_id=uid,
                workflow=request.workflow or "auto",
                latency_ms=intent_elapsed_ms,
            )
            logger.info("[Perf] query endpoint intent_classification: %d ms", intent_elapsed_ms)

            # 4. Empty guard
            if not (rewritten_query or "").strip():
                ans = "Please provide a non-empty query."
                _log_and_persist(
                    memory,
                    event_type="llm_answer",
                    content={"answer": ans, "workflow": "general"},
                    status="failed",
                    note="empty rewritten query",
                    answer=ans,
                    workflow="general",
                    **_lp,
                )
                return QueryResponse(
                    answer=ans,
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

            # 5. Build execution plan
            plan_start = time.perf_counter()
            plan_result = build_execution_plan(
                request,
                rewritten_query,
                intents=intents,
                conversation_context=ctx,
                classified_intents=classified_intents,
            )
            if isinstance(plan_result, tuple):
                execution_plan, field_clar = plan_result
            else:
                execution_plan, field_clar = plan_result, None

            if field_clar:
                _log_and_persist(
                    memory,
                    event_type="query_clarification",
                    content={"query": original_query, "clarification_question": field_clar},
                    note="missing required fields",
                    answer=field_clar,
                    workflow="clarification",
                    **_lp,
                )
                plan_elapsed_ms = int((time.perf_counter() - plan_start) * 1000)
                logger.info("[Perf] query endpoint plan_build (field_clar): %d ms", plan_elapsed_ms)
                GatewayEventLogger.log_runtime(
                    event_name="gateway_query_stage",
                    stage="plan_build",
                    message="plan build returned field clarification",
                    status="success",
                    request_id=request_id,
                    session_id=request.session_id,
                    user_id=uid,
                    workflow="clarification",
                    latency_ms=plan_elapsed_ms,
                )
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    "[Perf] Total pipeline use time (field clarification early return): %.3fs",
                    total_elapsed,
                )
                return QueryResponse(
                    answer=field_clar,
                    workflow="clarification",
                    routing_confidence=0.0,
                    sources=[],
                    request_id=request_id,
                    error=None,
                    debug={"original_query": original_query, "clarification_required": True},
                    clarification_required=True,
                    clarification_question=field_clar,
                    pending_query=original_query,
                )

            route_input_query = PlanHelper.extract_route_input_query(
                execution_plan,
                original_query,
            )

            # 6. Rewrite-only mode shortcut
            if GatewayConfig.is_rewrite_only_mode():
                _log_and_persist(
                    memory,
                    event_type="llm_answer",
                    content={"answer": rewritten_query, "workflow": "rewrite_only"},
                    note="rewrite-only mode",
                    answer=rewritten_query,
                    workflow="rewrite_only",
                    **_lp,
                )
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    "[Perf] Total pipeline use time (rewrite-only mode): %.3fs",
                    total_elapsed,
                )
                return QueryResponse(
                    answer=rewritten_query,
                    workflow="rewrite_only",
                    routing_confidence=1.0,
                    sources=[],
                    request_id=request_id,
                    error=None,
                    debug=DebugTraceBuilder.build(
                        original_query=original_query,
                        rewritten_query=rewritten_query,
                        rewrite_time_ms=rewrite_time_ms,
                        request=request,
                        route_input_query=route_input_query,
                        route_source="rewrite_only",
                    ),
                    plan=execution_plan,
                    task_results=[],
                    merged_answer="",
                )

            # Record successful plan build latency (no field clarification)
            plan_elapsed_ms = int((time.perf_counter() - plan_start) * 1000)
            GatewayEventLogger.log_runtime(
                event_name="gateway_query_stage",
                stage="plan_build",
                message="plan build completed",
                status="success",
                request_id=request_id,
                session_id=request.session_id,
                user_id=uid,
                workflow=request.workflow or "auto",
                latency_ms=plan_elapsed_ms,
            )
            logger.info("[Perf] query endpoint plan_build: %d ms", plan_elapsed_ms)

            # 7. Route metadata
            planned_task_count = sum(len(g.tasks) for g in execution_plan.task_groups)
            if planned_task_count > 1:
                route_source, routing_confidence = "planner", 1.0
                workflow = PlanHelper.derive_workflow(
                    execution_plan,
                    [],
                    fallback=workflow,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    workflow,
                    routing_confidence,
                    route_source,
                    route_backend,
                    route_llm_confidence,
                    _,
                ) = rewrite_and_route(
                    request,
                    enable_routing=True,
                    route_query=route_input_query,
                    rewritten_query=rewritten_query,
                    rewrite_intents=intents or [],
                    memory_rounds=memory_rounds_q,
                    memory_text_length=memory_text_length_q,
                )
                if execution_plan.task_groups and execution_plan.task_groups[0].tasks:
                    execution_plan.task_groups[0].tasks[0].workflow = workflow

            # 8. Dispatcher: execute + merge
            dispatch_start = time.perf_counter()
            task_results = DispatcherExecutor.execute_plan(execution_plan, request)
            merged_answer = PlanHelper.merge_task_answers(execution_plan, task_results)
            workflow = PlanHelper.derive_workflow(
                execution_plan,
                task_results,
                fallback=workflow,
            )

            dispatch_elapsed_ms = int((time.perf_counter() - dispatch_start) * 1000)
            GatewayEventLogger.log_runtime(
                event_name="gateway_query_stage",
                stage="dispatch_execute",
                message="dispatcher execution completed",
                status="success",
                request_id=request_id,
                session_id=request.session_id,
                user_id=uid,
                workflow=workflow,
                latency_ms=dispatch_elapsed_ms,
            )
            logger.info("[Perf] query endpoint dispatch_execute: %d ms", dispatch_elapsed_ms)

            aggregated_sources: List[Dict[str, Any]] = []
            for r in task_results:
                if r.status == "completed":
                    aggregated_sources.extend(r.sources)
            failed = [r for r in task_results if r.status == "failed"]
            top_error: str | None = None
            if failed and not merged_answer:
                top_error = failed[0].error or "All planned tasks failed."
            elif failed:
                top_error = f"{len(failed)} planned task(s) failed."

            meta = format_route_metadata(route_source, route_backend, route_llm_confidence)
            debug_trace = DebugTraceBuilder.build(
                original_query=original_query,
                rewritten_query=rewritten_query,
                rewrite_time_ms=rewrite_time_ms,
                request=request,
                route_input_query=route_input_query,
                route_source=route_source,
                route_backend=route_backend,
                route_llm_confidence=route_llm_confidence,
            )

            # 9a. All tasks failed
            if top_error and not merged_answer:
                _log_and_persist(
                    memory,
                    event_type="llm_answer",
                    content={"answer": "", "workflow": workflow, "error": top_error},
                    status="failed",
                    note="planned tasks failed",
                    answer=top_error or "error",
                    workflow=workflow,
                    **_lp,
                )
                GatewayEventLogger.log_error(
                    event_name="gateway_query_failed",
                    request_id=request_id,
                    session_id=request.session_id,
                    user_id=uid,
                    workflow=workflow,
                    query_raw=original_query,
                    query_rewritten=rewritten_query,
                    error_type="PlannedTaskFailure",
                    error_message=top_error or "All planned tasks failed.",
                    metadata={"failed_tasks": [t.task_id for t in failed]},
                )
                logger.warning(
                    "Planned execution failed for request %s (workflow=%s): %s %s",
                    request_id,
                    workflow,
                    top_error,
                    meta,
                )
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    "[Perf] Total pipeline use time (planned tasks failed): %.3fs",
                    total_elapsed,
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

            # 9b. Success
            if memory and uid and merged_answer and workflow not in ("clarification", "rewrite_only"):
                _log_and_persist(
                    memory,
                    event_type="llm_answer",
                    content={"answer": merged_answer, "workflow": workflow},
                    answer=merged_answer,
                    workflow=workflow,
                    **_lp,
                )
            total_elapsed = time.perf_counter() - total_start
            logger.info(
                "Gateway handled request %s with workflow=%s (confidence=%.2f) %s [total=%.3fs]",
                request_id,
                workflow,
                routing_confidence,
                meta,
                total_elapsed,
            )
            GatewayEventLogger.log_interaction(
                event_name="gateway_query_success", status="success",
                request_id=request_id, session_id=request.session_id,
                user_id=uid, workflow=workflow,
                query_raw=original_query, query_rewritten=rewritten_query,
                answer=merged_answer, latency_ms=rewrite_time_ms,
            )
            return QueryResponse(
                answer=merged_answer, workflow=workflow,
                routing_confidence=routing_confidence,
                sources=aggregated_sources, request_id=request_id,
                error=top_error, debug=debug_trace, plan=execution_plan,
                task_results=task_results, merged_answer=merged_answer,
            )

        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Gateway query handler failed: %s", exc)
            GatewayEventLogger.log_error(
                event_name="gateway_query_exception",
                request_id=request_id, session_id=request.session_id,
                user_id=uid, workflow=workflow,
                query_raw=original_query, query_rewritten=rewritten_query,
                error_type=type(exc).__name__, error_message=str(exc),
                metadata={"route_source": route_source},
            )
            if memory and uid:
                _log_and_persist(
                    memory, event_type="llm_answer",
                    content={"answer": "", "workflow": "error", "error": str(exc)},
                    status="failed", note=type(exc).__name__,
                    answer=str(exc), workflow="error", **_lp,
                )
            return QueryResponse(
                answer="", workflow=workflow, routing_confidence=routing_confidence,
                sources=[], request_id=request_id, error=str(exc),
                debug=DebugTraceBuilder.build(
                    original_query=original_query, rewritten_query=rewritten_query,
                    rewrite_time_ms=rewrite_time_ms, request=request,
                    route_input_query=route_input_query, route_source=route_source,
                    route_backend=route_backend, route_llm_confidence=route_llm_confidence,
                ),
            )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/api/v1/session/{session_id}")
async def get_session_history(session_id: str, last_n: int = 10) -> dict[str, Any]:
    """Return last N turns for a session (for UI or debugging)."""
    return ConversationHistoryHandler.get_session_history(
        gateway_memory, session_id, min(last_n, 50),
    )


@app.post("/api/v1/rewrite", response_model=RewriteResponse, summary="Route LLM only (no execution)")
async def rewrite(
    request: QueryRequest,
    _user: dict | None = Depends(_get_optional_user),
) -> RewriteResponse:
    """Run Route LLM pipeline only: clarification, rewriting, intent classification."""
    return RewritePipeline.run(request, _user, gateway_memory)


@app.post("/api/v1/query", response_model=QueryResponse, summary="Submit unified gateway query")
async def query(
    request: QueryRequest,
    _user: dict | None = Depends(_get_optional_user),
) -> QueryResponse:
    """Route LLM (planning) + Dispatcher (execution) → QueryResponse."""
    return QueryPipeline.run(request, _user, gateway_memory)


__all__ = ["app"]

