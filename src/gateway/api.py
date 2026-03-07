"""
Gateway FastAPI application.

Exposes a minimal REST API for the unified query gateway:
- POST /api/v1/query: accept QueryRequest, return QueryResponse (stubbed routing)
- GET /health: basic health check endpoint

Routing and backend integration will be implemented in later tasks; this
module focuses on the HTTP contract and app wiring.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import route_workflow, rewrite_query
from .logging_utils import format_route_metadata
from .services import (
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)
from .schemas import QueryRequest, QueryResponse

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


def _execute_workflow(
    workflow: str,
    routing_confidence: float,
    request: QueryRequest,
    rewritten_query: str,
) -> Dict[str, Any]:
    """
    Stub services layer for executing the chosen workflow.

    Future implementation will call underlying agents/services (UDS, RAG,
    SP-API, etc.). For now this returns a structured stub answer.

    Args:
        workflow: Selected workflow name.
        routing_confidence: Confidence score from router.
        request: Original QueryRequest.
        rewritten_query: Rewritten query used for routing.

    Returns:
        Dict with at least an 'answer' key, optional 'sources', or 'error'.
    """
    # Dispatch to the appropriate backend service.
    workflow_lower = (workflow or "general").lower()
    session_id = request.session_id

    if workflow_lower == "general":
        return call_general(rewritten_query, session_id)
    if workflow_lower == "amazon_docs":
        return call_amazon_docs(rewritten_query, session_id)
    if workflow_lower == "ic_docs":
        return call_ic_docs(rewritten_query, session_id)
    if workflow_lower == "sp_api":
        return call_sp_api(rewritten_query, session_id)
    if workflow_lower == "uds":
        return call_uds(rewritten_query, session_id)

    # Fallback to general workflow for unknown names.
    logger.warning("Unknown workflow '%s', falling back to general", workflow)
    return call_general(rewritten_query, session_id)


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

    try:
        # Step 1: rewrite query
        rewritten_query = rewrite_query(request)

        # Step 2: route workflow (returns metadata for logging)
        (
            workflow,
            routing_confidence,
            route_source,
            route_backend,
            route_llm_confidence,
        ) = route_workflow(rewritten_query, request)

        # Step 3: call services layer (stubbed)
        result: Dict[str, Any] = _execute_workflow(
            workflow=workflow,
            routing_confidence=routing_confidence,
            request=request,
            rewritten_query=rewritten_query,
        )

        meta = format_route_metadata(route_source, route_backend, route_llm_confidence)
        if "error" in result:
            # Backend-level error; surface in QueryResponse.error.
            error_msg = result.get("error")
            logger.warning(
                "Backend error for request %s (workflow=%s): %s %s",
                request_id,
                workflow,
                error_msg,
                meta,
            )
            return QueryResponse(
                answer="",
                workflow=workflow,
                routing_confidence=routing_confidence,
                sources=result.get("sources", []),
                request_id=request_id,
                error=error_msg,
            )

        response = QueryResponse(
            answer=result.get("answer", ""),
            workflow=workflow,
            routing_confidence=routing_confidence,
            sources=result.get("sources", []),
            request_id=request_id,
            error=None,
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
            workflow="auto",
            routing_confidence=0.0,
            sources=[],
            request_id=request_id,
            error=str(exc),
        )


__all__ = ["app"]

