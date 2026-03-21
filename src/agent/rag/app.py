"""
FastAPI RAG service: POST /query contract matches gateway RagWorkflowClient.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.chroma import bootstrap_project

from .service import RagQueryService

logger = logging.getLogger(__name__)

# Load .env and paths when the app module is imported (consistent with other services).
bootstrap_project()


class QueryModeEnum(str, Enum):
    """Allowed mode values (must match gateway payloads)."""

    general = "general"
    amazon_business = "amazon_business"
    documents = "documents"


class QueryBody(BaseModel):
    """Inbound JSON body for /query."""

    question: str = Field(..., min_length=1, description="User question text")
    mode: QueryModeEnum = Field(
        default=QueryModeEnum.general,
        description="RAG mode: general | amazon_business | documents",
    )


class QueryResponseModel(BaseModel):
    """Successful response shape."""

    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ErrorResponseModel(BaseModel):
    """Error envelope expected by gateway BackendHttpClient."""

    error: str
    error_type: str


app = FastAPI(
    title="Agent RAG API",
    description="DeepSeek + Chroma (documents) dual path for amazon_business",
    version="1.0.0",
)


@app.on_event("startup")
def _on_startup() -> None:
    """Log listen configuration; heavy clients stay lazy in RagQueryService."""
    host = os.getenv("RAG_API_HOST", "0.0.0.0")
    port = os.getenv("RAG_API_PORT", "8002")
    logger.info("Agent RAG API startup (lazy Chroma/embedder); host=%s port=%s", host, port)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe for orchestrators."""
    return {"status": "ok", "service": "agent_rag"}


@app.post("/query")
def post_query(
    body: QueryBody,
) -> Union[QueryResponseModel, ErrorResponseModel]:
    """
    Run RAG for the given mode.

    Returns HTTP 200 with either answer+sources or error+error_type so the
    gateway JSON client can parse the body without treating 4xx/5xx specially.
    """
    result = RagQueryService.run(body.question, body.mode.value)
    err = result.get("error")
    if err:
        return ErrorResponseModel(
            error=str(err),
            error_type=str(result.get("error_type") or "BackendError"),
        )
    return QueryResponseModel(
        answer=str(result.get("answer") or ""),
        sources=list(result.get("sources") or []),
    )
