"""
Gateway API request/response schemas.

Defines the contract used by the unified gateway:
- QueryRequest: incoming query from clients
- QueryResponse: normalized answer returned to clients
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    """
    Request model for unified gateway queries.

    Fields:
        query: Natural language question from the user.
        workflow: Routing workflow selector
                  (auto|general|amazon_docs|ic_docs|sp_api|uds).
        rewrite_enable: Whether to enable query rewriting.
        session_id: Optional session identifier for multi-turn context.
        stream: Whether client requests streaming response (SSE).
    """

    query: str = Field(..., description="Natural language user query")
    workflow: str = Field(
        default="auto",
        description=(
            "Workflow selector: auto|general|amazon_docs|ic_docs|sp_api|uds. "
            "Defaults to auto when not provided."
        ),
    )
    rewrite_enable: bool = Field(
        default=True,
        description="Enable or disable query rewriting.",
    )
    rewrite_backend: Optional[str] = Field(
        default=None,
        description=(
            "Rewrite backend when rewrite_enable=True: 'ollama' or 'deepseek'. "
            "Ignored when rewrite_enable=False. Defaults to GATEWAY_REWRITE_BACKEND env."
        ),
    )
    route_backend: Optional[str] = Field(
        default=None,
        description=(
            "Route LLM backend when workflow='auto' and Route LLM is enabled: "
            "'ollama' or 'deepseek'. Ignored when workflow is set explicitly. "
            "Defaults to GATEWAY_ROUTE_LLM_BACKEND env."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for multi-turn context.",
    )
    stream: bool = Field(
        default=False,
        description="If true, client prefers streaming responses (SSE).",
    )

    @field_validator("route_backend", mode="before")
    @classmethod
    def _normalize_route_backend(cls, v: Optional[str]) -> Optional[str]:
        """Normalize to lowercase; map unknown values to None (no client break)."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = str(v).strip().lower()
        if s in ("ollama", "deepseek"):
            return s
        return None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were my total Amazon sales last week?",
                "workflow": "uds",
                "rewrite_enable": True,
                "rewrite_backend": "ollama",
                "route_backend": None,
                "session_id": "session-1234",
                "stream": False,
            }
        },
    )


class QueryResponse(BaseModel):
    """
    Response model for unified gateway answers.

    Fields:
        answer: Final answer text for the user.
        workflow: Workflow that handled the query.
        routing_confidence: Confidence score for routing decision (0.0–1.0).
        sources: List of source metadata dicts (documents, tables, etc.).
        request_id: Unique identifier for this gateway request.
        error: Optional error message when the request fails.
    """

    answer: str = Field(
        default="",
        description="Final answer text for the user.",
    )
    workflow: str = Field(
        default="auto",
        description="Workflow that produced this answer.",
    )
    routing_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for routing decision (0.0–1.0).",
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of source metadata dicts (documents, tables, etc.).",
    )
    request_id: str = Field(
        ...,
        description="Unique identifier for this gateway request.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Optional error message when the request fails.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Your total Amazon sales last week were $12,345.",
                "workflow": "uds",
                "routing_confidence": 0.96,
                "sources": [
                    {
                        "type": "table",
                        "name": "amz_order",
                        "rows_scanned": 1024,
                    }
                ],
                "request_id": "req-1234",
                "error": None,
            }
        },
    )

