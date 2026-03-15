"""
UDS API Request/Response Schemas.

Pydantic models for all REST API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request model for query submission."""

    query: str = Field(..., description="Natural language business question")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "include_charts": True,
            "include_insights": True,
            "max_execution_time": 30,
        },
        description="Query options",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were total sales in October?",
                "options": {
                    "include_charts": True,
                    "include_insights": True,
                },
            }
        },
    )


class QueryResponse(BaseModel):
    """Response model for query results."""

    query_id: str = Field(..., description="Unique query identifier")
    status: str = Field(
        ...,
        description="Query status: pending, running, completed, failed",
    )
    query: str = Field(..., description="Original query text")
    intent: Optional[str] = Field(None, description="Classified intent domain")
    response: Optional[Dict[str, Any]] = Field(None, description="Formatted response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when query was created",
    )


class TableSchema(BaseModel):
    """Table schema information."""

    table_name: str = Field(..., description="Table name")
    database: str = Field(..., description="Database name")
    row_count: int = Field(..., description="Number of rows")
    columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Column definitions",
    )


class TableSampleResponse(BaseModel):
    """Sample data response."""

    table_name: str = Field(..., description="Table name")
    sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample rows",
    )
    limit: int = Field(..., description="Number of rows returned")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="healthy or unhealthy")
    database: Optional[str] = Field(None, description="Database connection status")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Check timestamp",
    )


class AgentStatusResponse(BaseModel):
    """Agent status response."""

    status: str = Field(..., description="Agent status")
    tools: int = Field(..., description="Number of registered tools")
    queries_processed: int = Field(..., description="Total queries processed")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    error_type: Optional[str] = Field(None, description="Error type/code")
