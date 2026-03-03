"""Pydantic schemas for Seller API."""
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))


class QueryResponse(BaseModel):
    response: str
    session_id: str
    iterations: int
    tools_used: List[str]


class SessionHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str
    iterations: int


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
