"""
Structured logger event models.

Pydantic models validate payloads before persistence to Redis/ClickHouse.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LogKind(str, Enum):
    """Supported log categories."""

    interaction = "interaction"
    runtime = "runtime"
    error = "error"


class LogStatus(str, Enum):
    """Lifecycle status of a log event."""

    started = "started"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class BaseLogEvent(BaseModel):
    """Base fields shared by all log events."""

    model_config = ConfigDict(extra="allow")

    event_kind: LogKind
    event_name: str = Field(default="", description="Fine-grained event identifier.")
    status: LogStatus = LogStatus.success

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow: Optional[str] = None

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert model to storage-friendly dict."""
        data = self.model_dump()
        data["ts"] = self.ts.isoformat()
        return data


class InteractionLog(BaseLogEvent):
    """User-facing interaction event payload."""

    event_kind: LogKind = LogKind.interaction

    query_raw: str = ""
    clarification_question: Optional[str] = None
    query_rewritten: Optional[str] = None
    intent_list: List[str] = Field(default_factory=list)
    intent_details: List[Dict[str, Any]] = Field(default_factory=list)
    answer: Optional[str] = None
    latency_ms: Optional[int] = None


class RuntimeLog(BaseLogEvent):
    """System runtime event payload."""

    event_kind: LogKind = LogKind.runtime

    stage: str = ""
    message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None


class ErrorLog(BaseLogEvent):
    """Error event payload."""

    event_kind: LogKind = LogKind.error
    status: LogStatus = LogStatus.failed

    error_type: str = "UnknownError"
    error_message: str = ""
    stacktrace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
