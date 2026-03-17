"""
IC-RAG logger package.

Public entry points:
- get_logger_facade(): singleton logging facade
- LoggerFacade: class for dependency injection/testing
"""

from .utils import enrich_log_fields, format_route_metadata
from .logger import LoggerFacade, get_logger_facade
from .models import ErrorLog, InteractionLog, LogKind, LogStatus, RuntimeLog
from .settings import LoggerSettings

__all__ = [
    "ErrorLog",
    "InteractionLog",
    "LogKind",
    "LogStatus",
    "LoggerFacade",
    "LoggerSettings",
    "RuntimeLog",
    "enrich_log_fields",
    "format_route_metadata",
    "get_logger_facade",
]
