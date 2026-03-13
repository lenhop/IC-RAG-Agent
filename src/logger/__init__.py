"""
IC-RAG logger package.

Public entry points:
- get_logger_facade(): singleton logging facade
- LoggerFacade: class for dependency injection/testing
"""

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
    "get_logger_facade",
]
