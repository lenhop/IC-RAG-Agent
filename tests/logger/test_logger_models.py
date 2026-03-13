"""
Unit tests for logger models and settings.
"""

from __future__ import annotations

from src.logger.models import ErrorLog, InteractionLog, LogKind, RuntimeLog
from src.logger.settings import LoggerSettings


def test_interaction_log_to_storage_dict_contains_required_fields():
    """InteractionLog should serialize to dict with expected defaults."""
    evt = InteractionLog(
        event_name="gateway_query_success",
        request_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        workflow="general",
        query_raw="what is fba",
        query_rewritten="what is fba",
        answer="fba means fulfillment by amazon",
    )
    data = evt.to_storage_dict()
    assert data["event_kind"] == LogKind.interaction
    assert data["request_id"] == "req-1"
    assert data["query_raw"] == "what is fba"
    assert "ts" in data


def test_runtime_and_error_log_defaults():
    """Runtime and error models should apply sensible defaults."""
    runtime_evt = RuntimeLog(event_name="rewrite_done", stage="rewriter", message="ok")
    error_evt = ErrorLog(event_name="gateway_exception", error_type="ValueError", error_message="bad input")
    assert runtime_evt.event_kind == LogKind.runtime
    assert error_evt.event_kind == LogKind.error
    assert error_evt.status == "failed"


def test_logger_settings_from_env_defaults(monkeypatch):
    """LoggerSettings.from_env should parse booleans and numeric defaults."""
    monkeypatch.delenv("LOGGER_ENABLED", raising=False)
    monkeypatch.delenv("LOGGER_CH_BATCH_SIZE", raising=False)
    settings = LoggerSettings.from_env()
    assert settings.enabled is True
    assert settings.clickhouse_batch_size >= 1
