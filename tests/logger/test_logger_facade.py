"""
Unit tests for logger facade dual-write behavior.
"""

from __future__ import annotations

from src.logger.logger import LoggerFacade
from src.logger.settings import LoggerSettings


class _FakeRedisClient:
    def __init__(self):
        self.calls = []

    def write_event(self, payload, *, user_id=None, session_id=None):
        self.calls.append((payload, user_id, session_id))
        return True

    def read_recent_by_user(self, user_id, last_n=20):
        return [{"user_id": user_id, "last_n": last_n}]

    def read_recent_by_session(self, session_id, last_n=20):
        return [{"session_id": session_id, "last_n": last_n}]


class _FakeCHClient:
    def __init__(self, should_fail=False):
        self.calls = []
        self.should_fail = should_fail

    def write_event(self, payload):
        if self.should_fail:
            raise RuntimeError("clickhouse unavailable")
        self.calls.append(payload)
        return True

    def read_events(self, **kwargs):
        return [{"read": True, **kwargs}]

    def flush(self):
        return True


def _settings() -> LoggerSettings:
    return LoggerSettings(
        enabled=True,
        redis_enabled=True,
        clickhouse_enabled=True,
        redis_url="redis://localhost:6379/0",
        redis_ttl_seconds=3600,
        redis_max_events_per_key=100,
        clickhouse_host="localhost",
        clickhouse_port=8123,
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_database="default",
        clickhouse_table="gateway_logs",
        clickhouse_connect_timeout=10,
        clickhouse_send_receive_timeout=30,
        clickhouse_batch_enabled=True,
        clickhouse_batch_size=10,
        retry_enabled=True,
        retry_attempts=2,
        retry_backoff_ms=1,
        redaction_enabled=True,
        redaction_fields=("token",),
    )


def test_log_interaction_dual_write_success():
    """Facade should write to both Redis and ClickHouse when available."""
    redis_client = _FakeRedisClient()
    ch_client = _FakeCHClient()
    facade = LoggerFacade(settings=_settings(), redis_client=redis_client, clickhouse_client=ch_client)

    result = facade.log_interaction(
        event_name="gateway_query_success",
        request_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        workflow="general",
        query_raw="what is fba",
        query_rewritten="what is fba",
        answer="ok",
    )

    assert result["redis"] is True
    assert result["clickhouse"] is True
    assert len(redis_client.calls) == 1
    assert len(ch_client.calls) == 1


def test_log_interaction_clickhouse_failure_does_not_raise():
    """ClickHouse failure should not break logging flow."""
    redis_client = _FakeRedisClient()
    ch_client = _FakeCHClient(should_fail=True)
    facade = LoggerFacade(settings=_settings(), redis_client=redis_client, clickhouse_client=ch_client)

    result = facade.log_interaction(
        event_name="gateway_query_success",
        session_id="sess-1",
        user_id="user-1",
        query_raw="q",
    )

    assert result["redis"] is True
    assert result["clickhouse"] is False


def test_read_short_term_prefers_user_key():
    """read_short_term should use user key when user_id is provided."""
    redis_client = _FakeRedisClient()
    facade = LoggerFacade(settings=_settings(), redis_client=redis_client, clickhouse_client=None)
    rows = facade.read_short_term(user_id="u1", session_id="s1", last_n=5)
    assert rows and rows[0]["user_id"] == "u1"
