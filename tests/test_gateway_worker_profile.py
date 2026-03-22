"""
Tests for gateway worker stub profiles (UDS vs SP-API split).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def clear_worker_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove worker profile env vars before each case."""
    for key in (
        "GATEWAY_WORKER_PROFILE",
        "GATEWAY_STUB_UDS_SP_API",
    ):
        monkeypatch.delenv(key, raising=False)


def test_rag_only_stubs_both(clear_worker_profile, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_only")
    from src.gateway.dispatcher.clients import worker_profile

    assert worker_profile.should_stub_uds() is True
    assert worker_profile.should_stub_sp_api() is True
    assert worker_profile.should_stub_uds_and_sp_api() is True


def test_rag_sp_api_stubs_uds_only(clear_worker_profile, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_sp_api")
    from src.gateway.dispatcher.clients import worker_profile

    assert worker_profile.should_stub_uds() is True
    assert worker_profile.should_stub_sp_api() is False
    assert worker_profile.should_stub_uds_and_sp_api() is False


def test_stub_flag_stubs_both(clear_worker_profile, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_sp_api")
    monkeypatch.setenv("GATEWAY_STUB_UDS_SP_API", "true")
    from src.gateway.dispatcher.clients import worker_profile

    assert worker_profile.should_stub_uds() is True
    assert worker_profile.should_stub_sp_api() is True


def test_default_no_stubs(clear_worker_profile) -> None:
    from src.gateway.dispatcher.clients import worker_profile

    assert worker_profile.should_stub_uds() is False
    assert worker_profile.should_stub_sp_api() is False


def test_uds_client_stubs_under_rag_sp_api(
    clear_worker_profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end client: UDS returns fixed stub text when profile is rag_sp_api."""
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_sp_api")
    from src.gateway.dispatcher.clients import uds_client

    out = uds_client.call_uds("any query", None)
    assert "answer" in out
    assert "UDS" in out["answer"] or "数据分析" in out["answer"]


def test_sp_api_client_calls_http_under_rag_sp_api(
    clear_worker_profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end client: SP-API posts to backend when profile is rag_sp_api."""
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_sp_api")
    from src.gateway.dispatcher.clients import sp_api_client

    monkeypatch.setattr(sp_api_client, "SP_API_URL", "http://127.0.0.1:9")
    with patch.object(
        sp_api_client.BackendHttpClient,
        "post_json",
        return_value={"response": "mocked_sp_answer"},
    ) as mock_post:
        out = sp_api_client.call_sp_api("order 123", "sess-1")
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url.endswith("/api/v1/seller/query")
    assert out == {"answer": "mocked_sp_answer", "sources": []}
