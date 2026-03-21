"""
Gateway worker profile: UDS/SP-API stubs when RAG-only mode is enabled.
"""

from __future__ import annotations

from typing import Generator

import pytest

from src.gateway.dispatcher.clients.sp_api_client import SpApiWorkflowClient
from src.gateway.dispatcher.clients.uds_client import UdsWorkflowClient
from src.gateway.dispatcher.clients.worker_profile import (
    STUB_SP_API_ANSWER,
    STUB_UDS_ANSWER,
    is_rag_only_worker_profile,
    should_stub_uds_and_sp_api,
    stub_response_for_workflow,
)


@pytest.fixture
def clear_worker_profile_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Remove profile-related env vars between tests."""
    for key in (
        "GATEWAY_WORKER_PROFILE",
        "GATEWAY_STUB_UDS_SP_API",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def test_default_no_stub(clear_worker_profile_env: None) -> None:
    """Without env, real HTTP path is selected (stub gate is false)."""
    assert should_stub_uds_and_sp_api() is False
    assert is_rag_only_worker_profile() is False


def test_rag_only_profile_enables_stub(
    monkeypatch: pytest.MonkeyPatch, clear_worker_profile_env: None
) -> None:
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_only")
    assert is_rag_only_worker_profile() is True
    assert should_stub_uds_and_sp_api() is True


def test_stub_flag_enables_stub(
    monkeypatch: pytest.MonkeyPatch, clear_worker_profile_env: None
) -> None:
    monkeypatch.setenv("GATEWAY_STUB_UDS_SP_API", "true")
    assert should_stub_uds_and_sp_api() is True


def test_stub_response_content() -> None:
    uds = stub_response_for_workflow("uds")
    assert uds["answer"] == STUB_UDS_ANSWER
    assert uds["sources"] == []
    sp = stub_response_for_workflow("sp_api")
    assert sp["answer"] == STUB_SP_API_ANSWER
    assert sp["sources"] == []


def test_uds_client_short_circuits_when_stub(
    monkeypatch: pytest.MonkeyPatch, clear_worker_profile_env: None
) -> None:
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_only")
    out = UdsWorkflowClient.call_uds("any query", "sid")
    assert out["answer"] == STUB_UDS_ANSWER
    assert out["sources"] == []


def test_sp_api_client_short_circuits_when_stub(
    monkeypatch: pytest.MonkeyPatch, clear_worker_profile_env: None
) -> None:
    monkeypatch.setenv("GATEWAY_STUB_UDS_SP_API", "1")
    out = SpApiWorkflowClient.call_sp_api("sku question", "session-1")
    assert out["answer"] == STUB_SP_API_ANSWER
    assert out["sources"] == []


def test_dispatcher_executor_uses_stub_for_uds(
    monkeypatch: pytest.MonkeyPatch, clear_worker_profile_env: None
) -> None:
    """End-to-end handler resolution returns stub without HTTP."""
    monkeypatch.setenv("GATEWAY_WORKER_PROFILE", "rag_only")
    from src.gateway.dispatcher.execution.executor import DispatcherExecutor

    result = DispatcherExecutor.call_workflow_backend("uds", "hello", None)
    assert "尚未" in result.get("answer", "")
    assert result.get("sources") == []
