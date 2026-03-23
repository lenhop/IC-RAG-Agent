"""
Unit tests for src.llm.chat_backend_policy: precedence, overrides, invalid env.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.llm.chat_backend_policy import (
    ENV_GATEWAY_CHAT_LLM_BACKEND,
    effective_backends_snapshot,
    resolve_chat_backend,
)


def _clear_chat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove chat-backend-related env vars for isolated tests."""
    keys = (
        "GATEWAY_REWRITE_BACKEND",
        "GATEWAY_CLARIFICATION_BACKEND",
        "GATEWAY_INTENT_DETECT_BACKEND",
        "GATEWAY_ROUTE_BACKEND",
        "GATEWAY_TEXT_GENERATION_BACKEND",
        ENV_GATEWAY_CHAT_LLM_BACKEND,
    )
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_global_only_unset_stage_vars_defaults_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    """When stage env and global are unset, every stage resolves to deepseek."""
    _clear_chat_env(monkeypatch)
    for stage in (
        "rewrite",
        "clarification",
        "intent_detect",
        "route",
        "text_generation",
    ):
        assert resolve_chat_backend(stage) == "deepseek"


def test_global_ollama_applies_when_stage_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """GATEWAY_CHAT_LLM_BACKEND=ollama drives all stages when per-stage vars are unset."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv(ENV_GATEWAY_CHAT_LLM_BACKEND, "ollama")
    assert resolve_chat_backend("clarification") == "ollama"
    assert resolve_chat_backend("intent_detect") == "ollama"


def test_stage_override_wins_over_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-stage env beats GATEWAY_CHAT_LLM_BACKEND."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv(ENV_GATEWAY_CHAT_LLM_BACKEND, "ollama")
    monkeypatch.setenv("GATEWAY_REWRITE_BACKEND", "deepseek")
    assert resolve_chat_backend("rewrite") == "deepseek"
    assert resolve_chat_backend("route") == "ollama"


def test_rewrite_request_override_only_on_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client rewrite_backend overrides env for rewrite stage only."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv(ENV_GATEWAY_CHAT_LLM_BACKEND, "ollama")
    monkeypatch.setenv("GATEWAY_REWRITE_BACKEND", "ollama")
    assert resolve_chat_backend("rewrite", request_override="deepseek") == "deepseek"
    # Override must not affect other stages.
    assert resolve_chat_backend("clarification", request_override="deepseek") == "ollama"


def test_invalid_stage_env_falls_back_to_global_then_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage stage env is ignored; global then deepseek is used."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_REWRITE_BACKEND", "not-a-backend")
    monkeypatch.setenv(ENV_GATEWAY_CHAT_LLM_BACKEND, "ollama")
    assert resolve_chat_backend("rewrite") == "ollama"

    monkeypatch.delenv(ENV_GATEWAY_CHAT_LLM_BACKEND, raising=False)
    assert resolve_chat_backend("rewrite") == "deepseek"


def test_normalize_aliases_ds_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """ds -> deepseek, local -> ollama in env strings."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_ROUTE_BACKEND", "ds")
    assert resolve_chat_backend("route") == "deepseek"
    monkeypatch.setenv("GATEWAY_ROUTE_BACKEND", "local")
    assert resolve_chat_backend("route") == "ollama"


def test_effective_backends_snapshot_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot includes per_stage and deepseek_api_key_set flag."""
    _clear_chat_env(monkeypatch)
    monkeypatch.setenv(ENV_GATEWAY_CHAT_LLM_BACKEND, "deepseek")
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "x"}, clear=False):
        snap = effective_backends_snapshot()
    assert snap["gateway_chat_llm_backend_effective"] == "deepseek"
    assert snap["deepseek_api_key_set"] is True
    assert set(snap["per_stage"].keys()) >= {
        "rewrite",
        "clarification",
        "intent_detect",
        "route",
        "text_generation",
    }
