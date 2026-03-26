"""Functional tests: clarification L1+L2 gate and check_ambiguity integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.gateway.keyword_regular_match import (
    ClarificationLayer12Gate,
    DefaultRouteRulesMatcherCache,
    reset_default_route_rules_matcher_cache,
)
from src.gateway.route_llm.clarification.clarification import check_ambiguity


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    (tmp_path / "amazon_business_intent_sentence.csv").write_text(
        "sentence,workflow\n"
        "known clear phrase,amazon_docs\n",
        encoding="utf-8",
    )
    (tmp_path / "regular_patterns.csv").write_text(
        "pattern,workflow,source,example\n"
        r"(?i)ord\d+,uds,order,ord1"
        "\n",
        encoding="utf-8",
    )
    (tmp_path / "clarification_signals.csv").write_text(
        "rule_id,tier,signal_category,locale,requires_history,pattern,description,example_hit\n"
        'CS-A-001,A,test,en,false,(?i)\\bit\\b,test,it\n',
        encoding="utf-8",
    )
    return tmp_path


def test_gate_returns_none_when_layer1_l2_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", "false")
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER3_FORCE", "false")
    assert (
        ClarificationLayer12Gate.try_resolve_without_l3(
            "any",
            has_conversation_history=False,
        )
        is None
    )


def test_gate_skips_l3_on_l1_whitelist(
    monkeypatch: pytest.MonkeyPatch,
    rules_dir: Path,
) -> None:
    monkeypatch.setenv("GATEWAY_ROUTE_RULES_DATA_DIR", str(rules_dir))
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER3_FORCE", "false")
    reset_default_route_rules_matcher_cache()
    out = ClarificationLayer12Gate.try_resolve_without_l3(
        "known clear phrase",
        has_conversation_history=False,
    )
    assert out is not None
    assert out["needs_clarification"] is False
    assert out["clarification_path"] == "l1_skip"
    DefaultRouteRulesMatcherCache.reset()


def test_gate_force_layer3(monkeypatch: pytest.MonkeyPatch, rules_dir: Path) -> None:
    monkeypatch.setenv("GATEWAY_ROUTE_RULES_DATA_DIR", str(rules_dir))
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER3_FORCE", "true")
    reset_default_route_rules_matcher_cache()
    out = ClarificationLayer12Gate.try_resolve_without_l3(
        "known clear phrase",
        has_conversation_history=False,
    )
    assert out is None
    DefaultRouteRulesMatcherCache.reset()


def test_check_ambiguity_disabled_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CLARIFICATION_ENABLED", "false")
    out = check_ambiguity("hello", conversation_context=None)
    assert out["clarification_path"] == "disabled"
    assert out["needs_clarification"] is False


@patch(
    "src.gateway.route_llm.clarification.clarification._ClarificationLLM.call",
)
def test_check_ambiguity_fastpath_no_llm(
    mock_llm,
    monkeypatch: pytest.MonkeyPatch,
    rules_dir: Path,
) -> None:
    monkeypatch.setenv("GATEWAY_CLARIFICATION_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_ROUTE_RULES_DATA_DIR", str(rules_dir))
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER3_FORCE", "false")
    reset_default_route_rules_matcher_cache()
    out = check_ambiguity("known clear phrase", conversation_context=None)
    mock_llm.assert_not_called()
    assert out.get("clarification_path") == "l1_skip"
    DefaultRouteRulesMatcherCache.reset()
