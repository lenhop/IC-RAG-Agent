"""Unit tests for gateway.keyword_regular_match (RouteRulesMatcher + §2.3 decision)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gateway.keyword_regular_match import (
    RouteRulesMatcher,
    clarification_skip_l3_decision,
    evaluate_l2_effective_ambiguity,
    is_clarification_layer1_l2_enabled,
    is_clarification_layer3_force,
    reset_default_route_rules_matcher_cache,
)


@pytest.fixture
def tiny_rules_dir(tmp_path: Path) -> Path:
    """Minimal CSV trio for deterministic tests."""
    (tmp_path / "amazon_business_intent_sentence.csv").write_text(
        "sentence,workflow\n"
        "hello whitelist,amazon_docs\n",
        encoding="utf-8",
    )
    (tmp_path / "regular_patterns.csv").write_text(
        "pattern,workflow,source,example\n"
        r"(?i)\bfoo\d+\b,uds,order,foo123"
        "\n",
        encoding="utf-8",
    )
    (tmp_path / "clarification_signals.csv").write_text(
        "rule_id,tier,signal_category,locale,requires_history,pattern,description,example_hit\n"
        'CS-A-001,A,test,en,false,(?i)\\bthat\\s+order\\b,test,example\n'
        'CS-X-001,exclusion,test,en,false,(?i)vs\\.,test,example\n',
        encoding="utf-8",
    )
    return tmp_path


def test_load_and_clear_sentence_match(tiny_rules_dir: Path) -> None:
    m = RouteRulesMatcher.load(tiny_rules_dir)
    hit = m.match_clear_sentence("Hello Whitelist")
    assert hit is not None
    assert hit.workflow == "amazon_docs"


def test_regular_pattern_match(tiny_rules_dir: Path) -> None:
    m = RouteRulesMatcher.load(tiny_rules_dir)
    hits = m.match_regular_patterns("prefix foo999 suffix")
    assert len(hits) == 1
    assert hits[0].workflow == "uds"
    assert hits[0].source == "order"


def test_clarification_l2_and_exclusion(tiny_rules_dir: Path) -> None:
    m = RouteRulesMatcher.load(tiny_rules_dir)
    assert evaluate_l2_effective_ambiguity(
        "show that order", has_conversation_history=False, matcher=m
    )
    # exclusion pattern wins when both could apply — here only exclusion matches "vs."
    ex = m.match_clarification_exclusions("A vs. B")
    assert len(ex) >= 1


def test_skip_l3_decision_cold_l1(tiny_rules_dir: Path) -> None:
    m = RouteRulesMatcher.load(tiny_rules_dir)
    skip, path = clarification_skip_l3_decision(
        "hello whitelist",
        has_conversation_history=False,
        matcher=m,
    )
    assert skip is True
    assert path == "l1_skip"


def test_skip_l3_decision_history_l2_forces_llm(tiny_rules_dir: Path) -> None:
    m = RouteRulesMatcher.load(tiny_rules_dir)
    skip, path = clarification_skip_l3_decision(
        "show that order",
        has_conversation_history=True,
        matcher=m,
    )
    assert skip is False
    assert path is None


def test_env_flags_read_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", raising=False)
    monkeypatch.delenv("GATEWAY_CLARIFICATION_LAYER3_FORCE", raising=False)
    assert is_clarification_layer1_l2_enabled() is False
    assert is_clarification_layer3_force() is False


def test_env_layer1_l2_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED", "true")
    assert is_clarification_layer1_l2_enabled() is True


def test_reset_matcher_cache(monkeypatch: pytest.MonkeyPatch, tiny_rules_dir: Path) -> None:
    monkeypatch.setenv("GATEWAY_ROUTE_RULES_DATA_DIR", str(tiny_rules_dir))
    reset_default_route_rules_matcher_cache()
    from src.gateway.keyword_regular_match import get_default_route_rules_matcher

    m1 = get_default_route_rules_matcher()
    assert m1 is not None
    m2 = get_default_route_rules_matcher()
    assert m1 is m2
    reset_default_route_rules_matcher_cache()
