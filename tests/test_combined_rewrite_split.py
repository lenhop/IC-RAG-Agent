"""Tests for unified rewrite JSON parsing and IntentDetailsBuilder override path."""

from __future__ import annotations

import pytest

from src.gateway.api.view_helpers import IntentDetailsBuilder
from src.gateway.route_llm.rewriting.rewrite_implement import RewriteSplitMethod


def test_rewrite_split_parse_json_object() -> None:
    """Parse strict JSON and embedded object in noise."""
    parsed = RewriteSplitMethod._parse_json_response('{"intents":["x"],"rewritten_display":"d"}')
    assert parsed is not None
    assert parsed.get("intents") == ["x"]
    assert parsed.get("rewritten_display") == "d"

    raw = 'prefix {"intents": ["a", "b"]} suffix'
    parsed2 = RewriteSplitMethod._parse_json_response(raw)
    assert parsed2 is not None
    assert parsed2.get("intents") == ["a", "b"]


def test_rewrite_split_parse_invalid() -> None:
    assert RewriteSplitMethod._parse_json_response("not json") is None


def test_intent_details_builder_intents_override_skips_extra_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """When intents_override is passed, classify_intents_batch runs; no split_intents on module."""

    def _fake_batch(intents: list, _ctx=None):
        return [
            {
                "query": intents[0],
                "workflow": "general",
                "intent_name": "general",
                "confidence": "high",
                "source": "keyword",
            }
        ]

    import src.gateway.route_llm.classification as clf

    monkeypatch.setattr(clf, "classify_intents_batch", _fake_batch)

    intents, details, workflows = IntentDetailsBuilder.build_intent_details(
        "ignored when override set",
        intents_override=["  hello world  "],
    )
    assert intents == ["hello world"]
    assert len(details) == 1
    assert workflows == ["general"]
