"""
Tests for src.retrieval.keyword_retrieval (pure matcher + classification rule store integration).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gateway.route_llm.classification.implement_methods import ClassificationKeywordRuleStore
from src.retrieval.keyword_retrieval import KeywordMatchResult, KeywordRetrieval, keyword_retrieve


def _sample_regex_rows() -> list[tuple[re.Pattern, str, str]]:
    """Minimal regex rules for unit tests (no disk)."""
    return [
        (re.compile(r"\d{3}-\d{7}-\d{7}", re.IGNORECASE), "sp_api", "get_order_status"),
        (re.compile(r"\b(inventory|stock)\b", re.IGNORECASE), "uds", "inventory"),
    ]


def test_keyword_retrieval_empty_returns_none() -> None:
    """Empty query returns None."""
    matcher = KeywordRetrieval([], [], [])
    assert matcher.match("") is None
    assert matcher.match("   ") is None


def test_keyword_retrieval_match_returns_result() -> None:
    """Stage 3 regex: order id matches sp_api."""
    matcher = KeywordRetrieval([], [], _sample_regex_rows())
    r = matcher.match("Get the latest status of order 112-1234567-8901234")
    assert r is not None
    assert isinstance(r, KeywordMatchResult)
    assert r.workflow == "sp_api"
    assert r.intent_name == "get_order_status"
    assert r.source == "keyword"


def test_keyword_retrieval_uds_style() -> None:
    """Stage 3 regex: inventory keyword."""
    matcher = KeywordRetrieval([], [], _sample_regex_rows())
    r = matcher.match("Show my inventory")
    assert r is not None
    assert r.workflow == "uds"
    assert r.intent_name == "inventory"


def test_keyword_retrieval_dict_exact_match() -> None:
    """Stage 1: exact canonical."""
    rows = [("hello world", "uds")]
    matcher = KeywordRetrieval(rows, [], [])
    r = matcher.match("Hello World")
    assert r is not None
    assert r.workflow == "uds"
    assert r.intent_name == "uds"


def test_keyword_retrieval_for_loop_phrase() -> None:
    """Stage 2: phrase substring."""
    for_rows = [("what is fba", "amazon_docs", "amazon_docs")]
    matcher = KeywordRetrieval([], for_rows, [])
    r = matcher.match("what is fba policy")
    assert r is not None
    assert r.workflow == "amazon_docs"


def test_keyword_retrieve_function() -> None:
    """keyword_retrieve injects rows without instantiating class manually."""
    r = keyword_retrieve("x", [], [], _sample_regex_rows())
    assert r is None
    r2 = keyword_retrieve("order 112-1234567-8901234", [], [], _sample_regex_rows())
    assert r2 is not None
    assert r2.workflow == "sp_api"


def test_classification_rule_store_integration_dict() -> None:
    """Real dict_sentences via ClassificationKeywordRuleStore."""
    m = ClassificationKeywordRuleStore.get_keyword_retrieval()
    r = m.match("tell me sales revenue last month")
    assert r is not None
    assert r.workflow == "uds"


def test_classification_rule_store_integration_for_loop() -> None:
    """Real YAML phrases via ClassificationKeywordRuleStore."""
    m = ClassificationKeywordRuleStore.get_keyword_retrieval()
    r = m.match("what is fba")
    assert r is not None
    assert r.workflow == "amazon_docs"


def test_classification_rule_store_integration_regex() -> None:
    """Real regular_rules.csv via ClassificationKeywordRuleStore."""
    m = ClassificationKeywordRuleStore.get_keyword_retrieval()
    r = m.match("Show my inventory")
    assert r is not None
    assert r.workflow == "uds"
    assert r.intent_name == "inventory"
