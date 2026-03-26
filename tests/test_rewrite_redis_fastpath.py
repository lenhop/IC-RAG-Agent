"""
Unit tests for Redis-backed rewrite L1/L2 indexing and unified rewrite fast-path wiring.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.gateway.route_llm.rewriting.rewrite_redis_fastpath import (
    RewriteRedisFastpath,
    build_clear_sentence_index,
    build_regular_pattern_list,
    match_clear_sentence_from_index,
    match_first_regular_pattern_from_list,
    normalize_clear_sentence_key,
)
from src.gateway.route_llm.rewriting.rewrite_implement import (
    RouterEnvConfig,
    _RewriteRouter,
)
from src.gateway.schemas import QueryRequest


class TestNormalizeAndClearIndex(unittest.TestCase):
    """L1 index: casefold + strip must align with CSV / Redis field semantics."""

    def test_normalize_clear_sentence_key(self) -> None:
        self.assertEqual(normalize_clear_sentence_key("  AbC  "), "abc")

    def test_build_and_match_clear_sentence(self) -> None:
        payload = json.dumps({"sentence": "Hello World", "workflow": "uds"})
        idx = build_clear_sentence_index({"Hello World": payload})
        hit = match_clear_sentence_from_index(idx, "hello world")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.get("workflow"), "uds")


class TestRegularPatternList(unittest.TestCase):
    """L2: compiled regex list and first-hit semantics."""

    def test_first_pattern_wins_stable_order(self) -> None:
        p_a = json.dumps({"pattern": r"foo", "workflow": "a"})
        p_b = json.dumps({"pattern": r"bar", "workflow": "b"})
        h = {r"bar": p_b, r"foo": p_a}
        compiled = build_regular_pattern_list(h)
        hit = match_first_regular_pattern_from_list(compiled, "x foo y")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.get("workflow"), "a")


class TestRouterEnvConfigRewriteLayers(unittest.TestCase):
    """Env parsing for rewrite layer flags."""

    @patch.dict(os.environ, {"GATEWAY_REWRITE_LAYER1_ENABLED": "true"}, clear=False)
    def test_layer1_true(self) -> None:
        self.assertTrue(RouterEnvConfig.is_rewrite_layer1_enabled())

    @patch.dict(os.environ, {"GATEWAY_REWRITE_LAYER1_ENABLED": "false"}, clear=False)
    def test_layer1_false(self) -> None:
        self.assertFalse(RouterEnvConfig.is_rewrite_layer1_enabled())

    @patch.dict(os.environ, {"GATEWAY_REWRITE_LAYER3_FORCE": "1"}, clear=False)
    def test_layer3_force(self) -> None:
        self.assertTrue(RouterEnvConfig.is_rewrite_layer3_force())


class TestRunUnifiedRewriteRedisFastPath(unittest.TestCase):
    """run_unified_rewrite skips LLM when L1 (or L2) Redis matches."""

    def setUp(self) -> None:
        RewriteRedisFastpath.reset_cache_for_tests()

    @patch.dict(
        os.environ,
        {
            "GATEWAY_REWRITE_LAYER1_ENABLED": "true",
            "GATEWAY_REWRITE_LAYER2_ENABLED": "true",
            "GATEWAY_REWRITE_LAYER3_FORCE": "false",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.call_unified_rewrite_llm",
        return_value='{"intents":["should not run"]}',
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.RewriteRedisFastpath.match_first_regular_pattern",
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.RewriteRedisFastpath.match_clear_sentence",
    )
    @patch.object(
        _RewriteRouter,
        "build_merged_context_for_rewrite",
        return_value=("Hello", None, 1, 10),
    )
    def test_l1_hit_skips_llm(
        self,
        _mock_build: object,
        mock_l1: object,
        mock_l2: object,
        mock_llm: object,
    ) -> None:
        mock_l1.return_value = {"sentence": "Hello", "workflow": "uds"}
        mock_l2.return_value = None
        req = QueryRequest(query="Hello", session_id="sess-rewrite-test")
        result = _RewriteRouter.run_unified_rewrite(req, gateway_memory=None)
        mock_llm.assert_not_called()
        self.assertEqual(result.rewrite_path, "l1_redis")
        self.assertEqual(len(result.intents), 1)

    @patch.dict(
        os.environ,
        {
            "GATEWAY_REWRITE_LAYER1_ENABLED": "true",
            "GATEWAY_REWRITE_LAYER2_ENABLED": "true",
            "GATEWAY_REWRITE_LAYER3_FORCE": "false",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.call_unified_rewrite_llm",
        return_value='{"intents":["no"]}',
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.RewriteRedisFastpath.match_first_regular_pattern",
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.RewriteRedisFastpath.match_clear_sentence",
    )
    @patch.object(
        _RewriteRouter,
        "build_merged_context_for_rewrite",
        return_value=("order 99999", None, 0, 0),
    )
    def test_l2_hit_when_l1_misses(
        self,
        _mock_build: object,
        mock_l1: object,
        mock_l2: object,
        mock_llm: object,
    ) -> None:
        mock_l1.return_value = None
        mock_l2.return_value = {"workflow": "uds", "_matched_pattern": r"order \d+"}
        req = QueryRequest(query="order 99999", session_id="sess-l2")
        result = _RewriteRouter.run_unified_rewrite(req, gateway_memory=None)
        mock_llm.assert_not_called()
        self.assertEqual(result.rewrite_path, "l2_redis")

    @patch.dict(
        os.environ,
        {
            "GATEWAY_REWRITE_LAYER1_ENABLED": "true",
            "GATEWAY_REWRITE_LAYER3_FORCE": "true",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.call_unified_rewrite_llm",
        return_value='{"rewritten_display":"OK","intents":["OK"]}',
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_implement.RewriteRedisFastpath.match_clear_sentence",
        return_value={"sentence": "x", "workflow": "uds"},
    )
    @patch.object(
        _RewriteRouter,
        "build_merged_context_for_rewrite",
        return_value=("force l3", None, 0, 0),
    )
    def test_layer3_force_always_calls_llm(
        self,
        _mock_build: object,
        _mock_l1: object,
        mock_llm: object,
    ) -> None:
        req = QueryRequest(query="force l3", session_id="sess-l3-force")
        result = _RewriteRouter.run_unified_rewrite(req, gateway_memory=None)
        mock_llm.assert_called_once()
        self.assertEqual(result.rewrite_path, "l3_llm")


if __name__ == "__main__":
    unittest.main()
