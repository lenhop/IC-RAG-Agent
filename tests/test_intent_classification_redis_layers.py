"""
Tests for intent classification Redis L1/L2 layers and L3_FORCE bypass.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.gateway.route_llm.classification.implement_methods import (
    ClassificationImplementMethod,
    IntentResult,
)


class TestIntentClassificationRedisLayers(unittest.TestCase):
    """Redis clear_sentence / regular_patterns short-circuit before keyword/vector/LLM."""

    @patch.dict(
        os.environ,
        {
            "GATEWAY_INTENT_CLASSIFICATION_LAYER1_ENABLED": "true",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER2_ENABLED": "true",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER3_FORCE": "false",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_first_regular_pattern",
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_clear_sentence",
    )
    def test_l1_hit_uses_redis_workflow(
        self,
        mock_l1: object,
        mock_l2: object,
    ) -> None:
        mock_l1.return_value = {"sentence": "hello", "workflow": "uds"}
        mock_l2.return_value = None
        result = ClassificationImplementMethod().detect("hello")
        self.assertEqual(result.workflow, "uds")
        self.assertEqual(result.source, "redis_clear_sentence")
        mock_l2.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "GATEWAY_INTENT_CLASSIFICATION_LAYER1_ENABLED": "true",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER2_ENABLED": "true",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER3_FORCE": "false",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_first_regular_pattern",
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_clear_sentence",
    )
    def test_l2_used_when_l1_misses(
        self,
        mock_l1: object,
        mock_l2: object,
    ) -> None:
        mock_l1.return_value = None
        mock_l2.return_value = {"workflow": "sp_api", "_matched_pattern": r"\d+"}
        result = ClassificationImplementMethod().detect("order 123")
        self.assertEqual(result.workflow, "sp_api")
        self.assertEqual(result.source, "redis_regular_patterns")

    @patch.dict(
        os.environ,
        {
            "GATEWAY_INTENT_CLASSIFICATION_LAYER1_ENABLED": "true",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER3_FORCE": "true",
        },
        clear=False,
    )
    @patch.object(ClassificationImplementMethod, "keyword_classification_method")
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_clear_sentence",
        return_value={"workflow": "uds_should_not_use"},
    )
    def test_layer3_force_skips_redis(
        self,
        _mock_l1: object,
        mock_kw: object,
    ) -> None:
        mock_kw.return_value = IntentResult(
            intent_name="kw",
            workflow="from_keyword",
            confidence="high",
            source="keyword",
        )
        result = ClassificationImplementMethod().detect("anything")
        self.assertEqual(result.workflow, "from_keyword")
        self.assertEqual(result.source, "keyword")

    @patch.dict(
        os.environ,
        {
            "GATEWAY_INTENT_CLASSIFICATION_LAYER1_ENABLED": "false",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER2_ENABLED": "false",
            "GATEWAY_INTENT_CLASSIFICATION_LAYER3_FORCE": "false",
        },
        clear=False,
    )
    @patch(
        "src.gateway.route_llm.rewriting.rewrite_redis_fastpath.RewriteRedisFastpath.match_clear_sentence",
        return_value={"workflow": "uds"},
    )
    @patch.object(ClassificationImplementMethod, "keyword_classification_method")
    def test_layers_disabled_skips_redis(
        self,
        mock_kw: object,
        _mock_l1: object,
    ) -> None:
        mock_kw.return_value = IntentResult(
            intent_name="k",
            workflow="kw_only",
            source="keyword",
        )
        result = ClassificationImplementMethod().detect("x")
        self.assertEqual(result.workflow, "kw_only")


if __name__ == "__main__":
    unittest.main()
