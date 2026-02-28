"""
Distance Threshold - Unit tests for the answer mode classifier.

Tests classify_answer_mode_sequential with distance threshold and
keyword fast path logic. No Chroma/Ollama required.

Run:
  pytest tests/test_intent_distance_threshold.py -v
  python -m pytest tests/test_intent_distance_threshold.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.query_pipeline import classify_answer_mode_sequential


# Mock doc for has_docs=True cases
class _MockDoc:
    """Minimal mock document for retrieval count."""

    def __init__(self):
        self.page_content = "mock"
        self.metadata = {}


def _mock_docs(n: int) -> list:
    """Return list of n mock docs."""
    return [_MockDoc() for _ in range(n)]


class TestKeywordFastPath:
    """Prefix + no domain -> general (skip retrieval)."""

    def test_what_is_no_domain_returns_general(self):
        """'what is X?' with no domain signals -> general."""
        result = classify_answer_mode_sequential(
            question="what is machine learning?",
            retrieved_docs=[],
            general_prefixes=["what is", "define", "什么是"],
            domain_signals=[],
        )
        assert result == "general"

    def test_define_no_domain_returns_general(self):
        """'define X' with no domain signals -> general."""
        result = classify_answer_mode_sequential(
            question="define algorithm",
            retrieved_docs=[],
            general_prefixes=["what is", "define", "什么是"],
            domain_signals=[],
        )
        assert result == "general"

    def test_what_is_with_domain_does_not_short_circuit(self):
        """'what is FBA?' has domain signal -> does not short-circuit to general."""
        result = classify_answer_mode_sequential(
            question="what is FBA fee?",
            retrieved_docs=_mock_docs(3),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[0.5, 0.6, 0.7],  # good matches
        )
        # Has domain + docs + good distance -> hybrid
        assert result == "hybrid"


class TestDistanceThreshold:
    """has_docs + min_dist > threshold -> general."""

    @patch.dict(os.environ, {"RAG_DISTANCE_THRESHOLD_ENABLED": "true"})
    def test_high_distance_returns_general(self):
        """Docs retrieved but min_dist > 1.0 -> general (docs not relevant)."""
        result = classify_answer_mode_sequential(
            question="What is Amazon's policy on quantum computing?",
            retrieved_docs=_mock_docs(3),
            general_prefixes=["what is", "define"],
            domain_signals=["amazon"],
            distances=[1.2, 1.5, 1.8],  # all above default threshold 1.0
        )
        assert result == "general"

    @patch.dict(os.environ, {"RAG_DISTANCE_THRESHOLD_ENABLED": "true"})
    def test_low_distance_returns_hybrid(self):
        """Docs retrieved and min_dist <= 1.0 -> hybrid."""
        result = classify_answer_mode_sequential(
            question="What is FBA fee?",
            retrieved_docs=_mock_docs(3),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[0.4, 0.5, 0.6],
        )
        assert result == "hybrid"

    @patch.dict(os.environ, {"RAG_DISTANCE_THRESHOLD_ENABLED": "true"})
    def test_distance_at_threshold_returns_hybrid(self):
        """min_dist == threshold (1.0) -> hybrid (boundary inclusive)."""
        result = classify_answer_mode_sequential(
            question="FBA inventory management",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[1.0, 1.1],
        )
        assert result == "hybrid"

    @patch.dict(
        os.environ,
        {
            "RAG_DISTANCE_THRESHOLD_ENABLED": "true",
            "RAG_MODE_DISTANCE_THRESHOLD_GENERAL": "0.8",
        },
    )
    def test_custom_threshold(self):
        """Custom threshold 0.8: min_dist 0.9 -> general."""
        result = classify_answer_mode_sequential(
            question="Amazon FBA",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["amazon"],
            distances=[0.9, 1.0],
        )
        assert result == "general"

    @patch.dict(os.environ, {"RAG_DISTANCE_THRESHOLD_ENABLED": "false"})
    def test_distance_disabled_falls_back_to_sequential(self):
        """When distance threshold disabled, has_docs -> hybrid regardless of distance."""
        result = classify_answer_mode_sequential(
            question="Amazon quantum computing",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["amazon"],
            distances=[1.5, 1.8],  # high distance
        )
        # Distance disabled: no distance check, has_docs -> hybrid
        assert result == "hybrid"

    def test_no_distances_falls_back_to_sequential(self):
        """When distances not provided, has_docs -> hybrid (sequential fallback)."""
        with patch.dict(os.environ, {"RAG_DISTANCE_THRESHOLD_ENABLED": "true"}):
            result = classify_answer_mode_sequential(
                question="Amazon FBA",
                retrieved_docs=_mock_docs(2),
                general_prefixes=["what is", "define"],
                domain_signals=["amazon"],
                distances=None,  # not provided
            )
        assert result == "hybrid"


class TestBaseLogic:
    """0 docs, domain signals, etc."""

    def test_zero_docs_with_domain_returns_documents(self):
        """0 docs + domain signals -> documents (surface 'no document found')."""
        result = classify_answer_mode_sequential(
            question="FBA fee structure",
            retrieved_docs=[],
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
        )
        assert result == "documents"

    def test_zero_docs_no_domain_returns_general(self):
        """0 docs + no domain -> general."""
        result = classify_answer_mode_sequential(
            question="How does pricing work?",
            retrieved_docs=[],
            general_prefixes=["what is", "define"],
            domain_signals=[],
        )
        assert result == "general"
