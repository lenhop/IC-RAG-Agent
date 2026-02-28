"""
Unit tests for FAQ similarity and LLM gray-zone judgment.

Tests load_faq_questions, _faq_similarity_score, and classify_answer_mode_sequential
with FAQ similarity and LLM gray-zone logic. No Chroma/Ollama required for most tests.
LLM gray-zone zero-shot tests mock the pipeline when transformers not installed.

Run:
  pytest tests/test_intent_faq_llm.py -v
  python -m pytest tests/test_intent_faq_llm.py -v -s
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

from src.rag.query_pipeline import _faq_similarity_score, classify_answer_mode_sequential
from src.rag.faq_loader import load_faq_questions


# Mock doc for has_docs=True cases
class _MockDoc:
    """Minimal mock document for retrieval count."""

    def __init__(self):
        self.page_content = "mock"
        self.metadata = {}


def _mock_docs(n: int) -> list:
    """Return list of n mock docs."""
    return [_MockDoc() for _ in range(n)]


class TestFaqLoader:
    """load_faq_questions from CSV."""

    def test_load_faq_questions_returns_questions(self):
        """load_faq_questions returns non-empty list when CSV exists."""
        questions = load_faq_questions(PROJECT_ROOT)
        # amazon_fqa.csv has 10 rows (excluding header)
        assert isinstance(questions, list)
        if Path(PROJECT_ROOT / "data/intent_classification/fqa/amazon_fqa.csv").exists():
            assert len(questions) >= 1
            assert all(isinstance(q, str) and q.strip() for q in questions)

    def test_load_faq_questions_missing_file_returns_empty(self):
        """load_faq_questions returns [] when path points to non-existent file."""
        with patch.dict(os.environ, {"RAG_FAQ_CSV": "data/nonexistent/faq.csv"}):
            questions = load_faq_questions(PROJECT_ROOT)
        assert questions == []


class TestFaqSimilarityScore:
    """_faq_similarity_score returns min L2 distance."""

    def test_empty_faq_returns_inf(self):
        """Empty faq_vectors -> inf."""
        result = _faq_similarity_score([0.1, 0.2, 0.3], [])
        assert result == float("inf")

    def test_empty_question_returns_inf(self):
        """Empty question_vector -> inf."""
        result = _faq_similarity_score([], [[0.1, 0.2, 0.3]])
        assert result == float("inf")

    def test_same_vector_returns_zero(self):
        """Identical vectors -> L2 distance 0."""
        v = [1.0, 0.0, 0.0]
        result = _faq_similarity_score(v, [v])
        assert result == 0.0

    def test_different_vectors_returns_positive(self):
        """Different vectors -> positive L2 distance."""
        q = [1.0, 0.0, 0.0]
        f = [0.0, 1.0, 0.0]
        result = _faq_similarity_score(q, [f])
        assert result > 0
        assert abs(result - (2 ** 0.5)) < 0.001  # L2 for orthogonal unit vectors


class TestFaqSimilarityOverride:
    """FAQ similarity overrides to hybrid when enabled."""

    @patch.dict(
        os.environ,
        {
            "RAG_FAQ_SIMILARITY_ENABLED": "true",
            "RAG_FAQ_SIMILARITY_THRESHOLD": "0.9",
            "RAG_DISTANCE_THRESHOLD_ENABLED": "true",
        },
    )
    def test_faq_match_overrides_to_hybrid(self):
        """When faq_min_dist < threshold, return hybrid (FAQ match)."""
        # Use identical vectors so L2=0 < 0.9
        qv = [0.5, 0.5, 0.0]
        fv = [qv]
        result = classify_answer_mode_sequential(
            question="What is FBA fee?",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[1.5, 1.8],  # would be general by distance
            question_vector=qv,
            faq_vectors=fv,
        )
        assert result == "hybrid"

    @patch.dict(
        os.environ,
        {
            "RAG_FAQ_SIMILARITY_ENABLED": "true",
            "RAG_FAQ_SIMILARITY_THRESHOLD": "0.5",
            "RAG_DISTANCE_THRESHOLD_ENABLED": "true",
        },
    )
    def test_faq_no_match_uses_distance(self):
        """When faq_min_dist >= threshold, use distance-based result."""
        # Orthogonal vectors: L2 = sqrt(2) > 0.5
        qv = [1.0, 0.0, 0.0]
        fv = [[0.0, 1.0, 0.0]]
        result = classify_answer_mode_sequential(
            question="What is FBA?",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[0.4, 0.5],  # good distance -> hybrid
            question_vector=qv,
            faq_vectors=fv,
        )
        assert result == "hybrid"

    def test_faq_disabled_ignores_faq_vectors(self):
        """When RAG_FAQ_SIMILARITY_ENABLED=false, FAQ vectors are ignored."""
        with patch.dict(os.environ, {"RAG_FAQ_SIMILARITY_ENABLED": "false", "RAG_DISTANCE_THRESHOLD_ENABLED": "true"}):
            result = classify_answer_mode_sequential(
                question="FBA fee",
                retrieved_docs=_mock_docs(2),
                general_prefixes=["what is", "define"],
                domain_signals=["fba"],
                distances=[1.5, 1.8],  # high -> general
                question_vector=[0.1] * 384,  # would match if FAQ enabled
                faq_vectors=[[0.1] * 384],
            )
        assert result == "general"


class TestLlmGrayZone:
    """Zero-shot when distance in gray zone."""

    @patch.dict(
        os.environ,
        {
            "RAG_DISTANCE_THRESHOLD_ENABLED": "true",
            "RAG_LLM_GRAY_ZONE_ENABLED": "true",
            "RAG_DISTANCE_GRAY_ZONE_MARGIN": "0.2",
            "RAG_MODE_DISTANCE_THRESHOLD_GENERAL": "1.0",
        },
    )
    @patch("src.rag.intent_classifier.classify_intent")
    def test_gray_zone_invokes_zero_shot(self, mock_zero_shot):
        """When min_dist in gray zone [0.8, 1.2], zero-shot is called."""
        mock_zero_shot.return_value = "documents"
        result = classify_answer_mode_sequential(
            question="FBA manual processing fee",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[1.0, 1.1],  # in gray zone [0.8, 1.2]
        )
        mock_zero_shot.assert_called_once()
        assert result == "documents"

    @patch.dict(
        os.environ,
        {
            "RAG_DISTANCE_THRESHOLD_ENABLED": "true",
            "RAG_LLM_GRAY_ZONE_ENABLED": "false",
            "RAG_DISTANCE_GRAY_ZONE_MARGIN": "0.2",
        },
    )
    def test_llm_disabled_uses_distance_in_gray_zone(self):
        """When RAG_LLM_GRAY_ZONE_ENABLED=false, gray zone uses distance (hybrid)."""
        result = classify_answer_mode_sequential(
            question="FBA fee",
            retrieved_docs=_mock_docs(2),
            general_prefixes=["what is", "define"],
            domain_signals=["fba"],
            distances=[1.0, 1.1],  # in gray zone, but LLM disabled
        )
        assert result == "hybrid"
