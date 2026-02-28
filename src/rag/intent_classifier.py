"""
Zero-shot 2-way intent classifier for documents/general.

Uses HuggingFace zero-shot-classification pipeline with NLI model
(e.g. distilbert-base-uncased-finetuned-mnli) for parallel intent workflow.
Per ANSWER_MODEL_IDENTITY_NEW.md: LLM method returns documents or general only (no hybrid).
"""

from __future__ import annotations

import os
from typing import Literal

IntentLabel = Literal["documents", "general"]

# Lazy-loaded pipeline cache
_pipeline_cache: dict[str, object] = {}


def classify_intent(question: str, model_name: str) -> IntentLabel | None:
    """
    Classify question into documents or general using zero-shot NLI (2-way).

    Args:
        question: User question string.
        model_name: HuggingFace model for zero-shot (e.g. distilbert-base-uncased-finetuned-mnli).

    Returns:
        Top label from pipeline ("documents" or "general"), or None if transformers
        not installed or pipeline fails.
    """
    try:
        from transformers import pipeline
    except ImportError:
        return None

    if not question or not str(question).strip():
        return None

    if model_name not in _pipeline_cache:
        try:
            _pipeline_cache[model_name] = pipeline(
                "zero-shot-classification",
                model=model_name,
            )
        except Exception:
            return None

    pipe = _pipeline_cache[model_name]
    candidate_labels = ["documents", "general"]

    try:
        result = pipe(question, candidate_labels=candidate_labels)
        if result and "labels" in result and result["labels"]:
            top = result["labels"][0]
            if top in candidate_labels:
                return top
    except Exception:
        pass
    return None


def llm_method_yes_no(question: str, model_name: str | None = None) -> bool:
    """
    LLM method for parallel intent workflow: returns True if documents, False if general.

    Args:
        question: User question string.
        model_name: HuggingFace model. Default from RAG_INTENT_CLASSIFIER_MODEL env.

    Returns:
        True if classifier outputs "documents", False otherwise (general or failure).
    """
    model = model_name or os.getenv("RAG_INTENT_CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-mnli")
    result = classify_intent(question, model)
    return result == "documents"


def invalidate_intent_classifier_cache() -> None:
    """Clear pipeline cache (for tests or model reload)."""
    global _pipeline_cache
    _pipeline_cache = {}
