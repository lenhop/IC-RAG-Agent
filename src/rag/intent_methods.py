"""
Parallel intent classification methods - four Yes/No signals.

Each method returns True (Yes) or False (No) for document-related intent.
Methods: Documents, Keywords, FAQ, LLM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # embedder, vector_store are generic LangChain objects

# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _faq_min_distance(question_vector: list, faq_vectors: list) -> float:
    """
    Return min L2 distance from question_vector to FAQ vectors (lower = more similar).

    Uses same L2 metric as Chroma. Returns float('inf') if faq_vectors empty.
    """
    if not question_vector or not faq_vectors:
        return float("inf")

    def l2_dist(a: list, b: list) -> float:
        if len(a) != len(b):
            return float("inf")
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    return min(l2_dist(question_vector, fv) for fv in faq_vectors)


def documents_method_yes_no(
    embedder,
    vector_store,
    question: str,
    retrieval_k: int = 5,
    threshold: float | None = None,
) -> bool:
    """
    Documents method: True if min_dist <= threshold (has relevant docs), else False.

    Args:
        embedder: LangChain embeddings model.
        vector_store: Chroma vector store.
        question: User question (already rewritten).
        retrieval_k: Number of docs to retrieve.
        threshold: Distance threshold. Default from RAG_MODE_DISTANCE_THRESHOLD_GENERAL.

    Returns:
        True if relevant documents found, False otherwise.
    """
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents

    thresh = threshold if threshold is not None else float(
        os.getenv("RAG_MODE_DISTANCE_THRESHOLD_GENERAL", "1.0")
    )
    question_vector = embedder.embed_query(question)
    collection = get_chroma_collection(vector_store)
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=retrieval_k,
        include=["documents", "metadatas", "distances"],
    )
    distances = results.get("distances", [[]])
    if not distances or not distances[0]:
        return False
    min_dist = min(distances[0])
    return min_dist <= thresh


def keywords_method_yes_no(question: str, project_root: Path | None = None) -> bool:
    """
    Keywords method: True if query matches domain keywords/phrases, else False.

    Args:
        question: User question.
        project_root: Optional project root for CSV path resolution.

    Returns:
        True if domain signals matched, False otherwise.
    """
    from src.rag.intent_keywords import match_domain_signals

    root = project_root or PROJECT_ROOT
    signals = match_domain_signals(question, root)
    return len(signals) > 0


def faq_method_yes_no(
    question_vector: list,
    faq_vectors: list,
    threshold: float | None = None,
) -> bool:
    """
    FAQ method: True if faq_min_dist < threshold (query similar to FAQ), else False.

    Args:
        question_vector: Embedded user question.
        faq_vectors: Pre-embedded FAQ question vectors.
        threshold: Similarity threshold. Default from RAG_FAQ_SIMILARITY_THRESHOLD.

    Returns:
        True if FAQ match, False otherwise.
    """
    if not faq_vectors:
        return False
    thresh = threshold if threshold is not None else float(
        os.getenv("RAG_FAQ_SIMILARITY_THRESHOLD", "0.9")
    )
    min_dist = _faq_min_distance(question_vector, faq_vectors)
    return min_dist < thresh


def llm_method_yes_no(question: str, model_name: str | None = None) -> bool:
    """
    LLM method: True if classifier outputs documents, False if general.

    Delegates to intent_classifier.llm_method_yes_no.
    """
    from src.rag.intent_classifier import llm_method_yes_no as _llm_yes_no

    return _llm_yes_no(question, model_name)
