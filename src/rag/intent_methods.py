"""
Parallel intent classification methods - four Yes/No signals.

Each method returns True (Yes) or False (No) for document-related intent.
Methods: Documents, Keywords, FAQ, LLM.

Central entry point: run_all_intent_methods() returns a unified dict with all
responses including Chroma retrieval data. Aggregation stays in query_pipeline.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    pass  # embedder, vector_store are generic LangChain objects

# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# TypedDict for response structure (type safety and documentation)
class DocumentsResponse(TypedDict):
    """Documents method response including Chroma retrieval data."""

    yes_no: bool
    min_dist: float
    retrieved_docs: list
    distances: list
    threshold: float


class KeywordsResponse(TypedDict):
    """Keywords method response."""

    yes_no: bool
    matched_signals: list


class FaqResponse(TypedDict):
    """FAQ method response."""

    yes_no: bool
    min_dist: float
    threshold: float


class LlmResponse(TypedDict):
    """LLM method response."""

    yes_no: bool


class IntentMethodsResponse(TypedDict):
    """Unified response from run_all_intent_methods."""

    documents: DocumentsResponse
    keywords: KeywordsResponse
    faq: FaqResponse
    llm: LlmResponse


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


# ---------------------------------------------------------------------------
# Dict-returning methods (used by run_all_intent_methods)
# ---------------------------------------------------------------------------


def documents_method_response(
    retrieved_docs: list,
    distances: list,
    threshold: float,
) -> DocumentsResponse:
    """
    Documents method: judge from pre-computed retrieval results.

    Args:
        retrieved_docs: Retrieved document chunks from Chroma.
        distances: L2 distances from query to each doc (lower = more similar).
        threshold: Distance threshold. Default from RAG_MODE_DISTANCE_THRESHOLD_GENERAL.

    Returns:
        Dict with yes_no, min_dist, retrieved_docs, distances, threshold.
    """
    min_dist = min(distances) if distances else float("inf")
    yes_no = min_dist <= threshold
    return {
        "yes_no": yes_no,
        "min_dist": min_dist,
        "retrieved_docs": retrieved_docs,
        "distances": list(distances),
        "threshold": threshold,
    }


def keywords_method_response(
    question: str,
    project_root: Path | None = None,
) -> KeywordsResponse:
    """
    Keywords method: True if query matches domain keywords/phrases.

    Args:
        question: User question.
        project_root: Optional project root for CSV path resolution.

    Returns:
        Dict with yes_no and matched_signals.
    """
    from src.rag.intent_keywords import match_domain_signals

    root = project_root or PROJECT_ROOT
    matched_signals = match_domain_signals(question, root)
    return {
        "yes_no": len(matched_signals) > 0,
        "matched_signals": matched_signals,
    }


def faq_method_response(
    question_vector: list,
    faq_vectors: list,
    enabled: bool,
    threshold: float | None = None,
) -> FaqResponse:
    """
    FAQ method: True if faq_min_dist < threshold (query similar to FAQ).

    Args:
        question_vector: Embedded user question.
        faq_vectors: Pre-embedded FAQ question vectors.
        enabled: Whether FAQ method is enabled.
        threshold: Similarity threshold. Default from RAG_FAQ_SIMILARITY_THRESHOLD.

    Returns:
        Dict with yes_no, min_dist, threshold.
    """
    if not enabled or not faq_vectors:
        return {
            "yes_no": False,
            "min_dist": float("inf"),
            "threshold": threshold or float(os.getenv("RAG_FAQ_SIMILARITY_THRESHOLD", "0.9")),
        }
    thresh = threshold if threshold is not None else float(
        os.getenv("RAG_FAQ_SIMILARITY_THRESHOLD", "0.9")
    )
    min_dist = _faq_min_distance(question_vector, faq_vectors)
    return {
        "yes_no": min_dist < thresh,
        "min_dist": min_dist,
        "threshold": thresh,
    }


def llm_method_response(question: str, enabled: bool) -> LlmResponse:
    """
    LLM method: True if classifier outputs documents, False if general.

    Args:
        question: User question.
        enabled: Whether LLM method is enabled.

    Returns:
        Dict with yes_no.
    """
    if not enabled:
        return {"yes_no": False}
    from src.rag.intent_classifier import llm_method_yes_no as _llm_yes_no

    return {"yes_no": _llm_yes_no(question)}


# ---------------------------------------------------------------------------
# Backward-compatible bool-returning methods (used by tests)
# ---------------------------------------------------------------------------


def documents_method_yes_no(
    embedder,
    vector_store,
    question: str,
    retrieval_k: int = 5,
    threshold: float | None = None,
) -> bool:
    """
    Documents method: True if min_dist <= threshold (has relevant docs), else False.

    Standalone variant that performs embed+retrieve internally. For pipeline use,
    prefer run_all_intent_methods with pre-computed retrieval results.
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
    """
    return keywords_method_response(question, project_root)["yes_no"]


def faq_method_yes_no(
    question_vector: list,
    faq_vectors: list,
    threshold: float | None = None,
) -> bool:
    """
    FAQ method: True if faq_min_dist < threshold (query similar to FAQ), else False.
    """
    resp = faq_method_response(
        question_vector, faq_vectors, enabled=True, threshold=threshold
    )
    return resp["yes_no"]


def llm_method_yes_no(question: str, model_name: str | None = None) -> bool:
    """
    LLM method: True if classifier outputs documents, False if general.
    Delegates to intent_classifier.llm_method_yes_no.
    """
    from src.rag.intent_classifier import llm_method_yes_no as _llm_yes_no

    return _llm_yes_no(question, model_name)


def _run_with_timing(func, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# Central entry point: run all methods in parallel, return unified dict
# ---------------------------------------------------------------------------


def run_all_intent_methods(
    question_vector: list,
    retrieved_docs: list,
    distances: list,
    question: str,
    faq_vectors: list,
    project_root: Path | None = None,
    threshold: float | None = None,
    faq_enabled: bool = False,
    llm_enabled: bool = False,
    verbose: bool = False,
) -> IntentMethodsResponse:
    """
    Run Documents, Keywords, FAQ, LLM methods in parallel; return unified dict.

    Documents uses pre-computed retrieval results (no embed/retrieve inside).
    FAQ and LLM respect faq_enabled/llm_enabled (return yes_no=False when disabled).

    Args:
        question_vector: Pre-embedded user question.
        retrieved_docs: Retrieved document chunks from Chroma.
        distances: L2 distances from retrieval.
        question: User question (for Keywords, LLM).
        faq_vectors: Pre-embedded FAQ question vectors.
        project_root: Optional project root for CSV path resolution.
        threshold: Distance threshold for Documents. Default from env.
        faq_enabled: Whether FAQ method is enabled.
        llm_enabled: Whether LLM method is enabled.
        verbose: If True, print step timing and four-method results.

    Returns:
        Dict with keys documents, keywords, faq, llm; each has yes_no and method-specific fields.
    """
    root = project_root or PROJECT_ROOT
    thresh = threshold if threshold is not None else float(
        os.getenv("RAG_MODE_DISTANCE_THRESHOLD_GENERAL", "1.0")
    )

    t0 = time.perf_counter()
    max_workers = int(os.getenv("RAG_PARALLEL_INTENT_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_doc = executor.submit(
            _run_with_timing,
            documents_method_response,
            retrieved_docs,
            distances,
            thresh,
        )
        future_kw = executor.submit(_run_with_timing, keywords_method_response, question, root)
        future_faq = executor.submit(
            _run_with_timing,
            faq_method_response,
            question_vector,
            faq_vectors,
            faq_enabled,
        )
        future_llm = executor.submit(
            _run_with_timing, llm_method_response, question, llm_enabled
        )

        doc_resp, doc_elapsed = future_doc.result()
        kw_resp, kw_elapsed = future_kw.result()
        faq_resp, faq_elapsed = future_faq.result()
        llm_resp, llm_elapsed = future_llm.result()

    if verbose:
        print(f"  [Step 5b] Parallel classify: {time.perf_counter() - t0:.2f}s")
        print(
            f"  [Four methods] Documents: {'Yes' if doc_resp['yes_no'] else 'No'} "
            f"(min_dist={doc_resp['min_dist']:.4f} <= {thresh}) ({doc_elapsed:.3f}s)"
        )
        print(
            f"  [Four methods] Keywords:  {'Yes' if kw_resp['yes_no'] else 'No'} "
            f"({kw_elapsed:.3f}s)"
        )
        print(
            f"  [Four methods] FAQ:      {'Yes' if faq_resp['yes_no'] else 'No'} "
            f"({faq_elapsed:.3f}s)"
        )
        print(
            f"  [Four methods] LLM:      {'Yes' if llm_resp['yes_no'] else 'No'} "
            f"({llm_elapsed:.3f}s)"
        )

    return {
        "documents": doc_resp,
        "keywords": kw_resp,
        "faq": faq_resp,
        "llm": llm_resp,
    }
