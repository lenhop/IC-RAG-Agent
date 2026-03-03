"""
RAG toolkit - Layer 2 of IC-RAG-Agent.

Combines ai-toolkit (Layer 1) with RAG-specific composition.
"""

from .document_cleaners import DocumentCleaner
from .embeddings import create_embeddings
from .document_loader import load_documents_from_files
from .file_search import search_files
from .ingest_pipeline import prepare_chunks_for_rag, rag_ingest_pipeline
from .query_pipeline import RAGPipeline, AnswerMode, QueryMode, get_collection_count
from .query_rewriting import rewrite_query, rewrite_query_lightweight
from .splitters import split_documents

# Evaluation (Phase 1.1+)
from .evaluation.generation_metrics import (
    evaluate_faithfulness,
    evaluate_relevance,
    GenerationEvaluator,
)
from .evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    RetrievalEvaluator,
)

__all__ = [
    "DocumentCleaner",
    "create_embeddings",
    "load_documents_from_files",
    "search_files",
    "split_documents",
    "rag_ingest_pipeline",
    "prepare_chunks_for_rag",
    "RAGPipeline",
    "get_collection_count",
    "rewrite_query",
    "rewrite_query_lightweight",
    # Evaluation
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_mrr",
    "RetrievalEvaluator",
    "evaluate_faithfulness",
    "evaluate_relevance",
    "GenerationEvaluator",
]
