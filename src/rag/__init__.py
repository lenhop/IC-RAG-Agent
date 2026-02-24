"""
RAG toolkit - Layer 2 of IC-RAG-Agent.

Combines ai-toolkit (Layer 1) with RAG-specific composition.
"""

from .cleaners import DocumentCleaner
from .embeddings import create_embeddings
from .document_loader import load_documents_from_files
from .file_search import search_files
from .ingest_pipeline import prepare_chunks_for_rag, rag_ingest_pipeline
from .query_pipeline import RAGPipeline, get_collection_count
from .splitters import split_documents

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
]
