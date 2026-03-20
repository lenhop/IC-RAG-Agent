"""
Chroma toolkit — public interface for loading and verifying local PersistentClient stores.

This package is the tool layer: scripts and application code should import from
``src.chroma`` for Chroma ingest/verify operations. Environment setup lives in
``src.utils`` (``bootstrap_project``, ``resolve_path``); they are re-exported here
for convenience. The ``src.rag`` package may compose these tools but must not be
required by load scripts.

Public API (stable):
    bootstrap_project, resolve_path (re-exported from src.utils)
    load_csv_column_from_file, load_csv_column_to_chroma
    load_documents_to_chroma, load_pdf_directory_to_chroma
    load_vector_registry_local, resolve_registry_csv_path, resolve_registry_chroma_path
    verify_collection_minimum, verify_default_project_chroma_stores
"""

from __future__ import annotations

from src.utils import bootstrap_project, resolve_path
from .ingest_csv import load_csv_column_from_file, load_csv_column_to_chroma
from .ingest_documents import load_documents_to_chroma, load_pdf_directory_to_chroma
from .intent_registry import (
    load_vector_registry_local,
    resolve_registry_chroma_path,
    resolve_registry_csv_path,
)
from .verify import verify_collection_minimum, verify_default_project_chroma_stores

__all__ = [
    "bootstrap_project",
    "resolve_path",
    "load_csv_column_from_file",
    "load_csv_column_to_chroma",
    "load_documents_to_chroma",
    "load_pdf_directory_to_chroma",
    "load_vector_registry_local",
    "resolve_registry_csv_path",
    "resolve_registry_chroma_path",
    "verify_collection_minimum",
    "verify_default_project_chroma_stores",
]
