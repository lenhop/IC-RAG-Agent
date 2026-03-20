"""
Load files from disk into Chroma via the RAG ingest pipeline.

Delegates document search, load, split, clean, and embed-store to
``src.rag.ingest_pipeline.rag_ingest_pipeline``. This module is the stable
Chroma-facing entry point; scripts should import from here, not from rag internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def load_documents_to_chroma(
    doc_root: str | Path,
    chroma_path: str | Path,
    collection_name: str = "documents",
    *,
    project_root: Path,
    extensions: Optional[List[str]] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    min_chunk_length: int = 20,
    embed_model: str = "ollama",
    embed_kwargs_extra: Optional[dict] = None,
    batch_size: int = 16,
    reset_db: bool = False,
    limit_files: Optional[int] = None,
) -> int:
    """
    Load documents (PDF, TXT, MD, CSV, JSON by default) into Chroma.

    Args:
        doc_root: Root directory to search.
        chroma_path: Chroma persist path.
        collection_name: Collection name.
        project_root: Project root for model paths.
        extensions: File extensions (default: .pdf, .txt, .md, .csv, .json).
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap.
        min_chunk_length: Minimum chunk length to keep.
        embed_model: minilm, ollama, or qwen3.
        embed_kwargs_extra: Extra args for create_embeddings (e.g. ollama_model).
        batch_size: Embedding batch size.
        reset_db: Remove Chroma DB directory before run.
        limit_files: Limit number of files (for testing).

    Returns:
        Number of chunks stored.
    """
    from src.rag.ingest_pipeline import rag_ingest_pipeline

    exts = extensions or [".pdf", ".txt", ".md", ".csv", ".json"]
    exts = [e if e.startswith(".") else f".{e}" for e in exts]

    embed_kwargs: dict = {"model_type": embed_model}
    if embed_kwargs_extra:
        embed_kwargs.update(embed_kwargs_extra)

    return rag_ingest_pipeline(
        root=doc_root,
        chroma_path=chroma_path,
        collection_name=collection_name,
        search_kwargs={"extensions": exts},
        split_kwargs={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        clean_kwargs={"min_length": min_chunk_length},
        embed_kwargs=embed_kwargs,
        embed_batch_size=batch_size,
        reset_db=reset_db,
        limit_files=limit_files,
        project_root=project_root,
    )


def load_pdf_directory_to_chroma(
    doc_root: str | Path,
    chroma_path: str | Path,
    collection_name: str = "documents",
    *,
    project_root: Path,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    min_chunk_length: int = 20,
    embed_model: str = "ollama",
    embed_kwargs_extra: Optional[dict] = None,
    batch_size: int = 16,
    reset_db: bool = False,
    limit_files: Optional[int] = None,
) -> int:
    """
    Convenience wrapper: ingest PDF files only from doc_root.

    Same parameters and return value as load_documents_to_chroma with
    extensions fixed to ``.pdf``.
    """
    return load_documents_to_chroma(
        doc_root,
        chroma_path,
        collection_name,
        project_root=project_root,
        extensions=[".pdf"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_chunk_length,
        embed_model=embed_model,
        embed_kwargs_extra=embed_kwargs_extra,
        batch_size=batch_size,
        reset_db=reset_db,
        limit_files=limit_files,
    )
