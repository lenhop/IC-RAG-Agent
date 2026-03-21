"""
RAG ingest pipeline orchestrator.

Layer 2: Composes search -> load -> clean -> split -> embed -> store.
Optimized for low memory: processes files in batches, avoids loading all at once.
"""

# [ANNOTATION] Disable Chroma telemetry before any chromadb import.
# Prevents "Failed to send telemetry event" errors (PostHog API compatibility issue).
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain_core.documents import Document


def _batch_iter(items: List, batch_size: int):
    """Yield successive batches from items for memory-efficient iteration."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def rag_ingest_pipeline(
    root: str | Path,
    chroma_path: str | Path,
    collection_name: str,
    *,
    search_kwargs: Optional[Dict[str, Any]] = None,
    clean_kwargs: Optional[Dict[str, Any]] = None,
    split_kwargs: Optional[Dict[str, Any]] = None,
    embed_kwargs: Optional[Dict[str, Any]] = None,
    embed_batch_size: int = 16,
    load_batch_size: int = 10,
    reset_db: bool = False,
    limit_files: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> int:
    """
    Full RAG ingest: search files -> load (batched) -> clean -> split -> embed -> store in Chroma.
    Processes files in batches to reduce memory usage.

    Args:
        root: Root directory to search for documents.
        chroma_path: Chroma persist directory.
        collection_name: Chroma collection name.
        search_kwargs: Passed to search_files (extensions, min_size, etc.).
        clean_kwargs: Passed to DocumentCleaner.clean_documents (min_length, etc.).
        split_kwargs: Passed to split_documents (chunk_size, chunk_overlap, language).
        embed_kwargs: Passed to create_embeddings (model_type, model_path, etc.).
        embed_batch_size: Batch size for embedding to avoid OOM.
        load_batch_size: Number of files to load per batch (low-memory mode).
        reset_db: Remove Chroma DB directory before run.
        limit_files: Limit number of files to process.
        project_root: Project root for default model paths.

    Returns:
        Number of chunks stored.
    """
    from .document_cleaners import DocumentCleaner
    from .embeddings import create_embeddings
    from .document_loader import load_documents_from_files
    from .file_search import search_files
    from .splitters import split_documents

    import chromadb
    from chromadb.config import Settings

    search_kwargs = search_kwargs or {}
    clean_kwargs = clean_kwargs or {}
    split_kwargs = split_kwargs or {}
    embed_kwargs = dict(embed_kwargs or {})

    # Step 1: Search files
    print("[Step 1/6] Searching files...")
    file_paths = search_files(root, limit=limit_files, **search_kwargs)
    if not file_paths:
        print("[Step 1/6] No files found.")
        return 0
    print(f"[Step 1/6] Found {len(file_paths)} files.")

    # Step 2: Initialize Chroma and embeddings (before batch loop)
    print("[Step 2/6] Initializing Chroma and embedding model...")
    chroma_path = Path(chroma_path)
    if reset_db and chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)

    # [ANNOTATION] Disable Chroma telemetry to avoid "Failed to send telemetry event" errors.
    # Alternative: set env ANONYMIZED_TELEMETRY=FALSE before running.
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name)
    collection = client.get_collection(name=collection_name)

    root_path = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    model_type = embed_kwargs.pop("model_type", "huggingface")
    embeddings = create_embeddings(model_type, project_root=root_path, **embed_kwargs)
    print("[Step 2/6] Chroma and embedding model ready.")

    # Step 3-6: Process per file batch (load -> split -> clean -> embed -> store)
    cleaner = DocumentCleaner()
    stored = 0
    chunk_id_offset = 0
    num_batches = (len(file_paths) + load_batch_size - 1) // load_batch_size

    for batch_idx, file_batch in enumerate(_batch_iter(file_paths, load_batch_size), 1):
        start_time = datetime.now()
        for file in file_batch:
            print(file)
        print(f"batch_idx: {batch_idx}, num_batches: {num_batches}")
        print(f"  [Step 3/6] Loading batch {batch_idx} ({len(file_batch)} files)...")
        batch_docs = load_documents_from_files(file_batch)
        if not batch_docs:
            continue

        print(f"  [Step 4/6] Splitting batch {batch_idx} ({len(batch_docs)} docs)...")
        chunks = split_documents(batch_docs, **split_kwargs)
        del batch_docs  # Free memory before next step

        if not chunks:
            continue

        print(f"  [Step 5/6] Cleaning batch {batch_idx} ({len(chunks)} chunks)...")
        chunks = cleaner.clean_documents(chunks, **clean_kwargs)
        if not chunks:
            continue

        # Embed and store in sub-batches to limit memory
        print(f"  [Step 6/6] Embedding and storing batch {batch_idx} ({len(chunks)} chunks)...")
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        del chunks  # Free chunks; keep only texts/metadatas for embedding

        for start in range(0, len(texts), embed_batch_size):
            end = min(start + embed_batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = [
                f"{collection_name}_{chunk_id_offset + start + i}"
                for i in range(len(batch_texts))
            ]
            vectors = embeddings.embed_documents(batch_texts)
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=vectors,
            )
            stored += len(batch_texts)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f"  Time taken for batch {batch_idx}: {total_time} seconds")
        chunk_id_offset += len(texts)
        del texts, metadatas  # Free before next file batch
        gc.collect()  # Encourage garbage collection between batches

    print(f"[Done] Stored {stored} chunks total.")
    return stored


def prepare_chunks_for_rag(
    root: str | Path,
    *,
    search_kwargs: Optional[Dict[str, Any]] = None,
    split_kwargs: Optional[Dict[str, Any]] = None,
    clean_kwargs: Optional[Dict[str, Any]] = None,
    limit_files: Optional[int] = None,
) -> List[Document]:
    """
    Search, load, clean, split - return chunks without embedding/storing.
    Useful for inspection or custom embedding flow.

    Args:
        root: Root directory to search.
        search_kwargs: Passed to search_files.
        split_kwargs: Passed to split_documents.
        clean_kwargs: Passed to DocumentCleaner.clean_documents.
        limit_files: Limit number of files.

    Returns:
        List of chunk Documents.
    """
    from .document_cleaners import DocumentCleaner
    from .document_loader import load_documents_from_files
    from .file_search import search_files
    from .splitters import split_documents

    search_kwargs = search_kwargs or {}
    split_kwargs = split_kwargs or {}
    clean_kwargs = clean_kwargs or {}

    print("[Step 1/4] Searching files...")
    file_paths = search_files(root, limit=limit_files, **search_kwargs)
    if not file_paths:
        print("[Step 1/4] No files found.")
        return []

    print(f"[Step 2/4] Loading {len(file_paths)} files...")
    all_docs = load_documents_from_files(file_paths)
    if not all_docs:
        return []

    print(f"[Step 3/4] Splitting {len(all_docs)} documents...")
    chunks = split_documents(all_docs, **split_kwargs)
    if not chunks:
        return []

    print(f"[Step 4/4] Cleaning {len(chunks)} chunks...")
    cleaner = DocumentCleaner()
    return cleaner.clean_documents(chunks, **clean_kwargs)
