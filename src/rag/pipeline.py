"""
RAG ingest pipeline orchestrator.

Layer 2: Composes search -> load -> clean -> split -> embed -> store.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


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
    reset_db: bool = False,
    limit_files: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> int:
    """
    Full RAG ingest: search files -> load -> clean -> split -> embed -> store in Chroma.

    Args:
        root: Root directory to search for documents.
        chroma_path: Chroma persist directory.
        collection_name: Chroma collection name.
        search_kwargs: Passed to search_files (extensions, min_size, etc.).
        clean_kwargs: Passed to DocumentCleaner.clean_documents (min_length, etc.).
        split_kwargs: Passed to split_documents (chunk_size, chunk_overlap, language).
        embed_kwargs: Passed to create_embeddings (model_type, model_path, etc.).
        embed_batch_size: Batch size for embedding to avoid OOM.
        reset_db: Remove Chroma DB directory before run.
        limit_files: Limit number of files to process.
        project_root: Project root for default model paths.

    Returns:
        Number of chunks stored.
    """
    from .cleaners import DocumentCleaner
    from .embeddings import create_embeddings
    from .document_loader import load_documents_from_files
    from .file_search import search_files
    from .splitters import split_documents

    import chromadb

    search_kwargs = search_kwargs or {}
    clean_kwargs = clean_kwargs or {}
    split_kwargs = split_kwargs or {}
    embed_kwargs = dict(embed_kwargs or {})

    # Step 1: Search files
    file_paths = search_files(root, limit=limit_files, **search_kwargs)
    if not file_paths:
        return 0

    # Step 2: Load documents
    all_docs = load_documents_from_files(file_paths)
    if not all_docs:
        return 0

    # Step 3: Split
    chunks = split_documents(all_docs, **split_kwargs)
    if not chunks:
        return 0

    # Step 4: Clean
    cleaner = DocumentCleaner()
    chunks = cleaner.clean_documents(chunks, **clean_kwargs)
    if not chunks:
        return 0

    # Step 5: Embed and store
    root_path = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    model_type = embed_kwargs.pop("model_type", "huggingface")
    embeddings = create_embeddings(model_type, project_root=root_path, **embed_kwargs)

    chroma_path = Path(chroma_path)
    if reset_db and chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name)
    collection = client.get_collection(name=collection_name)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    total = len(texts)
    stored = 0

    for start in range(0, total, embed_batch_size):
        end = min(start + embed_batch_size, total)
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = [f"{collection_name}_{start + i}" for i in range(len(batch_texts))]
        vectors = embeddings.embed_documents(batch_texts)
        collection.add(
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=vectors,
        )
        stored += len(batch_texts)

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
    from .cleaners import DocumentCleaner
    from .document_loader import load_documents_from_files
    from .file_search import search_files
    from .splitters import split_documents

    search_kwargs = search_kwargs or {}
    split_kwargs = split_kwargs or {}
    clean_kwargs = clean_kwargs or {}

    file_paths = search_files(root, limit=limit_files, **search_kwargs)
    if not file_paths:
        return []

    all_docs = load_documents_from_files(file_paths)
    if not all_docs:
        return []

    chunks = split_documents(all_docs, **split_kwargs)
    if not chunks:
        return []

    cleaner = DocumentCleaner()
    return cleaner.clean_documents(chunks, **clean_kwargs)
