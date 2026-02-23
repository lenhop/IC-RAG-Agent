#!/usr/bin/env python3
"""
Batch load local PDFs, split, embed, and store into Chroma.

Production script for IC-RAG-Agent. Flow:
  1. Clear existing data in Chroma collection
  2. Iterate over PDF files in a directory (recursive)
  3. Load each PDF, split into chunks
  4. Preprocess chunks (normalize whitespace, filter short/noise)
  5. Embed chunks with local embedding model (default: all-MiniLM-L6-v2)
  6. Store all chunks into Chroma

References: rag_01_load_split_embedding_chroma_qwen3.py, rag_02_chroma_common_operations.py

Embedding model: Use same model for load and RAG query. Default all-MiniLM-L6-v2 is
lightweight (~80MB) for Intel i5 / 16GB RAM. Use --embed-model qwen3 for Qwen3-VL-Embedding-2B.

For RAG (rag_03_05, rag_03_06): use --chroma-path <project>/data/chroma_db/amazon/fba
--collection amazon_fba_features so the RAG pipeline can query the same collection.
"""

import re
import shutil
import sqlite3
import sys
from pathlib import Path

# [ANNOTATION] Path setup: project root and ai-toolkit (scripts/ is under project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

# Config (align with rag_02, rag_03_05)
CHROMA_PERSIST_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / "ebay")
COLLECTION_NAME = "ebay"
PDF_ROOT = str(PROJECT_ROOT / "data" / "documents" / "sales_platform" / "ebay")
# Chunk size: 1024 reduces chunk count vs 512; all-MiniLM-L6-v2 max ~256 tokens (~1000 chars)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
# Embedding batch size: 16 for memory-constrained systems (Intel i5, 16GB RAM)
EMBED_BATCH_SIZE = 16
# Minimum chunk length to keep (filter noise from PDF extraction)
MIN_CHUNK_LENGTH = 20
# Embedding model: "minilm" (default) or "qwen3"
DEFAULT_EMBED_MODEL = "minilm"


def load_embedder(embed_model: str = DEFAULT_EMBED_MODEL):
    """Load embedding model via src.rag.create_embeddings (handles qwen3 path)."""
    from src.rag import create_embeddings
    return create_embeddings(embed_model, project_root=PROJECT_ROOT)


def preprocess_text(text: str) -> str:
    """
    Normalize text before embedding: collapse whitespace, strip, remove control chars.
    Reduces noise from PDF extraction and improves embedding quality.
    """
    if not text or not isinstance(text, str):
        return ""
    # Collapse multiple whitespace (spaces, newlines, tabs) to single space
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    # Remove control characters (optional, keeps printable chars)
    normalized = "".join(c for c in normalized if c.isprintable() or c.isspace())
    return normalized.strip()


def filter_and_preprocess_chunks(
    chunks: list, min_length: int = MIN_CHUNK_LENGTH
) -> list:
    """
    Preprocess chunk content and filter out very short chunks (noise).
    Returns list of (possibly merged) chunks with normalized page_content.
    """
    from langchain_core.documents import Document

    result = []
    for c in chunks:
        content = preprocess_text(c.page_content)
        if len(content) < min_length:
            continue
        # Create new doc with preprocessed content, preserve metadata
        result.append(
            Document(page_content=content, metadata=dict(c.metadata))
        )
    return result


def collect_pdf_paths(root: str) -> list[Path]:
    """Collect all PDF file paths under root (recursive)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.rglob("*.pdf"))


def _is_chroma_schema_error(exc: BaseException) -> bool:
    """Check if error is due to Chroma DB schema mismatch (e.g. old DB version)."""
    if isinstance(exc, sqlite3.OperationalError):
        return "no such column" in str(exc).lower()
    return False


def _reset_chroma_db(path: str) -> None:
    """Remove Chroma DB directory to allow fresh creation with current schema."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
        print(f"[OK] Removed incompatible Chroma DB at {path}")


def clear_chroma_collection(client, collection_name: str) -> None:
    """Remove all data from Chroma collection by deleting and recreating it."""
    try:
        client.delete_collection(name=collection_name)
        print(f"[OK] Deleted existing collection: {collection_name}")
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            print(f"[OK] Collection did not exist: {collection_name}")
        else:
            raise
    client.get_or_create_collection(name=collection_name)
    print(f"[OK] Created fresh collection: {collection_name}")


def load_split_embed_store(
    pdf_root: str,
    chroma_persist_path: str,
    collection_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    min_chunk_length: int = MIN_CHUNK_LENGTH,
    reset_db: bool = False,
    limit_pdfs: int | None = None,
) -> int:
    """
    Load PDFs from root, split, preprocess, embed, store into Chroma.
    Clears Chroma collection before inserting.
    Embeds and stores in batches to avoid OOM with large document sets.
    Returns total number of chunks stored.
    """
    import chromadb
    from ai_toolkit.rag.loaders import load_pdf_document
    from ai_toolkit.rag.splitters import split_document_recursive

    embeddings = load_embedder(embed_model=embed_model)
    pdf_paths = collect_pdf_paths(pdf_root)
    if not pdf_paths:
        print(f"[WARN] No PDF files found under: {pdf_root}")
        return 0
    total_pdfs = len(pdf_paths)
    if limit_pdfs is not None:
        pdf_paths = pdf_paths[:limit_pdfs]
        print(f"[OK] Found {total_pdfs} PDF(s), processing first {len(pdf_paths)} (--limit)")
    else:
        print(f"[OK] Found {len(pdf_paths)} PDF(s) under {pdf_root}")

    # [ANNOTATION] Step 1: Connect to Chroma and clear collection (retry on schema mismatch)
    if reset_db:
        _reset_chroma_db(chroma_persist_path)
    chroma_client = None
    for attempt in range(2):
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
            clear_chroma_collection(chroma_client, collection_name)
            break
        except Exception as e:
            if _is_chroma_schema_error(e):
                print(f"[WARN] Chroma DB schema mismatch: {e}")
                chroma_client = None  # Release reference before removing DB
                _reset_chroma_db(chroma_persist_path)
                if attempt == 0:
                    continue
            raise
    collection = chroma_client.get_collection(name=collection_name)

    all_chunks = []

    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"[OK] Processing {i} of {len(pdf_paths)}: {pdf_path}")
        try:
            docs = load_pdf_document(str(pdf_path))
            if not docs:
                print(f"[WARN] Empty PDF: {pdf_path}")
                continue
            chunks = split_document_recursive(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for c in chunks:
                c.metadata["source"] = str(pdf_path)
            all_chunks.extend(chunks)
            print(f"[OK] Loaded {pdf_path.name}: {len(docs)} pages -> {len(chunks)} chunks")
        except Exception as e:
            print(f"[FAIL] Error loading {pdf_path}: {e}")
            continue

    if not all_chunks:
        print("[WARN] No chunks to store")
        return 0

    # [ANNOTATION] Step 2: Preprocess and filter chunks before embedding
    all_chunks = filter_and_preprocess_chunks(all_chunks, min_length=min_chunk_length)
    if not all_chunks:
        print("[WARN] No chunks remaining after preprocessing")
        return 0
    print(f"[OK] Preprocessed: {len(all_chunks)} chunks (min length {min_chunk_length})")

    # [ANNOTATION] Step 3: Embed and store in batches to avoid OOM
    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    total = len(texts)
    stored = 0

    print(f"[OK] Embedding {total} chunks in batches of {embed_batch_size}...")
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
        print(f"[OK] Stored batch {start // embed_batch_size + 1}: {stored}/{total} chunks")

    print(f"[OK] Stored {stored} chunks into Chroma")
    print(f"[OK] Collection count: {collection.count()}")
    return stored


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load PDFs, embed, store to Chroma (clears before insert)")
    parser.add_argument("--pdf-root", default=PDF_ROOT, help="Root directory to scan for PDFs")
    parser.add_argument("--chroma-path", default=CHROMA_PERSIST_PATH, help="Chroma persist path")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size to avoid OOM (default: {EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Remove existing Chroma DB before run (fixes schema mismatch)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only first N PDFs (for testing)",
    )
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "qwen3"),
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model: minilm (lightweight) or qwen3 (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {CHUNK_OVERLAP})",
    )
    args = parser.parse_args()
    print(args.__dict__)

    load_split_embed_store(
        pdf_root=args.pdf_root,
        chroma_persist_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_batch_size=args.batch_size,
        embed_model=args.embed_model,
        reset_db=args.reset_db,
        limit_pdfs=args.limit,
    )
