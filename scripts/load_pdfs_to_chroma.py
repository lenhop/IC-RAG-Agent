#!/usr/bin/env python3
"""
Batch load local PDFs, split, embed, and store into Chroma.

Production script for IC-RAG-Agent. Flow:
  1. Clear existing data in Chroma collection
  2. Iterate over PDF files in a directory (recursive)
  3. Load each PDF, split into chunks
  4. Embed chunks with local Qwen3-VL-Embedding-2B
  5. Store all chunks into Chroma

References: rag_01_load_split_embedding_chroma_qwen3.py, rag_02_chroma_common_operations.py
"""

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

# [ANNOTATION] Qwen3-VL-Embedding scripts must be on path
_model_scripts = PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B" / "scripts"
if _model_scripts.is_dir():
    sys.path.insert(0, str(_model_scripts))

# Config (align with rag_02, rag_03_05)
CHROMA_PERSIST_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / "amazon")
COLLECTION_NAME = "amazon"
PDF_ROOT = str(PROJECT_ROOT / "data" / "documents" / "sales_platform" / "amazon")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _patch_torch_autocast() -> None:
    """Patch torch.is_autocast_enabled for PyTorch < 2.3 (Qwen embedding compatibility)."""
    import torch

    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return
    try:
        torch.is_autocast_enabled("cpu")
    except TypeError:
        original = torch.is_autocast_enabled

        def _patched(device_type=None):
            return original()

        _patched._qwen_patched = True
        torch.is_autocast_enabled = _patched


def load_embedder():
    """Load local Qwen3-VL-Embedding-2B via ai-toolkit."""
    _patch_torch_autocast()
    from ai_toolkit.models import LocalQwenEmbeddings

    model_path = str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B")
    return LocalQwenEmbeddings(model_path)


def collect_pdf_paths(root: str) -> list[Path]:
    """Collect all PDF file paths under root (recursive)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.rglob("*.pdf"))


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
) -> int:
    """
    Load PDFs from root, split, embed, store into Chroma.
    Clears Chroma collection before inserting.
    Returns total number of chunks stored.
    """
    import chromadb
    from ai_toolkit.rag.loaders import load_pdf_document
    from ai_toolkit.rag.splitters import split_document_recursive

    embeddings = load_embedder()
    pdf_paths = collect_pdf_paths(pdf_root)
    if not pdf_paths:
        print(f"[WARN] No PDF files found under: {pdf_root}")
        return 0

    print(f"[OK] Found {len(pdf_paths)} PDF(s) under {pdf_root}")

    # [ANNOTATION] Step 1: Clear Chroma before inserting
    chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
    clear_chroma_collection(chroma_client, collection_name)
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

    # [ANNOTATION] Step 2: Embed and store in batches
    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ids = [f"{collection_name}_{i}" for i in range(len(all_chunks))]

    print(f"[OK] Embedding {len(texts)} chunks...")
    vectors = embeddings.embed_documents(texts)

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=vectors,
    )
    print(f"[OK] Stored {len(texts)} chunks into Chroma")
    print(f"[OK] Collection count: {collection.count()}")
    return len(texts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load PDFs, embed, store to Chroma (clears before insert)")
    parser.add_argument("--pdf-root", default=PDF_ROOT, help="Root directory to scan for PDFs")
    parser.add_argument("--chroma-path", default=CHROMA_PERSIST_PATH, help="Chroma persist path")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name")
    args = parser.parse_args()

    load_split_embed_store(
        pdf_root=args.pdf_root,
        chroma_persist_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
