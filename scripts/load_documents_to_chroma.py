#!/usr/bin/env python3
"""
Batch load documents (PDF, TXT, MD, CSV, JSON), split, embed, and store into Chroma.

Uses src/rag toolkits: search_files, load_documents_from_files, DocumentCleaner,
split_documents, create_embeddings, rag_ingest_pipeline.

Flow:
  1. Search files by extension, size, etc.
  2. Load documents (dispatch by extension)
  3. Split into chunks
  4. Clean (normalize whitespace, filter short chunks)
  5. Embed with local model (HuggingFace or Ollama)
  6. Store in Chroma

Embedding model: Use same model for load and RAG query.
  - minilm: all-MiniLM-L6-v2 (lightweight, ~80MB)
  - ollama: Ollama all-minilm (requires ollama serve)
  - qwen3: Qwen3-VL-Embedding-2B (higher quality, more RAM)
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime

# [ANNOTATION] Path setup: project root and ai-toolkit
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

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Config: resolved from .env with fallbacks
def _resolve_path(env_key: str, default: str) -> str:
    """Resolve path from env; if relative, join with PROJECT_ROOT."""
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


CHROMA_PERSIST_PATH = _resolve_path(
    "CHROMA_DOCUMENTS_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "documents")
)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
DOC_ROOT = _resolve_path(
    "DOC_LOAD_ROOT", str(PROJECT_ROOT / "data" / "documents")
)
DEFAULT_EXTENSIONS = [
    e.strip() for e in os.getenv("DOC_LOAD_EXTENSIONS", ".pdf,.txt,.md,.csv,.json").split(",")
]
CHUNK_SIZE = int(os.getenv("DOC_LOAD_CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("DOC_LOAD_CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = int(os.getenv("DOC_LOAD_EMBED_BATCH_SIZE", "16"))
MIN_CHUNK_LENGTH = int(os.getenv("DOC_LOAD_MIN_CHUNK_LENGTH", "20"))


def main() -> int:
    """Run RAG ingest pipeline via src/rag toolkits."""
    start_time = datetime.now()
    import argparse

    parser = argparse.ArgumentParser(
        description="Load documents (PDF, TXT, MD, CSV, JSON), embed, store to Chroma"
    )
    parser.add_argument(
        "--doc-root",
        default=DOC_ROOT,
        help=f"Root directory to search (default: {DOC_ROOT})",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_PERSIST_PATH,
        help="Chroma persist path",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Chroma collection name",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help=f"File extensions to include (default: {DEFAULT_EXTENSIONS})",
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
        help=f"Chunk overlap (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "ollama", "qwen3"),
        default="minilm",
        help="Embedding model: minilm, ollama, or qwen3",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Remove Chroma DB before run (fixes schema mismatch)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only first N files (for testing)",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=MIN_CHUNK_LENGTH,
        help=f"Minimum chunk length to keep (default: {MIN_CHUNK_LENGTH})",
    )
    args = parser.parse_args()

    from src.rag import rag_ingest_pipeline, search_files

    # Normalize extensions: ensure leading dot
    exts = [e if e.startswith(".") else f".{e}" for e in args.extensions]

    file_paths = search_files(args.doc_root, extensions=exts, limit=args.limit)
    if not file_paths:
        print(f"[WARN] No files found under {args.doc_root} with extensions {exts}")
        return 1

    print(f"[OK] Doc root: {args.doc_root}")
    print(f"[OK] Found {len(file_paths)} file(s)")
    print(f"[OK] Extensions: {exts}")
    print(f"[OK] Embed model: {args.embed_model}")

    stored = rag_ingest_pipeline(
        root=args.doc_root,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        search_kwargs={"extensions": exts},
        split_kwargs={
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
        clean_kwargs={"min_length": args.min_chunk_length},
        embed_kwargs={"model_type": args.embed_model},
        embed_batch_size=args.batch_size,
        reset_db=args.reset_db,
        limit_files=args.limit,
        project_root=PROJECT_ROOT,
    )
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"[OK] Time taken: {total_time} seconds")  
    if stored == 0:
        print("[WARN] No chunks stored (empty documents or all filtered)")
        return 1
    print(f"[OK] Stored {stored} chunks into Chroma")
    return 0 if stored >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
