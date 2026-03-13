#!/usr/bin/env python3
"""
Load PDF documents into local ChromaDB (documents collection).

Behavior:
- Source root: data/documents (or DOC_LOAD_ROOT)
- File type: PDF only
- Target: local PersistentClient at data/chroma_db/documents
- Truncate before load: always enabled
- Default embedding: local Ollama (DOC_LOAD_EMBED_MODEL=ollama); requires ollama serve
  and an embed model (e.g. ollama pull all-minilm). Override with --embed-model minilm
  if Ollama is unavailable.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.chroma_loaders import bootstrap_project, load_documents_to_chroma, resolve_path

PROJECT_ROOT = bootstrap_project()

DEFAULT_DOC_ROOT = resolve_path("DOC_LOAD_ROOT", str(PROJECT_ROOT / "data" / "documents"), PROJECT_ROOT)
DEFAULT_CHROMA_PATH = resolve_path(
    "CHROMA_DOCUMENTS_PATH",
    str(PROJECT_ROOT / "data" / "chroma_db" / "documents"),
    PROJECT_ROOT,
)
DEFAULT_COLLECTION = os.getenv("CHROMA_DOCUMENTS_COLLECTION", "documents")
DEFAULT_CHUNK_SIZE = int(os.getenv("DOC_LOAD_CHUNK_SIZE", "1024"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DOC_LOAD_CHUNK_OVERLAP", "100"))
DEFAULT_MIN_CHUNK_LENGTH = int(os.getenv("DOC_LOAD_MIN_CHUNK_LENGTH", "20"))
DEFAULT_BATCH_SIZE = int(os.getenv("DOC_LOAD_EMBED_BATCH_SIZE", "16"))
# Default aligns with intent registry: local Ollama + same embed model family.
DEFAULT_EMBED_MODEL = os.getenv("DOC_LOAD_EMBED_MODEL", "ollama")


def _ollama_base_url() -> str:
    """Normalize GATEWAY_REWRITE_OLLAMA_URL to Ollama API base (no /api/generate)."""
    url = (os.getenv("GATEWAY_REWRITE_OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    for suffix in ("/api/generate", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def _embed_kwargs_for_model(embed_model: str) -> dict:
    """Build create_embeddings kwargs for Ollama (URL + model from env)."""
    if (embed_model or "").strip().lower() != "ollama":
        return {}
    model = (
        os.getenv("DOC_LOAD_OLLAMA_MODEL")
        or os.getenv("GATEWAY_INTENT_EMBEDDING_MODEL")
        or "all-minilm"
    )
    return {"ollama_model": model, "ollama_base_url": _ollama_base_url()}


def main() -> int:
    """Parse args and load PDFs into local ChromaDB with mandatory truncation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load PDF documents into local ChromaDB (truncate before load). "
        "Default embed backend: Ollama (run: ollama serve; ollama pull all-minilm)."
    )
    parser.add_argument("--doc-root", default=DEFAULT_DOC_ROOT, help="PDF root directory")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Local Chroma path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-length", type=int, default=DEFAULT_MIN_CHUNK_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "ollama", "qwen3"),
        default=DEFAULT_EMBED_MODEL,
        help="Embedding backend (default: ollama local)",
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    args = parser.parse_args()

    start_time = datetime.now()
    embed_extra = _embed_kwargs_for_model(args.embed_model)
    try:
        stored = load_documents_to_chroma(
            doc_root=args.doc_root,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            project_root=PROJECT_ROOT,
            extensions=[".pdf"],
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_length=args.min_chunk_length,
            embed_model=args.embed_model,
            embed_kwargs_extra=embed_extra or None,
            batch_size=args.batch_size,
            reset_db=True,
            limit_files=args.limit,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load documents: {exc}", file=sys.stderr)
        return 1

    elapsed = (datetime.now() - start_time).total_seconds()
    if stored <= 0:
        print("[WARN] No chunks stored.")
        return 1

    print(f"[OK] Stored {stored} chunks to local Chroma in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
