#!/usr/bin/env python3
"""
Load intent registry CSV into local ChromaDB (intent_registry collection).

Behavior:
- Source CSV: data/intent_classification/vector_retrieval/vector_intent_registry.csv
- Target: local PersistentClient at data/chroma_db/intent_registry
- Truncate before load: always enabled
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.chroma_loaders import bootstrap_project
from src.rag.vector_registry_loader import load_vector_registry_local, resolve_registry_csv_path

PROJECT_ROOT = bootstrap_project()
DEFAULT_CHROMA_PATH = Path(
    os.getenv("CHROMA_INTENT_REGISTRY_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "intent_registry"))
)
DEFAULT_COLLECTION = os.getenv("CHROMA_INTENT_REGISTRY_COLLECTION", "intent_registry")


def main() -> int:
    """Parse args and load intent registry CSV to local Chroma with truncation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load intent registry CSV into local ChromaDB (truncate before load)."
    )
    parser.add_argument("--csv-path", type=Path, default=resolve_registry_csv_path(PROJECT_ROOT))
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--embed-backend",
        choices=("ollama", "minilm"),
        default=os.getenv("INTENT_REGISTRY_EMBED_BACKEND", "ollama"),
        help="ollama (gateway-aligned) or minilm (no Ollama required)",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    try:
        stored = load_vector_registry_local(
            csv_path=args.csv_path,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            project_root=PROJECT_ROOT,
            batch_size=args.batch_size,
            truncate=True,
            embed_backend=args.embed_backend,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load intent registry: {exc}", file=sys.stderr)
        return 1

    elapsed = (datetime.now() - start_time).total_seconds()
    if stored <= 0:
        print("[WARN] No rows stored.")
        return 1

    print(f"[OK] Stored {stored} rows to local Chroma in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
