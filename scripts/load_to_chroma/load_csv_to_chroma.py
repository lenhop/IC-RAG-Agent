#!/usr/bin/env python3
"""
Load a single text column from a CSV into local Chroma (generic tool).

Uses src.chroma public API (load_csv_column_to_chroma). Embedding backend:
minilm (default), ollama, or qwen3 — same factory as RAG embeddings.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.chroma import load_csv_column_to_chroma
from src.utils import bootstrap_project

PROJECT_ROOT = bootstrap_project()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load one CSV column as Chroma documents (with embeddings)."
    )
    parser.add_argument("--csv-path", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--column",
        required=True,
        help="Column name whose values become document text",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(
            os.getenv(
                "CHROMA_GENERIC_CSV_PATH",
                str(PROJECT_ROOT / "data" / "chroma_db" / "generic_csv"),
            )
        ),
        help="Chroma persist directory",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_GENERIC_CSV_COLLECTION", "generic_csv"),
        help="Collection name",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "ollama", "qwen3"),
        default=os.getenv("CHROMA_GENERIC_CSV_EMBED", "minilm"),
    )
    args = parser.parse_args()

    csv_p = args.csv_path
    if not csv_p.is_absolute():
        csv_p = (PROJECT_ROOT / csv_p).resolve()
    if not csv_p.is_file():
        print(f"[ERROR] CSV not found: {csv_p}", file=sys.stderr)
        return 1

    chroma_p = args.chroma_path
    if not chroma_p.is_absolute():
        chroma_p = (PROJECT_ROOT / chroma_p).resolve()

    start = datetime.now()
    try:
        stored = load_csv_column_to_chroma(
            csv_p,
            args.column,
            args.collection,
            chroma_p,
            project_root=PROJECT_ROOT,
            embed_model=args.embed_model,
            batch_size=args.batch_size,
            reset_db=True,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    elapsed = (datetime.now() - start).total_seconds()
    if stored <= 0:
        print("[WARN] No rows stored (empty column or no data).")
        return 1
    print(f"[OK] Stored {stored} rows in {elapsed:.1f}s -> {chroma_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
