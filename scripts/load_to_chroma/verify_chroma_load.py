#!/usr/bin/env python3
"""
Verify local ChromaDB after load_documents_to_chroma + load_intent_registry_to_chroma.

Exits 0 only if documents and intent_registry collections have expected minimum counts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.chroma_loaders import bootstrap_project, resolve_path

PROJECT_ROOT = bootstrap_project()


def main() -> int:
    import chromadb
    from chromadb.config import Settings

    docs_path = resolve_path(
        "CHROMA_DOCUMENTS_PATH",
        str(PROJECT_ROOT / "data" / "chroma_db" / "documents"),
        PROJECT_ROOT,
    )
    reg_path = os.getenv("CHROMA_INTENT_REGISTRY_PATH") or os.getenv(
        "VECTOR_CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "intent_registry")
    )
    reg_path = str(Path(reg_path).resolve() if Path(reg_path).is_absolute() else (PROJECT_ROOT / reg_path).resolve())

    docs_coll = os.getenv("CHROMA_DOCUMENTS_COLLECTION", os.getenv("CHROMA_COLLECTION_NAME", "documents"))
    reg_coll = os.getenv("CHROMA_INTENT_REGISTRY_COLLECTION", "intent_registry")

    min_docs = int(os.getenv("VERIFY_CHROMA_MIN_DOCUMENTS", "1"))
    min_reg = int(os.getenv("VERIFY_CHROMA_MIN_INTENT_ROWS", "1"))

    ok = True
    for label, path, coll, minimum in [
        ("documents", docs_path, docs_coll, min_docs),
        ("intent_registry", reg_path, reg_coll, min_reg),
    ]:
        if not Path(path).exists():
            print(f"[FAIL] {label}: path missing: {path}")
            ok = False
            continue
        client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        try:
            c = client.get_collection(coll)
            n = c.count()
        except Exception as e:
            print(f"[FAIL] {label}: collection '{coll}' missing: {e}")
            ok = False
            continue
        if n < minimum:
            print(f"[FAIL] {label}: collection '{coll}' count={n} (need >= {minimum})")
            ok = False
        else:
            print(f"[OK] {label}: path={path} collection={coll} count={n}")
            if n > 0:
                peek = c.peek(limit=1)
                if peek.get("documents"):
                    doc0 = (peek["documents"][0] or "")[:120]
                    print(f"     sample doc: {doc0!r}...")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
