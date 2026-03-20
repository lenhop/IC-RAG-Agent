"""
Verify local Chroma PersistentClient stores (collection existence and counts).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from src.utils import resolve_path


def verify_collection_minimum(
    label: str,
    chroma_path: str | Path,
    collection_name: str,
    minimum: int,
) -> Tuple[bool, str]:
    """
    Check that chroma_path exists and collection has at least ``minimum`` documents.

    Args:
        label: Human-readable label for log lines.
        chroma_path: Chroma persist directory.
        collection_name: Collection name.
        minimum: Required minimum document count.

    Returns:
        (ok, message) where message describes success or failure.
    """
    import chromadb
    from chromadb.config import Settings

    path = Path(chroma_path)
    if not path.exists():
        msg = f"[FAIL] {label}: path missing: {path}"
        return False, msg

    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    try:
        coll = client.get_collection(collection_name)
        n = coll.count()
    except Exception as exc:
        msg = f"[FAIL] {label}: collection '{collection_name}' missing: {exc}"
        return False, msg

    if n < minimum:
        msg = (
            f"[FAIL] {label}: collection '{collection_name}' count={n} "
            f"(need >= {minimum})"
        )
        return False, msg

    msg = f"[OK] {label}: path={path} collection={collection_name} count={n}"
    if n > 0:
        try:
            peek = coll.peek(limit=1)
            if peek.get("documents"):
                doc0 = (peek["documents"][0] or "")[:120]
                msg = f"{msg}\n     sample doc: {doc0!r}..."
        except Exception:
            pass
    return True, msg


def verify_default_project_chroma_stores(project_root: Path) -> bool:
    """
    Verify documents + intent_registry stores using the same env defaults as load scripts.

    Environment:
        CHROMA_DOCUMENTS_PATH, CHROMA_DOCUMENTS_COLLECTION / CHROMA_COLLECTION_NAME
        CHROMA_INTENT_REGISTRY_PATH / VECTOR_CHROMA_PATH
        CHROMA_INTENT_REGISTRY_COLLECTION
        VERIFY_CHROMA_MIN_DOCUMENTS, VERIFY_CHROMA_MIN_INTENT_ROWS

    Args:
        project_root: Repository root from bootstrap_project().

    Returns:
        True if all checks pass.
    """
    docs_path = resolve_path(
        "CHROMA_DOCUMENTS_PATH",
        str(project_root / "data" / "chroma_db" / "documents"),
        project_root,
    )
    reg_path = os.getenv("CHROMA_INTENT_REGISTRY_PATH") or os.getenv(
        "VECTOR_CHROMA_PATH", str(project_root / "data" / "chroma_db" / "intent_registry")
    )
    reg_resolved = str(
        Path(reg_path).resolve()
        if Path(reg_path).is_absolute()
        else (project_root / reg_path).resolve()
    )

    docs_coll = os.getenv("CHROMA_DOCUMENTS_COLLECTION", os.getenv("CHROMA_COLLECTION_NAME", "documents"))
    reg_coll = os.getenv("CHROMA_INTENT_REGISTRY_COLLECTION", "intent_registry")

    min_docs = int(os.getenv("VERIFY_CHROMA_MIN_DOCUMENTS", "1"))
    min_reg = int(os.getenv("VERIFY_CHROMA_MIN_INTENT_ROWS", "1"))

    checks: List[Tuple[str, str, str, int]] = [
        ("documents", docs_path, docs_coll, min_docs),
        ("intent_registry", reg_resolved, reg_coll, min_reg),
    ]

    ok_all = True
    for label, path, coll, minimum in checks:
        ok, message = verify_collection_minimum(label, path, coll, minimum)
        print(message)
        if not ok:
            ok_all = False
    return ok_all
