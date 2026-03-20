"""
Intent registry CSV into local ChromaDB.

Canonical CSV shape: two columns ``text``, ``intent``. An optional ``workflow`` column
may override the value stored in Chroma metadata; if missing or empty, metadata
``workflow`` is set to ``intent`` so the gateway still receives both fields.

Gateway ``VectorRetrieval`` reads ``intent`` and ``workflow`` from metadata.

Embedding defaults align with gateway intent classification (Ollama /api/embed).
Optional minilm backend uses src.rag.embeddings inside this module only.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_DEFAULT_CSV_REL = "src/gateway/route_llm/classification/classification_data/vector_intent_registry.csv"
_DEFAULT_CHROMA_REL = "data/chroma_db/intent_registry"
_DEFAULT_COLLECTION = "intent_registry"


def _ollama_url() -> str:
    """Ollama base URL from OLLAMA_BASE_URL."""
    from src.llm.call_ollama import get_ollama_config

    return get_ollama_config().base_url.rstrip("/")


def _embed_model() -> str:
    from src.llm.call_ollama import get_ollama_config

    return get_ollama_config().embed_model


def _embed_batch(texts: List[str]) -> List[List[float]]:
    url = f"{_ollama_url()}/api/embed"
    resp = requests.post(
        url,
        json={"model": _embed_model(), "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}"
        )
    return embeddings


def _load_registry_rows(path: Path) -> List[Dict[str, str]]:
    """
    Parse registry CSV: require text + intent; workflow optional (falls back to intent).

    Two-column CSVs (text, intent) are valid. A third ``workflow`` column is used
    when present and non-empty (e.g. intent label vs dispatcher workflow differ).
    """
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not text or not intent:
                continue
            workflow = (row.get("workflow") or "").strip()
            if not workflow:
                workflow = intent
            rows.append({"text": text, "intent": intent, "workflow": workflow})
    return rows


def _row_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_vector_registry_local(
    csv_path: str | Path,
    chroma_path: str | Path,
    collection_name: str = _DEFAULT_COLLECTION,
    *,
    project_root: Path,
    batch_size: int = 64,
    truncate: bool = True,
    embed_backend: str = "ollama",
) -> int:
    """
    Load vector_intent_registry-style CSV into local PersistentClient Chroma.

    CSV columns: ``text`` and ``intent`` required; ``workflow`` optional (defaults to
    ``intent`` when missing).

    Returns:
        Number of rows added.
    """
    import chromadb
    from chromadb.config import Settings

    csv_p = Path(csv_path)
    if not csv_p.is_absolute():
        csv_p = project_root / csv_p
    if not csv_p.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = _load_registry_rows(csv_p)
    if not rows:
        raise ValueError(f"No valid rows (text, intent required) in {csv_p}")

    logger.info("Loaded %d rows from %s", len(rows), csv_p)

    chroma_p = Path(chroma_path)
    if not chroma_p.is_absolute():
        chroma_p = project_root / chroma_p
    chroma_p.mkdir(parents=True, exist_ok=True)
    logger.info("Using local ChromaDB at %s", chroma_p)

    client = chromadb.PersistentClient(
        path=str(chroma_p),
        settings=Settings(anonymized_telemetry=False),
    )
    if truncate:
        try:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception:
            pass
        collection = client.create_collection(name=collection_name)
        logger.info("Created new collection '%s'", collection_name)
    else:
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(
                "Found existing collection '%s' (%d docs) — will upsert",
                collection_name,
                collection.count(),
            )
        except Exception:
            collection = client.create_collection(name=collection_name)
            logger.info("Created new collection '%s'", collection_name)

    embed_backend = (embed_backend or "ollama").strip().lower()
    minilm_embedder = None
    if embed_backend == "minilm":
        from src.rag.embeddings import create_embeddings

        minilm_embedder = create_embeddings("minilm", project_root=project_root)

    total = len(rows)
    upserted = 0
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        texts = [r["text"] for r in batch]
        if embed_backend == "minilm" and minilm_embedder is not None:
            embeddings = minilm_embedder.embed_documents(texts)
        else:
            embeddings = _embed_batch(texts)
        ids = [_row_id(r["text"]) for r in batch]
        metadatas = [{"intent": r["intent"], "workflow": r["workflow"]} for r in batch]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        upserted += len(batch)
        logger.info("Upserted %d / %d", upserted, total)

    logger.info(
        "Done. Collection '%s' now has %d documents.",
        collection_name,
        collection.count(),
    )
    return upserted


def resolve_registry_csv_path(project_root: Path) -> Path:
    """Default or env-driven path to intent registry CSV."""
    env_val = os.getenv("VECTOR_REGISTRY_CSV") or os.getenv("CHROMA_INGEST_CSV")
    default = str(project_root / _DEFAULT_CSV_REL)
    path_str = env_val or default
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def resolve_registry_chroma_path(project_root: Path) -> Path:
    """Default or env-driven Chroma persist path for intent registry."""
    path_str = os.getenv("CHROMA_INTENT_REGISTRY_PATH") or os.getenv(
        "VECTOR_CHROMA_PATH", _DEFAULT_CHROMA_REL
    )
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p
