#!/usr/bin/env python3
"""
Load vector_intent_registry.csv into ChromaDB collection 'vector_intent_registry'.

Embeds each row's 'text' field via Ollama all-minilm and upserts into Chroma.
Supports local (PersistentClient) or remote ECS (HTTP v2 API directly).

Usage:
    python scripts/load_vector_registry.py              # Use CHA_HOST from .env if set, else local
    python scripts/load_vector_registry.py --local      # Force local ChromaDB
    python scripts/load_vector_registry.py --remote     # Force remote (CHA_HOST required)

Environment variables (all optional):
    VECTOR_REGISTRY_CSV     path to CSV  (default: src/prompts/retrieval/vector_intent_registry.csv)
    VECTOR_CHROMA_PATH      chroma db dir for local mode (default: data/chroma_db/intent_registry)
    VECTOR_COLLECTION_NAME  collection name (default: vector_intent_registry)
    CHA_HOST                ECS ChromaDB host (or CHROMA_ECS_HOST)
    CHA_PORT                ECS ChromaDB port (or CHROMA_ECS_PORT, default: 8000)
    VECTOR_CHROMA_LOCAL     if set (true/1/yes), use local even when CHA_HOST is set
    GATEWAY_REWRITE_OLLAMA_URL  ollama base url (default: http://localhost:11434)
    GATEWAY_INTENT_EMBEDDING_MODEL  embedding model (default: all-minilm)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import requests

# Disable Chroma telemetry before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

# Load .env for CHA_HOST, CHA_PORT
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

_DEFAULT_CSV = str(_PROJECT_ROOT / "src" / "prompts" / "retrieval" / "vector_intent_registry.csv")
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma_db" / "intent_registry")
_DEFAULT_COLLECTION = "vector_intent_registry"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_EMBED_MODEL = "all-minilm"
_TENANT = "default_tenant"
_DATABASE = "default_database"


def _cfg(key: str, default: str) -> str:
    return os.getenv(key, default)


def _ollama_url() -> str:
    return _cfg("GATEWAY_REWRITE_OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")


def _embed_model() -> str:
    return _cfg("GATEWAY_INTENT_EMBEDDING_MODEL", _DEFAULT_EMBED_MODEL)


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
        raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}")
    return embeddings


def _load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            workflow = (row.get("workflow") or "").strip()
            if text and intent and workflow:
                rows.append({"text": text, "intent": intent, "workflow": workflow})
    return rows


def _row_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load vector_intent_registry.csv into ChromaDB")
    parser.add_argument("--local", action="store_true", help="Force local ChromaDB (ignore CHA_HOST)")
    parser.add_argument("--remote", action="store_true", help="Force remote ChromaDB (requires CHA_HOST)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Remote ChromaDB v2 HTTP helpers
# ---------------------------------------------------------------------------

class ChromaV2Client:
    """Minimal ChromaDB v2 HTTP client (compatible with Chroma 1.0.0)."""

    def __init__(self, host: str, port: int):
        self.base = f"http://{host}:{port}/api/v2"
        self.coll_base = f"{self.base}/tenants/{_TENANT}/databases/{_DATABASE}/collections"

    def heartbeat(self) -> bool:
        resp = requests.get(f"{self.base}/heartbeat", timeout=5)
        resp.raise_for_status()
        return True

    def get_or_create_collection(self, name: str) -> str:
        """Return collection id, creating if needed."""
        # Try get first
        resp = requests.get(f"{self.coll_base}/{name}", timeout=10)
        if resp.status_code == 200:
            coll_id = resp.json()["id"]
            count = self.count(coll_id)
            logger.info("Found existing collection '%s' (id=%s, %d docs) — will upsert", name, coll_id, count)
            return coll_id
        # Create
        resp = requests.post(
            self.coll_base,
            json={"name": name, "get_or_create": True},
            timeout=10,
        )
        resp.raise_for_status()
        coll_id = resp.json()["id"]
        logger.info("Created collection '%s' (id=%s)", name, coll_id)
        return coll_id

    def count(self, collection_id: str) -> int:
        resp = requests.get(f"{self.coll_base}/{collection_id}/count", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return 0

    def upsert(self, collection_id: str, ids: List[str], documents: List[str],
               embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        resp = requests.post(
            f"{self.coll_base}/{collection_id}/upsert",
            json={
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            },
            timeout=60,
        )
        resp.raise_for_status()


def main() -> None:
    args = _parse_args()
    csv_path = _cfg("VECTOR_REGISTRY_CSV", _DEFAULT_CSV)
    chroma_path = _cfg("VECTOR_CHROMA_PATH", _DEFAULT_CHROMA_PATH)
    collection_name = _cfg("VECTOR_COLLECTION_NAME", _DEFAULT_COLLECTION)

    if not Path(csv_path).is_file():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    rows = _load_csv(csv_path)
    if not rows:
        logger.error("No valid rows found in %s", csv_path)
        sys.exit(1)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    # Determine mode: remote or local
    use_local = args.local
    if not use_local and not args.remote:
        local_env = os.getenv("VECTOR_CHROMA_LOCAL", "").strip().lower()
        use_local = local_env in ("true", "1", "yes")

    ecs_host = (os.getenv("CHA_HOST", "").strip() or os.getenv("CHROMA_ECS_HOST", "").strip())
    ecs_port = int(os.getenv("CHA_PORT") or os.getenv("CHROMA_ECS_PORT") or "8000")

    remote_client: ChromaV2Client | None = None

    if ecs_host and not use_local:
        logger.info("Using remote ChromaDB v2 at %s:%d", ecs_host, ecs_port)
        try:
            remote_client = ChromaV2Client(ecs_host, ecs_port)
            remote_client.heartbeat()
            logger.info("Remote ChromaDB connection OK")
        except Exception as exc:
            logger.warning("Remote ChromaDB unreachable (%s:%d): %s — falling back to local", ecs_host, ecs_port, exc)
            remote_client = None
            use_local = True

    if remote_client is None:
        # Local PersistentClient
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            sys.exit(1)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        logger.info("Using local ChromaDB at %s", chroma_path)
        local_chroma = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            collection = local_chroma.get_collection(name=collection_name)
            logger.info("Found existing collection '%s' (%d docs) — will upsert", collection_name, collection.count())
        except Exception:
            collection = local_chroma.create_collection(name=collection_name)
            logger.info("Created new collection '%s'", collection_name)

    # Embed and upsert in batches
    batch_size = 64
    total = len(rows)
    upserted = 0

    if remote_client:
        collection_id = remote_client.get_or_create_collection(collection_name)

    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        texts = [r["text"] for r in batch]

        try:
            embeddings = _embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding failed at batch %d: %s", i // batch_size, exc)
            sys.exit(1)

        ids = [_row_id(r["text"]) for r in batch]
        metadatas = [{"intent": r["intent"], "workflow": r["workflow"]} for r in batch]

        if remote_client:
            remote_client.upsert(collection_id, ids, texts, embeddings, metadatas)
        else:
            collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        upserted += len(batch)
        logger.info("Upserted %d / %d", upserted, total)

    if remote_client:
        final_count = remote_client.count(collection_id)
    else:
        final_count = collection.count()

    logger.info("Done. Collection '%s' now has %d documents.", collection_name, final_count)


if __name__ == "__main__":
    main()

