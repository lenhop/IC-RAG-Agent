#!/usr/bin/env python3
"""
Transfer ChromaDB data from local persistent storage to ECS ChromaDB server.

Reads from local PersistentClient paths and writes to remote HttpClient.
Preserves pre-computed embeddings (no re-embedding).

Usage:
  python scripts/transfer_chroma_to_ecs.py documents
  python scripts/transfer_chroma_to_ecs.py fqa keywords
  python scripts/transfer_chroma_to_ecs.py --all
  python scripts/transfer_chroma_to_ecs.py documents --dry-run

Env:
  CHROMA_ECS_HOST - ECS ChromaDB host (default: localhost)
  CHROMA_ECS_PORT - ECS ChromaDB port (default: 8001 for local Docker)
  CHROMA_ECS_SSL  - Use HTTPS (default: false)
  CHROMA_DOCUMENTS_PATH, CHROMA_FQA_PATH, CHROMA_KEYWORD_PATH - local paths
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Disable Chroma telemetry before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

# Project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.rag.chroma_loaders import bootstrap_project, resolve_path

# Bootstrap loads .env and sets up paths
PROJECT_ROOT = bootstrap_project()

# Preset: (collection_name, chroma_env, chroma_default)
PRESETS = {
    "documents": (
        os.getenv("CHROMA_COLLECTION_NAME", "documents"),
        "CHROMA_DOCUMENTS_PATH",
        str(PROJECT_ROOT / "data" / "chroma_db" / "documents"),
    ),
    "fqa": (
        "fqa_question",
        "CHROMA_FQA_PATH",
        str(PROJECT_ROOT / "data" / "chroma_db" / "fqa_question"),
    ),
    "keywords": (
        "keyword",
        "CHROMA_KEYWORD_PATH",
        str(PROJECT_ROOT / "data" / "chroma_db" / "keyword"),
    ),
}

# Default batch sizes (Chroma max ~416)
READ_BATCH_SIZE = 500
WRITE_BATCH_SIZE = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_remote_client(host: str, port: int, ssl: bool):
    """Create ChromaDB HttpClient for ECS."""
    import chromadb

    return chromadb.HttpClient(
        host=host,
        port=port,
        ssl=ssl,
    )


def _get_local_client(path: str):
    """Create ChromaDB PersistentClient for local path."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


def _transfer_collection(
    source_path: str,
    collection_name: str,
    target_host: str,
    target_port: int,
    target_ssl: bool,
    read_batch: int,
    write_batch: int,
    dry_run: bool,
    reset_remote: bool,
) -> int:
    """
    Transfer one collection from local to remote.

    Returns:
        Number of records transferred.
    """
    path = Path(source_path)
    if not path.exists():
        logger.warning("Source path does not exist: %s", source_path)
        return 0

    try:
        local_client = _get_local_client(source_path)
    except Exception as e:
        logger.error("Failed to connect to local ChromaDB at %s: %s", source_path, e)
        raise

    try:
        local_coll = local_client.get_collection(name=collection_name)
    except Exception as e:
        logger.warning("Collection %s not found at %s: %s", collection_name, source_path, e)
        return 0

    count = local_coll.count()
    if count == 0:
        logger.info("Collection %s is empty, skipping", collection_name)
        return 0

    logger.info("Transferring %s: %d records from %s", collection_name, count, source_path)

    if dry_run:
        logger.info("[DRY-RUN] Would transfer %d records to %s:%d", count, target_host, target_port)
        return count

    # Read all data with pagination (include embeddings for transfer)
    all_ids = []
    all_documents = []
    all_metadatas = []
    all_embeddings = []

    offset = 0
    while True:
        batch = local_coll.get(
            limit=read_batch,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        all_ids.extend(ids)
        all_documents.extend(batch.get("documents") or [])
        all_metadatas.extend(batch.get("metadatas") or [])
        emb = batch.get("embeddings")
        if emb:
            all_embeddings.extend(emb)
        offset += len(ids)
        if len(ids) < read_batch:
            break

    if len(all_ids) != count:
        logger.warning("Read %d records but count()=%d", len(all_ids), count)

    # Connect to remote and write in batches
    try:
        remote_client = _get_remote_client(target_host, target_port, target_ssl)
        remote_client.heartbeat()  # Verify connection
    except Exception as e:
        logger.error("Failed to connect to ECS ChromaDB at %s:%d: %s", target_host, target_port, e)
        raise

    # Delete existing collection if reset requested
    if reset_remote:
        try:
            remote_client.delete_collection(name=collection_name)
            logger.info("Deleted existing remote collection %s", collection_name)
        except Exception:
            pass

    remote_coll = remote_client.get_or_create_collection(name=collection_name)
    transferred = 0

    for start in range(0, len(all_ids), write_batch):
        end = min(start + write_batch, len(all_ids))
        batch_ids = all_ids[start:end]
        batch_docs = all_documents[start:end] if all_documents else None
        batch_meta = all_metadatas[start:end] if all_metadatas else None
        batch_emb = all_embeddings[start:end] if all_embeddings else None

        kwargs = {"ids": batch_ids}
        if batch_docs:
            kwargs["documents"] = batch_docs
        if batch_meta:
            kwargs["metadatas"] = batch_meta
        if batch_emb:
            kwargs["embeddings"] = batch_emb

        try:
            remote_coll.add(**kwargs)
            transferred += len(batch_ids)
            logger.info("  Wrote batch %d-%d (%d total)", start, end, transferred)
        except Exception as e:
            logger.error("Failed to add batch %d-%d: %s", start, end, e)
            raise

    # Verify
    remote_count = remote_coll.count()
    if remote_count != transferred:
        logger.warning("Remote count %d != transferred %d", remote_count, transferred)
    else:
        logger.info("Verified: remote collection has %d records", remote_count)

    return transferred


def main() -> int:
    """Parse args and run transfer."""
    parser = argparse.ArgumentParser(
        description="Transfer ChromaDB data from local to ECS",
    )
    parser.add_argument(
        "collections",
        nargs="*",
        choices=list(PRESETS.keys()),
        help="Collections to transfer: documents, fqa, keywords",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Transfer all preset collections",
    )
    parser.add_argument(
        "--target-host",
        default=os.getenv("CHROMA_ECS_HOST", "localhost"),
        help="ECS ChromaDB host (default: CHROMA_ECS_HOST or localhost)",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=int(os.getenv("CHROMA_ECS_PORT", "8001")),
        help="ECS ChromaDB port (default: CHROMA_ECS_PORT or 8001)",
    )
    parser.add_argument(
        "--target-ssl",
        action="store_true",
        default=os.getenv("CHROMA_ECS_SSL", "false").lower() in ("true", "1", "yes"),
        help="Use HTTPS for ECS connection",
    )
    parser.add_argument(
        "--read-batch",
        type=int,
        default=READ_BATCH_SIZE,
        help=f"Read batch size (default: {READ_BATCH_SIZE})",
    )
    parser.add_argument(
        "--write-batch",
        type=int,
        default=WRITE_BATCH_SIZE,
        help=f"Write batch size (default: {WRITE_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be transferred",
    )
    parser.add_argument(
        "--reset-remote",
        action="store_true",
        help="Delete remote collection before transfer (replaces existing data)",
    )

    args = parser.parse_args()

    if args.all:
        to_transfer = list(PRESETS.keys())
    elif args.collections:
        to_transfer = args.collections
    else:
        parser.error("Specify collections or --all")
        return 1

    total = 0
    for key in to_transfer:
        collection_name, chroma_env, chroma_default = PRESETS[key]
        source_path = resolve_path(chroma_env, chroma_default, PROJECT_ROOT)
        try:
            n = _transfer_collection(
                source_path=source_path,
                collection_name=collection_name,
                target_host=args.target_host,
                target_port=args.target_port,
                target_ssl=args.target_ssl,
                read_batch=args.read_batch,
                write_batch=args.write_batch,
                dry_run=args.dry_run,
                reset_remote=args.reset_remote,
            )
            total += n
        except Exception as e:
            logger.exception("Transfer failed for %s: %s", key, e)
            return 1

    logger.info("Total records transferred: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
