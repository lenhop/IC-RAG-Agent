"""
Intent Registry: load intent definitions from YAML and build Chroma collection.

On startup, reads data/intent_registry/intents.yaml, embeds all examples via
Ollama all-minilm, and upserts into a Chroma collection (intent_registry).

Each document in Chroma stores:
- document text: the example sentence
- metadata: intent_name, workflow, required_fields (JSON), clarification_template

Usage:
    from src.gateway.intent_registry import get_intent_collection, get_intent_metadata
    collection = get_intent_collection()  # lazy init, cached
    metadata = get_intent_metadata()      # {intent_name: {workflow, required_fields, ...}}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_REGISTRY_PATH = str(_PROJECT_ROOT / "data" / "intent_registry" / "intents.yaml")
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma_db" / "intent_registry")
_DEFAULT_COLLECTION_NAME = "intent_registry"
_DEFAULT_EMBEDDING_MODEL = "all-minilm"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Module-level cache
_collection = None
_intent_metadata: Optional[Dict[str, Dict[str, Any]]] = None


def _get_registry_path() -> str:
    return os.getenv("GATEWAY_INTENT_REGISTRY_PATH", _DEFAULT_REGISTRY_PATH)


def _get_chroma_path() -> str:
    return os.getenv("GATEWAY_INTENT_CHROMA_PATH", _DEFAULT_CHROMA_PATH)


def _get_collection_name() -> str:
    return os.getenv("GATEWAY_INTENT_CHROMA_COLLECTION", _DEFAULT_COLLECTION_NAME)


def _get_embedding_model() -> str:
    return os.getenv("GATEWAY_INTENT_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)


def _get_ollama_url() -> str:
    return os.getenv("GATEWAY_REWRITE_OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts via Ollama /api/embed endpoint."""
    url = f"{_get_ollama_url()}/api/embed"
    model = _get_embedding_model()
    try:
        resp = requests.post(
            url,
            json={"model": model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}")
        return embeddings
    except Exception as exc:
        logger.error("Ollama embed failed (%s): %s", url, exc)
        raise


def _load_intents_yaml() -> List[Dict[str, Any]]:
    """Load intent definitions from YAML file."""
    path = _get_registry_path()
    if not Path(path).is_file():
        raise FileNotFoundError(f"Intent registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError(f"No intents found in {path}")
    return intents


def _compute_registry_hash(intents: List[Dict[str, Any]]) -> str:
    """Compute a hash of the intent registry for change detection."""
    content = json.dumps(intents, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _build_collection():
    """Build or refresh the Chroma intent_registry collection."""
    global _collection, _intent_metadata

    import chromadb
    from chromadb.config import Settings

    intents = _load_intents_yaml()
    registry_hash = _compute_registry_hash(intents)

    chroma_path = _get_chroma_path()
    collection_name = _get_collection_name()
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    # Check if collection exists and is up-to-date via stored hash.
    try:
        existing = client.get_collection(name=collection_name)
        stored_meta = existing.metadata or {}
        if stored_meta.get("registry_hash") == registry_hash:
            logger.info(
                "Intent registry collection '%s' is up-to-date (hash=%s, count=%d)",
                collection_name, registry_hash, existing.count(),
            )
            _collection = existing
            _intent_metadata = _build_metadata_map(intents)
            return
        # Hash mismatch — delete and rebuild.
        logger.info("Intent registry changed (old=%s, new=%s); rebuilding collection",
                     stored_meta.get("registry_hash", "none"), registry_hash)
        client.delete_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist yet.
        pass

    # Build fresh collection.
    collection = client.create_collection(
        name=collection_name,
        metadata={"registry_hash": registry_hash},
    )

    # Gather all examples with metadata.
    all_ids: List[str] = []
    all_docs: List[str] = []
    all_meta: List[Dict[str, str]] = []

    for intent in intents:
        name = intent["name"]
        workflow = intent.get("workflow", "general")
        required_fields = json.dumps(intent.get("required_fields") or [])
        clarification_template = intent.get("clarification_template") or ""
        examples = intent.get("examples") or []

        for idx, example in enumerate(examples):
            example_text = (example or "").strip()
            if not example_text:
                continue
            doc_id = f"{name}_{idx}"
            all_ids.append(doc_id)
            all_docs.append(example_text)
            all_meta.append({
                "intent_name": name,
                "workflow": workflow,
                "required_fields": required_fields,
                "clarification_template": clarification_template,
            })

    if not all_docs:
        logger.warning("No examples found in intent registry; collection will be empty")
        _collection = collection
        _intent_metadata = _build_metadata_map(intents)
        return

    # Embed in batches (Ollama handles batches well).
    batch_size = 64
    all_embeddings: List[List[float]] = []
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        embeddings = _embed_texts(batch)
        all_embeddings.extend(embeddings)

    # Upsert into Chroma.
    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings,
        metadatas=all_meta,
    )

    logger.info(
        "Built intent registry collection '%s': %d examples across %d intents (hash=%s)",
        collection_name, len(all_docs), len(intents), registry_hash,
    )
    _collection = collection
    _intent_metadata = _build_metadata_map(intents)


def _build_metadata_map(intents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build {intent_name: {workflow, required_fields, clarification_template}} map."""
    result = {}
    for intent in intents:
        name = intent["name"]
        result[name] = {
            "workflow": intent.get("workflow", "general"),
            "required_fields": intent.get("required_fields") or [],
            "clarification_template": intent.get("clarification_template") or "",
        }
    return result


def get_intent_collection():
    """Get the Chroma intent_registry collection (lazy init, cached)."""
    global _collection
    if _collection is None:
        _build_collection()
    return _collection


def get_intent_metadata() -> Dict[str, Dict[str, Any]]:
    """Get intent metadata map (lazy init, cached)."""
    global _intent_metadata
    if _intent_metadata is None:
        _build_collection()
    return _intent_metadata or {}


def reload_registry() -> None:
    """Force reload the intent registry from YAML and rebuild Chroma collection."""
    global _collection, _intent_metadata
    _collection = None
    _intent_metadata = None
    _build_collection()
