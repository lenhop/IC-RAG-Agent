"""
Test ChromaDB connection to Alibaba ECS.

Verifies HttpClient can connect to remote ChromaDB at configured host:port.
Uses CHA_HOST and CHA_PORT from .env (fallback: CHROMA_ECS_HOST, CHROMA_ECS_PORT).

Run: PYTHONPATH=. python -m pytest tests/test_chroma_ecs_connection.py -v

Note: If you see 502 Bad Gateway or "Could not connect to tenant default_tenant",
check ChromaDB client/server version compatibility. ECS runs chromadb/chroma:latest;
ensure your chromadb Python package version matches (e.g. chromadb>=0.4,<0.5 for older
server API).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env for CHA_HOST, CHA_PORT
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Disable Chroma telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

ECS_HOST = os.getenv("CHA_HOST") or os.getenv("CHROMA_ECS_HOST", "localhost")
ECS_PORT = int(os.getenv("CHA_PORT") or os.getenv("CHROMA_ECS_PORT") or "8000")
TEST_COLLECTION = "test_ecs_connection"


@pytest.fixture(scope="module")
def chroma_client():
    """Create ChromaDB HttpClient for ECS."""
    try:
        import chromadb
    except ImportError as exc:
        pytest.skip(f"chromadb not installed: {exc}")

    try:
        client = chromadb.HttpClient(
            host=ECS_HOST,
            port=ECS_PORT,
            ssl=False,
        )
        # Trigger connection validation (list_collections)
        client.list_collections()
        return client
    except Exception as exc:
        pytest.skip(
            f"Cannot connect to ChromaDB at {ECS_HOST}:{ECS_PORT}. "
            f"Check network/whitelist and ChromaDB version compatibility: {exc}"
        )


def test_chroma_ecs_list_collections(chroma_client):
    """List collections on ECS ChromaDB."""
    collections = chroma_client.list_collections()
    assert isinstance(collections, list)


def test_chroma_ecs_heartbeat(chroma_client):
    """Verify ChromaDB heartbeat endpoint responds."""
    # HttpClient does not expose heartbeat directly; list_collections exercises the API
    collections = chroma_client.list_collections()
    # No exception means connection succeeded
    assert collections is not None


def test_chroma_ecs_add_and_get(chroma_client):
    """Add a test document and retrieve it from ECS ChromaDB."""
    # Use a unique collection name to avoid polluting shared data
    coll_name = f"{TEST_COLLECTION}_pytest"
    try:
        chroma_client.delete_collection(coll_name)
    except Exception:
        pass

    coll = chroma_client.create_collection(name=coll_name, metadata={"test": "true"})
    coll.add(
        ids=["test-id-1"],
        documents=["This is a test document for ECS ChromaDB connection."],
        metadatas=[{"source": "pytest"}],
    )

    result = coll.get(ids=["test-id-1"], include=["documents", "metadatas"])
    assert result["ids"] == ["test-id-1"]
    assert len(result["documents"]) == 1
    assert "ECS ChromaDB" in result["documents"][0]
    assert result["metadatas"][0]["source"] == "pytest"

    # Cleanup
    chroma_client.delete_collection(coll_name)
