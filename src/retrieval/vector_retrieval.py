"""
Vector retrieval for intent classification (Layer 2).

Uses existing Chroma DB at data/chroma_db/intent_registry to retrieve
top-K similar intents by embedding the query. Encapsulates connection,
embed, and threshold filtering in one class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Default Chroma path (align with load_intent_registry_to_chroma.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHROMA_PATH = _PROJECT_ROOT / "data" / "chroma_db" / "intent_registry"
_DEFAULT_COLLECTION = "intent_registry"
_DEFAULT_TOP_K = 5
_DEFAULT_THRESHOLD = 0.0  # No filtering by default; caller can set higher


@dataclass
class VectorCandidate:
    """Single candidate from vector retrieval."""

    text: str
    intent: str
    workflow: str
    score: float
    metadata: Optional[dict] = None


class VectorRetrieval:
    """
    Chroma-based vector retrieval over intent_registry.

    Connects to local Chroma at chroma_path, embeds query (via backend from env),
    runs similarity search, and returns top-K candidates above threshold.
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        collection_name: str = _DEFAULT_COLLECTION,
        top_k: int = _DEFAULT_TOP_K,
        score_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        """
        Args:
            chroma_path: Chroma persist directory (default: data/chroma_db/intent_registry).
            collection_name: Chroma collection name.
            top_k: Max number of candidates to return.
            score_threshold: Minimum similarity score (in [0,1] if normalized).
        """
        self._chroma_path = Path(
            os.getenv("CHROMA_INTENT_REGISTRY_PATH", str(chroma_path or _DEFAULT_CHROMA_PATH))
        )
        self._collection_name = os.getenv("CHROMA_INTENT_REGISTRY_COLLECTION", collection_name)
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._client = None
        self._collection = None

    def _ensure_client(self) -> None:
        """Lazy init Chroma client and collection to avoid import at module load."""
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError("chromadb is required for vector retrieval") from e
        if not self._chroma_path.exists():
            raise FileNotFoundError(f"Chroma path not found: {self._chroma_path}")
        self._client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_query(self, query: str) -> List[float]:
        """Embed single query using INTENT_REGISTRY_EMBED_BACKEND (e.g. ollama)."""
        backend = (os.getenv("INTENT_REGISTRY_EMBED_BACKEND") or "ollama").strip().lower()
        if backend == "ollama":
            return self._embed_via_ollama(query)
        # Fallback: minilm or other; could inject embedder later
        return self._embed_via_ollama(query)

    def _embed_via_ollama(self, query: str) -> List[float]:
        """Call Ollama /api/embed for one string."""
        import requests
        base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")
        model = (os.getenv("OLLAMA_EMBED_MODEL") or "all-minilm:latest").strip()
        url = f"{base_url}/api/embed"
        try:
            resp = requests.post(url, json={"model": model, "input": [query]}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embeddings")
            if not emb or len(emb) != 1:
                raise ValueError("Ollama embed returned invalid embeddings")
            return emb[0]
        except Exception as e:
            raise RuntimeError(f"Ollama embed failed: {e}") from e

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorCandidate]:
        """
        Retrieve top-K intent candidates from Chroma for the given query.

        Args:
            query: User query or intent clause.
            top_k: Override instance top_k.
            score_threshold: Override instance score_threshold.

        Returns:
            List of VectorCandidate (text, intent, workflow, score), ordered by score desc.
        """
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else self._top_k
        thresh = score_threshold if score_threshold is not None else self._score_threshold
        self._ensure_client()
        query_embedding = self._embed_query(query.strip())
        # Chroma query returns distances (cosine); convert to similarity if needed
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        candidates: List[VectorCandidate] = []
        for i, (docs, metas, dists) in enumerate(zip(documents, metadatas, distances)):
            for j, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                # Cosine distance: 0 = identical, 2 = opposite. Convert to similarity: 1 - dist/2
                sim = 1.0 - (float(dist) / 2.0) if dist is not None else 0.0
                if sim < thresh:
                    continue
                meta = meta or {}
                candidates.append(
                    VectorCandidate(
                        text=doc or "",
                        intent=meta.get("intent", ""),
                        workflow=meta.get("workflow", ""),
                        score=sim,
                        metadata=meta,
                    )
                )
        return candidates


def vector_retrieve(
    query: str,
    chroma_path: Optional[Path] = None,
    top_k: int = _DEFAULT_TOP_K,
) -> List[VectorCandidate]:
    """One-shot vector retrieval using default Chroma path."""
    retriever = VectorRetrieval(chroma_path=chroma_path, top_k=top_k)
    return retriever.retrieve(query)
