"""
Vector Retrieval — Public API Module (公共接口模块)

Architecture:
  vector_retrieval.py 仅提供与具体业务环节无关的公共能力：
    - VectorCandidate   检索结果数据类
    - VectorRetrieval   Chroma 向量检索（由调用方注入 chroma_path、collection_name 等）

  意图注册表路径、CHROMA_INTENT_REGISTRY_*、GATEWAY_VECTOR_* 等属于 gateway 分类环节，
  应在下游（如 implement_methods）解析后传入 VectorRetrieval。

Workflow:
  Query → embed(query) → Chroma query → 阈值过滤 → List[VectorCandidate]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class VectorCandidate:
    """Single candidate from vector retrieval (part of public API)."""

    text: str
    intent: str
    workflow: str
    score: float
    metadata: Optional[dict] = None


class VectorRetrieval:
    """
    Chroma-based vector retrieval; caller supplies store location and collection name.

    Embedding uses embed_backend (default ollama) and shared Ollama env:
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL — generic runtime config, not gateway-specific.
    """

    def __init__(
        self,
        chroma_path: Path,
        collection_name: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        embed_backend: str = "ollama",
    ) -> None:
        """
        Args:
            chroma_path: Chroma persist directory (must exist before non-empty retrieve).
            collection_name: Chroma collection name.
            top_k: Default max candidates per retrieve call.
            score_threshold: Default minimum similarity in [0, 1] after distance mapping.
            embed_backend: Embedding implementation name (currently supports ollama).
        """
        self._chroma_path = Path(chroma_path)
        self._collection_name = collection_name.strip()
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._embed_backend = (embed_backend or "ollama").strip().lower()
        self._client = None
        self._collection = None

    def _ensure_client(self) -> None:
        """Lazy init Chroma client and collection."""
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for vector retrieval") from exc
        if not self._chroma_path.exists():
            raise FileNotFoundError(f"Chroma path not found: {self._chroma_path}")
        self._client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_query(self, query: str) -> List[float]:
        """Vectorize query using configured embed_backend."""
        if self._embed_backend == "ollama":
            return self._embed_via_ollama(query)
        return self._embed_via_ollama(query)

    def _embed_via_ollama(self, query: str) -> List[float]:
        """Call Ollama /api/embed for one string (generic OLLAMA_* env)."""
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
        except Exception as exc:
            raise RuntimeError(f"Ollama embed failed: {exc}") from exc

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorCandidate]:
        """
        Retrieve top-K candidates from Chroma for the given query.

        Args:
            query: User query or intent clause.
            top_k: Override instance default.
            score_threshold: Override instance default.

        Returns:
            VectorCandidate list ordered by score descending.
        """
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else self._top_k
        thresh = score_threshold if score_threshold is not None else self._score_threshold
        self._ensure_client()
        query_embedding = self._embed_query(query.strip())
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        candidates: List[VectorCandidate] = []
        for docs, metas, dists in zip(documents, metadatas, distances):
            for doc, meta, dist in zip(docs, metas, dists):
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
    *,
    chroma_path: Path,
    collection_name: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    embed_backend: str = "ollama",
) -> List[VectorCandidate]:
    """
    One-shot retrieve; caller supplies chroma_path and collection_name.

    Args:
        query: Query text.
        chroma_path: Chroma persist directory.
        collection_name: Collection name.
        top_k: Max results.
        score_threshold: Minimum similarity.
        embed_backend: Embedding backend name.

    Returns:
        List of VectorCandidate.
    """
    retriever = VectorRetrieval(
        chroma_path,
        collection_name,
        top_k=top_k,
        score_threshold=score_threshold,
        embed_backend=embed_backend,
    )
    return retriever.retrieve(query)
