"""
Vector Retrieval — Public API Module (公共接口模块)

Architecture (强制):
  vector_retrieval.py = 公共接口模块（Layer 2 唯一入口），负责 Chroma 向量检索
  __init__.py         = 包入口，从本模块 re-export 公共 API

  下游模块（gateway、classification 等）应通过本模块或 src.retrieval 包导入，
  禁止直接依赖内部实现（_ensure_client、_embed_* 等）。

Workflow:
  Query → 连接 Chroma(intent_registry) → embed(query) → TopK 相似检索 → 阈值过滤 → List[VectorCandidate]

Internal (not exported):
  _PROJECT_ROOT, _DEFAULT_CHROMA_PATH, _DEFAULT_COLLECTION, _DEFAULT_TOP_K, _DEFAULT_THRESHOLD
  _ensure_client      — 懒加载 Chroma 客户端与 collection
  _embed_query        — 单条 query 向量化
  _embed_via_ollama   — 调用 Ollama /api/embed

Public API (exported via __init__.py):
  VectorCandidate      — 单条检索结果（text, intent, workflow, score, metadata）
  VectorRetrieval      — 主类；retrieve(query, top_k=..., score_threshold=...) → List[VectorCandidate]
  vector_retrieve(...) — 一次性便捷函数（测试/脚本用）
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


# ---------------------------------------------------------------------------
# Public API — 公共接口（唯一对外入口，由 __init__.py re-export）
#
# 下游业务模块（gateway、classification、dispatcher 等）必须通过这些类/函数访问
# Layer 2 向量检索能力。
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
    Chroma-based vector retrieval over intent_registry.

    Primary public entry point: ``retrieve(query, ...)``.

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
    """
    Optional one-shot helper (tests / scripts). Prefer ``VectorRetrieval`` for DI and tests.
    """
    retriever = VectorRetrieval(chroma_path=chroma_path, top_k=top_k)
    return retriever.retrieve(query)
