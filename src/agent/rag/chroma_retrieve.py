"""
Read-only Chroma retrieval for the documents collection.

Uses the same embedding factory as ingest (src.llm.embeddings) and chromadb
query distances; converts L2 distance to cosine similarity for normalized
vectors via: similarity = max(0, 1 - d^2 / 2).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Protocol

# Disable Chroma telemetry before chromadb import (project convention).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

logger = logging.getLogger(__name__)


class _SupportsEmbedQuery(Protocol):
    """Minimal protocol for LangChain-style embedders."""

    def embed_query(self, text: str) -> List[float]:
        """Return a single embedding vector for the query text."""


def l2_distance_to_cosine_similarity(distance: float) -> float:
    """
    Map Euclidean distance between L2-normalized unit vectors to cosine similarity.

    For unit vectors: ||u-v||^2 = 2 - 2 cos(theta), hence
    cos_sim = 1 - d^2 / 2. Clamped to [0, 1] for numerical safety.

    Args:
        distance: Euclidean distance reported by Chroma (same space as ingest).

    Returns:
        Estimated cosine similarity in [0, 1].
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    sim = 1.0 - (d * d) / 2.0
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


class ChromaRetriever:
    """
    Facade for querying the local Chroma documents collection with a gate on similarity.
    """

    @classmethod
    def retrieve(
        cls,
        collection: Any,
        embedder: _SupportsEmbedQuery,
        question: str,
        *,
        top_k: int,
        min_similarity: float,
        prefetch_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Embed the question, query Chroma, filter by similarity, keep up to top_k.

        Args:
            collection: chromadb Collection instance.
            embedder: Object with embed_query(str) -> vector.
            question: User query text.
            top_k: Maximum number of hits after filtering.
            min_similarity: Minimum cosine similarity (see l2_distance_to_cosine_similarity).
            prefetch_n: Number of neighbors to request from Chroma before filtering.

        Returns:
            List of dicts with keys: text, metadata, distance, similarity.

        Raises:
            ValueError: If question is empty or prefetch_n/top_k invalid.
            RuntimeError: If Chroma returns an unexpected shape.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty for Chroma retrieval")
        if top_k < 1 or prefetch_n < 1:
            raise ValueError("top_k and prefetch_n must be >= 1")

        try:
            query_embedding = embedder.embed_query(q)
        except Exception as exc:
            logger.error("embed_query failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Embedding query failed: {exc}") from exc

        n_fetch = max(prefetch_n, top_k)
        try:
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_fetch,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Chroma query failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Chroma query failed: {exc}") from exc

        docs_list = raw.get("documents") or []
        metas_list = raw.get("metadatas") or []
        dists_list = raw.get("distances") or []
        if not docs_list or not docs_list[0]:
            logger.info("Chroma returned no documents for query (empty collection or no neighbors)")
            return []

        documents = docs_list[0]
        metadatas = metas_list[0] if metas_list and metas_list[0] else [{}] * len(documents)
        distances = dists_list[0] if dists_list and dists_list[0] else [0.0] * len(documents)
        if len(documents) != len(distances):
            raise RuntimeError("Chroma result length mismatch between documents and distances")

        scored: List[Dict[str, Any]] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            sim = l2_distance_to_cosine_similarity(dist)
            scored.append(
                {
                    "text": text or "",
                    "metadata": meta if isinstance(meta, dict) else {},
                    "distance": float(dist),
                    "similarity": sim,
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        kept = [row for row in scored if row["similarity"] >= min_similarity][:top_k]

        logger.info(
            "Chroma retrieve: raw=%d after_sim_%.2f=%d cap_top_k=%d",
            len(scored),
            min_similarity,
            len(kept),
            top_k,
        )
        return kept

    @classmethod
    def hits_to_sources(cls, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize hit dicts into gateway-friendly source entries.

        Args:
            hits: Output rows from retrieve().

        Returns:
            List of serializable dicts for the HTTP ``sources`` field.
        """
        out: List[Dict[str, Any]] = []
        for i, h in enumerate(hits):
            meta = h.get("metadata") or {}
            out.append(
                {
                    "index": i + 1,
                    "text": (h.get("text") or "")[:2000],
                    "similarity": round(float(h.get("similarity", 0.0)), 4),
                    "metadata": {
                        "source": meta.get("source", ""),
                        "page": meta.get("page", ""),
                    },
                }
            )
        return out
