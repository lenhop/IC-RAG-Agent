"""
Runtime configuration for Agent RAG (Chroma documents + DeepSeek).

Loads paths and thresholds from the environment; validates inputs and raises
on invalid configuration (no silent fallbacks to null).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.chroma import bootstrap_project, resolve_path

logger = logging.getLogger(__name__)


def _embed_kwargs_for_ollama() -> Dict[str, Any]:
    """
    Build kwargs for create_embeddings when using the Ollama backend.

    Aligns with document load scripts: OLLAMA_BASE_URL and embed model from
    Ollama config / optional RAG_OLLAMA_EMBED_MODEL override.

    Returns:
        Dict with ollama_model and ollama_base_url.

    Raises:
        ValueError: If Ollama env is incomplete (from get_ollama_config).
    """
    from src.llm.call_ollama import get_ollama_config

    cfg = get_ollama_config()
    model = (os.getenv("RAG_OLLAMA_EMBED_MODEL") or "").strip() or cfg.embed_model
    base = cfg.base_url.rstrip("/")
    return {"ollama_model": model, "ollama_base_url": base}


@dataclass(frozen=True)
class RagRuntimeConfig:
    """
    Immutable RAG runtime settings resolved from env and project root.

    Attributes:
        project_root: Repository root from bootstrap_project().
        chroma_documents_path: Resolved absolute path to Chroma persist dir.
        chroma_collection_name: Collection name (default documents).
        embed_model: minilm | ollama | qwen3 (must match ingest).
        embed_extra: Extra kwargs passed to create_embeddings (e.g. Ollama URL).
        similarity_threshold: Minimum cosine similarity for kept chunks (0..1).
        chroma_top_k: Maximum chunks to return after filtering.
        chroma_query_prefetch: Neighbors to fetch before similarity filter.
    """

    project_root: Path
    chroma_documents_path: Path
    chroma_collection_name: str
    embed_model: str
    embed_extra: Dict[str, Any]
    similarity_threshold: float
    chroma_top_k: int
    chroma_query_prefetch: int

    @classmethod
    def from_env(cls) -> "RagRuntimeConfig":
        """
        Load configuration from environment variables.

        Returns:
            RagRuntimeConfig: Frozen config snapshot.

        Raises:
            ValueError: If numeric env vars are invalid or paths cannot resolve.
        """
        project_root = Path(bootstrap_project()).resolve()

        default_chroma = str(project_root / "data" / "chroma_db" / "documents")
        chroma_path_str = resolve_path(
            "CHROMA_DOCUMENTS_PATH", default_chroma, project_root
        )
        chroma_documents_path = Path(chroma_path_str).expanduser().resolve()

        collection = (
            os.getenv("CHROMA_DOCUMENTS_COLLECTION", "documents").strip() or "documents"
        )

        embed_model = (os.getenv("RAG_EMBED_MODEL", "minilm").strip() or "minilm").lower()
        if embed_model not in ("minilm", "ollama", "qwen3"):
            raise ValueError(
                f"RAG_EMBED_MODEL must be one of minilm, ollama, qwen3; got {embed_model!r}"
            )

        embed_extra: Dict[str, Any] = {}
        if embed_model == "ollama":
            embed_extra = _embed_kwargs_for_ollama()

        raw_thr = (os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7")).strip()
        try:
            similarity_threshold = float(raw_thr)
        except ValueError as exc:
            raise ValueError(
                f"RAG_SIMILARITY_THRESHOLD must be a float; got {raw_thr!r}"
            ) from exc
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("RAG_SIMILARITY_THRESHOLD must be in (0, 1]")

        raw_k = (os.getenv("RAG_CHROMA_TOP_K", "3")).strip()
        try:
            chroma_top_k = int(raw_k)
        except ValueError as exc:
            raise ValueError(f"RAG_CHROMA_TOP_K must be a positive int; got {raw_k!r}") from exc
        if chroma_top_k < 1:
            raise ValueError("RAG_CHROMA_TOP_K must be >= 1")

        raw_prefetch = (os.getenv("RAG_CHROMA_QUERY_PREFETCH", "")).strip()
        if raw_prefetch:
            try:
                chroma_query_prefetch = int(raw_prefetch)
            except ValueError as exc:
                raise ValueError(
                    f"RAG_CHROMA_QUERY_PREFETCH must be a positive int; got {raw_prefetch!r}"
                ) from exc
        else:
            chroma_query_prefetch = max(chroma_top_k * 5, 20)
        if chroma_query_prefetch < chroma_top_k:
            chroma_query_prefetch = chroma_top_k

        cfg = cls(
            project_root=project_root,
            chroma_documents_path=chroma_documents_path,
            chroma_collection_name=collection,
            embed_model=embed_model,
            embed_extra=embed_extra,
            similarity_threshold=similarity_threshold,
            chroma_top_k=chroma_top_k,
            chroma_query_prefetch=chroma_query_prefetch,
        )
        logger.debug(
            "RagRuntimeConfig: chroma_path=%s collection=%s embed=%s thr=%.3f top_k=%d",
            cfg.chroma_documents_path,
            cfg.chroma_collection_name,
            cfg.embed_model,
            cfg.similarity_threshold,
            cfg.chroma_top_k,
        )
        return cfg
