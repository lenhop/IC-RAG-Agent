"""
Orchestrates RAG modes: general, amazon_business (dual path), documents (Chroma + grounded LLM).

amazon_business runs Chroma retrieval and DeepSeek evidence in parallel threads, then merges.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from src.llm.embeddings import create_embeddings
from src.llm.text_generation_backend import resolve_text_generation_backend

from .chroma_retrieve import ChromaRetriever
from .config import RagRuntimeConfig
from .deepseek_retrieve import DeepSeekRetrieveFacade
from .merge_compose import MergeComposer

logger = logging.getLogger(__name__)

_ALLOWED_MODES = frozenset({"general", "amazon_business", "documents"})


def _format_chroma_context(hits: List[Dict[str, Any]]) -> str:
    """Build a single string of numbered excerpts for prompts."""
    parts: List[str] = []
    for i, h in enumerate(hits, 1):
        sim = float(h.get("similarity", 0.0))
        text = (h.get("text") or "").strip()
        meta = h.get("metadata") or {}
        src = meta.get("source", "")
        parts.append(f"[{i}] (similarity={sim:.3f}, source={src})\n{text}")
    return "\n\n".join(parts) if parts else ""


class RagQueryService:
    """
    Lazy-loaded embedder + Chroma collection; class-level cache for FastAPI workers.
    """

    _lock = threading.Lock()
    _config: Optional[RagRuntimeConfig] = None
    _embedder: Any = None
    _collection: Any = None

    @classmethod
    def reset_cache_for_testing(cls) -> None:
        """Drop cached clients (pytest hooks)."""
        with cls._lock:
            cls._config = None
            cls._embedder = None
            cls._collection = None

    @classmethod
    def _ensure_embedder(cls) -> RagRuntimeConfig:
        """
        Load config and embedding model (no Chroma required).

        Returns:
            RagRuntimeConfig snapshot.
        """
        with cls._lock:
            if cls._embedder is not None and cls._config is not None:
                return cls._config

            cfg = RagRuntimeConfig.from_env()
            embed_kwargs: Dict[str, Any] = {
                "model_type": cfg.embed_model,
                "project_root": cfg.project_root,
            }
            embed_kwargs.update(cfg.embed_extra)
            embedder = create_embeddings(**embed_kwargs)
            cls._config = cfg
            cls._embedder = embedder
            logger.info("RAG embedder ready model=%s", cfg.embed_model)
            return cfg

    @classmethod
    def _ensure_chroma_collection(cls) -> Tuple[RagRuntimeConfig, Any, Any]:
        """
        Ensure embedder and Chroma collection are available.

        Returns:
            Tuple of (config, embedder, collection).

        Raises:
            FileNotFoundError: If Chroma directory is missing.
            RuntimeError: If collection cannot be opened.
        """
        cfg = cls._ensure_embedder()
        with cls._lock:
            if cls._collection is not None:
                return cfg, cls._embedder, cls._collection

            if not cfg.chroma_documents_path.exists():
                raise FileNotFoundError(
                    f"Chroma persist path does not exist: {cfg.chroma_documents_path}. "
                    "Ingest documents before calling RAG retrieval modes."
                )

            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(cfg.chroma_documents_path),
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                collection = client.get_collection(name=cfg.chroma_collection_name)
            except Exception as exc:
                logger.error(
                    "Failed to open Chroma collection %r: %s",
                    cfg.chroma_collection_name,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Chroma collection {cfg.chroma_collection_name!r} unavailable: {exc}"
                ) from exc

            cls._collection = collection
            logger.info(
                "RAG Chroma ready path=%s collection=%s count=%s",
                cfg.chroma_documents_path,
                cfg.chroma_collection_name,
                collection.count(),
            )
            return cfg, cls._embedder, collection

    @classmethod
    def run(cls, question: str, mode: str) -> Dict[str, Any]:
        """
        Execute one RAG query for the given mode.

        Args:
            question: User question text.
            mode: general | amazon_business | documents.

        Returns:
            On success: {"answer": str, "sources": list}.
            On failure: {"error": str, "error_type": str} for gateway compatibility.
        """
        raw_mode = (mode or "").strip().lower()
        if raw_mode not in _ALLOWED_MODES:
            return {
                "error": f"unsupported mode {mode!r}; allowed: {sorted(_ALLOWED_MODES)}",
                "error_type": "ValueError",
            }

        q = (question or "").strip()
        if not q:
            return {
                "error": "question must be a non-empty string",
                "error_type": "ValueError",
            }

        try:
            if raw_mode == "general":
                return cls._run_general(q)
            if raw_mode == "amazon_business":
                return cls._run_amazon_business(q)
            return cls._run_documents(q)
        except ValueError as exc:
            logger.warning("RAG ValueError: %s", exc)
            return {"error": str(exc), "error_type": "ValueError"}
        except FileNotFoundError as exc:
            logger.warning("RAG FileNotFoundError: %s", exc)
            return {"error": str(exc), "error_type": "FileNotFoundError"}
        except RuntimeError as exc:
            logger.error("RAG RuntimeError: %s", exc, exc_info=True)
            return {"error": str(exc), "error_type": "RuntimeError"}
        except Exception as exc:
            logger.exception("RAG unexpected error: %s", exc)
            return {"error": str(exc), "error_type": type(exc).__name__}

    @classmethod
    def _run_general(cls, question: str) -> Dict[str, Any]:
        """DeepSeek only; no Chroma."""
        cls._ensure_embedder()
        answer = DeepSeekRetrieveFacade.general_answer(question)
        return {"answer": answer, "sources": []}

    @classmethod
    def _run_amazon_business(cls, question: str) -> Dict[str, Any]:
        """
        Chroma retrieval and DeepSeek evidence in parallel, then merge (Chroma wins conflicts).

        Both branches are I/O bound (vector DB + HTTP LLM); a small thread pool reduces
        wall-clock latency versus strict sequencing.
        """
        cfg, embedder, collection = cls._ensure_chroma_collection()

        def _chroma_branch() -> Tuple[List[Dict[str, Any]], str]:
            """Run Chroma query + format context in worker thread."""
            hits_local = ChromaRetriever.retrieve(
                collection,
                embedder,
                question,
                top_k=cfg.chroma_top_k,
                min_similarity=cfg.similarity_threshold,
                prefetch_n=cfg.chroma_query_prefetch,
            )
            return hits_local, _format_chroma_context(hits_local)

        def _deepseek_branch() -> str:
            """Run DeepSeek evidence pass in worker thread."""
            return DeepSeekRetrieveFacade.evidence_for_query(question)

        # Both tasks start immediately; result() order only affects which error is raised first.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_chroma = executor.submit(_chroma_branch)
            future_deepseek = executor.submit(_deepseek_branch)
            try:
                hits, chroma_context = future_chroma.result()
                evidence = future_deepseek.result()
            except Exception:
                logger.exception("amazon_business parallel branch failed")
                raise

        answer = MergeComposer.final_answer(
            question,
            chroma_context,
            evidence,
            text_generation_backend=resolve_text_generation_backend(),
        )
        sources = ChromaRetriever.hits_to_sources(hits)
        return {"answer": answer, "sources": sources}

    @classmethod
    def _run_documents(cls, question: str) -> Dict[str, Any]:
        """Chroma retrieval + single grounded DeepSeek answer (no dual model-evidence path)."""
        cfg, embedder, collection = cls._ensure_chroma_collection()
        hits = ChromaRetriever.retrieve(
            collection,
            embedder,
            question,
            top_k=cfg.chroma_top_k,
            min_similarity=cfg.similarity_threshold,
            prefetch_n=cfg.chroma_query_prefetch,
        )
        chroma_context = _format_chroma_context(hits)
        if not chroma_context.strip():
            chroma_context = (
                "(No document excerpts passed the similarity threshold; "
                "state that the indexed docs do not contain sufficient evidence.)"
            )
        answer = DeepSeekRetrieveFacade.answer_from_chroma_context_only(
            question,
            chroma_context,
        )
        sources = ChromaRetriever.hits_to_sources(hits)
        return {"answer": answer, "sources": sources}
