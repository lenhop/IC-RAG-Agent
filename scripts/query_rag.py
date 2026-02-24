#!/usr/bin/env python3
"""
RAG query script: retrieve from Chroma and generate answers using Ollama.

Queries the Chroma vector store populated by load_documents_to_chroma.py.
Uses Ollama llama3.2:latest for answer generation.

Flow:
  1. Load embedding model (must match ingest)
  2. Load Chroma vector store
  3. Create Ollama LLM (llama3.2:latest)
  4. User question -> embed_query -> question vector
  5. Retrieve top-k docs from Chroma
  6. Build RAG prompt (context + question)
  7. LLM generates answer

Usage:
  python scripts/query_rag.py --query "Your question"
  python scripts/query_rag.py --interactive
  python scripts/query_rag.py --mode documents --query "Question"
  python scripts/query_rag.py --mode general --query "What is RAG?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Path setup: project root and ai-toolkit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Import from Layer 2 (src/rag/query_pipeline)
from src.rag.query_pipeline import (
    RAGPipeline,
    AnswerMode,
    ANSWER_MODES,
    get_collection_count,
    print_result,
)


def _resolve_path(env_key: str, default: str) -> str:
    """Resolve path from env; if relative, join with PROJECT_ROOT."""
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


# Config from .env with fallbacks (must match load_documents_to_chroma.py)
CHROMA_PERSIST_PATH = _resolve_path(
    "CHROMA_DOCUMENTS_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "documents")
)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", os.getenv("MAX_RETRIEVAL_DOCS", "5")))
OLLAMA_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2:latest")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "minilm")
DEFAULT_QUERY = "What are the main topics in the documents?"


def _parse_mode_input(raw: str) -> AnswerMode | None:
    """Parse user input for mode: 1/doc, 2/general, 3/hybrid. Returns None if invalid."""
    s = raw.strip().lower()
    if s in ("1", "doc", "documents"):
        return "documents"
    if s in ("2", "gen", "general"):
        return "general"
    if s in ("3", "hybrid"):
        return "hybrid"
    return None


def main(
    query: str | None = None,
    interactive: bool = False,
    mode: AnswerMode = "hybrid",
    embed_model: str = EMBED_MODEL,
    chroma_path: str | None = None,
    collection_name: str | None = None,
    retrieval_k: int | None = None,
    llm_model: str | None = None,
    verbose: bool = True,
) -> str | None:
    """
    Run RAG pipeline. Build once, then single query or interactive loop.
    """
    chroma_path = chroma_path or CHROMA_PERSIST_PATH
    if not Path(chroma_path).exists():
        raise FileNotFoundError(
            f"Chroma path not found: {chroma_path}. "
            "Run load_documents_to_chroma.py first to ingest documents."
        )

    if verbose:
        print("Building RAG pipeline (one-time setup)...")
    pipeline = RAGPipeline.build(
        embed_model=embed_model,
        chroma_path=chroma_path,
        collection_name=collection_name or COLLECTION_NAME,
        retrieval_k=retrieval_k,
        llm_model=llm_model or OLLAMA_MODEL,
        verbose=verbose,
        project_root=PROJECT_ROOT,
    )

    if interactive:
        current_mode = mode
        last_answer = None
        print(
            "\nAnswer modes: 1=documents only, 2=general knowledge only, 3=hybrid (default)"
        )
        print("Type 1/2/3 or doc/general/hybrid to switch mode. Type question to ask. Type `exit` to exit.\n")
        while True:
            try:
                q = input(f"[{current_mode}] Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() == "exit":
                break
            if not q:
                continue
            parsed = _parse_mode_input(q)
            if parsed is not None:
                current_mode = parsed
                print(f"Mode set to: {current_mode}\n")
                continue
            if verbose:
                print(f"\nQuery: {q}\n")
            try:
                answer, docs = pipeline.query(q, mode=current_mode, verbose=verbose)
                coll_count = get_collection_count(pipeline.vector_store) if not docs else None
                print_result(answer, docs, mode=current_mode, collection_count=coll_count)
                last_answer = answer
            except Exception as e:
                print(f"[ERROR] {e}")
        return last_answer

    q = query or DEFAULT_QUERY
    if verbose:
        print(f"Query: {q}\n")
    answer, docs = pipeline.query(q, mode=mode, verbose=verbose)
    coll_count = get_collection_count(pipeline.vector_store) if not docs else None
    print_result(answer, docs, mode=mode, collection_count=coll_count)
    return answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RAG query: retrieve from Chroma, generate answer with Ollama llama3.2"
    )
    parser.add_argument("--query", type=str, default=None, help="Question (default: built-in)")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: multiple questions without restart",
    )
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "ollama", "qwen3"),
        default=EMBED_MODEL,
        help=f"Embedding model (must match ingest, default: {EMBED_MODEL})",
    )
    parser.add_argument("--chroma-path", type=str, default=None, help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    parser.add_argument("--retrieval-k", type=int, default=None, help="Number of docs to retrieve")
    parser.add_argument("--llm-model", type=str, default=None, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument(
        "--mode",
        choices=ANSWER_MODES,
        default="hybrid",
        help="Answer mode: documents, general, or hybrid (default: hybrid)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress timing output")
    args = parser.parse_args()

    try:
        main(
            query=args.query,
            interactive=args.interactive,
            mode=args.mode,
            embed_model=args.embed_model,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            retrieval_k=args.retrieval_k,
            llm_model=args.llm_model,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
