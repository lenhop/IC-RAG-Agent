"""
RAG pipeline for Amazon FBA: optimized with pipeline reuse and interactive mode.

Optimizations over rag_03_05:
  1. Pipeline reuse: Build embedder + vector_store + llm once, reuse for multiple queries
  2. Interactive mode: Ask multiple questions without restarting (--interactive)
  3. Per-step timing: Setup timed once; query steps timed per call

Same 7-step flow as rag_03_05, but steps 1/3/5 run once at build, steps 2/4/6/7 per query.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# [ANNOTATION] Path setup: project root and ai-toolkit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

_model_scripts = PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B" / "scripts"
if _model_scripts.is_dir():
    sys.path.insert(0, str(_model_scripts))

# Config (from rag_03_05)
CHROMA_PERSIST_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / "amazon")
COLLECTION_NAME = "amazon"
# Embedding model: must match load_pdfs_to_chroma.py (default minilm for low-memory)
EMBED_MODEL = "minilm"
MINILM_MODEL_PATH = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TEMPERATURE = 0.3
OLLAMA_NUM_CTX = 4096
RETRIEVAL_K = 4
DEFAULT_QUERY = "What is FBA Global Selling and what methods does it include?"


def _patch_torch_autocast() -> None:
    """Patch torch.is_autocast_enabled for PyTorch < 2.3 (Qwen embedding compatibility)."""
    import torch
    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return
    try:
        torch.is_autocast_enabled("cpu")
    except TypeError:
        original = torch.is_autocast_enabled

        def _patched(device_type=None):
            return original()

        _patched._qwen_patched = True
        torch.is_autocast_enabled = _patched


def _step1_load_embedder(embed_model: str = EMBED_MODEL):
    """Step 1: Load embedding model. Must match load_pdfs_to_chroma.py embedder."""
    if embed_model == "qwen3":
        _patch_torch_autocast()
        from ai_toolkit.models import LocalQwenEmbeddings
        return LocalQwenEmbeddings(str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B"))
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=MINILM_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _step2_embed_question(embedder, question: str) -> list:
    """Step 2: User question -> embedder.embed_query() -> question vector."""
    return embedder.embed_query(question)


def _step3_load_vector_store(embedder):
    """Step 3: ChromaDB + CHROMA_PERSIST_PATH -> load vector store."""
    from ai_toolkit.chroma import load_store
    return load_store(
        collection_name=COLLECTION_NAME,
        embeddings=embedder,
        persist_directory=CHROMA_PERSIST_PATH,
    )


def _step4_retrieve_docs(vector_store, question_vector: list) -> list:
    """Step 4: question vector + vector store -> retrieve top-k docs."""
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents
    collection = get_chroma_collection(vector_store)
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas"],
    )
    n = len(results["documents"][0])
    ids = [f"doc_{i}" for i in range(n)]
    return chroma_to_documents(
        ids=ids,
        documents=results["documents"][0],
        metadatas=results["metadatas"][0],
    )


def _step5_create_llm():
    """Step 5: Ollama + local model -> llm."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )


def _step6_build_rag_prompt(retrieved_docs: list, question: str) -> str:
    """Step 6: retrieved docs + user question -> build RAG prompt."""
    from langchain_core.prompts import PromptTemplate
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions about Amazon FBA based on the provided context.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    return prompt.format(context=context, question=question)


def _step7_generate_answer(llm, rag_prompt: str) -> str:
    """Step 7: rag prompt + llm -> answer."""
    return llm.invoke(rag_prompt) or ""


class RAGPipeline:
    """
    Reusable RAG pipeline: build once (steps 1, 3, 5), query many times (steps 2, 4, 6, 7).

    [ANNOTATION] Avoids reloading embedder (~31s) and LLM (~0.3s) per query; only steps 2/4/6/7 run.
    """

    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    @classmethod
    def build(cls, verbose: bool = True) -> "RAGPipeline":
        """
        Build pipeline once: load embedder, vector store, LLM (steps 1, 3, 5).

        Returns:
            RAGPipeline instance ready for multiple queries.
        """
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        embedder = _step1_load_embedder()
        if verbose:
            print(f"  [Build] Step 1 (load embedder): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        vector_store = _step3_load_vector_store(embedder)
        if verbose:
            print(f"  [Build] Step 3 (load vector store): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        llm = _step5_create_llm()
        if verbose:
            print(f"  [Build] Step 5 (create LLM): {time.perf_counter() - t0:.2f}s")

        if verbose:
            print(f"  [Build] Total setup: {time.perf_counter() - total_start:.2f}s\n")

        return cls(embedder, vector_store, llm)

    def query(self, question: str, verbose: bool = True) -> tuple[str, list]:
        """
        Run query (steps 2, 4, 6, 7). Reuses cached embedder, vector_store, llm.

        Args:
            question: User question about Amazon FBA.
            verbose: Print per-step timing.

        Returns:
            (answer, retrieved_docs)
        """
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        question_vector = _step2_embed_question(self.embedder, question)
        if verbose:
            print(f"  Step 2 (embed question): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        retrieved_docs = _step4_retrieve_docs(self.vector_store, question_vector)
        if verbose:
            print(f"  Step 4 (retrieve docs): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        rag_prompt = _step6_build_rag_prompt(retrieved_docs, question)
        if verbose:
            print(f"  Step 6 (build RAG prompt): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        answer = _step7_generate_answer(self.llm, rag_prompt)
        if verbose:
            print(f"  Step 7 (LLM generate): {time.perf_counter() - t0:.2f}s")

        if verbose:
            print(f"  Query total: {time.perf_counter() - total_start:.2f}s\n")

        return answer, retrieved_docs


def _print_result(answer: str, retrieved_docs: list) -> None:
    """Print answer and source references."""
    print("Answer:")
    print(answer)
    if retrieved_docs:
        print(f"\nSources ({len(retrieved_docs)} chunks):")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {src} (page {page})")


def main(
    query: str | None = None,
    interactive: bool = False,
    verbose: bool = True,
) -> str | None:
    """
    Run RAG pipeline. Build once, then single query or interactive loop.

    Args:
        query: Question (uses DEFAULT_QUERY if None and not interactive).
        interactive: If True, loop for multiple questions (empty input to exit).
        verbose: Print timing.

    Returns:
        Last answer, or None if interactive with no queries.
    """
    if not Path(CHROMA_PERSIST_PATH).exists():
        raise FileNotFoundError(
            f"Chroma path not found: {CHROMA_PERSIST_PATH}. "
            "Run rag_01_load_split_embedding_chroma_qwen3.py first."
        )

    if verbose:
        print("Building RAG pipeline (one-time setup)...")
    pipeline = RAGPipeline.build(verbose=verbose)

    if interactive:
        last_answer = None
        while True:
            q = input("\nQuestion (empty to exit): ").strip()
            if not q:
                break
            if verbose:
                print(f"\nQuery: {q}\n")
            answer, docs = pipeline.query(q, verbose=verbose)
            _print_result(answer, docs)
            last_answer = answer
        return last_answer

    q = query or DEFAULT_QUERY
    if verbose:
        print(f"Query: {q}\n")
    answer, docs = pipeline.query(q, verbose=verbose)
    _print_result(answer, docs)
    return answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RAG for Amazon FBA (optimized: pipeline reuse, interactive mode)"
    )
    parser.add_argument("--query", type=str, default=None, help="Question (default: built-in)")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: multiple questions without restart",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress timing output")
    args = parser.parse_args()
    main(query=args.query, interactive=args.interactive, verbose=not args.quiet)
