"""
RAG query pipeline - Layer 2.

Reusable RAG pipeline: build once (embedder, vector store, LLM), query many times.
Retrieve from Chroma, generate answers with Ollama.

Flow:
  1. Load embedding model (must match ingest)
  2. Load Chroma vector store
  3. Create Ollama LLM
  4. User question -> embed -> retrieve top-k -> build prompt -> LLM generate
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

# Path setup: project root and ai-toolkit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        import sys
        sys.path.insert(0, str(_path))
        break

# Answer modes: documents-only, general-knowledge-only, hybrid
AnswerMode = Literal["documents", "general", "hybrid"]
ANSWER_MODES: tuple[AnswerMode, ...] = ("documents", "general", "hybrid")


def _resolve_path(env_key: str, default: str, project_root: Path | None = None) -> str:
    """Resolve path from env; if relative, join with project_root."""
    root = project_root or PROJECT_ROOT
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = root / p
    return str(p.resolve())


# Config from .env (must match load_documents_to_chroma.py)
def _get_chroma_path(project_root: Path | None = None) -> str:
    root = project_root or PROJECT_ROOT
    return _resolve_path(
        "CHROMA_DOCUMENTS_PATH",
        str(root / "data" / "chroma_db" / "documents"),
        root,
    )


COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", os.getenv("MAX_RETRIEVAL_DOCS", "5")))
OLLAMA_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2:latest")
OLLAMA_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.3"))
OLLAMA_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "4096"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "minilm")


def _step1_load_embedder(embed_model: str = EMBED_MODEL, project_root: Path | None = None):
    """Step 1: Load embedding model. Must match load_documents_to_chroma.py."""
    from src.rag import create_embeddings
    root = project_root or PROJECT_ROOT
    model_type = embed_model if embed_model in ("ollama", "qwen3") else "minilm"
    return create_embeddings(model_type=model_type, project_root=root)


def _step2_load_vector_store(
    embedder,
    chroma_path: str | None = None,
    collection_name: str | None = None,
    project_root: Path | None = None,
):
    """Step 2: ChromaDB + persist path -> load vector store."""
    from ai_toolkit.chroma import load_store
    root = project_root or PROJECT_ROOT
    path = chroma_path or _get_chroma_path(root)
    return load_store(
        collection_name=collection_name or COLLECTION_NAME,
        embeddings=embedder,
        persist_directory=path,
    )


def _step3_create_llm(model: str = OLLAMA_MODEL):
    """Step 3: Ollama + local model -> llm."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=model,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )


def _step4_embed_question(embedder, question: str) -> list:
    """Step 4: User question -> embedder.embed_query() -> question vector."""
    return embedder.embed_query(question)


def get_collection_count(vector_store) -> int:
    """Return number of chunks in the Chroma collection."""
    from ai_toolkit.chroma import get_chroma_collection
    collection = get_chroma_collection(vector_store)
    return collection.count()


def _step5_retrieve_docs(vector_store, question_vector: list, retrieval_k: int = RETRIEVAL_K) -> list:
    """Step 5: question vector + vector store -> retrieve top-k docs."""
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents
    collection = get_chroma_collection(vector_store)
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=retrieval_k,
        include=["documents", "metadatas"],
    )
    n = len(results["documents"][0])
    ids = [f"doc_{i}" for i in range(n)]
    return chroma_to_documents(
        ids=ids,
        documents=results["documents"][0],
        metadatas=results["metadatas"][0],
    )


def _step6_build_rag_prompt(
    retrieved_docs: list, question: str, mode: AnswerMode = "hybrid"
) -> str:
    """Step 6: retrieved docs + user question + mode -> build RAG prompt."""
    from langchain_core.prompts import PromptTemplate
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if mode == "documents":
        template = """You are a helpful assistant. Answer ONLY based on the context below.
If the context is empty or does not contain relevant information, respond exactly:
"I cannot answer based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        return prompt.format(context=context, question=question)
    elif mode == "general":
        template = """You are a helpful assistant. Answer the question using your general knowledge only.
Do not use any document context.

Question: {question}

Answer:"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )
        return prompt.format(question=question)
    else:
        template = """You are a helpful assistant. Answer based on the provided context.
You may supplement with your general knowledge if the context is incomplete or does not fully cover the question.

Context:
{context}

Question: {question}

Answer:"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        return prompt.format(context=context, question=question)


def _step7_generate_answer(llm, rag_prompt: str) -> str:
    """Step 7: rag prompt + llm -> answer."""
    return llm.invoke(rag_prompt) or ""


class RAGPipeline:
    """
    Reusable RAG pipeline: build once (steps 1, 2, 3), query many times (steps 4, 5, 6, 7).
    """

    def __init__(self, embedder, vector_store, llm, retrieval_k: int = RETRIEVAL_K):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.retrieval_k = retrieval_k

    @classmethod
    def build(
        cls,
        embed_model: str = EMBED_MODEL,
        chroma_path: str | None = None,
        collection_name: str | None = None,
        retrieval_k: int | None = None,
        llm_model: str | None = None,
        verbose: bool = True,
        project_root: Path | None = None,
    ) -> "RAGPipeline":
        """
        Build pipeline once: load embedder, vector store, LLM (steps 1, 2, 3).
        """
        root = project_root or PROJECT_ROOT
        chroma_path = chroma_path or _get_chroma_path(root)
        collection_name = collection_name or COLLECTION_NAME
        retrieval_k = retrieval_k if retrieval_k is not None else RETRIEVAL_K
        llm_model = llm_model or OLLAMA_MODEL

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        embedder = _step1_load_embedder(embed_model, root)
        if verbose:
            print(f"  [Build] Step 1 (load embedder): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        vector_store = _step2_load_vector_store(
            embedder,
            chroma_path=chroma_path,
            collection_name=collection_name,
            project_root=root,
        )
        if verbose:
            print(f"  [Build] Step 2 (load vector store): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        llm = _step3_create_llm(llm_model)
        if verbose:
            print(f"  [Build] Step 3 (create LLM): {time.perf_counter() - t0:.2f}s")

        if verbose:
            coll_count = get_collection_count(vector_store)
            print(f"  [Build] Total setup: {time.perf_counter() - total_start:.2f}s")
            print(f"  [Build] Chroma: {chroma_path} | collection={collection_name} | chunks={coll_count}\n")

        return cls(embedder, vector_store, llm, retrieval_k)

    def query(
        self, question: str, mode: AnswerMode = "hybrid", verbose: bool = True
    ) -> tuple[str, list]:
        """Run query (steps 4, 5, 6, 7). Mode: documents, general, or hybrid."""
        total_start = time.perf_counter()
        retrieved_docs: list = []

        if mode == "general":
            rag_prompt = _step6_build_rag_prompt([], question, mode="general")
            if verbose:
                print(f"  Step 4 (embed question): skipped (general knowledge mode)")
                print(f"  Step 5 (retrieve docs): skipped (general knowledge mode)")
        else:
            t0 = time.perf_counter()
            question_vector = _step4_embed_question(self.embedder, question)
            if verbose:
                print(f"  Step 4 (embed question): {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            retrieved_docs = _step5_retrieve_docs(
                self.vector_store, question_vector, self.retrieval_k
            )
            if verbose:
                print(f"  Step 5 (retrieve docs): {time.perf_counter() - t0:.2f}s")

            if mode == "documents" and len(retrieved_docs) == 0:
                if verbose:
                    coll_count = get_collection_count(self.vector_store)
                    print(f"  Step 6 (build RAG prompt): skipped (no documents)")
                    print(f"  Step 7 (LLM generate): skipped (no documents)")
                    if coll_count == 0:
                        print(
                            f"  [Hint] Collection has 0 chunks. Run: python scripts/load_documents_to_chroma.py"
                        )
                    else:
                        print(
                            f"  [Hint] Collection has {coll_count} chunks but none matched. "
                            "Try hybrid mode or ensure --embed-model matches ingest."
                        )
                    print(f"  Query total: {time.perf_counter() - total_start:.2f}s\n")
                return (
                    "No relevant documents found. Cannot answer from documents.",
                    retrieved_docs,
                )

            t0 = time.perf_counter()
            rag_prompt = _step6_build_rag_prompt(retrieved_docs, question, mode=mode)
            if verbose:
                print(f"  Step 6 (build RAG prompt): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        answer = _step7_generate_answer(self.llm, rag_prompt)
        if verbose:
            print(f"  Step 7 (LLM generate): {time.perf_counter() - t0:.2f}s")

        if verbose:
            print(f"  Query total: {time.perf_counter() - total_start:.2f}s\n")

        return answer, retrieved_docs


def get_source_label(mode: AnswerMode, retrieved_docs: list) -> str:
    """Infer source label from mode and retrieval."""
    if mode == "general":
        return "General Knowledge"
    if len(retrieved_docs) == 0:
        return "General Knowledge" if mode == "hybrid" else "No documents found"
    return f"Document(s) ({len(retrieved_docs)} chunks)"


def print_result(
    answer: str,
    retrieved_docs: list,
    mode: AnswerMode = "hybrid",
    collection_count: int | None = None,
) -> None:
    """Print answer and source attribution."""
    print("Answer:")
    print(answer)
    source_label = get_source_label(mode, retrieved_docs)
    print(f"\nSource: {source_label}")
    if not retrieved_docs and collection_count is not None and collection_count == 0:
        print(
            "\n[Hint] Chroma collection is empty. Run: python scripts/load_documents_to_chroma.py"
        )
    elif not retrieved_docs and collection_count is not None and collection_count > 0:
        print(
            "\n[Hint] No chunks matched. Ensure --embed-model matches the model used during ingest."
        )
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs[:5], 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {src} (page {page})")
