"""
RAG pipeline for Amazon FBA questions: explicit 7-step flow.

Steps:
  1. Load embedding model (local) -> embedder
  2. User question -> embedder.embed_query() -> question vector
  3. ChromaDB + CHROMA_PERSIST_PATH -> load vector store
  4. question vector + vector store -> retrieve top-k docs from Chroma
  5. Ollama + local model -> llm
  6. retrieved docs + user question -> build RAG prompt
  7. rag prompt + llm -> answer

Uses ai-toolkit for embeddings, Chroma, and retrieval. Parameters from rag_03_04.
"""

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

# [ANNOTATION] Qwen3-VL-Embedding scripts must be on path for ai_toolkit.models
_model_scripts = PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B" / "scripts"
if _model_scripts.is_dir():
    sys.path.insert(0, str(_model_scripts))

# Config (from rag_03_04_chroma_llm_ollama_langchain.py)
CHROMA_PERSIST_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / "amazon" / "fba")
COLLECTION_NAME = "amazon_fba_features"
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


# ===================== Step 1: Load embedding model (local) -> embedder =====================
def step1_load_embedder():
    """Load local Qwen3-VL-Embedding-2B via ai-toolkit."""
    _patch_torch_autocast()
    from ai_toolkit.models import LocalQwenEmbeddings
    model_path = str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B")
    return LocalQwenEmbeddings(model_path)


# ===================== Step 2: User question -> embedder.embed_query() -> question vector =====================
def step2_embed_question(embedder, question: str) -> list:
    """Generate question embedding vector."""
    return embedder.embed_query(question)


# ===================== Step 3: ChromaDB + CHROMA_PERSIST_PATH -> load vector store =====================
def step3_load_vector_store(embedder):
    """Connect to existing Chroma collection via ai-toolkit load_store."""
    from ai_toolkit.chroma import load_store
    return load_store(
        collection_name=COLLECTION_NAME,
        embeddings=embedder,
        persist_directory=CHROMA_PERSIST_PATH,
    )


# ===================== Step 4: question vector + vector store -> retrieve top-k docs =====================
def step4_retrieve_docs(vector_store, question_vector: list) -> list:
    """Retrieve similar docs from Chroma using question vector."""
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents
    collection = get_chroma_collection(vector_store)
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas"],
    )
    # [ANNOTATION] Chroma returns list-of-lists; single query -> index [0]; ids not in include
    n = len(results["documents"][0])
    ids = [f"doc_{i}" for i in range(n)]
    docs = chroma_to_documents(
        ids=ids,
        documents=results["documents"][0],
        metadatas=results["metadatas"][0],
    )
    return docs


# ===================== Step 5: Ollama + local model -> llm =====================
def step5_create_llm():
    """Create Ollama LLM via langchain_ollama."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )


# ===================== Step 6: retrieved docs + user question -> build RAG prompt =====================
def step6_build_rag_prompt(retrieved_docs: list, question: str) -> str:
    """Build RAG prompt from retrieved docs and user question."""
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


# ===================== Step 7: rag prompt + llm -> answer =====================
def step7_generate_answer(llm, rag_prompt: str) -> str:
    """Generate answer from LLM."""
    return llm.invoke(rag_prompt) or ""


def rag_answer(question: str = DEFAULT_QUERY) -> str:
    """
    Run full RAG pipeline (steps 1-7).

    Args:
        question: User question about Amazon FBA.

    Returns:
        Generated answer.
    """
    if not Path(CHROMA_PERSIST_PATH).exists():
        raise FileNotFoundError(
            f"Chroma path not found: {CHROMA_PERSIST_PATH}. "
            "Run rag_01_load_split_embedding_chroma_qwen3.py first."
        )

    print(f"Query: {question}\n")
    total_start = time.perf_counter()

    # Step 1: load embedder (ai-toolkit LocalQwenEmbeddings)
    t0 = time.perf_counter()
    embedder = step1_load_embedder()
    print(f"  Step 1 (load embedder): {time.perf_counter() - t0:.2f}s")

    # Step 2: question -> embedding vector
    t0 = time.perf_counter()
    question_vector = step2_embed_question(embedder, question)
    print(f"  Step 2 (embed question): {time.perf_counter() - t0:.2f}s")

    # Step 3: load Chroma vector store (ai-toolkit load_store)
    t0 = time.perf_counter()
    vector_store = step3_load_vector_store(embedder)
    print(f"  Step 3 (load vector store): {time.perf_counter() - t0:.2f}s")

    # Step 4: question vector + vector store -> top-k retrieved docs (ai-toolkit query_collection)
    t0 = time.perf_counter()
    retrieved_docs = step4_retrieve_docs(vector_store, question_vector)
    print(f"  Step 4 (retrieve docs): {time.perf_counter() - t0:.2f}s")

    # Step 5: Ollama LLM
    t0 = time.perf_counter()
    llm = step5_create_llm()
    print(f"  Step 5 (create LLM): {time.perf_counter() - t0:.2f}s")

    # Step 6: retrieved docs + question -> RAG prompt
    t0 = time.perf_counter()
    rag_prompt = step6_build_rag_prompt(retrieved_docs, question)
    print(f"  Step 6 (build RAG prompt): {time.perf_counter() - t0:.2f}s")

    # Step 7: RAG prompt + LLM -> answer
    t0 = time.perf_counter()
    answer = step7_generate_answer(llm, rag_prompt)
    print(f"  Step 7 (LLM generate): {time.perf_counter() - t0:.2f}s")

    print(f"\n  Total: {time.perf_counter() - total_start:.2f}s\n")
    print("Answer:")
    print(answer)
    if retrieved_docs:
        print(f"\nSources ({len(retrieved_docs)} chunks):")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {src} (page {page})")

    return answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG for Amazon FBA questions")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Question about Amazon FBA")
    args = parser.parse_args()
    rag_answer(args.query)
