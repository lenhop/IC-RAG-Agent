"""
RAG pipeline for Amazon FBA questions using Chroma, Ollama, and LangChain.

Integrates:
  - Chroma: Vector store with Amazon FBA documents (from rag_01/rag_02)
  - Local Qwen3-VL-Embedding-2B: Embeddings for retrieval
  - Ollama (qwen3:1.7b): Local LLM for answer generation
  - LangChain: RetrievalQA chain

RAG workflow step order (flagged in code as [STEP N]):
  STEP 1: User sends query
  STEP 2: LocalQwenEmbeddings.embed_query(query) -> vector
  STEP 3: Chroma similarity_search -> top-k chunks
  STEP 4: RetrievalQA concatenates chunks into {context}
  STEP 5: PromptTemplate fills {context} + {question}
  STEP 6: OllamaLLM generates answer from prompt
  STEP 7: Result returns "result" + "source_documents"

Prerequisites:
  1. Run rag_01 to load/split/embed PDFs and store in Chroma
  2. Ollama running with: ollama pull qwen3:1.7b
  3. pip install langchain-ollama chromadb langchain-community
"""

import sys
from pathlib import Path

# [ANNOTATION] Path setup: resolve project root (src/draft -> IC-RAG-Agent) and locate ai-toolkit
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

# Chroma config (same as rag_01, rag_02)
CHROMA_PERSIST_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / "amazon" / "fba")
COLLECTION_NAME = "amazon_fba_features"

# Ollama config
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TEMPERATURE = 0.3
OLLAMA_NUM_CTX = 4096

# RAG config
RETRIEVAL_K = 4
DEFAULT_QUERY = "What is FBA Global Selling and what methods does it include?"


def _patch_torch_autocast() -> None:
    """Patch torch.is_autocast_enabled for PyTorch < 2.3 (Qwen embedding compatibility)."""
    import torch
    # [ANNOTATION] Skip if already patched (e.g. multiple imports)
    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return
    # [ANNOTATION] Newer PyTorch accepts device_type; older raises TypeError
    try:
        torch.is_autocast_enabled("cpu")
    except TypeError:
        original = torch.is_autocast_enabled

        def _patched(device_type=None):
            return original()

        _patched._qwen_patched = True
        torch.is_autocast_enabled = _patched


def _load_embeddings():
    """Load local Qwen3-VL-Embedding-2B for Chroma retrieval."""
    import torch
    from typing import List
    from langchain_core.embeddings import Embeddings

    _patch_torch_autocast()

    # [ANNOTATION] Qwen3-VL-Embedding scripts must be on path for qwen3_vl_embedding import
    model_scripts_path = PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B" / "scripts"
    if model_scripts_path.is_dir():
        sys.path.insert(0, str(model_scripts_path))

    from qwen3_vl_embedding import Qwen3VLEmbedder

    # [ANNOTATION] LangChain Embeddings interface: embed_documents (batch) and embed_query (single)
    class LocalQwenEmbeddings(Embeddings):
        """LangChain Embeddings wrapper for local Qwen3-VL-Embedding-2B (text-only)."""

        def __init__(self, model_path: str, max_length: int = 8192):
            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=model_path,
                max_length=max_length,
            )

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            inputs = [{"text": t} for t in texts]
            out = self.embedder.process(inputs, normalize=True)
            # [ANNOTATION] Convert bfloat16 to float32 for Chroma/NumPy compatibility
            out = out.to(dtype=torch.float32)
            arr = out.cpu().numpy()
            if arr.ndim == 1:
                return [arr.tolist()]
            return [arr[i].tolist() for i in range(len(arr))]

        def embed_query(self, text: str) -> List[float]:
            # [STEP 2] LocalQwenEmbeddings.embed_query(query) -> vector
            inputs = [{"text": text}]
            out = self.embedder.process(inputs, normalize=True)
            out = out.to(dtype=torch.float32)
            arr = out.cpu().numpy()
            if arr.ndim == 2:
                return arr[0].tolist()
            return arr.tolist()

    model_path = str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-2B")
    return LocalQwenEmbeddings(model_path)


def _load_chroma_vectorstore(embeddings):
    """Connect to existing Chroma collection (created by rag_01)."""
    from ai_toolkit.chroma import create_chroma_store

    # [ANNOTATION] documents=None -> connect to existing collection, do not add new docs
    return create_chroma_store(
        documents=None,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_PATH,
    )


def _create_rag_chain(vector_store, llm):
    """Create RetrievalQA chain for RAG."""
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate

    # [ANNOTATION] RAG prompt: {context} = retrieved chunks, {question} = user query
    prompt_template = """You are a helpful assistant that answers questions about Amazon FBA (Fulfillment by Amazon) based on the provided context.

Use the following context to answer the question. If the context does not contain relevant information, say so. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # [STEP 3] Chroma: similarity_search -> top-k chunks (via retriever)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    # [STEP 4] RetrievalQA: concatenates chunks into {context}
    # [STEP 5] PromptTemplate: fills {context} + {question}
    # [STEP 6] OllamaLLM: generates answer from prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def main(query: str = DEFAULT_QUERY) -> str:
    """
    Run RAG query for Amazon FBA questions.

    Args:
        query: User question about Amazon FBA.

    Returns:
        Generated answer based on retrieved context.
    """
    from langchain_ollama import OllamaLLM

    # [ANNOTATION] Pre-check: Chroma must be populated by rag_01 before running RAG
    chroma_path = Path(CHROMA_PERSIST_PATH)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: {CHROMA_PERSIST_PATH}. "
            "Run rag_01_load_split_embedding_chroma_qwen3.py first to populate the vector store."
        )

    print("Loading embeddings (Qwen3-VL-Embedding-2B)...")
    embeddings = _load_embeddings()

    print("Connecting to Chroma vector store...")
    vector_store = _load_chroma_vectorstore(embeddings)

    print(f"Initializing Ollama LLM ({OLLAMA_MODEL})...")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )

    print("Building RAG chain...")
    chain = _create_rag_chain(vector_store, llm)

    print(f"\nQuery: {query}\n")
    # [STEP 1] User query enters; chain runs steps 2-6 internally
    result = chain.invoke({"query": query})

    # [STEP 7] Result: "result" (answer text) + "source_documents" (retrieved chunks)
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    print("Answer:")
    print(answer)
    if source_docs:
        print(f"\nSources ({len(source_docs)} chunks):")
        for i, doc in enumerate(source_docs[:3], 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  {i}. {src} (page {page})")

    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG for Amazon FBA questions")
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Question about Amazon FBA",
    )
    args = parser.parse_args()
    main(query=args.query)
