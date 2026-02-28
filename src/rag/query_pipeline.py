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
# Query mode includes "auto" for automatic classification
QueryMode = Literal["documents", "general", "hybrid", "auto"]


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
OLLAMA_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:1.7b")
OLLAMA_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.3"))
OLLAMA_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "4096"))
# Request timeout (seconds) for Ollama API; increase for large models (e.g. 120-300s)
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("RAG_LLM_TIMEOUT", os.getenv("OLLAMA_REQUEST_TIMEOUT", "120")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "minilm")

# Query rewriting (extracted to query_rewriting.py)
from src.rag.query_rewriting import rewrite_query_lightweight


def _question_starts_with_prefix(question: str, prefixes: list[str]) -> bool:
    """Return True if question (lowercased, stripped) starts with any prefix."""
    if not question or not prefixes:
        return False
    q = question.strip().lower()
    return any(q.startswith(p) for p in prefixes if p)


def _faq_similarity_score(question_vector: list, faq_vectors: list) -> float:
    """
    Return min L2 distance from question_vector to FAQ vectors (lower = more similar).

    Uses same L2 metric as Chroma. Returns float('inf') if faq_vectors empty.
    """
    if not question_vector or not faq_vectors:
        return float("inf")

    def l2_dist(a: list, b: list) -> float:
        if len(a) != len(b):
            return float("inf")
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    return min(l2_dist(question_vector, fv) for fv in faq_vectors)


def classify_answer_mode_sequential(
    question: str,
    retrieved_docs: list,
    general_prefixes: list[str],
    domain_signals: list[str],
    project_root: Path | None = None,
    distances: list[float] | None = None,
    question_vector: list | None = None,
    faq_vectors: list | None = None,
) -> AnswerMode:
    """
    Sequential classifier: rewrite, keywords, retrieval, distance threshold, FAQ, LLM.

    Base logic:
    - If question starts with general prefix AND no domain signals -> general
    - Else if retrieved_docs > 0 -> hybrid (or general if distance threshold exceeded)
    - Else if domain signals but no docs -> documents (surface "no document found")
    - Else -> general

    Distance threshold (when RAG_DISTANCE_THRESHOLD_ENABLED and distances provided):
    - If has_docs and min_dist > threshold -> general (docs not relevant enough)
    - If has_docs and min_dist <= threshold -> hybrid

    FAQ similarity (when RAG_FAQ_SIMILARITY_ENABLED): overrides -> documents/hybrid.
    LLM gray-zone judgment (when RAG_LLM_GRAY_ZONE_ENABLED): zero-shot when distance in gray zone.

    Args:
        question: Rewritten question string.
        retrieved_docs: List of retrieved documents.
        general_prefixes: Prefixes that indicate general-knowledge intent.
        domain_signals: Matched domain keywords/phrases.
        project_root: Optional project root (for intent_keywords path resolution).
        distances: Optional list of L2 distances from retrieval.
        question_vector: Optional query embedding for FAQ similarity.
        faq_vectors: Optional pre-embedded FAQ vectors.

    Returns:
        AnswerMode: documents, general, or hybrid.
    """
    root = project_root or PROJECT_ROOT
    has_docs = len(retrieved_docs) > 0
    has_domain = len(domain_signals) > 0
    starts_general = _question_starts_with_prefix(question, general_prefixes)

    # Fast path: general prefix + no domain signals
    if starts_general and not has_domain:
        return "general"

    # FAQ similarity: if query matches FAQ, lean documents/hybrid
    faq_similarity_enabled = os.getenv("RAG_FAQ_SIMILARITY_ENABLED", "false").lower() in ("true", "1", "yes")
    if faq_similarity_enabled and question_vector and faq_vectors:
        faq_min_dist = _faq_similarity_score(question_vector, faq_vectors)
        faq_threshold = float(os.getenv("RAG_FAQ_SIMILARITY_THRESHOLD", "0.9"))
        if faq_min_dist < faq_threshold:
            return "hybrid"  # FAQ match -> document-related intent

    # Distance threshold when we have docs
    distance_threshold_enabled = os.getenv("RAG_DISTANCE_THRESHOLD_ENABLED", "true").lower() in ("true", "1", "yes")
    if has_docs and distance_threshold_enabled and distances:
        min_dist = min(distances) if distances else float("inf")
        threshold = float(os.getenv("RAG_MODE_DISTANCE_THRESHOLD_GENERAL", "1.0"))
        if min_dist > threshold:
            return "general"  # docs too far, not relevant

        # LLM gray-zone: distance near threshold, use zero-shot if enabled
        margin = float(os.getenv("RAG_DISTANCE_GRAY_ZONE_MARGIN", "0.2"))
        in_gray_zone = (threshold - margin) <= min_dist <= (threshold + margin)
        llm_gray_zone_enabled = os.getenv("RAG_LLM_GRAY_ZONE_ENABLED", "false").lower() in ("true", "1", "yes")
        if in_gray_zone and llm_gray_zone_enabled:
            from src.rag.intent_classifier import classify_intent

            model_name = os.getenv("RAG_INTENT_CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-mnli")
            llm_result = classify_intent(question, model_name)
            if llm_result in ("documents", "general"):
                return llm_result

        return "hybrid"  # min_dist <= threshold, clear or LLM fallback

    # has_docs without distance threshold check
    if has_docs:
        return "hybrid"
    if has_domain:
        return "documents"
    return "general"


def classify_answer_mode_parallel(
    embedder,
    vector_store,
    question: str,
    retrieval_k: int,
    faq_vectors: list,
    project_root: Path | None = None,
) -> tuple[AnswerMode, list, list]:
    """
    Parallel four-strategy intent classification.

    Runs Documents, Keywords, FAQ, LLM methods; aggregates Yes/No -> final mode.
    Embed and retrieve once; reuse for Documents + FAQ. Returns (effective_mode, retrieved_docs, question_vector).
    """
    from src.rag.intent_aggregator import aggregate_intent_signals
    from src.rag.intent_methods import faq_method_yes_no, keywords_method_yes_no, llm_method_yes_no

    root = project_root or PROJECT_ROOT
    question_vector = _step4_embed_question(embedder, question)
    retrieved_docs, distances = _step5_retrieve_docs(
        vector_store, question_vector, retrieval_k, include_distances=True
    )

    threshold = float(os.getenv("RAG_MODE_DISTANCE_THRESHOLD_GENERAL", "1.0"))
    min_dist = min(distances) if distances else float("inf")
    doc_yes = min_dist <= threshold

    kw_yes = keywords_method_yes_no(question, root)

    faq_enabled = os.getenv("RAG_FAQ_SIMILARITY_ENABLED", "false").lower() in ("true", "1", "yes")
    faq_yes = faq_method_yes_no(question_vector, faq_vectors) if (faq_enabled and faq_vectors) else False

    llm_enabled = os.getenv("RAG_LLM_GRAY_ZONE_ENABLED", "false").lower() in ("true", "1", "yes")
    llm_yes = llm_method_yes_no(question) if llm_enabled else False

    signals = [doc_yes, kw_yes, faq_yes, llm_yes]
    effective_mode = aggregate_intent_signals(signals)

    if effective_mode == "general" and retrieved_docs:
        retrieved_docs = []

    return effective_mode, retrieved_docs, question_vector


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
    """Step 3: Ollama + local model -> llm.

    Uses OLLAMA_REQUEST_TIMEOUT (default 120s) to avoid premature disconnects
    when the model takes long to generate. Increase for large models (e.g. 300s).
    """
    from langchain_ollama import OllamaLLM
    client_kwargs = {"timeout": OLLAMA_REQUEST_TIMEOUT}
    return OllamaLLM(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
        client_kwargs=client_kwargs,
    )


def _step4_embed_question(embedder, question: str) -> list:
    """Step 4: User question -> embedder.embed_query() -> question vector."""
    return embedder.embed_query(question)


def get_collection_count(vector_store) -> int:
    """Return number of chunks in the Chroma collection."""
    from ai_toolkit.chroma import get_chroma_collection
    collection = get_chroma_collection(vector_store)
    return collection.count()


def _step5_retrieve_docs(
    vector_store, question_vector: list, retrieval_k: int = RETRIEVAL_K, include_distances: bool = True
) -> tuple[list, list[float]]:
    """
    Step 5: question vector + vector store -> retrieve top-k docs.

    Includes distances for threshold-based classification when include_distances=True.
    Returns (docs, distances). Distances are L2 (lower = more similar).
    """
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents
    collection = get_chroma_collection(vector_store)
    include = ["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"]
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=retrieval_k,
        include=include,
    )
    docs_list = results["documents"][0]
    n = len(docs_list)
    ids = [f"doc_{i}" for i in range(n)]
    docs = chroma_to_documents(
        ids=ids,
        documents=docs_list,
        metadatas=results["metadatas"][0],
    )
    distances: list[float] = []
    if include_distances and "distances" in results and results["distances"]:
        distances = list(results["distances"][0]) if results["distances"][0] else []
    return docs, distances


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
    """Step 7: rag prompt + llm -> answer.

    Retries on transient errors (RemoteProtocolError, ConnectError) that
    often occur when Ollama is loading the model or under memory pressure.
    """
    import httpx

    max_retries = int(os.getenv("RAG_LLM_MAX_RETRIES", "2"))
    retry_delay = float(os.getenv("RAG_LLM_RETRY_DELAY", "5.0"))
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(rag_prompt) or ""
        except (httpx.RemoteProtocolError, httpx.ConnectError, ConnectionError) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))
                continue
            hint = (
                "Ollama may have disconnected (OOM, model loading, or server restart). "
                "Check: ollama list, ollama serve. Try RAG_LLM_TIMEOUT=300 or a smaller model."
            )
            raise RuntimeError(f"LLM failed after {max_retries + 1} attempts: {e}. {hint}") from last_exc

    return ""  # unreachable


class RAGPipeline:
    """
    Reusable RAG pipeline: build once (steps 1, 2, 3), query many times (steps 4, 5, 6, 7).
    """

    def __init__(
        self,
        embedder,
        vector_store,
        llm,
        retrieval_k: int = RETRIEVAL_K,
        faq_questions: list | None = None,
        faq_vectors: list | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.retrieval_k = retrieval_k
        self.faq_questions = faq_questions or []
        self.faq_vectors = faq_vectors or []

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

        # Load and embed FAQ questions if enabled
        faq_questions: list = []
        faq_vectors: list = []
        faq_similarity_enabled = os.getenv("RAG_FAQ_SIMILARITY_ENABLED", "false").lower() in ("true", "1", "yes")
        if faq_similarity_enabled:
            from src.rag.faq_loader import load_faq_questions

            faq_questions = load_faq_questions(root)
            if faq_questions:
                t0 = time.perf_counter()
                faq_vectors = embedder.embed_documents(faq_questions)
                if verbose:
                    print(f"  [Build] FAQ: {len(faq_questions)} questions embedded ({time.perf_counter() - t0:.2f}s)")

        if verbose:
            coll_count = get_collection_count(vector_store)
            print(f"  [Build] Total setup: {time.perf_counter() - total_start:.2f}s")
            print(f"  [Build] Chroma: {chroma_path} | collection={collection_name} | chunks={coll_count}\n")

        return cls(embedder, vector_store, llm, retrieval_k, faq_questions=faq_questions, faq_vectors=faq_vectors)

    def query(
        self, question: str, mode: QueryMode = "hybrid", verbose: bool = True
    ) -> tuple[str, list, AnswerMode]:
        """
        Run query (steps 4, 5, 6, 7). Mode: documents, general, hybrid, or auto.

        When mode is "auto", sequential classifier selects documents/general/hybrid
        based on rewrite, keywords, retrieval count, and distance threshold.
        Returns (answer, docs, selected_mode).
        """
        from src.rag.intent_keywords import get_general_prefixes, match_domain_signals

        total_start = time.perf_counter()
        retrieved_docs: list = []
        root = PROJECT_ROOT

        # Apply lightweight query rewrite when enabled (env-controlled)
        effective_question = rewrite_query_lightweight(question)

        # Resolve mode: "auto" -> classify; else use explicit mode
        effective_mode: AnswerMode
        if mode == "auto":
            auto_enabled = os.getenv("RAG_AUTO_MODE_ENABLED", "true").lower() in ("true", "1", "yes")
            if not auto_enabled:
                effective_mode = "hybrid"
            else:
                use_parallel = os.getenv("RAG_USE_PARALLEL_INTENT", "true").lower() in ("true", "1", "yes")
                if use_parallel:
                    t0 = time.perf_counter()
                    effective_mode, retrieved_docs, _ = classify_answer_mode_parallel(
                        self.embedder,
                        self.vector_store,
                        effective_question,
                        self.retrieval_k,
                        self.faq_vectors,
                        root,
                    )
                    if verbose:
                        print(f"  Step 4+5 (embed + retrieve + parallel classify): {time.perf_counter() - t0:.2f}s")
                        print(f"  [Auto] Classified as {effective_mode} (parallel four-strategy)")
                else:
                    general_prefixes = get_general_prefixes()
                    domain_signals = match_domain_signals(effective_question, root)
                    if _question_starts_with_prefix(effective_question, general_prefixes) and not domain_signals:
                        effective_mode = "general"
                        if verbose:
                            print(f"  [Auto] Classified as general (prefix + no domain signals)")
                    else:
                        t0 = time.perf_counter()
                        question_vector = _step4_embed_question(self.embedder, effective_question)
                        if verbose:
                            print(f"  Step 4 (embed question): {time.perf_counter() - t0:.2f}s")
                        t0 = time.perf_counter()
                        retrieved_docs, distances = _step5_retrieve_docs(
                            self.vector_store, question_vector, self.retrieval_k, include_distances=True
                        )
                        if verbose:
                            print(f"  Step 5 (retrieve docs): {time.perf_counter() - t0:.2f}s")
                        effective_mode = classify_answer_mode_sequential(
                            effective_question,
                            retrieved_docs,
                            general_prefixes,
                            domain_signals,
                            root,
                            distances=distances,
                            question_vector=question_vector,
                            faq_vectors=self.faq_vectors,
                        )
                        if effective_mode == "general" and len(retrieved_docs) > 0:
                            retrieved_docs = []
                        if verbose:
                            print(f"  [Auto] Classified as {effective_mode} (docs={len(retrieved_docs)}, signals={len(domain_signals)})")
        else:
            effective_mode = mode

        # Now run the standard flow with effective_mode
        if effective_mode == "general":
            rag_prompt = _step6_build_rag_prompt([], effective_question, mode="general")
            if verbose and mode != "auto":
                print(f"  Step 4 (embed question): skipped (general knowledge mode)")
                print(f"  Step 5 (retrieve docs): skipped (general knowledge mode)")
        else:
            if mode != "auto" or len(retrieved_docs) == 0:
                # For explicit mode or auto: need to embed/retrieve if we don't have docs yet
                if len(retrieved_docs) == 0:
                    t0 = time.perf_counter()
                    question_vector = _step4_embed_question(self.embedder, effective_question)
                    if verbose:
                        print(f"  Step 4 (embed question): {time.perf_counter() - t0:.2f}s")
                    t0 = time.perf_counter()
                    retrieved_docs, _ = _step5_retrieve_docs(
                        self.vector_store, question_vector, self.retrieval_k, include_distances=False
                    )
                    if verbose:
                        print(f"  Step 5 (retrieve docs): {time.perf_counter() - t0:.2f}s")

            if effective_mode == "documents" and len(retrieved_docs) == 0:
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
                    effective_mode,
                )

            t0 = time.perf_counter()
            rag_prompt = _step6_build_rag_prompt(retrieved_docs, effective_question, mode=effective_mode)
            if verbose:
                print(f"  Step 6 (build RAG prompt): {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        answer = _step7_generate_answer(self.llm, rag_prompt)
        if verbose:
            print(f"  Step 7 (LLM generate): {time.perf_counter() - t0:.2f}s")

        if verbose:
            print(f"  Query total: {time.perf_counter() - total_start:.2f}s\n")

        return answer, retrieved_docs, effective_mode


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
