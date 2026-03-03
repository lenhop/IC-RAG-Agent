"""
Retrieval Layer Evaluation Metrics

Implements Recall@5, Precision@5, and MRR per RAG_EVALUATION_IMPLEMENTATION_PLAN.md.
Uses ai_toolkit.chroma public API (get_chroma_collection, query_collection).
"""

from typing import Any, Dict, List

from langchain_core.documents import Document


def calculate_recall_at_k(
    retrieved_docs: List[Document],
    relevant_contexts: List[str],
    k: int = 5,
) -> float:
    """Calculate Recall@K: (relevant chunks in top-k) / (total relevant chunks).

    Args:
        retrieved_docs: List of retrieved documents from RAG pipeline.
        relevant_contexts: List of relevant chunk IDs or content snippets.
        k: Top-k documents to consider (default: 5).

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_contexts:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    retrieved_content = [doc.page_content for doc in top_k_docs]

    # Count how many relevant contexts appear in top-k
    relevant_found = 0
    for context in relevant_contexts:
        if any(context.lower() in content.lower() for content in retrieved_content):
            relevant_found += 1

    return relevant_found / len(relevant_contexts)


def calculate_precision_at_k(
    retrieved_docs: List[Document],
    relevant_contexts: List[str],
    k: int = 5,
) -> float:
    """Calculate Precision@K: (relevant chunks in top-k) / k.

    Args:
        retrieved_docs: List of retrieved documents from RAG pipeline.
        relevant_contexts: List of relevant chunk IDs or content snippets.
        k: Top-k documents to consider (default: 5).

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if k == 0:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    retrieved_content = [doc.page_content for doc in top_k_docs]

    # Count how many top-k docs are relevant
    relevant_found = 0
    for content in retrieved_content:
        if any(context.lower() in content.lower() for context in relevant_contexts):
            relevant_found += 1

    return relevant_found / k


def calculate_mrr(
    retrieved_docs: List[Document],
    relevant_contexts: List[str],
) -> float:
    """Calculate Mean Reciprocal Rank: 1 / rank of first relevant document.

    Args:
        retrieved_docs: List of retrieved documents from RAG pipeline.
        relevant_contexts: List of relevant chunk IDs or content snippets.

    Returns:
        MRR score (0.0 to 1.0).
    """
    if not relevant_contexts:
        return 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        if any(context.lower() in doc.page_content.lower() for context in relevant_contexts):
            return 1.0 / rank

    return 0.0


def _get_relevant_contexts(case: Dict[str, Any], use_ground_truth_fallback: bool = True) -> List[str]:
    """Extract relevant contexts from test case.

    Prefer 'contexts' field (ideal retrieval chunks). Fallback to ground_truth
    prefix when contexts missing (weak proxy; manual annotation preferred).

    Args:
        case: Test case dict with 'contexts' or 'ground_truth'.
        use_ground_truth_fallback: If True, use ground_truth[:100] when contexts missing.

    Returns:
        List of context strings for metric calculation.
    """
    if "contexts" in case and case["contexts"]:
        ctx = case["contexts"]
        return ctx if isinstance(ctx, list) else [ctx]
    if use_ground_truth_fallback and case.get("ground_truth"):
        return [str(case["ground_truth"])[:100]]
    return []


def _retrieve_docs(pipeline: Any, question: str, k: int) -> tuple[List[Document], List[float]]:
    """Retrieve top-k docs using public Chroma API.

    Uses get_chroma_collection + query_collection (no private _collection).

    Args:
        pipeline: RAGPipeline with embedder and vector_store.
        question: User question.
        k: Number of documents to retrieve.

    Returns:
        (docs, distances) tuple.
    """
    from ai_toolkit.chroma import get_chroma_collection, query_collection, chroma_to_documents

    collection = get_chroma_collection(pipeline.vector_store)
    question_vector = pipeline.embedder.embed_query(question)
    results = query_collection(
        collection,
        query_embeddings=[question_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = results.get("documents", [[]])[0] or []
    metadatas_list = results.get("metadatas", [[]])[0] or []
    distances_list = results.get("distances", [[]])[0] or []

    n = len(docs_list)
    ids = [f"doc_{i}" for i in range(n)]
    docs = chroma_to_documents(ids=ids, documents=docs_list, metadatas=metadatas_list)
    distances = list(distances_list)

    return docs, distances


class RetrievalEvaluator:
    """Evaluator for retrieval metrics (Recall@K, Precision@K, MRR)."""

    def evaluate_batch(
        self,
        pipeline: Any,
        test_cases: List[Dict[str, Any]],
        k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate retrieval metrics on a batch of test cases.

        Args:
            pipeline: RAGPipeline instance (must have embedder, vector_store).
            test_cases: List of dicts with 'question', 'ground_truth', optional 'contexts'.
            k: Top-k for Recall@K and Precision@K.
            verbose: Print per-case and aggregate results.

        Returns:
            Dict with avg_recall, avg_precision, avg_mrr, per_case list, and raw lists.
        """
        results: Dict[str, Any] = {
            "recall_at_k": [],
            "precision_at_k": [],
            "mrr": [],
            "per_case": [],
        }

        for case in test_cases:
            question = case.get("question", "")
            case_id = case.get("id", "unknown")

            # Retrieve via public API
            docs, distances = _retrieve_docs(pipeline, question, k)

            # Get relevant contexts (prefer contexts, fallback to ground_truth)
            relevant_contexts = _get_relevant_contexts(case)
            if not relevant_contexts and verbose:
                print(f"  [Warn] {case_id}: No contexts or ground_truth; metrics will be 0")

            # Calculate metrics
            recall = calculate_recall_at_k(docs, relevant_contexts, k)
            precision = calculate_precision_at_k(docs, relevant_contexts, k)
            mrr = calculate_mrr(docs, relevant_contexts)

            results["recall_at_k"].append(recall)
            results["precision_at_k"].append(precision)
            results["mrr"].append(mrr)

            min_dist = min(distances) if distances else float("inf")
            results["per_case"].append({
                "id": case_id,
                "question": question[:80] + "..." if len(question) > 80 else question,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "min_distance": min_dist,
            })

            if verbose:
                print(f"\n{case_id}: {question[:60]}...")
                print(f"  Recall@{k}: {recall:.2f} | Precision@{k}: {precision:.2f} | MRR: {mrr:.2f}")
                print(f"  Min distance: {min_dist:.4f}")

        # Aggregate
        n = len(results["recall_at_k"])
        results["avg_recall"] = sum(results["recall_at_k"]) / n if n else 0.0
        results["avg_precision"] = sum(results["precision_at_k"]) / n if n else 0.0
        results["avg_mrr"] = sum(results["mrr"]) / n if n else 0.0

        if verbose:
            print("\n" + "=" * 60)
            print(f"Average Recall@{k}: {results['avg_recall']:.2f}")
            print(f"Average Precision@{k}: {results['avg_precision']:.2f}")
            print(f"Average MRR: {results['avg_mrr']:.2f}")
            print("=" * 60)

        return results
