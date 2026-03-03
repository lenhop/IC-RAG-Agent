"""
RAGAS Integration for RAG Evaluation (Phase 3.2).

Wrapper for RAGAS library metrics: context_precision, answer_relevancy, faithfulness.
Provides correlation analysis with custom LLM-as-Judge metrics.
"""

from typing import Any, Dict, List, Optional
import warnings


def evaluate_with_ragas(
    test_cases: List[Dict[str, Any]],
    pipeline: Any,
    mode: str = "hybrid",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate test cases using RAGAS metrics.

    Args:
        test_cases: List of test case dicts with question, ground_truth, optional contexts.
        pipeline: RAGPipeline with LLM for generation.
        mode: Answer mode (documents, general, hybrid, auto).
        verbose: Print progress.

    Returns:
        Dict with keys:
            - context_precision: float (0-1)
            - answer_relevancy: float (0-1)
            - faithfulness: float (0-1)
            - per_case: List of per-case results
            - total_count: int
    """
    # Check for empty cases before importing RAGAS
    if not test_cases:
        return {
            "context_precision": 0.0,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "per_case": [],
            "total_count": 0,
        }

    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            answer_relevancy,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "RAGAS not installed. Run: pip install ragas datasets"
        ) from e

    # Prepare RAGAS dataset format
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for case in test_cases:
        question = case.get("question", "")
        if not question:
            continue

        # Generate answer using pipeline
        try:
            result = pipeline.query(question, mode=mode)
            answer = result.get("answer", "")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to generate answer for {case.get('id', '?')}: {e}")
            continue

        # Get contexts (retrieved chunks)
        contexts = case.get("contexts", [])
        if not contexts:
            # Fallback: retrieve from pipeline
            try:
                from ai_toolkit.chroma import get_chroma_collection, query_collection
                collection = get_chroma_collection(pipeline.vector_store)
                q_vec = pipeline.embedder.embed_query(question)
                res = query_collection(
                    collection,
                    query_embeddings=[q_vec],
                    n_results=5,
                    include=["documents"],
                )
                contexts = res.get("documents", [[]])[0] or []
            except Exception:
                contexts = []

        if not isinstance(contexts, list):
            contexts = [contexts]

        ground_truth = case.get("ground_truth", "")

        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(ground_truth)

    if not ragas_data["question"]:
        return {
            "context_precision": 0.0,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "per_case": [],
            "total_count": 0,
        }

    # Create RAGAS dataset
    dataset = Dataset.from_dict(ragas_data)

    # Run RAGAS evaluation
    if verbose:
        print(f"[RAGAS] Evaluating {len(ragas_data['question'])} cases...")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(
                dataset,
                metrics=[context_precision, answer_relevancy, faithfulness],
            )
    except Exception as e:
        if verbose:
            print(f"[ERROR] RAGAS evaluation failed: {e}")
        return {
            "context_precision": 0.0,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "per_case": [],
            "total_count": 0,
            "error": str(e),
        }

    # Extract metrics
    scores = result.to_pandas()
    avg_context_precision = scores["context_precision"].mean()
    avg_answer_relevancy = scores["answer_relevancy"].mean()
    avg_faithfulness = scores["faithfulness"].mean()

    # Build per-case results
    per_case = []
    for i, case in enumerate(test_cases[:len(scores)]):
        per_case.append({
            "id": case.get("id", f"case_{i}"),
            "question": case.get("question", ""),
            "context_precision": float(scores.iloc[i]["context_precision"]),
            "answer_relevancy": float(scores.iloc[i]["answer_relevancy"]),
            "faithfulness": float(scores.iloc[i]["faithfulness"]),
        })

    return {
        "context_precision": float(avg_context_precision),
        "answer_relevancy": float(avg_answer_relevancy),
        "faithfulness": float(avg_faithfulness),
        "per_case": per_case,
        "total_count": len(per_case),
    }


def compare_metrics(
    custom_results: Dict[str, Any],
    ragas_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare custom LLM-as-Judge metrics with RAGAS metrics.

    Args:
        custom_results: Output from GenerationEvaluator.evaluate_batch.
        ragas_results: Output from evaluate_with_ragas.

    Returns:
        Dict with keys:
            - faithfulness_correlation: Pearson coefficient
            - relevance_correlation: Pearson coefficient (relevance vs answer_relevancy)
            - discrepancies: List of cases with large differences
            - recommendation: str (which metric to trust)
    """
    try:
        import numpy as np
        from scipy.stats import pearsonr
    except ImportError as e:
        raise ImportError(
            "scipy required for correlation analysis. Run: pip install scipy"
        ) from e

    custom_per_case = custom_results.get("per_case", [])
    ragas_per_case = ragas_results.get("per_case", [])

    if not custom_per_case or not ragas_per_case:
        return {
            "faithfulness_correlation": 0.0,
            "relevance_correlation": 0.0,
            "discrepancies": [],
            "recommendation": "Insufficient data for comparison",
        }

    # Align cases by ID
    custom_map = {c["id"]: c for c in custom_per_case}
    ragas_map = {r["id"]: r for r in ragas_per_case}
    common_ids = set(custom_map.keys()) & set(ragas_map.keys())

    if len(common_ids) < 2:
        return {
            "faithfulness_correlation": 0.0,
            "relevance_correlation": 0.0,
            "discrepancies": [],
            "recommendation": "Too few common cases for correlation",
        }

    # Extract faithfulness scores
    custom_faith = []
    ragas_faith = []
    for case_id in common_ids:
        custom_faith.append(1.0 if custom_map[case_id].get("is_faithful", False) else 0.0)
        ragas_faith.append(ragas_map[case_id].get("faithfulness", 0.0))

    # Extract relevance scores (normalize custom to 0-1)
    custom_rel = []
    ragas_rel = []
    for case_id in common_ids:
        custom_rel.append(custom_map[case_id].get("relevance_score", 0) / 5.0)
        ragas_rel.append(ragas_map[case_id].get("answer_relevancy", 0.0))

    # Calculate correlations
    faith_corr, faith_p = pearsonr(custom_faith, ragas_faith)
    rel_corr, rel_p = pearsonr(custom_rel, ragas_rel)

    # Identify discrepancies (|custom - ragas| > 0.3)
    discrepancies = []
    for case_id in common_ids:
        custom_f = 1.0 if custom_map[case_id].get("is_faithful", False) else 0.0
        ragas_f = ragas_map[case_id].get("faithfulness", 0.0)
        custom_r = custom_map[case_id].get("relevance_score", 0) / 5.0
        ragas_r = ragas_map[case_id].get("answer_relevancy", 0.0)

        faith_diff = abs(custom_f - ragas_f)
        rel_diff = abs(custom_r - ragas_r)

        if faith_diff > 0.3 or rel_diff > 0.3:
            discrepancies.append({
                "id": case_id,
                "question": custom_map[case_id].get("question", "")[:80],
                "faithfulness_diff": faith_diff,
                "relevance_diff": rel_diff,
                "custom_faithful": custom_f,
                "ragas_faithful": ragas_f,
                "custom_relevance": custom_r,
                "ragas_relevance": ragas_r,
            })

    # Recommendation
    if faith_corr > 0.7 and rel_corr > 0.7:
        recommendation = "High correlation - both metrics reliable"
    elif faith_corr > 0.5 and rel_corr > 0.5:
        recommendation = "Moderate correlation - review discrepancies manually"
    else:
        recommendation = "Low correlation - metrics disagree significantly, manual review required"

    return {
        "faithfulness_correlation": float(faith_corr),
        "faithfulness_p_value": float(faith_p),
        "relevance_correlation": float(rel_corr),
        "relevance_p_value": float(rel_p),
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "total_compared": len(common_ids),
        "recommendation": recommendation,
    }
