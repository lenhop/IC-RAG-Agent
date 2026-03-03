"""
Retrieval Layer Evaluation - Integration Test

Uses src/rag/evaluation/retrieval_metrics and dataset_loader.
Runs against real RAG pipeline and amazon_fqa.csv.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.evaluation.dataset_loader import load_fqa_dataset, validate_dataset
from src.rag.evaluation.retrieval_metrics import RetrievalEvaluator
from src.rag.query_pipeline import RAGPipeline


if __name__ == "__main__":
    try:
        test_cases = load_fqa_dataset(limit=10, project_root=PROJECT_ROOT)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    validate_dataset(test_cases)
    print(f"Loaded {len(test_cases)} test cases from amazon_fqa.csv\n")

    print("Building RAG pipeline...")
    pipeline = RAGPipeline.build(verbose=False)

    print("\nEvaluating retrieval metrics...")
    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate_batch(pipeline, test_cases, k=5, verbose=True)

    output_path = PROJECT_ROOT / "tests" / "evaluation" / "retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
