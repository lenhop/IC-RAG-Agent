#!/usr/bin/env python3
"""
End-to-End RAG Evaluation Script (Phase 3.1).

Single command runs full evaluation: load dataset, retrieval metrics,
generation metrics, UMAP visualization, HTML report.

Prerequisites:
  - amazon_fqa.csv at RAG_FAQ_CSV or data/intent_classification/fqa/
  - Chroma populated: python scripts/load_to_chroma/load_documents_to_chroma.py
  - LLM API (for generation eval): DEEPSEEK_API_KEY or QWEN_API_KEY

Usage:
  python scripts/run_evaluation.py --limit 10
  python scripts/run_evaluation.py --limit 5 --skip-umap
  python scripts/run_evaluation.py --mode documents --limit 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths from RAG_FAQ_CSV
DEFAULT_DATASET = str(PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv")


def main() -> int:
    """Run full evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run full RAG evaluation: retrieval, generation, UMAP, report"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Path to CSV (default: RAG_FAQ_CSV or {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of test cases (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=("documents", "general", "hybrid", "auto"),
        default="hybrid",
        help="Answer mode for generation eval (default: hybrid)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: tests/evaluation/results_{timestamp})",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP visualization (faster)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-case results (default: True)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.no_verbose

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "tests" / "evaluation" / f"results_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset or "RAG_FAQ_CSV/default",
        "limit": args.limit,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "skip_umap": args.skip_umap,
    }

    try:
        from src.rag.evaluation.dataset_loader import load_fqa_dataset, validate_dataset
        from src.rag.evaluation.retrieval_metrics import RetrievalEvaluator
        from src.rag.evaluation.generation_metrics import GenerationEvaluator
        from src.rag.evaluation.visualize_umap import generate_umap_plot
        from src.rag.evaluation.report_generator import generate_html_report
        from src.rag.query_pipeline import RAGPipeline
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}", file=sys.stderr)
        return 1

    # Step 1: Load dataset
    if verbose:
        print("[1/6] Loading dataset...")
    try:
        test_cases = load_fqa_dataset(
            csv_path=args.dataset,
            limit=args.limit,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if not validate_dataset(test_cases, warn_missing=verbose):
        print("[ERROR] Dataset validation failed", file=sys.stderr)
        return 1

    if verbose:
        print(f"  Loaded {len(test_cases)} test cases\n")

    # Step 2: Build RAG pipeline
    if verbose:
        print("[2/6] Building RAG pipeline...")
    try:
        pipeline = RAGPipeline.build(verbose=False)
    except Exception as e:
        print(f"[ERROR] Pipeline build failed: {e}", file=sys.stderr)
        return 1

    if verbose:
        print("  Pipeline ready\n")

    # Step 3: Retrieval evaluation
    if verbose:
        print("[3/6] Running retrieval evaluation...")
    retrieval_results = {}
    try:
        retrieval_evaluator = RetrievalEvaluator()
        retrieval_results = retrieval_evaluator.evaluate_batch(
            pipeline, test_cases, k=5, verbose=verbose
        )
    except Exception as e:
        import traceback
        print(f"[WARN] Retrieval eval failed: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        retrieval_results = {"avg_recall": 0, "avg_precision": 0, "avg_mrr": 0, "per_case": []}

    retrieval_path = output_dir / "retrieval_results.json"
    with open(retrieval_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  Saved to {retrieval_path}\n")

    # Step 4: Generation evaluation
    if verbose:
        print("[4/6] Running generation evaluation (LLM-as-Judge)...")
    try:
        generation_evaluator = GenerationEvaluator()
        generation_results = generation_evaluator.evaluate_batch(
            pipeline,
            test_cases,
            mode=args.mode,
            verbose=verbose,
            progress=verbose,
        )
    except Exception as e:
        print(f"[WARN] Generation eval failed: {e}", file=sys.stderr)
        generation_results = {
            "faithfulness_rate": 0,
            "faithful_count": 0,
            "total_count": 0,
            "avg_relevance_score": 0,
            "per_case": [],
        }

    generation_path = output_dir / "generation_results.json"
    with open(generation_path, "w", encoding="utf-8") as f:
        json.dump(generation_results, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  Saved to {generation_path}\n")

    # Step 5: UMAP visualization (optional)
    umap_path = None
    if not args.skip_umap:
        if verbose:
            print("[5/6] Generating UMAP visualization...")
        try:
            umap_out = output_dir / "umap_visualization.html"
            generate_umap_plot(
                test_cases,
                pipeline,
                output_path=umap_out,
            )
            umap_path = umap_out
            if verbose:
                print(f"  Saved to {umap_path}\n")
        except Exception as e:
            print(f"[WARN] UMAP failed: {e}", file=sys.stderr)
    else:
        if verbose:
            print("[5/6] Skipping UMAP (--skip-umap)\n")

    # Step 6: HTML report
    if verbose:
        print("[6/6] Generating HTML report...")
    report_path = output_dir / "evaluation_report.html"
    generate_html_report(
        retrieval_results=retrieval_results,
        generation_results=generation_results,
        umap_path=umap_path,
        output_path=report_path,
        config=config,
    )
    if verbose:
        print(f"  Saved to {report_path}\n")

    # Console summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - retrieval_results.json")
    print(f"  - generation_results.json")
    if umap_path:
        print(f"  - umap_visualization.html")
    print(f"  - evaluation_report.html")
    print()

    avg_recall = retrieval_results.get("avg_recall", 0)
    avg_precision = retrieval_results.get("avg_precision", 0)
    avg_mrr = retrieval_results.get("avg_mrr", 0)
    print("Retrieval:")
    print(f"  Recall@5: {avg_recall:.2f} | Precision@5: {avg_precision:.2f} | MRR: {avg_mrr:.2f}")

    if generation_results:
        faith = generation_results.get("faithfulness_rate", 0)
        rel = generation_results.get("avg_relevance_score", 0)
        n = generation_results.get("total_count", 0)
        fc = generation_results.get("faithful_count", 0)
        print("Generation:")
        print(f"  Faithfulness: {fc}/{n} ({faith:.1%})")
        print(f"  Avg Relevance: {rel:.2f}/5")
        print()
        retrieval_ok = avg_recall >= 0.4
        gen_ok = faith >= 0.85 and rel >= 4.0
        if retrieval_ok and gen_ok:
            print("Overall: PASS")
        else:
            print("Overall: FAIL")
            if not retrieval_ok:
                print("  - Retrieval below target (Recall@5 >= 0.4)")
            if not gen_ok:
                if faith < 0.85:
                    print("  - Faithfulness below target (>= 85%)")
                if rel < 4.0:
                    print("  - Relevance below target (>= 4.0/5)")
    else:
        print("Generation: (skipped or failed)")
        print("Overall: (incomplete)")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
