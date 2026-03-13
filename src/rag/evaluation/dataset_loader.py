"""
Evaluation Dataset Loader

Loads and validates test cases from FAQ CSV for RAG evaluation.
Per RAG_EVALUATION_IMPLEMENTATION_PLAN.md Phase 1.3.

Note: This is for evaluation (in-memory test cases), NOT for Chroma ingestion.
- scripts/load_to_chroma/load_documents_to_chroma.py: documents -> Chroma (RAG)
- scripts/load_to_chroma/load_intent_registry_to_chroma.py: intent registry CSV -> Chroma
- dataset_loader: CSV -> List[Dict] for retrieval/generation metrics
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path aligned with RAG_FAQ_CSV
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = os.getenv(
    "RAG_FAQ_CSV",
    str(PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv"),
)


def _resolve_csv_path(csv_path: str, project_root: Optional[Path] = None) -> Path:
    """Resolve CSV path: if relative, join with project root."""
    p = Path(csv_path)
    if not p.is_absolute():
        root = project_root or PROJECT_ROOT
        p = root / p
    return p.resolve()


def load_fqa_dataset(
    csv_path: Optional[str] = None,
    limit: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load evaluation test cases from FAQ CSV.

    Args:
        csv_path: Path to CSV. Default: RAG_FAQ_CSV or data/.../amazon_fqa.csv.
        limit: Max number of rows (None = all).
        project_root: For resolving relative paths.

    Returns:
        List of dicts: id, question, ground_truth, category, source, contexts (if present).
    """
    path = _resolve_csv_path(csv_path or DEFAULT_CSV_PATH, project_root)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    test_cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break

            # Map CSV columns to evaluation format
            ground_truth = row.get("answer") or row.get("ground_truth") or ""
            case: Dict[str, Any] = {
                "id": f"faq_{i+1:03d}",
                "question": (row.get("question") or "").strip(),
                "ground_truth": ground_truth.strip(),
                "category": row.get("category", "unknown"),
                "source": row.get("source", "Amazon SellerCentral"),
            }

            # Optional: contexts column (ideal retrieval chunks for Recall/Precision/MRR)
            ctx = row.get("contexts")
            if ctx:
                # Support comma-separated or pipe-separated
                case["contexts"] = [s.strip() for s in ctx.replace("|", ",").split(",") if s.strip()]

            test_cases.append(case)

    if not test_cases:
        warnings.warn("[WARN] No test cases loaded from CSV")

    return test_cases


def validate_dataset(
    test_cases: List[Dict[str, Any]],
    warn_missing: bool = True,
) -> bool:
    """Validate test cases have required fields.

    Required: question, ground_truth.
    Optional: contexts, category, source, id. Missing contexts triggers warning only.

    Args:
        test_cases: List of test case dicts.
        warn_missing: If True, emit warnings for missing contexts.

    Returns:
        True if all cases have question and ground_truth; False otherwise.
    """
    if not test_cases:
        if warn_missing:
            warnings.warn("[ERROR] Empty test dataset")
        return False

    valid = True
    missing_contexts_count = 0

    for case in test_cases:
        case_id = case.get("id", "?")
        if not (case.get("question") or "").strip():
            if warn_missing:
                warnings.warn(f"[ERROR] {case_id}: Missing 'question' field")
            valid = False
        if not (case.get("ground_truth") or "").strip():
            if warn_missing:
                warnings.warn(f"[ERROR] {case_id}: Missing 'ground_truth' field")
            valid = False
        if "contexts" not in case or not case["contexts"]:
            missing_contexts_count += 1
            if warn_missing:
                warnings.warn(
                    f"[WARN] {case_id}: Missing 'contexts' - will use ground_truth fallback"
                )

    if warn_missing and missing_contexts_count == len(test_cases):
        warnings.warn(
            "No cases have 'contexts'; retrieval metrics will use ground_truth as fallback "
            "(weak proxy). Manual contexts preferred for reliable Recall/Precision/MRR."
        )

    return valid


def add_relevant_contexts(
    test_cases: List[Dict[str, Any]],
    pipeline: Any,
    k: int = 5,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    """Add retrieved contexts when missing (fallback for retrieval evaluation).

    Uses pipeline to retrieve top-k docs per question. If case already has
    'contexts', skip unless overwrite=True. Manual contexts are preferred.
    Auto-retrieved contexts are less reliable than manual annotation.

    Args:
        test_cases: List of test case dicts (modified in place if overwrite).
        pipeline: RAGPipeline with embedder and vector_store.
        k: Number of docs to retrieve per question.
        overwrite: If True, replace existing contexts.

    Returns:
        Same test_cases list (modified in place).
    """
    from ai_toolkit.chroma import get_chroma_collection, query_collection

    print("[WARN] Auto-retrieved contexts are less reliable than manual annotation")

    collection = get_chroma_collection(pipeline.vector_store)
    auto_retrieved_count = 0

    for case in test_cases:
        if "contexts" in case and case["contexts"] and not overwrite:
            continue

        question = case.get("question", "")
        if not question:
            continue

        try:
            question_vector = pipeline.embedder.embed_query(question)
            results = query_collection(
                collection,
                query_embeddings=[question_vector],
                n_results=k,
                include=["documents"],
            )
            docs_list = results.get("documents", [[]])[0] or []
            case["contexts"] = [d for d in docs_list if d]
            case["contexts_source"] = "auto_retrieved"
            auto_retrieved_count += 1
        except Exception:
            case["contexts"] = []
            case["contexts_source"] = "auto_retrieved"
            auto_retrieved_count += 1

    print(f"[INFO] Auto-retrieved contexts for {auto_retrieved_count} cases")
    return test_cases
