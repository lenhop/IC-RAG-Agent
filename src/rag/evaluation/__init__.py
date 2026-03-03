"""
RAG Evaluation Framework

Implements retrieval and generation metrics per RAG_EVALUATION_IMPLEMENTATION_PLAN.md.
"""

from .dataset_loader import (
    add_relevant_contexts,
    load_fqa_dataset,
    validate_dataset,
)
from .generation_metrics import (
    evaluate_faithfulness,
    evaluate_relevance,
    GenerationEvaluator,
)
from .retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    RetrievalEvaluator,
)
from .visualize_umap import generate_umap_plot
from .report_generator import generate_html_report
from .ragas_metrics import (
    evaluate_with_ragas,
    compare_metrics,
)

__all__ = [
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_mrr",
    "RetrievalEvaluator",
    "evaluate_faithfulness",
    "evaluate_relevance",
    "GenerationEvaluator",
    "load_fqa_dataset",
    "validate_dataset",
    "add_relevant_contexts",
    "generate_umap_plot",
    "generate_html_report",
    "evaluate_with_ragas",
    "compare_metrics",
]
