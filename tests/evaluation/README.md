# RAG Evaluation Tests

Unit and integration tests for the RAG evaluation framework.

## How to Run Evaluation

### Prerequisites

- `amazon_fqa.csv` at `data/intent_classification/fqa/amazon_fqa.csv` or `RAG_FAQ_CSV`
- Chroma populated: `python scripts/load_to_chroma.py documents`
- LLM API keys (for generation eval): `DEEPSEEK_API_KEY` or `QWEN_API_KEY`

### Full Evaluation (Single Command)

```bash
# From project root
python scripts/run_evaluation.py --limit 10

# Quick run without UMAP
python scripts/run_evaluation.py --limit 5 --skip-umap

# Custom output directory
python scripts/run_evaluation.py --limit 10 --output ./eval_results
```

### Output

Results are written to `tests/evaluation/results_{timestamp}/` (or `--output` path):

- `retrieval_results.json` - Recall@5, Precision@5, MRR per case
- `generation_results.json` - Faithfulness and relevance scores
- `umap_visualization.html` - Embedding space plot (unless `--skip-umap`)
- `evaluation_report.html` - Full HTML report with issue list

## Running Tests

```bash
# All evaluation tests
pytest tests/evaluation/ -v

# Unit tests only (fast, no Chroma/LLM)
pytest tests/evaluation/test_metrics.py tests/evaluation/test_dataset_loader.py -v

# Integration tests (requires Chroma, dataset, optional LLM)
pytest tests/evaluation/test_end_to_end.py tests/evaluation/test_run_evaluation.py -v
```

Integration tests may skip if dataset, Chroma, or LLM are unavailable.
