# Rewriting Evaluation Toolkit

This folder contains utilities for rewrite quality evaluation.

## Pipeline

`run_rewrite_eval.py` performs:

1. Bulk rewrite requests from input CSV (`query` column required)
2. Embedding-based similarity/distance calculation between:
   - original `query`
   - `rewritten_query`
3. Final CSV generation with rewrite and metric fields

## Usage

```bash
/opt/miniconda3/bin/python tools/rewriting/run_rewrite_eval.py \
  --input tests/data/test_queries.csv \
  --output tests/data/test_queries_eval.csv \
  --endpoint http://127.0.0.1:8000/api/v1/rewrite \
  --rewrite-backend ollama \
  --embed-model models/all-MiniLM-L6-v2
```

## Output fields

- Rewrite fields:
  - `rewritten_query`
  - `rewrite_time_ms`
  - `rewrite_backend_used`
  - `rewrite_status` (`ok|skipped|error`)
  - `rewrite_error`
- Distance fields:
  - `cosine_similarity`
  - `cosine_distance`
  - `distance_flag` (`ok|too_similar|too_different|skipped|error`)

## Defaults

- `too_similar` threshold: `0.98`
- `too_different` threshold: `0.75`
- blank or whitespace-only query: skipped

## Notes

- Requires `sentence-transformers` to compute embeddings.
- `embed-model` can be a local path or model name.

