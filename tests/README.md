# RAG API Tests and Examples

## Prerequisites

1. Start the RAG API server:
   ```bash
   ./scripts/run_rag_api.sh
   ```
2. Ensure Chroma DB is loaded (run `python scripts/load_documents_to_chroma.py` if needed).

## Shell Examples (curl)

Run the example script to see API usage:

```bash
./tests/examples_rag_api.sh
```

Or with a custom base URL:

```bash
RAG_API_URL=http://localhost:9000 ./tests/examples_rag_api.sh
```

### Manual curl Examples

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```

**Query (hybrid mode):**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "mode": "hybrid"}'
```

**Query (documents only):**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the main topic.", "mode": "documents"}'
```

**Query (general knowledge only):**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2 + 2?", "mode": "general"}'
```

## Python Tests (pytest)

Install test dependencies:

```bash
pip install pytest requests
```

Run tests (API must be running):

```bash
pytest tests/test_rag_api.py -v
```

With verbose output:

```bash
pytest tests/test_rag_api.py -v -s
```

Custom API URL:

```bash
RAG_API_URL=http://localhost:9000 pytest tests/test_rag_api.py -v
```

Tests are skipped automatically if the API is not reachable.
