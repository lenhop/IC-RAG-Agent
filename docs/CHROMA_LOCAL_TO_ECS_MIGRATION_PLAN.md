# ChromaDB Migration Plan: Local to ECS

## Goal

Transfer ChromaDB data from local persistent storage to ECS ChromaDB server, enabling RAG and intent classification to work in production.

## Current State

| Component | Local | ECS |
|-----------|-------|-----|
| **ChromaDB** | `PersistentClient` with local paths | `chromadb/chroma` Docker image (HTTP server, port 8000) |
| **Documents** | `data/chroma_db/documents` (collection: `documents`) | Not populated |
| **FAQ** | `data/chroma_db/fqa_question` (collection: `fqa_question`) | Not populated |
| **Keywords** | `data/chroma_db/keyword` (collection: `keyword`) | Not populated |

## Source Collections

| Collection | Env Var | Default Path | Purpose |
|------------|---------|--------------|---------|
| `documents` | `CHROMA_DOCUMENTS_PATH` | `data/chroma_db/documents` | RAG document retrieval |
| `fqa_question` | `CHROMA_FQA_PATH` | `data/chroma_db/fqa_question` | Intent classification (FAQ similarity) |
| `keyword` | `CHROMA_KEYWORD_PATH` | `data/chroma_db/keyword` | Intent classification (keyword matching) |

## Target (ECS ChromaDB)

- **Docker**: `chromadb/chroma:latest` in [docker-compose.yml](docker/docker-compose.yml)
- **Port**: 8000 (host 8001 in compose to avoid conflict)
- **Connection**: `chromadb.HttpClient(host="<ECS_HOST>", port=8000)` or `chromadb.HttpClient(host="chromadb", port=8000)` when gateway runs in same Docker network
- **ECS host**: Use `8.163.3.40` or container name `chromadb` if in same stack

## Migration Strategy

### High-Level Flow

```
Local PersistentClient          ECS HttpClient
┌─────────────────────┐        ┌─────────────────────┐
│ data/chroma_db/     │        │ chromadb:8000       │
│  documents/         │  --->   │  documents          │
│  fqa_question/      │  copy   │  fqa_question       │
│  keyword/           │  --->   │  keyword            │
└─────────────────────┘        └─────────────────────┘
```

### Per-Collection Transfer

1. **Connect** to local path with `chromadb.PersistentClient(path=local_path)`
2. **List** collections in that path (or use known collection name)
3. **Read** all records: `collection.get(include=["documents","metadatas","embeddings"])` with pagination (limit=500, offset loop)
4. **Connect** to ECS with `chromadb.HttpClient(host=CHROMA_ECS_HOST, port=CHROMA_ECS_PORT)`
5. **Create** or get collection on remote
6. **Add** records in batches (Chroma max batch size ~416; use 200–300 for safety)
7. **Verify** count matches

### Batch Size Considerations

- ChromaDB `get_max_batch_size()` typically returns ~416
- Use `limit=500` for read batches, `add()` in batches of 200–300
- Large collections (e.g. documents) may have 10k+ chunks; paginate read and write

### Embedding Preservation

- **Critical**: Transfer **pre-computed embeddings** from local to remote. Do NOT re-embed on transfer.
- Local embeddings were created with `minilm`, `ollama`, or `qwen3` per collection
- Remote must receive `ids`, `documents`, `metadatas`, `embeddings` as-is

## Implementation Plan

### 1. Create migration script

**New file:** `scripts/transfer_chroma_to_ecs.py`

- CLI: `--source-path`, `--collection`, `--target-host`, `--target-port`, `--batch-size`, `--dry-run`
- Support transferring one collection or all three (documents, fqa_question, keyword)
- Use `PersistentClient` for source, `HttpClient` for target
- Paginated read: `collection.get(limit=batch_size, offset=offset, include=["documents","metadatas","embeddings"])`
- Batched add: `remote_collection.add(ids=..., documents=..., metadatas=..., embeddings=...)`
- Progress logging and count verification
- Error handling: retry on transient HTTP errors, skip/abort on schema mismatch

### 2. Configuration

**Env vars:**

| Var | Description | Example |
|-----|-------------|---------|
| `CHROMA_ECS_HOST` | ECS ChromaDB host | `8.163.3.40` or `chromadb` |
| `CHROMA_ECS_PORT` | ECS ChromaDB port | `8000` |
| `CHROMA_ECS_SSL` | Use HTTPS | `false` |

**Files to update:**

- [.env.example](.env.example) – add `CHROMA_ECS_HOST`, `CHROMA_ECS_PORT`
- [docker/.env.prod](docker/.env.prod) – add ECS ChromaDB URL for production

### 3. RAG/Gateway config for ECS

After migration, RAG API and gateway must use **HttpClient** when running against ECS:

- Add `CHROMA_MODE=remote` (or `CHROMA_ECS_URL`) to switch from `PersistentClient` to `HttpClient`
- Update [src/rag/query_pipeline.py](src/rag/query_pipeline.py) and [ai_toolkit.chroma](external) to support HTTP client when `CHROMA_ECS_HOST` is set
- Gateway `IC_DOCS_ENABLED=true` + RAG pointing to ECS ChromaDB

### 4. Pre-migration checklist

- [ ] Local ChromaDB populated: `python scripts/load_to_chroma.py documents`, `fqa`, `keywords`
- [ ] ECS ChromaDB container running and reachable (port 8000)
- [ ] Network: local machine can reach ECS host (VPN/firewall if needed)
- [ ] Disk: ECS ChromaDB has sufficient space for migrated data

### 5. Post-migration verification

- Run `scripts/transfer_chroma_to_ecs.py --verify` (or similar) to compare counts
- Smoke test: query RAG API with a sample question, confirm retrieval from ECS ChromaDB
- Intent classification: test FAQ and keyword paths against migrated collections

## Files to Create/Modify

| Action | File |
|--------|------|
| Create | `scripts/transfer_chroma_to_ecs.py` |
| Modify | `.env.example` – add `CHROMA_ECS_HOST`, `CHROMA_ECS_PORT` |
| Modify | `docker/.env.prod` – add ECS ChromaDB config |
| Optional | `src/rag/query_pipeline.py` – support HttpClient when `CHROMA_ECS_HOST` set |
| Optional | `ai-toolkit` / `langchain-chroma` – support remote Chroma in load_store |

## ECS Deployment Notes

- **docker-compose.yml**: ChromaDB on port 8001 (host) -> 8000 (container)
- **docker-compose.uds.yml**: No ChromaDB; add ChromaDB service if RAG runs on ECS
- **Standalone ECS**: If ChromaDB runs on a separate ECS instance, use `CHROMA_ECS_HOST=<ECS_PUBLIC_IP>`, ensure port 8000 is open in security group

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Large collection timeout | Paginate read/write; add retry with backoff |
| Embedding dimension mismatch | Verify collection metadata; fail fast if mismatch |
| Network interruption | Resume capability (track last offset; optional) |
| Duplicate IDs on add | Use `upsert` if supported, or delete remote collection before add |
