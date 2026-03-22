# Gateway lightweight stack: RAG + SP-API (dispatcher profile)

## Goal

- Route LLM + Dispatcher behavior stays the same.
- **RAG** calls the Agent RAG HTTP API (port **8002**).
- **SP-API** calls the Seller Agent HTTP API (port **8003**).
- **UDS** is not started; the gateway returns a fixed Chinese notice instead of calling port **8001**.

## Environment

`./bin/ic.sh restart --dispatcher-rag-only` sets on the gateway process:

| Variable | Value | Effect |
|----------|-------|--------|
| `GATEWAY_WORKER_PROFILE` | `rag_sp_api` | Stub **UDS** only; **live** SP-API client to `SP_API_URL` |
| `GATEWAY_REWRITE_ONLY_MODE` | `false` | Ensures full `/query` runs planner + dispatcher (overrides a stale `.env`) |
| `GATEWAY_ROUTE_ONLY_MODE` | `false` | Same as above |

To stub **both** UDS and SP-API (no HTTP to 8001/8003), set **one** of:

| Variable | Value | Effect |
|----------|-------|--------|
| `GATEWAY_WORKER_PROFILE` | `rag_only` (or `rag-only`, `rag`) | Stub UDS + SP-API |
| `GATEWAY_STUB_UDS_SP_API` | `true` / `1` / `yes` / `on` | Stub both workers |

Stubs return `{"answer": "...", "sources": []}` so tasks **complete** and merge can include the text.

Aliases for `rag_sp_api`: `rag-sp-api`, `dispatcher-rag-sp-api`, `dispatcher_rag_sp_api`.

## Processes to run

### Option A: one command (recommended)

From repo root:

```bash
./bin/ic.sh restart --dispatcher-rag-only --wait
```

Starts **rag**, **sp_api**, then **gateway**, with strict health wait on **8002**, **8003**, and **8000**.

With unified chat (**7862**):

```bash
./bin/ic.sh restart --dispatcher-rag-only --with-ui --wait
./bin/ic.sh restart --dispatcher-rag-only --with-ui --no-login --wait
```

### Option B: manual

1. **Agent RAG** (8002):

   ```bash
   RAG_API_PORT=8002 python -m uvicorn src.agent.rag.app:app --host 0.0.0.0 --port 8002
   ```

2. **SP-API Agent** (8003):

   ```bash
   python -m uvicorn src.agent.sp_api.app:app --host 0.0.0.0 --port 8003
   ```

3. **Gateway** (8000):

   ```bash
   export GATEWAY_WORKER_PROFILE=rag_sp_api
   export RAG_API_URL=http://127.0.0.1:8002
   export SP_API_URL=http://127.0.0.1:8003
   python scripts/run_gateway.py
   ```

4. Ensure RAG env (e.g. `DEEPSEEK_API_KEY`, Chroma) matches [tasks/RAG_WORKFLOW.md](../../tasks/RAG_WORKFLOW.md). For SP-API without Amazon credentials, use `SP_API_TEST_MODE=true` for smoke tests.

### Dispatcher (section 4) missing in chat?

- **Gateway**: `project_stack.sh restart --dispatcher-rag-only` forces `GATEWAY_REWRITE_ONLY_MODE=false` and `GATEWAY_ROUTE_ONLY_MODE=false` on the gateway process so a `true` value in `.env` does not block Dispatcher + workers.
- **Unified Chat** always calls full `/query` after the optional `/rewrite` preview. If the gateway is started with `--route-only` / `GATEWAY_REWRITE_ONLY_MODE=true`, `/query` returns before workers run, so dispatcher timings and task results are empty.

## Smoke checks

- `curl -s http://127.0.0.1:8002/health`
- `curl -s http://127.0.0.1:8003/api/v1/health`
- `curl -s http://127.0.0.1:8000/health`
- Classified `general` / `amazon_docs` queries return RAG answers; forced `uds` returns the UDS stub; `sp_api` hits the SP-API service (or stub if `rag_only` / `GATEWAY_STUB_UDS_SP_API`).

## Tests

```bash
pytest tests/test_gateway_worker_profile.py tests/test_sp_api_order_listing.py -q
```
