# Gateway RAG-only deployment (three-layer stack)

## Goal

- Route LLM + Dispatcher stay **unchanged in behavior**.
- Worker layer: **RAG** calls the Agent RAG HTTP API; **UDS** and **SP-API** return a fixed Chinese notice instead of calling ports 8001 / 8003.

## Environment

Set **one** of the following on the gateway process:

| Variable | Value | Effect |
|----------|-------|--------|
| `GATEWAY_WORKER_PROFILE` | `rag_only` (or `rag-only`, `rag`) | Stub UDS + SP-API |
| `GATEWAY_STUB_UDS_SP_API` | `true` / `1` / `yes` / `on` | Same stub (explicit flag) |

Stubs return `{"answer": "...", "sources": []}` so tasks are **completed** and appear in rule-based merge. Errors would be **failed** and omitted from merge.

## Processes to run

### Option A: one command (recommended)

From repo root:

```bash
./bin/ic.sh restart --dispatcher-rag-only --wait
```

Starts RAG then Gateway with `GATEWAY_WORKER_PROFILE=rag_only` and **strict** health wait on 8002 and 8000.

With unified chat (7862):

```bash
./bin/ic.sh restart --dispatcher-rag-only --with-ui --wait
./bin/ic.sh restart --dispatcher-rag-only --with-ui --no-login --wait
```

### Option B: manual

1. **Agent RAG** (port 8002 recommended):

   ```bash
   RAG_API_PORT=8002 python -m uvicorn src.agent.rag.app:app --host 0.0.0.0 --port 8002
   ```

2. **Gateway** (port 8000):

   ```bash
   export GATEWAY_WORKER_PROFILE=rag_only
   export RAG_API_URL=http://127.0.0.1:8002
   python scripts/run_gateway.py
   ```

3. Ensure `DEEPSEEK_API_KEY` and Chroma env vars match [tasks/RAG_WORKFLOW.md](../../tasks/RAG_WORKFLOW.md).

### Dispatcher (section 4) missing in chat?

- **Gateway**: `project_stack.sh restart --dispatcher-rag-only` forces `GATEWAY_REWRITE_ONLY_MODE=false` and `GATEWAY_ROUTE_ONLY_MODE=false` on the gateway process so a `true` value in `.env` does not block Dispatcher + RAG.
- **Unified Chat** always calls full `/query` after the optional `/rewrite` preview. If the gateway is started with `--route-only` / `GATEWAY_REWRITE_ONLY_MODE=true`, `/query` returns before workers run, so dispatcher timings and task results are empty.

## Smoke checks

- `curl -s http://127.0.0.1:8002/health`
- `curl -s http://127.0.0.1:8000/docs` (if gateway exposes OpenAPI)
- Classified `general` / `amazon_docs` queries return RAG answers; forced `uds` / `sp_api` tasks return the stub text.

## Tests

```bash
pytest tests/test_gateway_worker_profile.py tests/test_agent_rag_service.py -q
```
