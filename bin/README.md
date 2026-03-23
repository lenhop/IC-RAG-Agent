# Bin Scripts

This folder contains executable shell entrypoints for local runtime and operations.

## Local stack (recommended)

Use **one** of these (they are equivalent):


| Script                       | Role                                        |
| ---------------------------- | ------------------------------------------- |
| `**./bin/ic.sh`**            | Short name; forwards to `project_stack.sh`. |
| `**./bin/project_stack.sh`** | Full implementation.                        |


### Stack commands (all services in a profile)

```text
./bin/ic.sh start|restart|stop|status [options]
```


| Option                  | Meaning                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `--with-ui`             | Start unified chat (Gradio) on **7862**.                                                           |
| `--route-only`          | Gateway only (+ optional UI); no `uds` / `rag` / `sp_api` processes.                               |
| `--dispatcher-rag-only` | **rag (8002)** + **sp_api (8003)** + **gateway (8000)**; **UDS** stubbed in gateway (live SP-API). |
| `--no-login`            | With `--with-ui`: skip login (dev).                                                                |
| `--wait`                | After start/restart, **fail** if any service does not become healthy (strict).                     |


Examples:

```bash
./bin/ic.sh restart --with-ui
./bin/ic.sh restart --with-ui --no-login --wait
./bin/ic.sh restart --dispatcher-rag-only --with-ui --wait --no-login 
./bin/ic.sh restart --route-only --with-ui --no-login --wait
./bin/ic.sh stop
./bin/ic.sh status
```

### Per-service commands (single process)

Ignore profile flags; start/stop/restart **one** service:

```text
./bin/ic.sh service <start|restart|stop|status> <gateway|uds|rag|sp_api|ui>
```

Examples:

```bash
./bin/ic.sh service restart rag
./bin/ic.sh service stop ui
./bin/ic.sh service status gateway
```

Ports: gateway **8000**, uds **8001**, rag **8002**, sp_api **8003**, ui **7862**.

Runtime logs: `logs/runtime/`. PID files: `.runtime/*.pid`.

### Legacy `src.rag.app` (not the stack Agent RAG)

Gateway stack uses `**src.agent.rag.app:app`** on **8002** (`ic.sh service start rag`). If you still need the older `**src.rag.app`** API, run Uvicorn directly (pick a free port, e.g. 8004):

```bash
python -m uvicorn src.rag.app:app --host 0.0.0.0 --port 8004
```

## Model Utilities

- `download_models_from_hf.sh`
  - Download HuggingFace models used by this project.
- `ollama_pull_with_proxy.sh`
  - Pull Ollama models with proxy-friendly setup.

## ChromaDB / Vector Registry

- `load_vector_registry.sh`
  - Loads `vector_intent_registry.csv` (columns `text`, `intent`; optional `workflow`) into **local** ChromaDB only (`VECTOR_CHROMA_PATH`). Requires Ollama for embeddings.
  - Usage:
    - `./bin/load_vector_registry.sh`
    - `OLLAMA_BASE_URL=http://${CH_HOST}:11434 ./bin/load_vector_registry.sh` — use remote Ollama when local Ollama is not running
  - Target is local Chroma only; no ECS load/transfer in the new script structure.

## Deployment and Ops

- `uds_ops.sh`
  - Unified ECS/prod operations helper (status, logs, deploy, rollback, setup).
  - Usage:
    - `./bin/uds_ops.sh status`
    - `./bin/uds_ops.sh logs`
    - `./bin/uds_ops.sh deploy [version]`
    - `./bin/uds_ops.sh rollback <version>`

## Notes

- Prefer `**./bin/ic.sh`** for local development and testing.
- Runtime logs are written to `logs/runtime/` when started via the stack scripts.
- Process IDs are tracked in `.runtime/` when started via the stack scripts.

