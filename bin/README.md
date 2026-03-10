# Bin Scripts

This folder contains executable shell entrypoints for local runtime and operations.

## Local Runtime

- `project_stack.sh`
  - Start, restart, stop, and check status for the local multi-service stack.
  - Managed services:
    - Gateway: `8000`
    - UDS API: `8001`
    - RAG API: `8002`
    - SP-API API: `8003`
  - Optional UI:
    - Unified Chat UI: `7862` with `--with-ui`
  - Route LLM only (quick testing):
    - Use `--route-only` to run a minimal test stack: gateway only, no downstream backends (`uds`, `rag`, `sp_api`).
    - Gateway runs Route LLM (clarification, rewriting, intent classification) + Dispatcher plan building; returns early without worker execution.
    - Planner prompt mode is enabled automatically to test task grouping and per-task breakdown rewrites.
  - Usage:
    - `./bin/project_stack.sh start`
    - `./bin/project_stack.sh restart`
    - `./bin/project_stack.sh stop`
    - `./bin/project_stack.sh status`
    - `./bin/project_stack.sh start --with-ui`
    - `./bin/project_stack.sh start --route-only`
    - `./bin/project_stack.sh restart --route-only`
    - `./bin/project_stack.sh start --with-ui --route-only`
- `run_rag_api.sh`
  - Run RAG API service with Uvicorn.
  - Usage:
    - `./bin/run_rag_api.sh`
    - `RAG_API_PORT=8002 ./bin/run_rag_api.sh`

## Model Utilities

- `download_models_from_hf.sh`
  - Download HuggingFace models used by this project.
- `ollama_pull_with_proxy.sh`
  - Pull Ollama models with proxy-friendly setup.

## Deployment and Ops

- `uds_ops.sh`
  - Unified ECS/prod operations helper (status, logs, deploy, rollback, setup).
  - Usage:
    - `./bin/uds_ops.sh status`
    - `./bin/uds_ops.sh logs`
    - `./bin/uds_ops.sh deploy [version]`
    - `./bin/uds_ops.sh rollback <version>`

## Notes

- Prefer `project_stack.sh` for local development and testing.
- Runtime logs are written to `logs/runtime/` when started via `project_stack.sh`.
- Process IDs are tracked in `.runtime/` when started via `project_stack.sh`.

