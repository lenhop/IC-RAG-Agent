# Query Rewriting Guide

**Version:** 1.1.0  
**Last Updated:** 2026-03-08

---

## Overview

Query rewriting improves search quality by transforming user questions into clearer, more structured queries before routing to backend workflows. The gateway supports two backends:

- **Local (Ollama):** Runs on your machine; no API key required.
- **DeepSeek:** Cloud API; requires `DEEPSEEK_API_KEY`.

On failure (connection, timeout, API error), the gateway returns the normalized query and logs a warning. No exception is raised.

---

## Planner Mode and Multi-Intent Split

When `GATEWAY_REWRITE_PLANNER_ENABLED=true` (set by `bin/project_stack.sh` for the gateway), the gateway uses a **two-phase flow** to handle complex queries with multiple sub-questions:

1. **Phase 1 – Unified rewrite (rewriting):** One LLM call rewrites and emits sub-questions. Expected output: `{"intents": ["...", "..."], "rewritten_display": "..."}`. Prompt: `src/gateway/route_llm/rewriting/rewrite_prompt.md` (implementation in `rewrite_implement.py` / `rewriters.py`).

2. **Phase 2 – Task building:** For each intent, LLM prompt-based classification assigns a workflow (sp_api, uds, amazon_docs, general). One task is created per intent.

**Classification fallback:** When classification is unavailable, the dispatcher assigns "general" workflow to each intent.

Example: `"what is FBA get order 112-123 which table stores fee show me trend"` → 4 tasks (amazon_docs, sp_api, uds, uds).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **GATEWAY_CHAT_LLM_BACKEND** | (unset, effective **deepseek**) | Global default for all **chat** stages when the stage-specific variable is unset or invalid: `rewrite`, `clarification`, `intent_detect`, `route`, `text_generation`. Does **not** apply to embedding (`OLLAMA_*`, `INTENT_REGISTRY_EMBED_BACKEND`, `RAG_EMBED_MODEL`). |
| **GATEWAY_REWRITE_BACKEND** | (unset) | Overrides global for the unified rewrite stage when set to `ollama` or `deepseek`. Aliases: `local` (ollama), `ds` (deepseek). |
| **OLLAMA_BASE_URL** | http://localhost:11434 | Ollama API base (append /api/generate internally). |
| **OLLAMA_GENERATE_MODEL** | qwen3:1.7b | Model for generate (rewrite, clarification, intent split). |
| **OLLAMA_REQUEST_TIMEOUT** | 120 | HTTP timeout seconds for Ollama. |
| **OLLAMA_EMBED_MODEL** | all-minilm:latest | Model for /api/embed. |
| **GATEWAY_REWRITE_PLANNER_ENABLED** | true (gateway) | When true, uses two-phase intent classification for multi-question queries. |
| **DEEPSEEK_API_KEY** | (required) | Required for any DeepSeek Route LLM call. |
| **DEEPSEEK_LLM_MODEL** | deepseek-chat | Model for all DeepSeek chat calls (clarification, rewrite, intent split). |
| **DEEPSEEK_BASE_URL** | https://api.deepseek.com | OpenAI-compatible API base URL. |
| **DEEPSEEK_REQUEST_TIMEOUT** | 60 | HTTP timeout (seconds) for DeepSeek requests. |

**Unified clients:**
- **DeepSeek:** `src/llm/call_deepseek.py` — `DeepSeekChat.complete()`.
- **Ollama:** `src/llm/call_ollama.py` — `OllamaClient.generate()` / `.embed()`. Config is **only** from env (no hardcoded host/model/timeout): set at least one URL key (`OLLAMA_URL`, `OLLAMA_BASE_URL`, or `GATEWAY_REWRITE_OLLAMA_URL`), one model key (`OLLAMA_MODEL` or `GATEWAY_REWRITE_OLLAMA_MODEL`), and one timeout key (`OLLAMA_TIMEOUT` or `GATEWAY_REWRITE_TIMEOUT`). Clarification may still override URL/model/timeout per call.

**Precedence (chat backends):** Request `rewrite_backend` (when valid) overrides env for the rewrite stage only. Otherwise: stage-specific env (e.g. `GATEWAY_REWRITE_BACKEND`, `GATEWAY_CLARIFICATION_BACKEND`, `GATEWAY_INTENT_DETECT_BACKEND`, `GATEWAY_ROUTE_BACKEND`, `GATEWAY_TEXT_GENERATION_BACKEND`) then `GATEWAY_CHAT_LLM_BACKEND`, then **deepseek**. Invalid values are ignored with a warning. Optional: `GET /api/v1/gateway/llm-env-hints` returns effective per-stage labels (no secrets).

**Text generation:** `GATEWAY_TEXT_GENERATION_BACKEND` follows the same chain; `complete_chat` may still fall back from DeepSeek to Ollama at runtime if DeepSeek is unavailable.

---

## Enabling and Disabling Rewriting in the UI

1. Open the unified chat UI (e.g. `python scripts/run_unified_chat.py`).
2. Expand the **Options** accordion.
3. Use the **Rewriting Enable** checkbox:
   - **Checked:** Query rewriting is enabled; the selected backend is used.
   - **Unchecked:** Only normalization (trim, collapse whitespace) is applied; no LLM call.

---

## Switching Between Local (Ollama) and DeepSeek

1. Ensure **Rewriting Enable** is checked.
2. In the **Rewrite backend** dropdown, choose:
   - **Local (Ollama):** Uses the local Ollama server.
   - **DeepSeek:** Uses the DeepSeek API.

**Requirements:**

| Backend | Requirement |
|---------|-------------|
| **Local (Ollama)** | Ollama must be running (e.g. `ollama serve`). Model must be pulled: `ollama pull qwen3:1.7b`. |
| **DeepSeek** | `DEEPSEEK_API_KEY` must be set in `.env`. |

---

## Prerequisites

### Local (Ollama)

1. Install and start Ollama:
   ```bash
   ollama serve
   ```

2. Pull the rewrite model:
   ```bash
   ollama pull qwen3:1.7b
   ```

3. Verify:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### DeepSeek

1. Obtain an API key from [DeepSeek](https://platform.deepseek.com/).
2. Add to `.env`:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

---

## Troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| Rewriting has no effect | Rewriting Enable unchecked | Enable the checkbox in Options. |
| "Connection refused" with Local | Ollama not running | Start Ollama: `ollama serve`. |
| "DEEPSEEK_API_KEY not set" | Key missing or empty | Set `DEEPSEEK_API_KEY` in `.env`. |
| Timeout | LLM too slow or unreachable | Increase `OLLAMA_REQUEST_TIMEOUT` or switch backend. |
| Original query returned | LLM failed (connection, timeout, API error) | Check logs; gateway falls back to normalized query. |

---

## API Usage

When calling `POST /api/v1/query` directly:

```json
{
  "query": "What were my sales last week?",
  "workflow": "auto",
  "rewrite_backend": "ollama",
  "session_id": "optional-session-id",
  "stream": false
}
```

- `rewrite_backend` is optional; when omitted, `GATEWAY_REWRITE_BACKEND` is used.
- The gateway always runs the unified rewrite+split stage when the query pipeline proceeds past clarification (requires `session_id` for memory merge).

---

## Related Documentation

- [FRAMEWORK.md](../FRAMEWORK.md) - Section 5.2 Multi-Task Execution Flow (Two-Phase Intent Split)
- [tasks/IMPROVEMENT.md](../../tasks/IMPROVEMENT.md) - Section 5.3 Multi-Intent Split Logic
- [OPERATIONS.md](../OPERATIONS.md) - Production operations
