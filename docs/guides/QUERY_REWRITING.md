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

1. **Phase 1 – Intent classification:** The LLM is asked to list each distinct sub-question. Expected output: `{"intents": ["...", "..."]}`. Implemented in `rewrite_intents_only()` in `src/gateway/rewriters.py`.

2. **Phase 2 – Task building:** For each intent, the heuristic router assigns a workflow (ic_docs, sp_api, uds, etc.). One task is created per intent.

**Heuristic split fallback:** When Phase 1 fails (LLM returns invalid format or error), `_split_multi_intent_clauses()` in `src/gateway/router.py` splits the query by question-starter patterns: `get order`, `which`, `show me`, `what is`, `what table`, `check`, etc.

Example: `"what is FBA get order 112-123 which table stores fee show me trend"` → 4 tasks (ic_docs, sp_api, uds, uds).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **GATEWAY_REWRITE_BACKEND** | ollama | Default backend when the client does not send `rewrite_backend`. Use `ollama` or `deepseek`. |
| **GATEWAY_REWRITE_OLLAMA_URL** | http://localhost:11434/api/generate | Ollama generate endpoint URL. |
| **GATEWAY_REWRITE_OLLAMA_MODEL** | qwen2.5:1.5b | Model used for local rewriting. |
| **GATEWAY_REWRITE_DEEPSEEK_MODEL** | deepseek-chat | Model used for DeepSeek API. |
| **GATEWAY_REWRITE_TIMEOUT** | 10 | Timeout in seconds for LLM calls. |
| **GATEWAY_REWRITE_PLANNER_ENABLED** | true (gateway) | When true, uses two-phase intent classification for multi-question queries. |
| **DEEPSEEK_API_KEY** | (from .env) | Required for DeepSeek backend. |

**Precedence:** If the client sends `rewrite_backend` in the request, it overrides `GATEWAY_REWRITE_BACKEND`.

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
| **Local (Ollama)** | Ollama must be running (e.g. `ollama serve`). Model must be pulled: `ollama pull qwen2.5:1.5b`. |
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
   ollama pull qwen2.5:1.5b
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
| Timeout | LLM too slow or unreachable | Increase `GATEWAY_REWRITE_TIMEOUT` or switch backend. |
| Original query returned | LLM failed (connection, timeout, API error) | Check logs; gateway falls back to normalized query. |

---

## API Usage

When calling `POST /api/v1/query` directly:

```json
{
  "query": "What were my sales last week?",
  "workflow": "auto",
  "rewrite_enable": true,
  "rewrite_backend": "ollama",
  "session_id": "optional-session-id",
  "stream": false
}
```

- `rewrite_backend` is optional; when omitted, `GATEWAY_REWRITE_BACKEND` is used.
- When `rewrite_enable` is `false`, `rewrite_backend` is ignored.

---

## Related Documentation

- [FRAMEWORK.md](../FRAMEWORK.md) - Section 5.2 Multi-Task Execution Flow (Two-Phase Intent Split)
- [tasks/IMPROVEMENT.md](../../tasks/IMPROVEMENT.md) - Section 5.3 Multi-Intent Split Logic
- [OPERATIONS.md](../OPERATIONS.md) - Production operations
