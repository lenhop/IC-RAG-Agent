# Clarification module

## Ollama backend

When `GATEWAY_CLARIFICATION_BACKEND=ollama`, calls go through `OllamaClient` using the same four Ollama variables as the rest of Route LLM:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | e.g. `http://localhost:11434` |
| `OLLAMA_GENERATE_MODEL` | Generate model for clarification |
| `OLLAMA_REQUEST_TIMEOUT` | HTTP timeout (seconds) |
| `OLLAMA_EMBED_MODEL` | Required by `get_ollama_config()` (embed not used in clarification) |

## DeepSeek backend

| Variable | Purpose |
|----------|---------|
| `GATEWAY_CLARIFICATION_BACKEND` | `deepseek` |
| `DEEPSEEK_API_KEY` | Required |
| `GATEWAY_CLARIFICATION_DEEPSEEK_BASE_URL` or `DEEPSEEK_BASE_URL` | API base |

## Other

| Variable | Purpose |
|----------|---------|
| `GATEWAY_CLARIFICATION_ENABLED` | `true` / `false` |
| `GATEWAY_CLARIFICATION_MEMORY_ROUNDS` | Conversation rounds for context |
