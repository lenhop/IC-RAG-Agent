# clarification.py – 10 Functions and Downstream Usage

## Overview

| # | Function | Purpose | Where Used |
|---|----------|---------|------------|
| 1 | `_is_concrete_documentation_query` | Skip clarification for docs/policy questions | `check_ambiguity` |
| 2 | `_build_user_input` | Build user input string with optional history | `_generate_clarification_question_ollama`, `_call_clarification_ollama`, `_call_clarification_deepseek` |
| 3 | `_generate_clarification_question_ollama` | LLM generates contextual question (heuristic path) | `check_ambiguity` |
| 4 | `_heuristic_needs_clarification` | Fast-path pattern match for known ambiguous queries | `check_ambiguity` |
| 5 | `_get_timeout` | Read timeout from env | `_generate_clarification_question_ollama`, `_call_clarification_ollama`, `_call_clarification_deepseek` |
| 6 | `_strip_markdown_fences` | Remove ``` fences from LLM output | `_generate_clarification_question_ollama`, `check_ambiguity` |
| 7 | `_extract_first_json_object` | Extract first `{...}` from text | `_generate_clarification_question_ollama`, `check_ambiguity` |
| 8 | `_call_clarification_ollama` | Call Ollama API for ambiguity detection | `check_ambiguity` |
| 9 | `_call_clarification_deepseek` | Call DeepSeek API for ambiguity detection | `check_ambiguity` |
| 10 | `check_ambiguity` | Public API – main entry point | `api.py` (rewrite + query endpoints) |

---

## Call Graph (Internal)

```
check_ambiguity (public)
├── _is_concrete_documentation_query          # Skip docs/policy queries
├── _heuristic_needs_clarification             # Fast path: pattern match
│   └── (uses _HEURISTIC_AMBIGUOUS constant)
├── _generate_clarification_question_ollama   # When heuristic hits + ollama backend
│   ├── _build_user_input
│   ├── _get_timeout
│   ├── _strip_markdown_fences
│   └── _extract_first_json_object
├── _call_clarification_ollama                 # Full LLM ambiguity check (ollama)
│   ├── _build_user_input
│   └── _get_timeout
├── _call_clarification_deepseek              # Full LLM ambiguity check (deepseek)
│   ├── _build_user_input
│   └── _get_timeout
├── _strip_markdown_fences                    # Parse LLM response
└── _extract_first_json_object                # Parse LLM response
```

---

## Downstream Usage (External)

### `check_ambiguity` – called from

| File | Location | Context |
|------|----------|---------|
| `src/gateway/api_and_auth/api.py` | L596 | Rewrite endpoint – when clarification enabled, before rewrite |
| `src/gateway/api_and_auth/api.py` | L818 | Query endpoint – when clarification enabled, before rewrite |

### `check_ambiguity` – exported from

| File | Export |
|------|--------|
| `src/gateway/route_llm/clarification/__init__.py` | `from .clarification import check_ambiguity` |

### Tests that use clarification

| File | Test |
|------|------|
| `tests/gateway/test_clarification.py` | Direct tests of `check_ambiguity` |
| `tests/gateway/test_gateway_api.py` | Mocks `check_ambiguity` for rewrite/query flows |
| `tests/gateway/test_gateway_memory.py` | `test_query_clarification_triggers_save_turn` |
| `tests/gateway/test_gateway_logger_integration.py` | Mocks `_clarification_enabled` |

---

## Function Details

### 1. `_is_concrete_documentation_query(query: str) -> bool`
- **Purpose:** Detect docs/policy/compliance/requirements questions that should not trigger clarification.
- **Used by:** `check_ambiguity` (L278).
- **Returns:** `True` if query matches patterns like "documentation requirements", "what does Amazon", "guidelines", etc.

### 2. `_build_user_input(query, conversation_context=None) -> str`
- **Purpose:** Build the prompt user input string, optionally prepending conversation history.
- **Used by:** `_generate_clarification_question_ollama`, `_call_clarification_ollama`, `_call_clarification_deepseek` (L99, L199, L234).

### 3. `_generate_clarification_question_ollama(query, conversation_context=None) -> Optional[str]`
- **Purpose:** Call Ollama to generate a contextual clarification question when heuristic matches but we want a richer question.
- **Used by:** `check_ambiguity` (L292) – only when heuristic path + ollama backend.

### 4. `_heuristic_needs_clarification(query: str) -> Optional[dict]`
- **Purpose:** Fast path: pattern-match known ambiguous queries (inventory, order, fees, sales) without required identifiers.
- **Used by:** `check_ambiguity` (L285) – when no conversation_context provided.

### 5. `_get_timeout() -> int`
- **Purpose:** Read `GATEWAY_REWRITE_TIMEOUT` from env (default 10).
- **Used by:** `_generate_clarification_question_ollama`, `_call_clarification_ollama`, `_call_clarification_deepseek`.

### 6. `_strip_markdown_fences(text: str) -> str`
- **Purpose:** Remove ``` code fences from LLM output.
- **Used by:** `_generate_clarification_question_ollama` (L119), `check_ambiguity` (L315).

### 7. `_extract_first_json_object(text: str) -> Optional[str]`
- **Purpose:** Extract the first balanced `{...}` JSON object from text.
- **Used by:** `_generate_clarification_question_ollama` (L121), `check_ambiguity` (L318).

### 8. `_call_clarification_ollama(query, conversation_context=None) -> str`
- **Purpose:** Call Ollama with `CLARIFICATION_PROMPT` to detect ambiguity; return raw response.
- **Used by:** `check_ambiguity` (L303-305) when backend is `ollama`.

### 9. `_call_clarification_deepseek(query, conversation_context=None) -> str`
- **Purpose:** Call DeepSeek API with clarification prompt; return raw response.
- **Used by:** `check_ambiguity` (L307) when backend is `deepseek`.

### 10. `check_ambiguity(query, backend=None, conversation_context=None) -> dict`
- **Purpose:** Main entry point. Decide if query needs clarification.
- **Used by:** `api.py` rewrite endpoint (L596), query endpoint (L818).
- **Returns:** `{"needs_clarification": True, "clarification_question": "..."}` or `{"needs_clarification": False}`.
