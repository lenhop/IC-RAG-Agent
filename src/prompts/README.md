# Prompts Directory

Externalized LLM prompts for the IC-RAG-Agent gateway Route LLM pipeline.

Each `.txt` file contains one prompt used by a specific pipeline step.
Code loads these via `src/prompts/loader.py` at startup and caches them in memory.

## Prompt Files

| File | Pipeline Step | Module | Used When |
|------|--------------|--------|-----------|
| `clarification.txt` | Ambiguity detection | `gateway/clarification.py` | Always (before rewrite) |
| `clarification_generate_question.txt` | Generate clarification question | `gateway/clarification.py` | When heuristic detects ambiguity |
| `rewrite.txt` | Query rewriting | `gateway/rewriters.py` | Always (rewrite is always on) |
| `intent_classification.txt` | Intent decomposition | `gateway/rewriters.py` | When intent classification enabled |
| `rewrite_planner.txt` | Task planning (hybrid) | `gateway/rewriters.py` | When planner mode enabled |
| `route_classification.txt` | Workflow routing | `gateway/route_llm.py` | When Route LLM enabled |

## Editing Prompts

- Edit the `.txt` file directly; restart the gateway to pick up changes.
- Prompts are loaded once at import time and cached.
- Use `{placeholder}` syntax for dynamic substitution if needed in the future.

## Adding New Prompts

1. Create a new `.txt` file in this directory.
2. Use `from src.prompts.loader import load_prompt` in your module.
3. Call `load_prompt("your_prompt_name")` (filename without `.txt`).
