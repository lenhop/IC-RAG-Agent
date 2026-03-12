"""
Unit tests for gateway refactor:
- Prompt files load correctly from new paths
- route_llm is gone (no import, no env flag)
- route_backend removed from QueryRequest schema
- route_workflow always returns heuristic
- rewriters / intent_classifier import cleanly
"""
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# 1. Prompt files exist at new paths
# ---------------------------------------------------------------------------

PROMPTS_ROOT = Path(__file__).parent.parent.parent / "src" / "prompts"

@pytest.mark.parametrize("rel_path", [
    "query_clarification/clarification_detect_ambiguity.txt",
    "query_clarification/clarification_generate_question.txt",
    "query_rewriting/rewrite_query_clean.txt",
    "intent_classification/intent_verify_candidate.txt",
])
def test_prompt_file_exists(rel_path):
    p = PROMPTS_ROOT / rel_path
    assert p.is_file(), f"Missing prompt file: {p}"
    assert p.stat().st_size > 0, f"Prompt file is empty: {p}"


def test_retired_prompts_not_in_active_dir():
    """Retired prompts must not exist outside bak/."""
    for name in ("intent_classification.txt", "intent_verify.txt",
                 "route_classification.txt", "intent_route_single.txt"):
        active = PROMPTS_ROOT / name
        assert not active.exists(), f"Retired prompt still in active dir: {active}"
        # Also not in intent_classification/ subfolder
        sub = PROMPTS_ROOT / "intent_classification" / name
        assert not sub.exists(), f"Retired prompt still in intent_classification/: {sub}"


# ---------------------------------------------------------------------------
# 2. prompt_loader loads all active prompts without error
# ---------------------------------------------------------------------------

def test_prompt_loader_loads_all_active():
    from src.gateway.prompt_loader import load_prompt, clear_cache
    clear_cache()
    names = [
        "query_clarification/clarification_detect_ambiguity",
        "query_clarification/clarification_generate_question",
        "query_rewriting/rewrite_query_clean",
        "intent_classification/intent_verify_candidate",
    ]
    for name in names:
        text = load_prompt(name)
        assert isinstance(text, str) and len(text) > 10, f"Prompt '{name}' loaded empty"


def test_prompt_loader_raises_for_retired():
    from src.gateway.prompt_loader import load_prompt, clear_cache
    clear_cache()
    for name in ("intent_classification/intent_route_single",):
        with pytest.raises(FileNotFoundError):
            load_prompt(name)


def test_intent_split_query_prompt_exists():
    """intent_split_query.txt is still active (used by split_intents)."""
    from src.gateway.prompt_loader import load_prompt, clear_cache
    clear_cache()
    text = load_prompt("intent_classification/intent_split_query")
    assert isinstance(text, str) and len(text) > 10


# ---------------------------------------------------------------------------
# 3. route_llm module is gone
# ---------------------------------------------------------------------------

def test_route_llm_module_deleted():
    route_llm_path = Path(__file__).parent.parent.parent / "src" / "gateway" / "route_llm.py"
    assert not route_llm_path.exists(), "route_llm.py should have been deleted"


def test_route_llm_not_importable():
    with pytest.raises(ModuleNotFoundError):
        import src.gateway.route_llm  # noqa: F401


# ---------------------------------------------------------------------------
# 4. QueryRequest schema has no route_backend field
# ---------------------------------------------------------------------------

def test_schema_no_route_backend():
    from src.gateway.schemas import QueryRequest
    fields = QueryRequest.model_fields
    assert "route_backend" not in fields, "route_backend should be removed from QueryRequest"


def test_schema_still_has_rewrite_backend():
    from src.gateway.schemas import QueryRequest
    assert "rewrite_backend" in QueryRequest.model_fields


def test_schema_accepts_request_without_route_backend():
    from src.gateway.schemas import QueryRequest
    req = QueryRequest(query="test query")
    assert req.query == "test query"
    assert req.workflow == "auto"


# ---------------------------------------------------------------------------
# 5. router.route_workflow always uses heuristic (no LLM call)
# ---------------------------------------------------------------------------

def test_route_workflow_manual_override():
    from src.gateway.router import route_workflow
    from src.gateway.schemas import QueryRequest
    req = QueryRequest(query="test", workflow="uds")
    wf, conf, source, backend, llm_conf = route_workflow("test", req)
    assert wf == "uds"
    assert conf == 1.0
    assert source == "manual"


def test_route_workflow_auto_uses_heuristic():
    from src.gateway.router import route_workflow
    from src.gateway.schemas import QueryRequest
    req = QueryRequest(query="what is FBA", workflow="auto")
    wf, conf, source, backend, llm_conf = route_workflow("what is FBA", req)
    assert source == "heuristic"
    assert wf in {"general", "amazon_docs", "ic_docs", "sp_api", "uds"}
    assert backend is None
    assert llm_conf is None


def test_route_workflow_no_llm_even_with_env_flag():
    """Even if someone sets GATEWAY_ROUTE_LLM_ENABLED, it must be ignored."""
    from src.gateway.router import route_workflow
    from src.gateway.schemas import QueryRequest
    req = QueryRequest(query="check order status", workflow="auto")
    with patch.dict(os.environ, {"GATEWAY_ROUTE_LLM_ENABLED": "true"}):
        wf, conf, source, backend, llm_conf = route_workflow("check order status", req)
    # Should still be heuristic — the flag no longer exists in router
    assert source == "heuristic"


# ---------------------------------------------------------------------------
# 6. rewriters module imports and loads prompts correctly
# ---------------------------------------------------------------------------

def test_rewriters_import():
    from src.gateway import rewriters
    assert hasattr(rewriters, "REWRITE_PROMPT")
    assert not hasattr(rewriters, "INTENT_CLASSIFICATION_PROMPT")
    assert not hasattr(rewriters, "intent_classification_enabled")
    assert not hasattr(rewriters, "rewrite_intents_only")
    assert len(rewriters.REWRITE_PROMPT) > 10


# ---------------------------------------------------------------------------
# 7. intent_classifier imports and loads prompt correctly
# ---------------------------------------------------------------------------

def test_intent_classifier_import():
    from src.gateway import intent_classifier
    assert hasattr(intent_classifier, "classify_intent")
    assert hasattr(intent_classifier, "IntentResult")


def test_intent_classifier_prompt_path():
    """Verify _llm_verify loads from the new path (no FileNotFoundError at import)."""
    from src.gateway.prompt_loader import load_prompt, clear_cache
    clear_cache()
    # This is what intent_classifier._llm_verify calls
    text = load_prompt("intent_classification/intent_verify_candidate")
    assert "{candidates}" in text
    assert "{query}" in text


# ---------------------------------------------------------------------------
# 8. Keyword intent matching
# ---------------------------------------------------------------------------

def test_keyword_match_single_workflow():
    from src.gateway.intent_classifier import _keyword_match_intent
    assert _keyword_match_intent("check order status for 112-1234567") == "sp_api"
    assert _keyword_match_intent("show me FBA fees for last month") == "uds"
    assert _keyword_match_intent("what is RAG") == "general"
    assert _keyword_match_intent("FBA storage fee policy") == "amazon_docs"


def test_keyword_match_hybrid():
    from src.gateway.intent_classifier import _keyword_match_intent
    # Hits both sp_api (order status) and uds (fba fees)
    result = _keyword_match_intent("check order status and show fba fees last month")
    assert result == "hybrid"


def test_keyword_match_empty():
    from src.gateway.intent_classifier import _keyword_match_intent
    assert _keyword_match_intent("") == "general"
    assert _keyword_match_intent("   ") == "general"


# ---------------------------------------------------------------------------
# 9. resolve_intent fallback logic
# ---------------------------------------------------------------------------

def test_resolve_intent_consistent():
    from src.gateway.intent_classifier import resolve_intent
    assert resolve_intent("uds", "uds") == "uds"
    assert resolve_intent("sp_api", "sp_api") == "sp_api"


def test_resolve_intent_keyword_priority():
    from src.gateway.intent_classifier import resolve_intent
    assert resolve_intent("uds", "sp_api") == "uds"
    assert resolve_intent("uds", "hybrid") == "uds"


def test_resolve_intent_vector_fallback():
    from src.gateway.intent_classifier import resolve_intent
    assert resolve_intent("hybrid", "uds") == "uds"
    assert resolve_intent("hybrid", "sp_api") == "sp_api"


def test_resolve_intent_both_hybrid():
    from src.gateway.intent_classifier import resolve_intent
    assert resolve_intent("hybrid", "hybrid") == "general"


# ---------------------------------------------------------------------------
# 10. IntentResult has vector_distance field
# ---------------------------------------------------------------------------

def test_intent_result_has_vector_distance():
    from src.gateway.intent_classifier import IntentResult
    r = IntentResult(
        intent_name="fee_analysis",
        workflow="uds",
        distance=0.25,
        confidence="high",
        source="keyword",
        vector_distance=0.25,
    )
    assert r.vector_distance == 0.25
    assert r.source == "keyword"


# ---------------------------------------------------------------------------
# 11. Intent split fallback behavior (LLM single item / invalid JSON)
# ---------------------------------------------------------------------------


def test_split_intents_llm_single_item_uses_heuristic(monkeypatch):
    """When LLM returns one long item, split_intents should apply heuristic multi-clause split."""
    from src.gateway import intent_classifier as ic

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "response": (
                    '{"intents":["what is few-shot learning and when is it useful, '
                    "what are amazon's fba small and light program eligibility and fee structure, "
                    "get competitive buy box pricing for asin b08pqr7890, "
                    "what columns does amz_order have for tracking shipment status, "
                    "compare fba vs fbm fulfillment cost per unit for the past quarter, "
                    'and list all pending customer returns for my account"]}'
                )
            }

    monkeypatch.setattr(ic, "load_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(ic.requests, "post", lambda *_args, **_kwargs: _Resp())

    query = (
        "what is few-shot learning and when is it useful, what are amazon's fba "
        "small and light program eligibility and fee structure, get competitive "
        "buy box pricing for asin b08pqr7890, what columns does amz_order have "
        "for tracking shipment status, compare fba vs fbm fulfillment cost per "
        "unit for the past quarter, and list all pending customer returns for my account"
    )
    intents = ic.split_intents(query)
    assert isinstance(intents, list)
    assert len(intents) >= 6
    assert any("few-shot learning" in part for part in intents)
    assert any("buy box pricing" in part for part in intents)
    assert any("pending customer returns" in part for part in intents)


def test_split_intents_invalid_json_uses_heuristic(monkeypatch):
    """When LLM output is not parseable JSON, split_intents should still split long multi-clause query."""
    from src.gateway import intent_classifier as ic

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"response": "not-a-json-response"}

    monkeypatch.setattr(ic, "load_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(ic.requests, "post", lambda *_args, **_kwargs: _Resp())

    query = "show fba fees last month, get order status for 112-123, and list pending returns"
    intents = ic.split_intents(query)
    assert len(intents) >= 3
    assert intents[0].lower().startswith("show fba fees")
    assert any("order status" in part.lower() for part in intents)
    assert any("pending returns" in part.lower() for part in intents)
