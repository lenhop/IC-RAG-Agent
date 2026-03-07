"""
Tests for gateway.api (POST /api/v1/query) with mocked services.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.gateway.api import app


client = TestClient(app)


def _base_payload() -> dict:
    return {
        "query": "test query",
        "workflow": "auto",
        "rewrite_enable": True,
        "session_id": "sess-1",
        "stream": False,
    }


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten general query")
def test_query_general_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call general service and return unified response."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["routing_confidence"] == 0.95
    assert data["answer"] == "general answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_amazon_docs", return_value={"answer": "amazon docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("amazon_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten amazon docs query")
def test_query_amazon_docs_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call amazon_docs service when routed to amazon_docs."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "amazon_docs"
    assert data["routing_confidence"] == 0.9
    assert data["answer"] == "amazon docs answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_ic_docs", return_value={"answer": "ic docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten ic docs query")
def test_query_ic_docs_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call ic_docs service when routed to ic_docs."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["routing_confidence"] == 0.9
    assert data["answer"] == "ic docs answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_sp_api", return_value={"answer": "sp api answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("sp_api", 0.85, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten sp api query")
def test_query_sp_api_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call sp_api service when routed to sp_api."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "sp_api"
    assert data["routing_confidence"] == 0.85
    assert data["answer"] == "sp api answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_uds", return_value={"answer": "uds answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("uds", 0.88, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten uds query")
def test_query_uds_success(mock_rewrite, mock_route, mock_call):
    """Gateway should call uds service when routed to uds."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "uds"
    assert data["routing_confidence"] == 0.88
    assert data["answer"] == "uds answer"
    assert data["error"] is None
    mock_call.assert_called_once()


@patch("src.gateway.api.call_general", return_value={"error": "backend failure", "error_type": "ConnectionError"})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten query")
def test_query_backend_error_propagated_to_response(mock_rewrite, mock_route, mock_call):
    """When backend returns error dict, gateway should surface it in QueryResponse.error."""
    resp = client.post("/api/v1/query", json=_base_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["answer"] == ""
    assert data["error"] == "backend failure"
    mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# Rewrite backend integration (rewrite_enable + rewrite_backend)
# ---------------------------------------------------------------------------


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.rewriters.rewrite_with_ollama", return_value="rewritten by ollama")
def test_query_rewrite_backend_ollama(mock_rewrite_ollama, mock_route, mock_call):
    """POST with rewrite_enable=True, rewrite_backend=ollama uses rewrite_with_ollama."""
    payload = {
        "query": "  what are my sales?  ",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "ollama",
        "session_id": "sess-1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general answer"
    assert data["error"] is None
    mock_rewrite_ollama.assert_called_once()
    mock_rewrite_ollama.assert_called_with("what are my sales?")
    mock_call.assert_called_once_with("rewritten by ollama", "sess-1")


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.95, "manual", None, None),
)
@patch("src.gateway.rewriters.rewrite_with_deepseek", return_value="rewritten by deepseek")
def test_query_rewrite_backend_deepseek(mock_rewrite_deepseek, mock_route, mock_call):
    """POST with rewrite_enable=True, rewrite_backend=deepseek uses rewrite_with_deepseek."""
    payload = {
        "query": "show revenue by month",
        "workflow": "auto",
        "rewrite_enable": True,
        "rewrite_backend": "deepseek",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "general answer"
    assert data["error"] is None
    mock_rewrite_deepseek.assert_called_once()
    mock_rewrite_deepseek.assert_called_with("show revenue by month")
    mock_call.assert_called_once_with("rewritten by deepseek", None)


@patch("src.gateway.rewriters.rewrite_with_ollama")
@patch("src.gateway.rewriters.rewrite_with_deepseek")
@patch("src.gateway.api.call_general", return_value={"answer": "ok", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.9, "manual", None, None),
)
def test_query_rewrite_disabled_no_rewriter_call(mock_route, mock_call, mock_deepseek, mock_ollama):
    """POST with rewrite_enable=False does not call rewrite_with_ollama or rewrite_with_deepseek."""
    payload = {
        "query": "  test query  ",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    mock_ollama.assert_not_called()
    mock_deepseek.assert_not_called()
    mock_call.assert_called_once_with("test query", "s1")


# ---------------------------------------------------------------------------
# Route LLM: /api/v1/query with workflow=auto and route_backend (mocked routing)
# ---------------------------------------------------------------------------


@patch("src.gateway.api.call_uds", return_value={"answer": "uds answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("uds", 0.92, "llm", "ollama", 0.92),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten query")
def test_query_route_llm_enabled_returns_llm_workflow_in_response(
    mock_rewrite, mock_route, mock_call_uds
):
    """POST with workflow=auto and route_backend: when routing uses LLM, response reflects LLM workflow."""
    payload = {
        "query": "what were my sales?",
        "workflow": "auto",
        "rewrite_enable": True,
        "route_backend": "ollama",
        "session_id": "s1",
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "uds"
    assert data["routing_confidence"] == 0.92
    assert data["answer"] == "uds answer"
    assert data["error"] is None
    mock_route.assert_called_once()
    mock_call_uds.assert_called_once()


@patch("src.gateway.api.call_general", return_value={"answer": "general answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("general", 0.7, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="rewritten query")
def test_query_route_llm_disabled_or_fallback_returns_heuristic_workflow(
    mock_rewrite, mock_route, mock_call_general
):
    """POST with workflow=auto when routing uses heuristic: response has heuristic workflow/confidence."""
    payload = {
        "query": "random question",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "general"
    assert data["routing_confidence"] == 0.7
    assert data["answer"] == "general answer"
    mock_route.assert_called_once()
    mock_call_general.assert_called_once()


@patch("src.gateway.api.call_ic_docs", return_value={"answer": "ic docs answer", "sources": []})
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 1.0, "manual", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="query")
def test_query_explicit_workflow_ignores_route_backend(mock_rewrite, mock_route, mock_call_ic):
    """POST with explicit workflow=ic_docs: response uses that workflow; route_backend is irrelevant."""
    payload = {
        "query": "anything",
        "workflow": "ic_docs",
        "rewrite_enable": False,
        "route_backend": "ollama",
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["routing_confidence"] == 1.0
    assert data["answer"] == "ic docs answer"
    mock_call_ic.assert_called_once()


# ---------------------------------------------------------------------------
# IC docs skip when IC_DOCS_ENABLED is false (no RAG call)
# ---------------------------------------------------------------------------


@patch("src.gateway.services._ic_docs_enabled", return_value=False)
@patch(
    "src.gateway.api.route_workflow",
    return_value=("ic_docs", 0.9, "heuristic", None, None),
)
@patch("src.gateway.api.rewrite_query", return_value="framework query")
def test_query_ic_docs_disabled_returns_friendly_message_no_rag_call(
    mock_rewrite, mock_route, mock_ic_enabled
):
    """When IC docs is disabled, gateway returns friendly message and does not call RAG."""
    payload = {
        "query": "Explain IC-RAG-Agent framework.",
        "workflow": "auto",
        "rewrite_enable": False,
        "session_id": None,
        "stream": False,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ic_docs"
    assert data["error"] is None
    assert "not ready" in data["answer"].lower()
    assert "Amazon docs" in data["answer"] or "general knowledge" in data["answer"].lower()
    mock_ic_enabled.assert_called()

