"""
SP-API Seller Agent — FastAPI service for gateway POST /api/v1/seller/query.

Read-only ReAct agent over Orders v0 getOrder, Catalog Items v2022-04-01 getCatalogItem (ASIN),
and Listings 2021-08-01 getListingsItem (seller SKU).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .exceptions import SPAPIAuthError
from .sp_api_agent import build_sp_api_react_agent
from .sp_api_client import SPAPIClient, SPAPICredentials

# Mirror scripts/run_gateway.py: raw ``uvicorn src.agent.sp_api.app:app`` does not load .env;
# without this, SP_API_REFRESH_TOKEN from repo-root .env is invisible (unlike test_get_amazon_order.py).
try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Ollama Python client raises this when the HTTP API returns non-2xx (e.g. 502 from proxy/daemon).
try:
    from ollama import ResponseError as OllamaResponseError
except ImportError:
    OllamaResponseError = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SP-API Seller Agent",
    description="ReAct agent with Amazon SP-API read-only tools (orders, listings).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SellerQueryRequest(BaseModel):
    """Body for gateway-compatible seller query."""

    query: str = Field(..., description="Natural language question for the agent.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id (logged only; reserved for future use).",
    )


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str = "ok"
    service: str = "sp_api_agent"


# Lazy singletons (avoid failing import when credentials missing until first request)
_stack: Dict[str, Any] = {}


def _test_mode() -> bool:
    return os.getenv("SP_API_TEST_MODE", "").strip().lower() in ("true", "1", "yes")


def _ollama_base_url() -> str:
    """
    Base URL for LangChain OllamaLLM (same convention as ``OLLAMA_BASE_URL`` in ``call_ollama``).

    Returns:
        Stripped URL without trailing slash, e.g. http://127.0.0.1:11434
    """
    raw = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip().rstrip("/")
    return raw


def _make_ollama_llm(model: str) -> Any:
    """Build OllamaLLM with explicit ``base_url`` so SP-API matches project Ollama config."""
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model,
        base_url=_ollama_base_url(),
        temperature=0.1,
        num_predict=2048,
    )


def _create_sp_api_llm() -> Any:
    """
    LLM for ReAct: SP_API_LLM_PROVIDER overrides UDS_LLM_PROVIDER; default ollama.

    Returns:
        LangChain-compatible model or callable.
    """
    provider = (
        os.getenv("SP_API_LLM_PROVIDER") or os.getenv("UDS_LLM_PROVIDER") or "ollama"
    ).lower()

    if provider == "ollama":
        model = (
            os.getenv("SP_API_LLM_MODEL")
            or os.getenv("UDS_LLM_MODEL")
            or os.getenv("UDS_OLLAMA_MODEL")
            or "qwen3:1.7b"
        )
        return _make_ollama_llm(model)

    model = (
        os.getenv("SP_API_LLM_MODEL")
        or os.getenv("UDS_LLM_MODEL")
        or os.getenv("RAG_LLM_MODEL", "deepseek-chat")
    )
    provider_api_key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "QWEN_API_KEY",
        "glm": "GLM_API_KEY",
    }
    required_api_key = provider_api_key_map.get(provider)
    if required_api_key and not os.getenv(required_api_key, "").strip():
        logger.warning(
            "SP-API agent provider '%s' needs %s; falling back to Ollama.",
            provider,
            required_api_key,
        )
        fallback = os.getenv("SP_API_LLM_FALLBACK_MODEL", "qwen3:1.7b")
        return _make_ollama_llm(fallback)

    try:
        from ai_toolkit.models import ModelManager

        manager = ModelManager()
        return manager.create_model(
            provider=provider,
            model=model if ":" not in str(model) else None,
            temperature=0.1,
            max_tokens=2048,
        )
    except ImportError:
        return _make_ollama_llm(str(model))


def _get_stack() -> Any:
    """Return cached ReActAgent (builds client + credentials + LLM once)."""
    if _stack.get("agent") is not None:
        return _stack["agent"]
    creds = SPAPICredentials.from_env()
    client = SPAPIClient(creds)
    llm = _create_sp_api_llm()
    agent = build_sp_api_react_agent(client, creds, llm)
    _stack["client"] = client
    _stack["credentials"] = creds
    _stack["agent"] = agent
    return agent


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness for stack scripts and operators."""
    return HealthResponse()


@app.post("/api/v1/seller/query")
async def seller_query(body: SellerQueryRequest) -> Dict[str, Any]:
    """
    Run the ReAct SP-API agent on the user query.

    Returns:
        Success: ``{"response": "<text>"}``.
        Failure: ``{"error": "...", "error_type": "..."}`` (HTTP 200 for gateway parser).
    """
    q = (body.query or "").strip()
    if not q:
        return {"error": "query must be non-empty", "error_type": "ValidationError"}

    if _test_mode():
        sid = (body.session_id or "").strip()
        extra = f" session_id={sid}" if sid else ""
        return {
            "response": f"[SP_API_TEST_MODE] Received query (no live SP-API call).{extra}\n{q}",
        }

    if body.session_id:
        logger.debug("seller/query session_id=%s", body.session_id)

    try:
        agent = _get_stack()
        answer = agent.run(q)
        return {"response": answer}
    except SPAPIAuthError as exc:
        logger.warning("SP-API auth failed: %s", exc)
        return {"error": str(exc), "error_type": "SPAPIAuthError"}
    except ValueError as exc:
        logger.warning("SP-API config error: %s", exc)
        return {"error": str(exc), "error_type": "ValueError"}
    except Exception as exc:
        # LangChain -> ollama client: HTTP 502/503 often means daemon down or bad reverse proxy.
        if OllamaResponseError is not None and isinstance(exc, OllamaResponseError):
            logger.exception("seller/query Ollama HTTP error: %s", exc)
            return {
                "error": (
                    "Ollama LLM request failed (HTTP error from Ollama, e.g. 502). "
                    "Check: `ollama serve` is running; OLLAMA_BASE_URL matches that server; "
                    "model is pulled (`ollama pull` for SP_API_LLM_MODEL / UDS_LLM_MODEL). "
                    f"Detail: {exc}"
                ),
                "error_type": "OllamaResponseError",
            }
        logger.exception("seller/query failed: %s", exc)
        return {"error": str(exc), "error_type": type(exc).__name__}
