"""
RAG REST API - FastAPI service for multi-user RAG queries.

Phase 1+2 of multi-user service plan. Wraps RAGPipeline with HTTP endpoints.
Build pipeline once at startup, reuse for all requests.
Phase 2: Concurrency (semaphore), reliability (timeouts, 503 on overload).

Endpoints:
  POST /query - Single RAG query
  GET /health - Health check

FastAPI concepts used in this module:
  - lifespan: Run code at startup (build pipeline) and shutdown (cleanup).
  - app.state: Store shared objects (e.g. pipeline) accessible in route handlers.
  - Pydantic BaseModel: Define request/response schemas; FastAPI validates JSON automatically.
  - response_model: Ensures response matches schema and generates OpenAPI docs.
  - HTTPException: Return non-2xx status codes with error details.
  - CORSMiddleware: Allow browser requests from other origins (e.g. chat UI).
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

# Path setup: project root and ai-toolkit (must run from project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

# Load .env before other imports
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Config from .env
RAG_API_HOST = os.getenv("RAG_API_HOST", "0.0.0.0")
RAG_API_PORT = int(os.getenv("RAG_API_PORT", "8000"))
RAG_QUERY_TIMEOUT = int(os.getenv("RAG_QUERY_TIMEOUT", "120"))
# Phase 2: Concurrency and reliability
RAG_MAX_CONCURRENT_QUERIES = int(os.getenv("RAG_MAX_CONCURRENT_QUERIES", "10"))
RAG_QUEUE_TIMEOUT = int(os.getenv("RAG_QUEUE_TIMEOUT", "30"))

# Pipeline instance (set at startup)
_pipeline: Any = None


def _get_source_label(mode: str, retrieved_docs: list) -> str:
    """Infer source label from mode and retrieval."""
    if mode == "general":
        return "General Knowledge"
    if len(retrieved_docs) == 0:
        return "General Knowledge" if mode == "hybrid" else "No documents found"
    return f"Document(s) ({len(retrieved_docs)} chunks)"


def _docs_to_sources(docs: list) -> list[dict[str, str]]:
    """Convert retrieved docs to source list for JSON response."""
    sources = []
    for doc in docs[:5]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        # Use basename for cleaner display
        if isinstance(src, str) and "/" in src:
            src = src.split("/")[-1]
        sources.append({"file": src, "page": str(page)})
    return sources


@asynccontextmanager
async def lifespan(app: Any):
    """
    Lifespan context: runs once when the app starts, yields to serve requests,
    then runs cleanup after 'yield' when the app shuts down.
    Use this for heavy one-time setup (e.g. loading models, DB connections).
    """
    global _pipeline
    from src.rag.query_pipeline import RAGPipeline

    chroma_path = os.getenv("CHROMA_DOCUMENTS_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "documents"))
    if not Path(chroma_path).is_absolute():
        chroma_path = str(PROJECT_ROOT / chroma_path)

    if not Path(chroma_path).exists():
        raise FileNotFoundError(
            f"Chroma path not found: {chroma_path}. "
            "Run load_documents_to_chroma.py first."
        )

    _pipeline = RAGPipeline.build(
        embed_model=os.getenv("RAG_EMBED_MODEL", "minilm"),
        chroma_path=chroma_path,
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
        retrieval_k=int(os.getenv("RAG_RETRIEVAL_K", os.getenv("MAX_RETRIEVAL_DOCS", "5"))),
        llm_model=os.getenv("RAG_LLM_MODEL", "llama3.2:latest"),
        verbose=False,
    )
    # app.state: shared storage for all route handlers (request-scoped data goes in request.state)
    app.state.pipeline = _pipeline
    # Phase 2: Semaphore limits concurrent LLM calls to avoid OOM; excess requests wait or get 503
    app.state.query_semaphore = asyncio.Semaphore(RAG_MAX_CONCURRENT_QUERIES)
    yield  # App runs here; code below runs on shutdown
    _pipeline = None


# --- FastAPI app ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="RAG Query API",
    version="1.0.0",
    description="RAG query service for document and general knowledge Q&A",
    lifespan=lifespan,  # Hooks startup/shutdown to our lifespan function
)

# CORS: Allow browser requests from other origins (e.g. chat UI at localhost:5173).
# Without this, fetch() from a different origin would be blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RAG_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """
    Pydantic model for POST /query body. FastAPI validates incoming JSON
    against this schema; invalid requests get 422 Unprocessable Entity.
    Field(..., ...) means required; Field(default=...) means optional.
    """

    question: str = Field(..., min_length=1, max_length=4096, description="User question")
    mode: Literal["documents", "general", "hybrid"] = Field(
        default="hybrid",
        description="Answer mode: documents only, general knowledge only, or hybrid",
    )


class QueryResponse(BaseModel):
    """
    Response schema for POST /query. response_model=QueryResponse ensures
    the return value matches this shape and appears in OpenAPI docs.
    """

    answer: str = Field(..., description="Generated answer")
    source: str = Field(..., description="Source label (Document(s) or General Knowledge)")
    sources: list[dict[str, str]] = Field(default_factory=list, description="Source file and page references")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="ok or error")
    pipeline_ready: bool = Field(..., description="Whether RAG pipeline is loaded")
    chunks: int | None = Field(default=None, description="Chroma collection chunk count")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    POST /query: Run a single RAG query.
    - request: FastAPI injects parsed JSON as QueryRequest (validated).
    - response_model: FastAPI serializes return value to JSON and validates.
    - Phase 2: Semaphore limits concurrent queries; 503 if queue wait exceeds RAG_QUEUE_TIMEOUT.
    - Runs sync pipeline.query() in thread pool to avoid blocking the event loop.
    """
    pipeline = app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    semaphore = app.state.query_semaphore
    # Phase 2: Acquire slot; 503 if server is overloaded and wait exceeds queue timeout
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=RAG_QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy. Try again later (queue timeout {RAG_QUEUE_TIMEOUT}s)",
        )

    def _run_query() -> tuple[str, list]:
        return pipeline.query(
            request.question,
            mode=request.mode,
            verbose=False,
        )

    try:
        loop = asyncio.get_event_loop()
        answer, docs = await asyncio.wait_for(
            loop.run_in_executor(None, _run_query),
            timeout=RAG_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Query timed out after {RAG_QUERY_TIMEOUT}s")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        semaphore.release()

    source_label = _get_source_label(request.mode, docs)
    sources = _docs_to_sources(docs)

    return QueryResponse(
        answer=answer,
        source=source_label,
        sources=sources,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    GET /health: Health check for load balancers and monitoring.
    Returns pipeline status and Chroma chunk count when available.
    """
    pipeline = app.state.pipeline
    if pipeline is None:
        return HealthResponse(status="error", pipeline_ready=False, chunks=None)

    try:
        from src.rag.query_pipeline import get_collection_count
        chunks = get_collection_count(pipeline.vector_store)
    except Exception:
        chunks = None

    return HealthResponse(
        status="ok",
        pipeline_ready=True,
        chunks=chunks,
    )
