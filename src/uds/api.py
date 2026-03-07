"""
UDS Agent REST API.

FastAPI application exposing the UDS Agent via REST endpoints.
Supports synchronous and streaming query execution.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import UDSConfig
from .schemas import (
    AgentStatusResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TableSchema,
    TableSampleResponse,
)
from .uds_client import UDSClient, QueryError
from .cache import UDSCache

logger = logging.getLogger(__name__)

# Project root for resolving paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="UDS Agent API",
    description="Business Intelligence API for Amazon Data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized components (avoid startup failures when DB/LLM unavailable)
_uds_client: Optional[UDSClient] = None
_uds_agent: Optional[Any] = None
_uds_cache: Optional[UDSCache] = None
_query_results: Dict[str, Dict[str, Any]] = {}


def _get_uds_client() -> UDSClient:
    """Get or create UDS client."""
    global _uds_client, _uds_cache
    if _uds_cache is None:
        _uds_cache = UDSCache()
    if _uds_client is None:
        _uds_client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE,
            cache=_uds_cache,
        )
    return _uds_client


def _get_uds_agent():
    """Get or create UDS agent (lazy init with LLM)."""
    global _uds_agent, _uds_cache
    if _uds_agent is None:
        from .uds_agent import UDSAgent

        llm = _create_llm()
        client = _get_uds_client()
        _uds_agent = UDSAgent(uds_client=client, llm_client=llm, cache=_uds_cache)
    return _uds_agent


def _create_llm():
    """Create LLM for UDS Agent (Ollama or env-configured)."""
    import os

    provider = (
        os.getenv("UDS_LLM_PROVIDER")
        or os.getenv("RAG_LLM_PROVIDER", "ollama")
    ).lower()
    model = (
        os.getenv("UDS_LLM_MODEL")
        or os.getenv("RAG_LLM_MODEL", "qwen3:1.7b")
    )

    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model,
            temperature=0.1,
            num_predict=2048,
        )
    # Remote provider via ModelManager
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
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=model, temperature=0.1, num_predict=2048)


def _formatted_response_to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Convert FormattedResponse dataclass to JSON-serializable dict."""
    if obj is None:
        return None
    if hasattr(obj, "__dict__"):
        return {
            k: v
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj) if hasattr(obj, "items") else {"raw": str(obj)}


# ---------------------------------------------------------------------------
# Query Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/uds/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    """Submit analytical question (synchronous)."""
    query_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        agent = _get_uds_agent()
        result = agent.process_query(request.query)
        execution_time = time.time() - start_time

        if result.get("success"):
            response_obj = result.get("response")
            response_dict = _formatted_response_to_dict(response_obj)
            if response_dict is None and response_obj is not None:
                response_dict = {
                    "summary": getattr(response_obj, "summary", str(response_obj)),
                    "insights": getattr(response_obj, "insights", []),
                    "data": getattr(response_obj, "data", None),
                    "charts": getattr(response_obj, "charts", []),
                    "recommendations": getattr(response_obj, "recommendations", []),
                    "metadata": getattr(response_obj, "metadata", {}),
                }

            metadata = result.get("metadata") or {}
            if isinstance(response_dict, dict) and "metadata" in response_dict:
                metadata = {**metadata, **response_dict.get("metadata", {})}
            metadata["execution_time"] = execution_time

            _query_results[query_id] = {
                "query": request.query,
                "intent": result.get("intent"),
                "response": response_dict,
                "metadata": metadata,
            }

            return QueryResponse(
                query_id=query_id,
                status="completed",
                query=request.query,
                intent=result.get("intent"),
                response=response_dict,
                metadata=metadata,
            )
        else:
            return QueryResponse(
                query_id=query_id,
                status="failed",
                query=request.query,
                error=result.get("error", "Unknown error"),
                metadata={"execution_time": time.time() - start_time},
            )

    except Exception as e:
        logger.exception("Query failed: %s", e)
        return QueryResponse(
            query_id=query_id,
            status="failed",
            query=request.query,
            error=str(e),
            metadata={"execution_time": time.time() - start_time},
        )


@app.post("/api/v1/uds/query/stream")
async def submit_query_stream(request: QueryRequest):
    """Submit question with streaming response (SSE)."""

    async def event_generator():
        query_id = str(uuid.uuid4())
        yield f"data: {json.dumps({'event': 'start', 'query_id': query_id})}\n\n"

        try:
            agent = _get_uds_agent()

            # Run process_query in thread pool (it's synchronous)
            import asyncio

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, agent.process_query, request.query
            )

            if result.get("success"):
                response_obj = result.get("response")
                response_dict = _formatted_response_to_dict(response_obj)
                if response_dict is None and response_obj is not None:
                    response_dict = {
                        "summary": getattr(response_obj, "summary", str(response_obj)),
                        "insights": getattr(response_obj, "insights", []),
                        "data": getattr(response_obj, "data", None),
                        "charts": getattr(response_obj, "charts", []),
                        "recommendations": getattr(response_obj, "recommendations", []),
                        "metadata": getattr(response_obj, "metadata", {}),
                    }

                _query_results[query_id] = {
                    "query": request.query,
                    "intent": result.get("intent"),
                    "response": response_dict,
                    "metadata": result.get("metadata", {}),
                }

                yield f"data: {json.dumps({'event': 'complete', 'query_id': query_id, 'data': response_dict})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'error', 'query_id': query_id, 'error': result.get('error', 'Unknown error')})}\n\n"

        except Exception as e:
            logger.exception("Streaming query failed: %s", e)
            yield f"data: {json.dumps({'event': 'error', 'query_id': query_id, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/uds/query/{query_id}", response_model=QueryResponse)
async def get_query_result(query_id: str) -> QueryResponse:
    """Get query status and results."""
    if query_id not in _query_results:
        raise HTTPException(status_code=404, detail="Query not found")

    result = _query_results[query_id]
    return QueryResponse(
        query_id=query_id,
        status="completed",
        query=result.get("query", ""),
        intent=result.get("intent"),
        response=result.get("response"),
        metadata=result.get("metadata"),
    )


@app.delete("/api/v1/uds/query/{query_id}")
async def cancel_query(query_id: str) -> Dict[str, str]:
    """Cancel running query (removes from cache)."""
    if query_id in _query_results:
        del _query_results[query_id]
        return {"message": "Query cancelled"}
    raise HTTPException(status_code=404, detail="Query not found")


# ---------------------------------------------------------------------------
# Metadata Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/uds/tables")
async def list_tables() -> Dict[str, list]:
    """List all available tables."""
    try:
        client = _get_uds_client()
        tables = client.list_tables()
        return {"tables": [{"name": t} for t in tables]}
    except QueryError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/v1/uds/tables/{table_name}", response_model=TableSchema)
async def get_table_schema(table_name: str) -> TableSchema:
    """Get table schema."""
    try:
        client = _get_uds_client()
        tables = client.list_tables()
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

        schema = client.get_table_schema(table_name)
        return TableSchema(
            table_name=schema["table_name"],
            database=schema["database"],
            row_count=schema["row_count"],
            columns=schema["columns"],
        )
    except QueryError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/v1/uds/tables/{table_name}/sample", response_model=TableSampleResponse)
async def get_table_sample(table_name: str, limit: int = 10) -> TableSampleResponse:
    """Get sample data from table."""
    if limit < 1 or limit > 1000:
        limit = 10

    try:
        client = _get_uds_client()
        tables = client.list_tables()
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

        df = client.query(
            f"SELECT * FROM {client.database}.{table_name} LIMIT {limit}",
            as_dataframe=True,
        )
        # Convert to records; handle non-JSON-serializable types
        records = []
        for _, row in df.iterrows():
            rec = {}
            for col in df.columns:
                val = row[col]
                if hasattr(val, "isoformat"):
                    val = val.isoformat()
                elif isinstance(val, (float,)) and (val != val):
                    val = None
                rec[col] = val
            records.append(rec)

        return TableSampleResponse(
            table_name=table_name,
            sample=records,
            limit=len(records),
        )
    except QueryError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/v1/uds/statistics")
async def get_statistics() -> Dict[str, Any]:
    """Get database statistics from uds_statistics.json."""
    stats_path = UDSConfig.project_path(UDSConfig.STATISTICS_PATH)
    if not stats_path.exists():
        return {"tables": {}, "message": "Statistics file not found"}

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health & Monitoring
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        client = _get_uds_client()
        if client.ping():
            return HealthResponse(status="healthy", database="connected")
        return HealthResponse(status="unhealthy", database="disconnected")
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            database="disconnected",
            error=str(e),
        )


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Prometheus-style metrics (placeholder)."""
    return {
        "uds_queries_total": len(_query_results),
        "uds_agent_status": "running",
    }


@app.get("/api/v1/uds/status", response_model=AgentStatusResponse)
async def agent_status() -> AgentStatusResponse:
    """Get agent status."""
    try:
        agent = _get_uds_agent()
        tool_count = len(agent._registry) if hasattr(agent, "_registry") else 0
    except Exception:
        tool_count = 0
    return AgentStatusResponse(
        status="running",
        tools=tool_count,
        queries_processed=len(_query_results),
    )
