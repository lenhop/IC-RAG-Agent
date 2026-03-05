"""FastAPI REST API for Seller Operations Agent."""
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_toolkit.errors import ValidationError as ToolkitValidationError

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None

from .schemas import HealthResponse, QueryRequest, QueryResponse, SessionHistoryItem
from .sp_api_agent import SellerOperationsAgent
from .short_term_memory import ConversationMemory
from .workflow import create_app

# Optional long-term memory import
try:
    from .long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None  # type: ignore


# Global app state (set in lifespan)
_agent: SellerOperationsAgent = None
_memory: ConversationMemory = None
_long_term_memory: Optional["LongTermMemory"] = None
_workflow_app = None


def get_agent() -> SellerOperationsAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


def get_memory() -> ConversationMemory:
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return _memory


def get_long_term_memory() -> Optional["LongTermMemory"]:
    """Get long-term memory instance (may be None if not available)."""
    return _long_term_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create agent, memory, SP-API client, and long-term memory at startup."""
    global _agent, _memory, _long_term_memory, _workflow_app
    import os

    # Initialize long-term memory (optional, graceful degradation)
    if LongTermMemory is not None and os.environ.get("SP_API_LONG_TERM_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes"):
        try:
            _long_term_memory = LongTermMemory()
        except Exception:
            _long_term_memory = None
    else:
        _long_term_memory = None

    if os.environ.get("SP_API_TEST_MODE") == "true":
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = []
        mock_redis.rpush.return_value = 1
        mock_redis.expire.return_value = True
        mock_redis.delete.return_value = 1
        _memory = ConversationMemory(mock_redis)
        mock_llm = lambda p: "Thought: I will help.\nFinal Answer: Mock response for tests."
        _agent = SellerOperationsAgent(
            llm=mock_llm,
            sp_api_client=mock_client,
            memory=_memory,
            max_iterations=15,
            long_term_memory=_long_term_memory,
        )
        _workflow_app = create_app(_agent)
    else:
        try:
            from .sp_api_client import SPAPIClient, SPAPICredentials
            import redis
            creds = SPAPICredentials.from_env()
            redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            client = SPAPIClient(creds, redis_client)
            _memory = ConversationMemory(redis_client)
            mock_llm = lambda p: "Final Answer: Data retrieved successfully."
            _agent = SellerOperationsAgent(
                llm=mock_llm,
                sp_api_client=client,
                memory=_memory,
                max_iterations=15,
                long_term_memory=_long_term_memory,
            )
            _workflow_app = create_app(_agent)
        except Exception:
            try:
                import redis
                redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
                _memory = ConversationMemory(redis_client)
            except Exception:
                _memory = None
            _agent = None
            _workflow_app = None
    yield
    _agent = None
    _memory = None
    _long_term_memory = None
    _workflow_app = None


app = FastAPI(title="Seller Operations API", version="1.0.0", lifespan=lifespan)

# Fix CORS: allow cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ToolkitValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error_code": "ValidationError", "message": str(exc)},
    )


@app.exception_handler(PermissionError)
async def permission_exception_handler(request, exc):
    return JSONResponse(
        status_code=403,
        content={"error_code": "PermissionError", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"error_code": type(exc).__name__, "message": str(exc)},
    )


@app.post("/api/v1/seller/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    agent = get_agent()
    result = agent.query(req.query, req.session_id, user_id=req.user_id)
    history = agent._memory.get_history(req.session_id, last_n=1)
    iterations = history[-1].get("iterations", 0) if history and isinstance(history, list) and history else 0
    return QueryResponse(response=result, session_id=req.session_id, iterations=iterations, tools_used=[])


@app.post("/api/v1/seller/query/stream")
async def query_stream(req: QueryRequest):
    if EventSourceResponse is None:
        raise HTTPException(503, "sse-starlette not installed")
    agent = get_agent()

    async def event_generator():
        final_response = ""
        for chunk in agent.run_streaming(req.query):
            if chunk.get("type") == "final":
                final_response = chunk.get("response", "")
            yield {"data": json.dumps(chunk)}
        if final_response and agent._memory:
            from src.agent.models import AgentState
            state = AgentState(query=req.query)
            state.iteration = 0
            agent._memory.save_turn(req.session_id, req.query, final_response, state)

    return EventSourceResponse(event_generator())


@app.get("/api/v1/seller/session/{session_id}", response_model=List[SessionHistoryItem])
async def get_session(session_id: str):
    memory = get_memory()
    history = memory.get_history(session_id)
    return [SessionHistoryItem(**h) for h in history]


@app.delete("/api/v1/seller/session/{session_id}")
async def delete_session(session_id: str):
    memory = get_memory()
    memory.clear_session(session_id)
    return {"cleared": True}


@app.post("/api/v1/seller/session/{session_id}/save")
async def save_session(session_id: str, user_id: Optional[str] = None):
    """Manually trigger session summary storage to long-term memory.

    Requires user_id to be provided (either in request body or from QueryRequest).
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required for long-term memory storage")

    agent = get_agent()
    memory_id = agent.store_session_summary(session_id, user_id)

    if memory_id is None:
        return {
            "saved": False,
            "message": "Session summary could not be stored. Long-term memory may be unavailable or session has no history."
        }

    return {
        "saved": True,
        "memory_id": memory_id,
        "message": "Session summary stored successfully"
    }


@app.get("/api/v1/seller/tools")
async def list_tools():
    agent = get_agent()
    return agent.list_tools()


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")
