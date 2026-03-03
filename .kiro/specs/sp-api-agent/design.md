# SP-API Agent — Design

**Feature:** sp-api-agent  
**Part:** 2 of IC-Agent

---

## Architecture Overview

```
User Query
    ↓
FastAPI (api.py)
    ↓
SellerOperationsAgent  ← subclass of ReActAgent (Part 1)
    ↓
LangGraph Workflow (workflow.py)
    ├── classify_intent
    ├── select_tools
    ├── execute_react_loop  ← ReActAgent.run()
    ├── format_response
    └── store_memory  ← Redis (memory.py)
    ↓
SP-API Tools (10 tools, tools/)
    ↓
SPAPIClient (sp_api_client.py)
    ├── LWA OAuth2 token refresh
    ├── Per-endpoint rate limiter
    └── Redis cache (TTL 1hr)
    ↓
Amazon SP-API (US marketplace)
```

---

## Module Design

### `sp_api_client.py`

```python
@dataclass
class SPAPICredentials:
    refresh_token: str
    client_id: str
    client_secret: str
    marketplace_id: str = "ATVPDKIKX0DER"
    seller_id: str = ""
    region: str = "us-east-1"

class SPAPIClient:
    def __init__(self, credentials: SPAPICredentials, redis_client=None)
    def get(self, path: str, params: dict = None) -> dict
    def post(self, path: str, body: dict = None) -> dict
    def _refresh_token(self) -> None
    def _get_auth_header(self) -> dict
    def _cache_get(self, key: str) -> Optional[dict]
    def _cache_set(self, key: str, value: dict, ttl: int = 3600) -> None
```

Rate limiting: one `RateLimiter` instance per endpoint path, using token bucket algorithm. Limits sourced from SP-API documentation.

### `tools/` — 10 Tools

All inherit from `ai_toolkit.tools.BaseTool`. Each implements:
- `name: str` — tool name (snake_case)
- `description: str` — for LLM prompt
- `_get_parameters() -> List[ToolParameter]`
- `validate_parameters(**kwargs) -> None`
- `execute(**kwargs) -> ToolResult`

```python
# Example: ProductCatalogTool
class ProductCatalogTool(BaseTool):
    name = "product_catalog"
    description = "Look up a product by ASIN or SKU. Returns title, price, category, status."
    
    def _get_parameters(self):
        return [
            ToolParameter("identifier", str, "ASIN or SKU", required=True),
            ToolParameter("identifier_type", str, "asin or sku", required=True),
        ]
    
    def execute(self, identifier: str, identifier_type: str) -> ToolResult:
        ...
```

### `seller_operations_agent.py`

```python
class SellerOperationsAgent(ReActAgent):
    """
    ReActAgent subclass with all 10 SP-API tools pre-registered.
    Adds intent classification and IC-RAG documentation lookup.
    """
    def __init__(self, llm, sp_api_client: SPAPIClient, memory: ConversationMemory, ...):
        tools = self._build_tools(sp_api_client)
        super().__init__(llm=llm, tools=tools, max_iterations=15)
        self._memory = memory
    
    def query(self, query: str, session_id: str) -> str:
        history = self._memory.get_history(session_id)
        result = self.run(query)  # ReActAgent.run()
        self._memory.save_turn(session_id, query, result, ...)
        return result
```

### `workflow.py` — LangGraph

```python
class SellerAgentState(TypedDict):
    query: str
    session_id: str
    intent: str          # "query" | "action" | "report"
    selected_tools: List[str]
    agent_state: AgentState
    response: str
    formatted_response: str

# Nodes
def classify_intent(state: SellerAgentState) -> SellerAgentState: ...
def select_tools(state: SellerAgentState) -> SellerAgentState: ...
def execute_react_loop(state: SellerAgentState) -> SellerAgentState: ...
def format_response(state: SellerAgentState) -> SellerAgentState: ...
def store_memory(state: SellerAgentState) -> SellerAgentState: ...

# Graph
workflow = StateGraph(SellerAgentState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("select_tools", select_tools)
workflow.add_node("execute_react_loop", execute_react_loop)
workflow.add_node("format_response", format_response)
workflow.add_node("store_memory", store_memory)
workflow.set_entry_point("classify_intent")
workflow.add_edge("classify_intent", "select_tools")
workflow.add_edge("select_tools", "execute_react_loop")
workflow.add_edge("execute_react_loop", "format_response")
workflow.add_edge("format_response", "store_memory")
workflow.add_edge("store_memory", END)
```

### `memory.py`

```python
class ConversationMemory:
    SESSION_TTL = 86400  # 24 hours
    KEY_PATTERN = "session:{session_id}:history"
    
    def __init__(self, redis_client): ...
    def save_turn(self, session_id: str, query: str, response: str, agent_state: AgentState) -> None: ...
    def get_history(self, session_id: str, last_n: int = 10) -> List[dict]: ...
    def clear_session(self, session_id: str) -> None: ...
```

### `api.py` — FastAPI

```python
# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))

class QueryResponse(BaseModel):
    response: str
    session_id: str
    iterations: int
    tools_used: List[str]

# Endpoints
POST /api/v1/seller/query          → QueryResponse
POST /api/v1/seller/query/stream   → EventSourceResponse (SSE)
GET  /api/v1/seller/session/{id}   → List[dict]
DELETE /api/v1/seller/session/{id} → {"cleared": True}
GET  /api/v1/seller/tools          → List[dict]
GET  /api/v1/health                → {"status": "ok"}
```

---

## Error Handling

| Error | Source | Handling |
|-------|--------|----------|
| `AuthenticationError` | LWA token refresh fails | Raise immediately, log credentials issue |
| `RateLimitError` | 429 from SP-API | Retry up to 3x with backoff via ToolExecutor |
| `SPAPIError` | 4xx/5xx from SP-API | Wrap in ToolResult with error message |
| `ValidationError` | Bad tool parameters | Return ToolResult(success=False, error=...) |
| `ToolNotFoundError` | Agent selects unknown tool | Inherited from ReActAgent |
| `MaxIterationsError` | Agent loop exceeds limit | Return partial result with warning |

---

## Correctness Properties

The following properties must hold and will be validated with Hypothesis property-based tests.

**Validates: Requirements 10.2**
Property 1: For any sequence of N requests to the same endpoint, the rate limiter never allows more than `burst` requests in any 1-second window.

**Validates: Requirements 10.1**
Property 2: For any valid credentials, `_get_auth_header()` always returns a dict with key `"Authorization"` containing a non-empty Bearer token.

**Validates: Requirements 1.1, 2.1, 3.1, 4.1, 5.1, 6.1**
Property 3: For any tool, `execute(**valid_params)` always returns a `ToolResult` with either `success=True` and non-None `output`, or `success=False` and non-None `error`.

**Validates: Requirements 1.1, 2.1, 3.1**
Property 4: For any tool, `validate_parameters(**invalid_params)` raises `ValidationError` when required parameters are missing or have wrong types.

**Validates: Requirements 8.1**
Property 5: For any sequence of `save_turn()` calls, `get_history()` returns turns in insertion order.

**Validates: Requirements 8.1**
Property 6: For any session, `get_history(last_n=N)` returns at most N turns.

**Validates: Requirements 9.1**
Property 7: For any valid `QueryRequest`, `POST /api/v1/seller/query` returns HTTP 200 with a `QueryResponse` containing a non-empty `response` string.

**Validates: Requirements 9.2**
Property 8: For any SP-API error response (4xx/5xx), the API endpoint returns a structured JSON error with `error_code` and `message` fields.

**Validates: Requirements 1.1**
Property 9: Redis cache hit for the same (identifier, identifier_type) always returns the same data as the original API response.

**Validates: Requirements 9.1 (streaming)**
Property 10: For any valid query, all SSE chunks from the streaming endpoint, when concatenated, equal the full response from the sync endpoint.

---

## File Structure

```
IC-RAG-Agent/
└── src/seller_operations/
    ├── __init__.py
    ├── sp_api_client.py
    ├── seller_operations_agent.py
    ├── workflow.py
    ├── memory.py
    ├── api.py
    ├── schemas.py
    ├── tools/
    │   ├── __init__.py
    │   ├── catalog.py
    │   ├── inventory.py
    │   ├── orders.py
    │   ├── shipments.py
    │   ├── fba.py
    │   ├── financials.py
    │   └── reports.py
    └── tests/
        ├── __init__.py
        ├── test_sp_api_client.py
        ├── test_tools.py
        ├── test_agent.py
        ├── test_workflow.py
        └── test_properties.py
```

---

## Dependencies

```txt
# Add to requirements.txt
httpx>=0.25.0          # Async HTTP for SP-API calls
langgraph>=0.1.0       # Workflow orchestration
fastapi>=0.110.0       # REST API
uvicorn>=0.27.0        # ASGI server
sse-starlette>=1.8.0   # Server-sent events
redis>=5.0.0           # Conversation memory
ratelimit>=2.2.1       # Rate limiting
```
