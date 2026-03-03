# SP-API Agent - Implementation Plan

**Phase:** Part 2 of IC-Agent  
**Duration:** 6 weeks  
**Goal:** Build the Amazon SP-API Agent for autonomous seller operations — catalog, inventory, orders, shipments, FBA, and financials

**Dependencies:** Part 1 (ReAct Agent Core) ✅ Complete

---

## Overview

The SP-API Agent gives the IC-Agent system the ability to take real actions on Amazon: query product catalog, check inventory, retrieve orders, manage shipments, and analyze financials. It wraps the Amazon Selling Partner API behind a natural language interface powered by the ReAct Agent Core.

```
User Query
    ↓
SellerOperationsAgent (ReActAgent subclass)
    ↓
LangGraph Workflow (intent → tool selection → execution → response)
    ↓
SP-API Tools (10 tools, real API calls)
    ↓
Amazon SP-API (US marketplace)
```

---

## Architecture

```
src/seller_operations/
├── sp_api_client.py          ← Auth, rate limiting, caching, retry
├── seller_operations_agent.py ← Main agent (ReActAgent subclass)
├── workflow.py               ← LangGraph state machine
├── memory.py                 ← Redis conversation memory
├── api.py                    ← FastAPI REST endpoints
├── tools/
│   ├── catalog.py            ← ProductCatalogTool
│   ├── inventory.py          ← InventoryTool
│   ├── orders.py             ← OrdersTool, OrderDetailsTool
│   ├── shipments.py          ← ShipmentsTool, InboundShipmentTool
│   ├── fba.py                ← FBAFeeTool, FBAEligibilityTool
│   ├── financials.py         ← FinancialsTool
│   └── reports.py            ← ReportRequestTool
└── tests/
    ├── test_sp_api_client.py
    ├── test_tools.py
    ├── test_agent.py
    ├── test_workflow.py
    └── test_properties.py
```

---

## SP-API Tools (10 Tools)

All tools inherit from `ai_toolkit.tools.BaseTool`.

| Tool | Class | SP-API Endpoint | Description |
|------|-------|-----------------|-------------|
| `product_catalog` | `ProductCatalogTool` | Catalog Items v2022-04-01 | Search/get product by ASIN or SKU |
| `inventory_summary` | `InventoryTool` | FBA Inventory v1 | FBA inventory levels by SKU |
| `list_orders` | `ListOrdersTool` | Orders v0 | List orders by date range, status |
| `order_details` | `OrderDetailsTool` | Orders v0 | Get order + line items by order ID |
| `list_shipments` | `ListShipmentsTool` | FBA Inbound v0 | List inbound FBA shipments |
| `create_shipment` | `CreateShipmentTool` | FBA Inbound v0 | Create inbound shipment plan |
| `fba_fees` | `FBAFeeTool` | Product Fees v0 | Estimate FBA fulfillment fees |
| `fba_eligibility` | `FBAEligibilityTool` | FBA Small and Light | Check FBA eligibility for ASIN |
| `financials` | `FinancialsTool` | Finances v0 | Settlement transactions, P&L |
| `request_report` | `ReportRequestTool` | Reports v2021-06-30 | Request + download SP-API reports |

---

## SP-API Client Design

```python
# src/seller_operations/sp_api_client.py

class SPAPIClient:
    """
    Authenticated SP-API client with rate limiting, caching, and retry.
    
    - Auth: LWA (Login with Amazon) OAuth2 token refresh
    - Rate limiting: Per-endpoint token bucket (respects SP-API quotas)
    - Caching: Redis cache for catalog/product data (TTL: 1 hour)
    - Retry: Exponential backoff on 429/503 (uses ai-toolkit ToolExecutor)
    """
    
    def __init__(self, credentials: SPAPICredentials, redis_client=None): ...
    def get(self, path: str, params: dict) -> dict: ...
    def post(self, path: str, body: dict) -> dict: ...
    def refresh_token(self) -> None: ...
```

**Credentials (from environment):**
```env
SP_API_CLIENT_ID=...
SP_API_CLIENT_SECRET=...
SP_API_REFRESH_TOKEN=...
SP_API_MARKETPLACE_ID=ATVPDKIKX0DER   # US
SP_API_REGION=us-east-1
SP_API_ROLE_ARN=...
SP_API_AWS_ACCESS_KEY=...
SP_API_AWS_SECRET_KEY=...
SP_API_APP_ID=...
```

---

## LangGraph Workflow

The agent uses a LangGraph state machine for structured multi-step reasoning:

```
START
  ↓
classify_intent          ← Is this a query, action, or report request?
  ↓
select_tools             ← Which SP-API tools are needed?
  ↓
execute_react_loop       ← ReActAgent.run() with selected tools
  ↓
format_response          ← Format for user (table, summary, raw)
  ↓
store_memory             ← Save to Redis session
  ↓
END
```

---

## Conversation Memory (Redis)

```python
# src/seller_operations/memory.py

class ConversationMemory:
    """
    Redis-backed session memory for multi-turn conversations.
    
    - Session TTL: 24 hours
    - Stores: query history, tool results, user preferences
    - Key pattern: session:{session_id}:history
    """
    
    def save_turn(self, session_id: str, query: str, response: str, state: AgentState): ...
    def get_history(self, session_id: str, last_n: int = 10) -> list: ...
    def clear_session(self, session_id: str): ...
```

---

## FastAPI REST API

```
POST /api/v1/seller/query          ← Single query (sync)
POST /api/v1/seller/query/stream   ← Streaming response (SSE)
GET  /api/v1/seller/session/{id}   ← Get session history
DELETE /api/v1/seller/session/{id} ← Clear session
GET  /api/v1/seller/tools          ← List available tools
GET  /api/v1/health                ← Health check
```

---

## Implementation Plan

### Week 1: SP-API Client + Auth

**Goal:** Authenticated SP-API client with rate limiting and caching

- [ ] `SPAPICredentials` dataclass (from env vars)
- [ ] LWA token refresh (POST to `https://api.amazon.com/auth/o2/token`)
- [ ] `SPAPIClient` with `get()` / `post()` methods
- [ ] Per-endpoint rate limiter (token bucket)
- [ ] Redis cache layer for catalog/product responses (TTL 1hr)
- [ ] Retry on 429/503 via ai-toolkit ToolExecutor
- [ ] Unit tests with mocked HTTP responses
- [ ] Integration test against SP-API sandbox

**Files:**
- `src/seller_operations/__init__.py`
- `src/seller_operations/sp_api_client.py`
- `src/seller_operations/tests/test_sp_api_client.py`

---

### Week 2: SP-API Tools (10 tools)

**Goal:** All 10 tools implemented, validated, tested

- [ ] `ProductCatalogTool` — search by ASIN/SKU, return title/price/category
- [ ] `InventoryTool` — FBA inventory levels, days of supply
- [ ] `ListOrdersTool` — orders by date range + status filter
- [ ] `OrderDetailsTool` — order + line items by order ID
- [ ] `ListShipmentsTool` — inbound FBA shipments list
- [ ] `CreateShipmentTool` — create inbound shipment plan
- [ ] `FBAFeeTool` — estimate fulfillment fees for ASIN
- [ ] `FBAEligibilityTool` — check FBA eligibility
- [ ] `FinancialsTool` — settlement transactions
- [ ] `ReportRequestTool` — request + poll + download SP-API reports
- [ ] Unit tests for all 10 tools (mock SP-API responses)
- [ ] Parameter validation tests

**Files:**
- `src/seller_operations/tools/` (10 files)
- `src/seller_operations/tests/test_tools.py`

---

### Week 3: SellerOperationsAgent + ReAct Integration

**Goal:** Agent that uses ReActAgent core with SP-API tools

- [ ] `SellerOperationsAgent` class (subclass of `ReActAgent`)
- [ ] Register all 10 SP-API tools on init
- [ ] Intent classification (query vs action vs report)
- [ ] Tool selection based on intent
- [ ] Integration with IC-RAG for Amazon documentation lookup
- [ ] Integration tests: full query → tool execution → response

**Files:**
- `src/seller_operations/seller_operations_agent.py`
- `src/seller_operations/tests/test_agent.py`

---

### Week 4: LangGraph Workflow + Memory

**Goal:** Structured multi-step workflow with conversation memory

- [ ] LangGraph state definition (`SellerAgentState`)
- [ ] Nodes: `classify_intent`, `select_tools`, `execute_react_loop`, `format_response`, `store_memory`
- [ ] Edges and conditional routing
- [ ] `ConversationMemory` class (Redis-backed)
- [ ] Session management (create, retrieve, expire)
- [ ] Multi-turn conversation tests

**Files:**
- `src/seller_operations/workflow.py`
- `src/seller_operations/memory.py`
- `src/seller_operations/tests/test_workflow.py`

---

### Week 5: FastAPI REST API

**Goal:** Production-ready REST API with streaming

- [ ] FastAPI app setup with CORS, auth middleware
- [ ] `POST /api/v1/seller/query` — sync query endpoint
- [ ] `POST /api/v1/seller/query/stream` — SSE streaming endpoint
- [ ] `GET/DELETE /api/v1/seller/session/{id}` — session management
- [ ] `GET /api/v1/seller/tools` — tool listing
- [ ] `GET /api/v1/health` — health check
- [ ] Request/response Pydantic models
- [ ] API tests

**Files:**
- `src/seller_operations/api.py`
- `src/seller_operations/schemas.py`

---

### Week 6: Testing + Documentation

**Goal:** 80%+ coverage, property tests, docs

- [ ] Property-based tests (Hypothesis) — 10+ properties
- [ ] Integration tests against SP-API sandbox
- [ ] Load test: 10 concurrent queries
- [ ] API documentation (OpenAPI/Swagger auto-generated)
- [ ] SP-API tool usage guide
- [ ] Deployment guide (Docker)

**Files:**
- `src/seller_operations/tests/test_properties.py`
- `docs/ic_agent_docs/SP_API_AGENT_PLAN.md` (this file, updated)

---

## Property-Based Tests (Hypothesis)

| # | Property |
|---|----------|
| 1 | Tool parameter validation rejects invalid inputs |
| 2 | All tools return ToolResult with success/error |
| 3 | Rate limiter never exceeds per-endpoint quota |
| 4 | Token refresh produces valid auth header |
| 5 | Redis cache hit returns same data as API call |
| 6 | Session history preserves insertion order |
| 7 | Agent terminates within max_iterations |
| 8 | Tool registry has no duplicate names |
| 9 | Streaming response chunks reassemble to full response |
| 10 | SP-API error codes map to correct exception types |

---

## Python Dependencies to Add

```txt
# SP-API
sp-api>=1.0.0              # python-amazon-sp-api wrapper
# OR raw HTTP
httpx>=0.25.0              # Async HTTP client

# LangGraph
langgraph>=0.1.0

# FastAPI
fastapi>=0.110.0
uvicorn>=0.27.0
sse-starlette>=1.8.0       # Server-sent events

# Redis
redis>=5.0.0

# Rate limiting
ratelimit>=2.2.1
```

---

## SP-API Rate Limits Reference

| Endpoint | Rate | Burst |
|----------|------|-------|
| Catalog Items | 2 req/s | 2 |
| FBA Inventory | 2 req/s | 2 |
| Orders | 0.0167 req/s | 20 |
| Finances | 0.5 req/s | 30 |
| Reports | 0.0222 req/s | 10 |

---

## Next Steps

1. Create Kiro spec at `.kiro/specs/sp-api-agent/`
2. Assign Week 1–2 to Claude Code (SP-API client + tools)
3. Assign Week 3–4 to Cursor (agent + workflow)
4. Assign Week 5–6 to both in parallel (API + tests)

---

**Document Owner:** Kiro (AI Project Manager)  
**Last Updated:** 2026-03-03
