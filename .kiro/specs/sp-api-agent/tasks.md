# SP-API Agent — Tasks

**Feature:** sp-api-agent  
**Spec:** `.kiro/specs/sp-api-agent/`

---

## Task List

### Week 1: SP-API Client + Auth

- [ ] 1. Implement `SPAPICredentials` dataclass (load from env vars)
  - Fields: refresh_token, client_id, client_secret, marketplace_id, seller_id, region
  - Validates all required fields are non-empty on init
  - Raises `ValueError` if any required field is missing

- [ ] 2. Implement `SPAPIClient` with LWA token refresh
  - `_refresh_token()` — POST to LWA endpoint, store access token + expiry
  - `_get_auth_header()` — returns `{"Authorization": "Bearer <token>"}`, auto-refreshes if expired
  - `get(path, params)` and `post(path, body)` — authenticated HTTP methods
  - Uses `httpx` for HTTP calls

- [ ] 3. Implement per-endpoint rate limiter (token bucket)
  - One `RateLimiter` per endpoint path
  - Respects SP-API quotas (see design.md rate limits table)
  - Queues requests that exceed rate, does not drop them

- [ ] 4. Implement Redis cache layer in `SPAPIClient`
  - `_cache_get(key)` / `_cache_set(key, value, ttl=3600)`
  - Cache key: `spapi:{path}:{hash(params)}`
  - Only caches GET responses with 200 status

- [ ] 5. Unit tests for `SPAPIClient` (mock HTTP + mock Redis)
  - Test token refresh on expiry
  - Test rate limiter blocks excess requests
  - Test cache hit returns cached data
  - Test cache miss calls API

---

### Week 2: SP-API Tools (10 Tools)

- [ ] 6. Implement `ProductCatalogTool` (`tools/catalog.py`)
  - Parameters: `identifier` (str), `identifier_type` ("asin" | "sku")
  - Calls Catalog Items v2022-04-01
  - Returns: title, price, category, status, ASIN, SKU

- [ ] 7. Implement `InventoryTool` (`tools/inventory.py`)
  - Parameters: `sku` (str, optional), `next_token` (str, optional)
  - Calls FBA Inventory v1
  - Returns: fulfillable_quantity, reserved_quantity, inbound_quantity, days_of_supply per SKU

- [ ] 8. Implement `ListOrdersTool` + `OrderDetailsTool` (`tools/orders.py`)
  - `ListOrdersTool`: params `created_after` (ISO date), `created_before` (ISO date, optional), `order_statuses` (list, optional)
  - `OrderDetailsTool`: param `order_id` (str)
  - Both call Orders v0

- [ ] 9. Implement `ListShipmentsTool` + `CreateShipmentTool` (`tools/shipments.py`)
  - `ListShipmentsTool`: param `shipment_status_list` (list, optional)
  - `CreateShipmentTool`: params `items` (list of {sku, quantity}), `ship_from_address` (dict)
  - Both call FBA Inbound v0

- [ ] 10. Implement `FBAFeeTool` + `FBAEligibilityTool` (`tools/fba.py`)
  - `FBAFeeTool`: params `asin` (str), `price` (float)
  - `FBAEligibilityTool`: param `asin` (str)

- [ ] 11. Implement `FinancialsTool` (`tools/financials.py`)
  - Parameters: `posted_after` (ISO date), `posted_before` (ISO date, optional)
  - Calls Finances v0
  - Returns transactions grouped by type with net totals

- [ ] 12. Implement `ReportRequestTool` (`tools/reports.py`)
  - Parameters: `report_type` (str), `data_start_time` (ISO date, optional), `data_end_time` (ISO date, optional)
  - Requests report, polls until DONE (max 10 polls, 30s interval), returns document URL
  - Calls Reports v2021-06-30

- [ ] 13. Unit tests for all 10 tools (`tests/test_tools.py`)
  - Mock SP-API responses for each tool
  - Test valid parameters → correct ToolResult
  - Test invalid parameters → ValidationError
  - Test SP-API error response → ToolResult(success=False)

---

### Week 3: SellerOperationsAgent

- [ ] 14. Implement `SellerOperationsAgent` (`seller_operations_agent.py`)
  - Subclass of `ReActAgent`
  - `__init__` registers all 10 tools
  - `query(query, session_id)` — runs agent, saves to memory, returns response
  - `max_iterations=15`

- [ ] 15. Implement intent classification in agent
  - Classify query as "query" (read-only), "action" (write), or "report" (async)
  - Use LLM with a short classification prompt
  - Log intent for observability

- [ ] 16. Integration tests for `SellerOperationsAgent` (`tests/test_agent.py`)
  - Mock SP-API client + mock LLM
  - Test full query → tool execution → response cycle
  - Test multi-turn conversation with memory
  - Test max_iterations guard

---

### Week 4: LangGraph Workflow + Memory

- [ ] 17. Implement `ConversationMemory` (`memory.py`)
  - Redis-backed, session TTL 24 hours
  - `save_turn(session_id, query, response, agent_state)`
  - `get_history(session_id, last_n=10)` — returns list of dicts in insertion order
  - `clear_session(session_id)`

- [ ] 18. Implement LangGraph workflow (`workflow.py`)
  - `SellerAgentState` TypedDict
  - 5 nodes: classify_intent, select_tools, execute_react_loop, format_response, store_memory
  - Compile and expose `app = workflow.compile()`

- [ ] 19. Tests for workflow + memory (`tests/test_workflow.py`)
  - Test each node in isolation
  - Test full graph execution with mock agent
  - Test session history ordering (property: insertion order preserved)
  - Test session TTL expiry

---

### Week 5: FastAPI REST API

- [ ] 20. Implement Pydantic schemas (`schemas.py`)
  - `QueryRequest`, `QueryResponse`, `SessionHistoryItem`, `ToolSchema`, `HealthResponse`

- [ ] 21. Implement FastAPI app (`api.py`)
  - All 6 endpoints (see design.md)
  - Dependency injection for agent, memory, SP-API client
  - CORS middleware
  - Global exception handler → structured JSON errors

- [ ] 22. Implement SSE streaming endpoint
  - `POST /api/v1/seller/query/stream`
  - Yields chunks as agent produces thoughts/observations
  - Final chunk contains complete response

- [ ] 23. API tests
  - Test all endpoints with mock agent
  - Test error responses (400, 404, 500)
  - Test SSE stream reassembles to full response

---

### Week 6: Property Tests + Documentation

- [ ] 24. Write property-based tests (`tests/test_properties.py`)
  - Implement all 10 properties from design.md using Hypothesis
  - Each property runs ≥ 100 iterations
  - Annotate each test with `# Validates: Requirements X.Y`

- [ ] 25. Integration tests against SP-API sandbox
  - Test `ProductCatalogTool` with real sandbox ASIN
  - Test `ListOrdersTool` with sandbox date range
  - Test `InventoryTool` with sandbox SKU
  - Skip in CI if `SP_API_SANDBOX=false`

- [ ] 26. Update `requirements.txt` with new dependencies

- [ ] 27. Update `IC_AGENT_OUTLINE.md` — mark Part 2 complete

---

## Acceptance Criteria

- [ ] All 10 SP-API tools implemented and tested
- [ ] `SellerOperationsAgent` runs full ReAct loop with real tools
- [ ] LangGraph workflow executes all 5 nodes
- [ ] Conversation memory persists across turns within session
- [ ] FastAPI serves all 6 endpoints
- [ ] SSE streaming works end-to-end
- [ ] All 10 property tests pass (≥ 100 iterations each)
- [ ] Test coverage ≥ 80% for `src/seller_operations/`
- [ ] SP-API sandbox integration tests pass
- [ ] `IC_AGENT_OUTLINE.md` updated to ✅ Complete
