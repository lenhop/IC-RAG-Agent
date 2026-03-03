# SP-API Agent — Requirements

**Feature:** sp-api-agent  
**Part:** 2 of IC-Agent  
**Depends on:** react-agent-core (Part 1) ✅

---

## Overview

The SP-API Agent enables autonomous Amazon seller operations through natural language. It wraps the Amazon Selling Partner API behind a ReAct agent, allowing users to query catalog, inventory, orders, shipments, FBA fees, and financials conversationally.

---

## User Stories

### 1. Product Catalog

**1.1** As a seller, I want to look up a product by ASIN or SKU so I can get title, price, category, and listing status.

Acceptance criteria:
- Given a valid ASIN or SKU, the agent returns product title, price, category, and status
- Given an invalid ASIN/SKU, the agent returns a clear error message
- Results are cached in Redis for 1 hour to avoid redundant API calls

**1.2** As a seller, I want to search the catalog by keyword so I can find ASINs for products I sell.

Acceptance criteria:
- Given a keyword, the agent returns up to 20 matching products with ASIN and title
- Results include relevance ranking from SP-API

---

### 2. Inventory

**2.1** As a seller, I want to check FBA inventory levels for any SKU so I can know fulfillable quantity and days of supply.

Acceptance criteria:
- Given a SKU, the agent returns fulfillable quantity, reserved quantity, inbound quantity, and days of supply
- Given a SKU with zero inventory, the agent flags it as a stockout risk

**2.2** As a seller, I want to see all SKUs with low inventory (< 14 days supply) so I can prioritize replenishment.

Acceptance criteria:
- Agent returns a ranked list of at-risk SKUs sorted by days of supply ascending
- Each result includes SKU, fulfillable quantity, and days of supply

---

### 3. Orders

**3.1** As a seller, I want to list orders by date range and status so I can monitor sales activity.

Acceptance criteria:
- Given a date range and optional status filter, the agent returns matching orders
- Each order includes order ID, purchase date, status, total, and item count
- Supports status filters: PENDING, SHIPPED, DELIVERED, CANCELED, RETURNED

**3.2** As a seller, I want to get full details for a specific order so I can see line items, buyer info, and shipping details.

Acceptance criteria:
- Given an order ID, the agent returns order header + all line items
- Line items include SKU, ASIN, quantity, price, and shipping info

---

### 4. Inbound Shipments

**4.1** As a seller, I want to list my inbound FBA shipments so I can track what's in transit.

Acceptance criteria:
- Agent returns shipments with ID, name, destination FC, status, and unit count
- Supports status filters: WORKING, SHIPPED, IN_TRANSIT, RECEIVING, CLOSED

**4.2** As a seller, I want to create an inbound shipment plan so I can send inventory to FBA.

Acceptance criteria:
- Given a list of SKUs and quantities, the agent creates a shipment plan via SP-API
- Returns shipment ID, destination FC, and label requirements

---

### 5. FBA Fees

**5.1** As a seller, I want to estimate FBA fulfillment fees for an ASIN so I can calculate margins.

Acceptance criteria:
- Given an ASIN and price, the agent returns estimated fulfillment fee and referral fee
- Fee breakdown includes per-unit fulfillment fee and referral fee rate

**5.2** As a seller, I want to check FBA eligibility for an ASIN so I can know if it qualifies for FBA Small and Light.

Acceptance criteria:
- Given an ASIN, the agent returns eligibility status and reason if ineligible

---

### 6. Financials

**6.1** As a seller, I want to query settlement transactions so I can understand my P&L.

Acceptance criteria:
- Given a date range, the agent returns settlement transactions grouped by type
- Transaction types: product sales, referral fees, FBA fees, refunds, storage fees
- Returns net total per transaction type and overall net

---

### 7. Reports

**7.1** As a seller, I want to request and download SP-API reports so I can get bulk data exports.

Acceptance criteria:
- Given a report type, the agent requests the report, polls until ready, and returns the download URL or content
- Supports common report types: inventory, orders, financials, FBA fees
- Reports are stored to Alibaba Cloud OSS on completion

---

### 8. Conversation Memory

**8.1** As a seller, I want the agent to remember context within a session so I can ask follow-up questions without repeating myself.

Acceptance criteria:
- Agent maintains conversation history for the duration of a session (24-hour TTL)
- Follow-up queries can reference previous results (e.g., "show me the details for the first order")
- Session history is stored in Redis

---

### 9. REST API

**9.1** As a developer, I want a REST API so I can integrate the SP-API Agent into other systems.

Acceptance criteria:
- `POST /api/v1/seller/query` accepts a query string and session ID, returns a response
- `POST /api/v1/seller/query/stream` returns a streaming SSE response
- `GET /api/v1/seller/session/{id}` returns session history
- `DELETE /api/v1/seller/session/{id}` clears a session
- `GET /api/v1/health` returns service health status

**9.2** As a developer, I want the API to handle errors gracefully so I can build reliable integrations.

Acceptance criteria:
- SP-API errors return structured JSON with error code and message
- Rate limit errors (429) are retried automatically before returning to caller
- All endpoints return appropriate HTTP status codes

---

### 10. Authentication & Rate Limiting

**10.1** As a system, I need to authenticate with Amazon SP-API using LWA OAuth2 so API calls are authorized.

Acceptance criteria:
- LWA access token is refreshed automatically before expiry
- Credentials are loaded from environment variables, never hardcoded
- Token refresh failures raise a clear `AuthenticationError`

**10.2** As a system, I need to respect SP-API rate limits so the seller account is not throttled.

Acceptance criteria:
- Each endpoint has a per-endpoint rate limiter (token bucket)
- Requests that exceed the rate limit are queued, not dropped
- Rate limit state is logged for observability

---

## Non-Functional Requirements

- All SP-API calls use HTTPS
- No PII (buyer email, address) stored beyond session TTL
- Test coverage ≥ 80% for all modules
- Property-based tests for all core invariants (≥ 10 properties)
- SP-API sandbox used for all integration tests (no production calls in CI)
