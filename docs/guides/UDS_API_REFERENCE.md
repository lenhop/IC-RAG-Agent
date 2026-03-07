# UDS Agent API Reference

Complete reference for the UDS Agent REST API.

---

## Base URL

```
http://localhost:8000
```

Production: Replace with your deployed host.

---

## Authentication

Currently no authentication. For production, add API key or OAuth2:

```http
X-API-Key: your-api-key
Authorization: Bearer <token>
```

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/uds/query` | Submit query (sync) |
| POST | `/api/v1/uds/query/stream` | Submit query (streaming) |
| GET | `/api/v1/uds/query/{id}` | Get query result |
| DELETE | `/api/v1/uds/query/{id}` | Cancel query |
| GET | `/api/v1/uds/tables` | List tables |
| GET | `/api/v1/uds/tables/{name}` | Get table schema |
| GET | `/api/v1/uds/tables/{name}/sample` | Get sample data |
| GET | `/api/v1/uds/statistics` | Get statistics |
| GET | `/health` | Health check |
| GET | `/metrics` | Metrics |
| GET | `/api/v1/uds/status` | Agent status |

---

## 1. POST /api/v1/uds/query

Submit a natural language business question. Blocks until processing completes.

### Request Schema

```json
{
  "query": "string (required)",
  "options": {
    "include_charts": true,
    "include_insights": true,
    "max_execution_time": 30
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Natural language question |
| options | object | No | Query options |
| options.include_charts | boolean | No | Include chart suggestions (default: true) |
| options.include_insights | boolean | No | Include insights (default: true) |
| options.max_execution_time | number | No | Max seconds (default: 30) |

### Response Schema (200)

```json
{
  "query_id": "uuid",
  "status": "completed | failed",
  "query": "string",
  "intent": "string | null",
  "response": null | {
    "summary": "string",
    "insights": ["string"],
    "data": {},
    "charts": ["object"],
    "recommendations": ["string"],
    "metadata": {}
  },
  "metadata": {},
  "error": "string | null",
  "created_at": "datetime"
}
```

### Code Examples

**Python (requests):**

```python
import requests

url = "http://localhost:8000/api/v1/uds/query"
payload = {
    "query": "What were total sales in October?",
    "options": {"include_charts": True, "include_insights": True},
}
response = requests.post(url, json=payload)
data = response.json()
print(data["query_id"], data["status"])
if data["status"] == "completed":
    print(data["response"]["summary"])
```

**cURL:**

```bash
curl -X POST http://localhost:8000/api/v1/uds/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were total sales in October?"}'
```

**JavaScript (fetch):**

```javascript
const response = await fetch("http://localhost:8000/api/v1/uds/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What were total sales in October?" }),
});
const data = await response.json();
console.log(data.query_id, data.status);
if (data.status === "completed") {
  console.log(data.response.summary);
}
```

---

## 2. POST /api/v1/uds/query/stream

Submit a question with Server-Sent Events (SSE) streaming.

### Request

Same as POST /api/v1/uds/query.

### Response

- **Content-Type:** `text/event-stream`
- **Events:**

| Event | Payload | Description |
|-------|---------|-------------|
| start | `{event, query_id}` | Query started |
| complete | `{event, query_id, data}` | Full response |
| error | `{event, query_id, error}` | Error message |

### Code Examples

**Python (requests):**

```python
import requests
import json

url = "http://localhost:8000/api/v1/uds/query/stream"
payload = {"query": "What were total sales in October?"}
with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line and line.startswith(b"data: "):
            data = json.loads(line[6:].decode())
            if data.get("event") == "complete":
                print(data.get("data", {}).get("summary"))
            elif data.get("event") == "error":
                print("Error:", data.get("error"))
```

**cURL:**

```bash
curl -X POST http://localhost:8000/api/v1/uds/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What were total sales in October?"}' \
  --no-buffer
```

**JavaScript (EventSource not supported for POST; use fetch):**

```javascript
const response = await fetch("http://localhost:8000/api/v1/uds/query/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What were total sales in October?" }),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value);
  const lines = buffer.split("\n\n");
  buffer = lines.pop();
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.slice(6));
      if (data.event === "complete") console.log(data.data?.summary);
      if (data.event === "error") console.error(data.error);
    }
  }
}
```

---

## 3. GET /api/v1/uds/query/{query_id}

Retrieve status and results for a completed query.

### Path Parameters

| Name | Type | Description |
|------|------|-------------|
| query_id | string | UUID from query/stream response |

### Response (200)

Same schema as POST /api/v1/uds/query response.

### Response (404)

```json
{"detail": "Query not found"}
```

### Code Examples

**Python:**

```python
response = requests.get(f"http://localhost:8000/api/v1/uds/query/{query_id}")
data = response.json()
```

**cURL:**

```bash
curl http://localhost:8000/api/v1/uds/query/550e8400-e29b-41d4-a716-446655440000
```

**JavaScript:**

```javascript
const res = await fetch(`http://localhost:8000/api/v1/uds/query/${queryId}`);
const data = await res.json();
```

---

## 4. DELETE /api/v1/uds/query/{query_id}

Remove a query from the cache.

### Response (200)

```json
{"message": "Query cancelled"}
```

### Response (404)

```json
{"detail": "Query not found"}
```

### Code Examples

**Python:**

```python
requests.delete(f"http://localhost:8000/api/v1/uds/query/{query_id}")
```

**cURL:**

```bash
curl -X DELETE http://localhost:8000/api/v1/uds/query/{query_id}
```

---

## 5. GET /api/v1/uds/tables

List all available tables.

### Response (200)

```json
{
  "tables": [
    {"name": "amz_order"},
    {"name": "amz_transaction"},
    {"name": "amz_fee"}
  ]
}
```

### Response (503)

Database unavailable.

### Code Examples

**Python:**

```python
response = requests.get("http://localhost:8000/api/v1/uds/tables")
tables = response.json()["tables"]
```

**cURL:**

```bash
curl http://localhost:8000/api/v1/uds/tables
```

---

## 6. GET /api/v1/uds/tables/{table_name}

Get table schema (columns, types, row count).

### Path Parameters

| Name | Type | Description |
|------|------|-------------|
| table_name | string | Table name (e.g. amz_order) |

### Response (200)

```json
{
  "table_name": "amz_order",
  "database": "ic_agent",
  "row_count": 4175232,
  "columns": [
    {
      "name": "amazon_order_id",
      "type": "String",
      "default_kind": "",
      "comment": ""
    },
    {
      "name": "start_date",
      "type": "Date",
      "default_kind": "",
      "comment": ""
    }
  ]
}
```

### Response (404)

```json
{"detail": "Table 'xyz' not found"}
```

### Code Examples

**Python:**

```python
response = requests.get("http://localhost:8000/api/v1/uds/tables/amz_order")
schema = response.json()
```

**cURL:**

```bash
curl http://localhost:8000/api/v1/uds/tables/amz_order
```

---

## 7. GET /api/v1/uds/tables/{table_name}/sample

Get sample rows from a table.

### Path Parameters

| Name | Type | Description |
|------|------|-------------|
| table_name | string | Table name |

### Query Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| limit | integer | 10 | Rows to return (1–1000) |

### Response (200)

```json
{
  "table_name": "amz_order",
  "sample": [
    {
      "amazon_order_id": "123-456",
      "start_date": "2025-10-01",
      "item_price": 29.99
    }
  ],
  "limit": 5
}
```

### Code Examples

**Python:**

```python
response = requests.get(
    "http://localhost:8000/api/v1/uds/tables/amz_order/sample",
    params={"limit": 5}
)
data = response.json()
```

**cURL:**

```bash
curl "http://localhost:8000/api/v1/uds/tables/amz_order/sample?limit=5"
```

---

## 8. GET /api/v1/uds/statistics

Get precomputed database statistics from `uds_statistics.json`.

### Response (200)

```json
{
  "tables": {
    "amz_order": {
      "row_count": 4175232,
      "date_range": "2025-10-01 to 2025-10-31"
    }
  }
}
```

### Response (500)

If statistics file is missing or invalid.

### Code Examples

**Python:**

```python
response = requests.get("http://localhost:8000/api/v1/uds/statistics")
stats = response.json()
```

**cURL:**

```bash
curl http://localhost:8000/api/v1/uds/statistics
```

---

## 9. GET /health

Health check for load balancers and monitoring.

### Response (200)

```json
{
  "status": "healthy",
  "database": "connected",
  "error": null,
  "timestamp": "2026-03-06T12:00:00"
}
```

### Response (200, unhealthy)

```json
{
  "status": "unhealthy",
  "database": "disconnected",
  "error": "Connection refused",
  "timestamp": "2026-03-06T12:00:00"
}
```

### Code Examples

**Python:**

```python
response = requests.get("http://localhost:8000/health")
health = response.json()
if health["status"] == "healthy":
    print("Ready")
```

**cURL:**

```bash
curl http://localhost:8000/health
```

---

## 10. GET /metrics

Prometheus-style metrics (placeholder).

### Response (200)

```json
{
  "uds_queries_total": 42,
  "uds_agent_status": "running"
}
```

---

## 11. GET /api/v1/uds/status

Agent status (tools, queries processed).

### Response (200)

```json
{
  "status": "running",
  "tools": 4,
  "queries_processed": 42
}
```

---

## Error Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 404 | Resource not found (query, table) |
| 422 | Validation error (request body) |
| 500 | Internal server error |
| 503 | Service unavailable (e.g. database down) |

### Error Response Format

```json
{
  "detail": "Error message"
}
```

### Troubleshooting

| Error | Cause | Action |
|-------|-------|--------|
| Query not found | Query ID expired or invalid | Submit new query |
| Table not found | Invalid table name | List tables, use correct name |
| 503 | Database connection failed | Check ClickHouse, network, credentials |
| 500 | Internal error | Check logs, retry |

---

## Integration Examples

### Python Client Class

```python
import requests

class UDSClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def query(self, question: str):
        r = requests.post(
            f"{self.base_url}/api/v1/uds/query",
            json={"query": question}
        )
        r.raise_for_status()
        return r.json()

    def query_stream(self, question: str):
        r = requests.post(
            f"{self.base_url}/api/v1/uds/query/stream",
            json={"query": question},
            stream=True
        )
        r.raise_for_status()
        return r.iter_lines()

    def list_tables(self):
        r = requests.get(f"{self.base_url}/api/v1/uds/tables")
        r.raise_for_status()
        return r.json()["tables"]

    def health(self):
        r = requests.get(f"{self.base_url}/health")
        return r.json()

# Usage
client = UDSClient()
result = client.query("What were total sales in October?")
print(result["response"]["summary"])
```

### JavaScript Integration

```javascript
class UDSClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async query(question) {
    const res = await fetch(`${this.baseUrl}/api/v1/uds/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async listTables() {
    const res = await fetch(`${this.baseUrl}/api/v1/uds/tables`);
    const data = await res.json();
    return data.tables;
  }

  async health() {
    const res = await fetch(`${this.baseUrl}/health`);
    return res.json();
  }
}

// Usage
const client = new UDSClient();
const result = await client.query("What were total sales in October?");
console.log(result.response.summary);
```

---

## Related Documentation

- [UDS_USER_GUIDE.md](UDS_USER_GUIDE.md) – User guide and query examples
- [UDS_API_GUIDE.md](UDS_API_GUIDE.md) – Quick start and configuration
- [UDS_DEVELOPER_GUIDE.md](UDS_DEVELOPER_GUIDE.md) – Architecture and extensions
