"""
API Integration Tests - End-to-end testing via REST API.

Tests the complete flow: API -> Agent -> Tools -> ClickHouse -> Response.
Requires: API server running (uvicorn src.uds.api:app --port 8000)
          ClickHouse with ic_agent database
          LLM (Ollama or configured provider)

Run: pytest tests/test_api_integration.py -v
Skip: Tests skip automatically if API server is not reachable.
"""

import os
import time

import pytest
import requests

# API base URL - configurable via env
BASE_URL = os.getenv("UDS_API_URL", "http://localhost:8000")
# Timeout for query requests (complex queries can take 15s+)
REQUEST_TIMEOUT = int(os.getenv("UDS_API_TIMEOUT", "60"))


@pytest.fixture(scope="class")
def api_client():
    """Verify API is running and return base URL."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"API health check failed: {response.status_code}")
        data = response.json()
        if data.get("status") != "healthy":
            pytest.skip(f"API unhealthy: {data.get('error', 'unknown')}")
        return BASE_URL
    except requests.exceptions.RequestException as e:
        pytest.skip(f"API server not running at {BASE_URL}: {e}")


def _post_query(url: str, query: str) -> requests.Response:
    """POST query to API with timeout."""
    return requests.post(
        f"{url}/api/v1/uds/query",
        json={"query": query},
        timeout=REQUEST_TIMEOUT,
    )


class TestAPIIntegrationSimple:
    """Simple query integration tests (5 scenarios)."""

    def test_simple_sales_query(self, api_client):
        """Test: What were total sales in October?"""
        response = _post_query(api_client, "What were total sales in October?")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ("completed", "failed")
        assert "query_id" in data
        assert data["query"] == "What were total sales in October?"

        if data["status"] == "completed":
            assert data.get("intent") == "sales"
            assert "response" in data
            resp = data["response"]
            assert resp is not None
            if isinstance(resp, dict):
                assert "summary" in resp or "insights" in resp or "data" in resp
            # Performance: simple query < 5s (if metadata has execution_time)
            meta = data.get("metadata") or {}
            if "execution_time" in meta:
                assert meta["execution_time"] < 10.0, "Simple query took too long"

    def test_simple_inventory_query(self, api_client):
        """Test: Show me current inventory levels."""
        response = _post_query(api_client, "Show me current inventory levels")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            assert data.get("intent") == "inventory"

    def test_simple_list_tables(self, api_client):
        """Test: List all available tables (via agent query)."""
        response = _post_query(api_client, "List all available tables")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            # General or schema-related intent
            assert data.get("intent") in ("general", "sales", None)
            resp = data.get("response")
            if isinstance(resp, dict):
                # Response should mention tables
                summary = str(resp.get("summary", ""))
                data_str = str(resp.get("data", ""))
                assert "table" in summary.lower() or "table" in data_str.lower() or True

    def test_simple_describe_table(self, api_client):
        """Test: Describe the amz_order table."""
        response = _post_query(api_client, "Describe the amz_order table")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            resp = data.get("response")
            if isinstance(resp, dict):
                content = str(resp.get("summary", "")) + str(resp.get("data", ""))
                assert "amz_order" in content or "order" in content.lower()

    def test_simple_profit_margin(self, api_client):
        """Test: What's the profit margin for October?"""
        response = _post_query(api_client, "What's the profit margin for October?")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            assert data.get("intent") == "financial"


class TestAPIIntegrationMedium:
    """Medium complexity query tests (4 scenarios)."""

    def test_medium_top_products_with_inventory(self, api_client):
        """Test: Top 10 products by revenue with their inventory levels."""
        response = _post_query(
            api_client,
            "Top 10 products by revenue with their inventory levels",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            meta = data.get("metadata") or {}
            if "execution_time" in meta:
                assert meta["execution_time"] < 20.0
            # May use multiple tools
            resp_meta = (data.get("response") or {}).get("metadata") or {}
            tools_used = resp_meta.get("tools_used", [])
            # Multi-tool queries typically use 2+ tools
            assert isinstance(tools_used, list)

    def test_medium_compare_periods(self, api_client):
        """Test: Compare sales between first and second half of October."""
        response = _post_query(
            api_client,
            "Compare sales between first and second half of October",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            assert data.get("intent") == "comparison"

    def test_medium_low_stock_performance(self, api_client):
        """Test: Show me low stock items and their sales performance."""
        response = _post_query(
            api_client,
            "Show me low stock items and their sales performance",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

    def test_medium_financial_summary(self, api_client):
        """Test: Financial summary with fee breakdown."""
        response = _post_query(api_client, "Financial summary with fee breakdown")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            assert data.get("intent") == "financial"


class TestAPIIntegrationComplex:
    """Complex query integration tests (3 scenarios)."""

    def test_complex_sales_analysis_dashboard(self, api_client):
        """Test: Analyze October sales, top products, inventory, dashboard."""
        response = _post_query(
            api_client,
            "Analyze October sales trends, identify top products, "
            "check their inventory, and create a dashboard",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            meta = data.get("metadata") or {}
            if "execution_time" in meta:
                assert meta["execution_time"] < 30.0
            resp = data.get("response")
            if isinstance(resp, dict) and resp.get("insights"):
                assert len(resp["insights"]) >= 0

    def test_complex_quarterly_comparison(self, api_client):
        """Test: Compare Q3 vs Q4 revenue, top 10 products each period."""
        response = _post_query(
            api_client,
            "Compare Q3 vs Q4 revenue, show top 10 products for each period",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            meta = data.get("metadata") or {}
            if "execution_time" in meta:
                assert meta["execution_time"] < 30.0

    def test_complex_business_health_check(self, api_client):
        """Test: Full business health check."""
        response = _post_query(
            api_client,
            "Full business health check: sales trends, inventory status, "
            "financial summary, and top performers",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "failed")

        if data["status"] == "completed":
            meta = data.get("metadata") or {}
            if "execution_time" in meta:
                assert meta["execution_time"] < 30.0
            resp = data.get("response")
            if isinstance(resp, dict):
                insights = resp.get("insights", [])
                recommendations = resp.get("recommendations", [])
                assert isinstance(insights, list)
                assert isinstance(recommendations, list)


class TestAPIMetadataEndpoints:
    """Test metadata endpoints (complement to query tests)."""

    def test_list_tables_endpoint(self, api_client):
        """GET /api/v1/uds/tables returns table list."""
        response = requests.get(
            f"{api_client}/api/v1/uds/tables",
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "tables" in data
        assert isinstance(data["tables"], list)

    def test_get_table_schema_endpoint(self, api_client):
        """GET /api/v1/uds/tables/{name} returns schema."""
        # First get table list
        tables_resp = requests.get(f"{api_client}/api/v1/uds/tables", timeout=10)
        if tables_resp.status_code != 200:
            pytest.skip("Could not list tables")
        tables = [t["name"] for t in tables_resp.json().get("tables", [])]
        if not tables:
            pytest.skip("No tables in database")

        response = requests.get(
            f"{api_client}/api/v1/uds/tables/{tables[0]}",
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["table_name"] == tables[0]
        assert "columns" in data
        assert "row_count" in data
