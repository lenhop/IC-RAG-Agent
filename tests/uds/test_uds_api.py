"""
UDS API Tests.

Tests for REST API endpoints in src/uds/api.py.
Uses mocks to avoid requiring live ClickHouse or LLM.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.uds.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset API module globals before each test."""
    import src.uds.api as api_module

    api_module._uds_client = None
    api_module._uds_agent = None
    api_module._query_results.clear()
    yield
    api_module._uds_client = None
    api_module._uds_agent = None
    api_module._query_results.clear()


# ---------------------------------------------------------------------------
# Health & Monitoring
# ---------------------------------------------------------------------------


def test_health_check_healthy():
    """Test health endpoint when database is connected."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "timestamp" in data


def test_health_check_unhealthy():
    """Test health endpoint when database is disconnected."""
    mock_client = MagicMock()
    mock_client.ping.return_value = False

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"


def test_health_check_connection_error():
    """Test health endpoint when client creation fails."""
    with patch("src.uds.api._get_uds_client", side_effect=Exception("Connection refused")):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    assert "error" in data


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uds_queries_total" in data
    assert "uds_agent_status" in data


def test_agent_status():
    """Test agent status endpoint."""
    mock_agent = MagicMock()
    mock_agent._registry = {"tool1": MagicMock(), "tool2": MagicMock()}

    with patch("src.uds.api._get_uds_agent", return_value=mock_agent):
        response = client.get("/api/v1/uds/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["tools"] == 2
    assert "queries_processed" in data


# ---------------------------------------------------------------------------
# Metadata Endpoints
# ---------------------------------------------------------------------------


def test_list_tables():
    """Test list tables endpoint."""
    mock_client = MagicMock()
    mock_client.list_tables.return_value = ["amz_order", "amz_transaction"]

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables")

    assert response.status_code == 200
    data = response.json()
    assert "tables" in data
    assert len(data["tables"]) == 2
    assert data["tables"][0]["name"] == "amz_order"


def test_list_tables_query_error():
    """Test list tables when query fails."""
    from src.uds.uds_client import QueryError

    mock_client = MagicMock()
    mock_client.list_tables.side_effect = QueryError("Connection failed")

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables")

    assert response.status_code == 503


def test_get_table_schema():
    """Test get table schema endpoint."""
    mock_client = MagicMock()
    mock_client.list_tables.return_value = ["amz_order"]
    mock_client.get_table_schema.return_value = {
        "table_name": "amz_order",
        "database": "ic_agent",
        "row_count": 1000,
        "columns": [
            {"name": "order_id", "type": "String"},
            {"name": "amount", "type": "Float64"},
        ],
    }

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables/amz_order")

    assert response.status_code == 200
    data = response.json()
    assert data["table_name"] == "amz_order"
    assert data["database"] == "ic_agent"
    assert data["row_count"] == 1000
    assert len(data["columns"]) == 2


def test_get_table_schema_not_found():
    """Test get table schema for non-existent table."""
    mock_client = MagicMock()
    mock_client.list_tables.return_value = ["amz_order"]

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables/nonexistent")

    assert response.status_code == 404


def test_get_table_sample():
    """Test get table sample endpoint."""
    import pandas as pd

    mock_client = MagicMock()
    mock_client.list_tables.return_value = ["amz_order"]
    mock_client.database = "ic_agent"
    mock_client.query.return_value = pd.DataFrame(
        [{"order_id": "o1", "amount": 100.0}, {"order_id": "o2", "amount": 200.0}]
    )

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables/amz_order/sample?limit=5")

    assert response.status_code == 200
    data = response.json()
    assert data["table_name"] == "amz_order"
    assert "sample" in data
    assert len(data["sample"]) == 2
    assert data["limit"] == 2


def test_get_table_sample_not_found():
    """Test get table sample for non-existent table."""
    mock_client = MagicMock()
    mock_client.list_tables.return_value = ["amz_order"]

    with patch("src.uds.api._get_uds_client", return_value=mock_client):
        response = client.get("/api/v1/uds/tables/nonexistent/sample")

    assert response.status_code == 404


def test_get_statistics():
    """Test statistics endpoint.

    API returns either:
    - When file exists: raw JSON with table names as keys (e.g. amz_order, amz_fee)
    - When file missing: {"tables": {}, "message": "Statistics file not found"}
    """
    from pathlib import Path

    # Use same path resolution as API (project root)
    project_root = Path(__file__).resolve().parent.parent.parent
    stats_path = project_root / "src" / "uds" / "data" / "uds_statistics.json"

    response = client.get("/api/v1/uds/statistics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

    if stats_path.exists():
        # File found: table names as top-level keys
        assert "amz_order" in data or "tables" in data or len(data) > 0
    else:
        # File not found: structured response
        assert "tables" in data or "message" in data


# ---------------------------------------------------------------------------
# Query Endpoints
# ---------------------------------------------------------------------------


def test_submit_query_success():
    """Test successful query submission."""
    mock_agent = MagicMock()
    mock_agent.process_query.return_value = {
        "success": True,
        "query": "What were total sales?",
        "intent": "sales",
        "response": MagicMock(
            summary="Total sales: $1000",
            insights=["Strong growth"],
            data={"total": 1000},
            charts=[],
            recommendations=[],
            metadata={},
        ),
        "metadata": {},
    }

    with patch("src.uds.api._get_uds_agent", return_value=mock_agent):
        response = client.post(
            "/api/v1/uds/query",
            json={"query": "What were total sales in October?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "query_id" in data
    assert data["query"] == "What were total sales in October?"
    assert data["intent"] == "sales"


def test_submit_query_failure():
    """Test query submission when agent fails."""
    mock_agent = MagicMock()
    mock_agent.process_query.return_value = {
        "success": False,
        "query": "What were total sales?",
        "error": "Database connection failed",
    }

    with patch("src.uds.api._get_uds_agent", return_value=mock_agent):
        response = client.post(
            "/api/v1/uds/query",
            json={"query": "What were total sales?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "error" in data


def test_submit_query_exception():
    """Test query submission when agent raises exception."""
    mock_agent = MagicMock()
    mock_agent.process_query.side_effect = RuntimeError("Unexpected error")

    with patch("src.uds.api._get_uds_agent", return_value=mock_agent):
        response = client.post(
            "/api/v1/uds/query",
            json={"query": "What were total sales?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "error" in data


def test_get_query_result():
    """Test get query result endpoint."""
    import src.uds.api as api_module

    api_module._query_results["test-id"] = {
        "query": "What were total sales?",
        "intent": "sales",
        "response": {"summary": "Total: $1000"},
        "metadata": {},
    }

    response = client.get("/api/v1/uds/query/test-id")

    assert response.status_code == 200
    data = response.json()
    assert data["query_id"] == "test-id"
    assert data["status"] == "completed"
    assert data["response"]["summary"] == "Total: $1000"


def test_get_query_result_not_found():
    """Test get query result for non-existent query."""
    response = client.get("/api/v1/uds/query/nonexistent-id")

    assert response.status_code == 404


def test_cancel_query():
    """Test cancel query endpoint."""
    import src.uds.api as api_module

    api_module._query_results["cancel-id"] = {
        "query": "test",
        "intent": None,
        "response": None,
        "metadata": {},
    }

    response = client.delete("/api/v1/uds/query/cancel-id")

    assert response.status_code == 200
    assert response.json()["message"] == "Query cancelled"
    assert "cancel-id" not in api_module._query_results


def test_cancel_query_not_found():
    """Test cancel query for non-existent query."""
    response = client.delete("/api/v1/uds/query/nonexistent-id")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_submit_query_stream():
    """Test streaming query endpoint."""
    mock_agent = MagicMock()
    mock_agent.process_query.return_value = {
        "success": True,
        "query": "What were total sales?",
        "intent": "sales",
        "response": MagicMock(
            summary="Total: $1000",
            insights=[],
            data={},
            charts=[],
            recommendations=[],
            metadata={},
        ),
        "metadata": {},
    }

    with patch("src.uds.api._get_uds_agent", return_value=mock_agent):
        response = client.post(
            "/api/v1/uds/query/stream",
            json={"query": "What were total sales?"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Parse SSE events
    lines = response.text.strip().split("\n\n")
    assert len(lines) >= 1
    first_event = json.loads(lines[0].replace("data: ", ""))
    assert "event" in first_event
    assert first_event["event"] in ("start", "complete")


# ---------------------------------------------------------------------------
# OpenAPI / Docs
# ---------------------------------------------------------------------------


def test_swagger_ui_accessible():
    """Test Swagger UI is accessible at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/api/v1/uds/query" in data["paths"]
    assert "/health" in data["paths"]
