"""
Tests for UDS Client.
Requires clickhouse-connect and pandas. Run with: pytest tests/test_uds_client.py -v
"""

import pytest

from src.uds.uds_client import UDSClient, QueryError
from src.uds.config import UDSConfig


@pytest.fixture
def uds_client():
    """Create UDS client for testing."""
    return UDSClient(
        host=UDSConfig.CH_HOST,
        port=UDSConfig.CH_PORT,
        user=UDSConfig.CH_USER,
        password=UDSConfig.CH_PASSWORD,
        database=UDSConfig.CH_DATABASE,
    )


def test_connection(uds_client):
    """Test database connection."""
    assert uds_client.ping() is True


def test_list_tables(uds_client):
    """Test listing tables."""
    tables = uds_client.list_tables()
    assert len(tables) >= 9
    assert "amz_order" in tables
    assert "amz_transaction" in tables


def test_get_table_schema(uds_client):
    """Test getting table schema."""
    schema = uds_client.get_table_schema("amz_order")
    assert schema["table_name"] == "amz_order"
    assert schema["row_count"] >= 0
    assert len(schema["columns"]) > 0


def test_simple_query(uds_client):
    """Test simple query execution."""
    df = uds_client.query(
        f"SELECT COUNT(*) as count FROM {UDSConfig.CH_DATABASE}.amz_order"
    )
    assert len(df) == 1
    assert df["count"].iloc[0] >= 0


def test_query_as_tuples(uds_client):
    """Test query returning list of tuples."""
    rows = uds_client.query(
        f"SELECT 1 as x, 'a' as y",
        as_dataframe=False,
    )
    assert len(rows) == 1
    assert rows[0] == (1, "a")


def test_streaming_query(uds_client):
    """Test streaming results (synthetic data for speed)."""
    total_rows = 0
    for chunk in uds_client.query_stream(
        "SELECT number as id FROM numbers(2500)",
        chunk_size=1000,
    ):
        total_rows += len(chunk)
        assert len(chunk) <= 1000
    assert total_rows == 2500


def test_query_error_raises(uds_client):
    """Test that invalid query raises QueryError."""
    with pytest.raises(QueryError):
        uds_client.query("SELECT * FROM nonexistent_table_xyz_123")


def test_close(uds_client):
    """Test client close."""
    uds_client.close()
    # After close, ping might fail
    try:
        uds_client.ping()
    except Exception:
        pass
