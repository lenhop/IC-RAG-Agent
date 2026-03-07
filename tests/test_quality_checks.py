"""Unit tests for uds/quality_checks.py."""

import pandas as pd
import pytest

from src.uds.maintenance.quality_checks import DataQualityChecker
from src.uds.uds_client import UDSClient


class DummyClient:
    """A minimal stub replicating UDSClient interface for testing."""

    def __init__(self):
        self.queries = []

    def query(self, sql: str, as_dataframe: bool = True, params=None):
        self.queries.append(sql)
        # return dummy DataFrame depending on query content
        if "COUNT(*) as count FROM" in sql:
            return pd.DataFrame([{"count": 10}])
        if "orphan_count" in sql:
            return pd.DataFrame([{"orphan_count": 2}])
        if "order_revenue" in sql:
            return pd.DataFrame([{"order_revenue": 100.0, "transaction_revenue": 99.0}])
        # for completeness null checks
        if "null_percentage" in sql:
            return pd.DataFrame([{"column_name": "col1", "total_rows": 10, "null_count": 1, "null_percentage": 10.0}])
        if "order_skus" in sql and "listing_skus" in sql:
            # SKU consistency query
            return pd.DataFrame([{"order_skus": 5, "listing_skus": 4}])
        # schema queries
        if "FROM system.tables" in sql:
            return [("amz_order",), ("amz_transaction",)]
        # timeliness queries with MIN/MAX/unique_dates
        if "MIN(" in sql and "MAX(" in sql and "unique_dates" in sql:
            return pd.DataFrame([{"min_date": "2025-10-01", "max_date": "2025-10-30", "unique_dates": 30}])
        if "negative_count" in sql:
            return pd.DataFrame([{"negative_count": 0}])
        if "outlier_count" in sql:
            return pd.DataFrame([{"outlier_count": 1}])
        # generic fallback
        return pd.DataFrame([])

    def get_table_schema(self, table_name: str):
        # return minimal schema
        return {"table_name": table_name, "columns": [{"name": "col1", "type": "String"}]}

    def list_tables(self):
        return ["amz_order", "amz_transaction"]


@pytest.fixture
def dummy_checker():
    client = DummyClient()
    return DataQualityChecker(client)


def test_check_completeness(dummy_checker):
    result = dummy_checker.check_completeness("ic_agent.amz_order")
    assert result["table"] == "ic_agent.amz_order"
    assert result["total_rows"] == 10
    assert result["overall_completeness"] == pytest.approx(90.0)
    assert isinstance(result["columns"], list)


def test_check_consistency(dummy_checker):
    result = dummy_checker.check_consistency()
    assert "orphan_orders" in result
    assert result["orphan_orders"]["count"] == 2
    assert result["revenue_reconciliation"]["difference_pct"] == pytest.approx(1.0)


def test_run_all_checks(dummy_checker):
    report = dummy_checker.run_all_checks()
    assert "completeness" in report
    assert "consistency" in report
    assert "timeliness" in report
    assert "accuracy" in report
    # completeness should contain entries for each table
    assert "amz_order" in report["completeness"]


def test_generate_quality_report_function():
    client = DummyClient()
    from src.uds.maintenance.quality_checks import generate_quality_report
    # generation should complete and return a dict
    report = generate_quality_report(client, output_file="test.json")
    assert isinstance(report, dict)

    # ensure report is JSON-serializable by coercing numpy types
    import json
    json.dumps(report, default=lambda o: int(o) if hasattr(o, 'item') else str(o))

