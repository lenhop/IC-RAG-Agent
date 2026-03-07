"""Unit tests for uds/statistics.py."""

import pandas as pd
import pytest

from src.uds.maintenance.statistics import StatisticalAnalyzer


class DummyClient:
    def query(self, sql: str, as_dataframe: bool = True, params=None):
        # simulate different query responses based on keywords
        if "COUNT(*) as count" in sql:
            return pd.DataFrame([{"count": 5}])
        if "MIN(" in sql and "MAX(" in sql:
            # return numeric stats
            return pd.DataFrame([{
                "min_val": 1,
                "max_val": 10,
                "avg_val": 5.5,
                "q25": 2,
                "median": 5,
                "q75": 8,
                "std_dev": 2.5
            }])
        if "COUNT(DISTINCT" in sql:
            return pd.DataFrame([{"unique_count": 3, "total_count": 5}])
        if "MIN(" in sql and "unique_dates" in sql:
            return pd.DataFrame([{"min_date": "2025-10-01", "max_date": "2025-10-30", "unique_dates": 30}])
        # fallback
        return pd.DataFrame([])

    def get_table_schema(self, table):
        # return a schema with two columns
        return {"columns": [{"name": "num", "type": "Int32"}, {"name": "cat", "type": "String"}, {"name": "dt", "type": "Date"}]}


def test_get_table_statistics():
    client = DummyClient()
    analyzer = StatisticalAnalyzer(client)
    stats = analyzer.get_table_statistics("amz_order")
    assert stats["row_count"] == 5
    assert "numeric_columns" in stats
    assert "categorical_columns" in stats
    assert "date_columns" in stats


def test_analyze_time_series():
    client = DummyClient()
    analyzer = StatisticalAnalyzer(client)
    # monkeypatch query to return a simple df
    df = pd.DataFrame({"date": ["2025-10-01", "2025-10-02"], "count": [10, 20]})
    client.query = lambda sql, as_dataframe=True, params=None: df
    analysis = analyzer.analyze_time_series("amz_order", "start_date", "quantity")
    assert analysis["total_days"] == 2
    assert analysis["trend"] == "increasing"


def test_find_correlations():
    client = DummyClient()
    analyzer = StatisticalAnalyzer(client)
    # create a small dataframe with numeric correlation
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    client.query = lambda sql, as_dataframe=True, params=None: df
    corr = analyzer.find_correlations("amz_order", ["a", "b"])
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (2, 2)
