"""
Unit tests for UDS schema and visualization tools.

Tests ListTablesTool, DescribeTableTool, GetTableRelationshipsTool,
SearchColumnsTool, CreateChartTool, CreateDashboardTool, ExportVisualizationTool.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_toolkit.tools import ToolResult
from ai_toolkit.errors import ValidationError

from src.uds.tools import (
    ListTablesTool,
    DescribeTableTool,
    GetTableRelationshipsTool,
    SearchColumnsTool,
    CreateChartTool,
    CreateDashboardTool,
    ExportVisualizationTool,
    UDSToolRegistry,
)


# ---------------------------------------------------------------------------
# Schema Tools - ListTablesTool
# ---------------------------------------------------------------------------


class TestListTablesTool:
    """Tests for ListTablesTool."""

    def test_list_tables_returns_tool_result(self):
        """ListTablesTool.execute returns ToolResult."""
        tool = ListTablesTool()
        result = tool.execute(include_stats=True)
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_list_tables_has_tables(self):
        """ListTablesTool returns tables with metadata."""
        tool = ListTablesTool()
        result = tool.execute(include_stats=True)
        assert result.success
        data = result.output
        assert "tables" in data
        assert "total_tables" in data
        assert data["total_tables"] >= 1
        assert len(data["tables"]) == data["total_tables"]

    def test_list_tables_with_stats(self):
        """ListTablesTool with include_stats=True includes row_count, date_range."""
        tool = ListTablesTool()
        result = tool.execute(include_stats=True)
        assert result.success
        first_table = result.output["tables"][0]
        assert "name" in first_table
        assert "description" in first_table
        assert "row_count" in first_table
        assert "date_range" in first_table
        assert "primary_key" in first_table

    def test_list_tables_without_stats(self):
        """ListTablesTool with include_stats=False omits stats."""
        tool = ListTablesTool()
        result = tool.execute(include_stats=False)
        assert result.success
        first_table = result.output["tables"][0]
        assert "name" in first_table
        assert "description" in first_table
        assert "row_count" not in first_table
        assert "date_range" not in first_table


# ---------------------------------------------------------------------------
# Schema Tools - DescribeTableTool
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_uds_client():
    """Mock UDSClient for DescribeTableTool."""
    client = MagicMock()
    client.query.return_value = pd.DataFrame(
        [{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}]
    )
    return client


class TestDescribeTableTool:
    """Tests for DescribeTableTool."""

    @patch("src.uds.tools.schema_tools.UDSClient")
    def test_describe_table_success(self, mock_client_cls, mock_uds_client):
        """DescribeTableTool returns schema for valid table."""
        mock_client_cls.return_value = mock_uds_client
        tool = DescribeTableTool()
        result = tool.execute(table_name="amz_order", include_sample=True)
        assert isinstance(result, ToolResult)
        assert result.success
        data = result.output
        assert data["table_name"] == "amz_order"
        assert "columns" in data
        assert "description" in data
        assert "sample_data" in data
        assert len(data["sample_data"]) == 2

    @patch("src.uds.tools.schema_tools.UDSClient")
    def test_describe_table_not_found(self, mock_client_cls, mock_uds_client):
        """DescribeTableTool returns error for unknown table."""
        mock_client_cls.return_value = mock_uds_client
        tool = DescribeTableTool()
        result = tool.execute(table_name="nonexistent_table_xyz")
        assert result.success is False
        assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# Schema Tools - GetTableRelationshipsTool
# ---------------------------------------------------------------------------


class TestGetTableRelationshipsTool:
    """Tests for GetTableRelationshipsTool."""

    def test_get_relationships_all(self):
        """GetTableRelationshipsTool returns all relationships when no table specified."""
        tool = GetTableRelationshipsTool()
        result = tool.execute()
        assert isinstance(result, ToolResult)
        assert result.success
        data = result.output
        assert "relationships" in data
        assert "total_relationships" in data

    def test_get_relationships_for_table(self):
        """GetTableRelationshipsTool returns relationships for specific table."""
        tool = GetTableRelationshipsTool()
        result = tool.execute(table_name="amz_order")
        assert result.success
        data = result.output
        assert "table" in data
        assert data["table"] == "amz_order"
        assert "relationships" in data

    def test_get_relationships_table_not_found(self):
        """GetTableRelationshipsTool returns error for unknown table."""
        tool = GetTableRelationshipsTool()
        result = tool.execute(table_name="nonexistent_xyz")
        assert result.success is False


# ---------------------------------------------------------------------------
# Schema Tools - SearchColumnsTool
# ---------------------------------------------------------------------------


class TestSearchColumnsTool:
    """Tests for SearchColumnsTool."""

    def test_search_columns_by_name(self):
        """SearchColumnsTool finds columns by name."""
        tool = SearchColumnsTool()
        result = tool.execute(search_term="asin", search_in="name")
        assert result.success
        data = result.output
        assert "matches" in data
        assert "total_matches" in data
        assert "search_term" in data
        assert data["search_term"] == "asin"

    def test_search_columns_by_description(self):
        """SearchColumnsTool finds columns by description."""
        tool = SearchColumnsTool()
        result = tool.execute(search_term="amazon", search_in="description")
        assert result.success
        assert "matches" in result.output

    def test_search_columns_both(self):
        """SearchColumnsTool searches both name and description."""
        tool = SearchColumnsTool()
        result = tool.execute(search_term="date", search_in="both")
        assert result.success
        for m in result.output.get("matches", []):
            assert "table" in m
            assert "column_name" in m
            assert "type" in m

    def test_search_columns_requires_term(self):
        """SearchColumnsTool validates search_term."""
        tool = SearchColumnsTool()
        with pytest.raises(ValidationError):
            tool.validate_parameters(search_term="")


# ---------------------------------------------------------------------------
# Visualization Tools - CreateChartTool
# ---------------------------------------------------------------------------


class TestCreateChartTool:
    """Tests for CreateChartTool."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for chart tests."""
        return {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 15, 25, 30],
        }

    def test_create_line_chart(self, sample_data):
        """CreateChartTool creates line chart."""
        tool = CreateChartTool()
        result = tool.execute(
            data=sample_data,
            chart_type="line",
            x_column="x",
            y_column="y",
            title="Test Line",
        )
        assert result.success
        assert "chart_html" in result.output
        assert "plotly" in result.output["chart_html"].lower() or "chart" in result.output["chart_html"].lower()
        assert result.output["chart_type"] == "line"

    def test_create_bar_chart(self, sample_data):
        """CreateChartTool creates bar chart."""
        tool = CreateChartTool()
        result = tool.execute(
            data=sample_data,
            chart_type="bar",
            x_column="x",
            y_column="y",
        )
        assert result.success
        assert result.output["chart_type"] == "bar"

    def test_create_pie_chart(self):
        """CreateChartTool creates pie chart."""
        tool = CreateChartTool()
        data = {"category": ["A", "B", "C"], "value": [10, 20, 30]}
        result = tool.execute(
            data=data,
            chart_type="pie",
            x_column="category",
            y_column="value",
        )
        assert result.success
        assert result.output["chart_type"] == "pie"

    def test_create_scatter_chart(self, sample_data):
        """CreateChartTool creates scatter chart."""
        tool = CreateChartTool()
        result = tool.execute(
            data=sample_data,
            chart_type="scatter",
            x_column="x",
            y_column="y",
        )
        assert result.success
        assert result.output["chart_type"] == "scatter"

    def test_create_chart_invalid_type(self, sample_data):
        """CreateChartTool rejects invalid chart type."""
        tool = CreateChartTool()
        result = tool.execute(
            data=sample_data,
            chart_type="invalid",
            x_column="x",
            y_column="y",
        )
        assert result.success is False

    def test_create_chart_empty_data(self, sample_data):
        """CreateChartTool rejects empty data."""
        tool = CreateChartTool()
        result = tool.execute(
            data={},
            chart_type="line",
            x_column="x",
            y_column="y",
        )
        assert result.success is False

    def test_create_chart_returns_figure_json(self, sample_data):
        """CreateChartTool returns figure_json for export."""
        tool = CreateChartTool()
        result = tool.execute(
            data=sample_data,
            chart_type="line",
            x_column="x",
            y_column="y",
        )
        assert result.success
        assert "figure_json" in result.output
        # Should be valid JSON
        parsed = json.loads(result.output["figure_json"])
        assert "data" in parsed or "layout" in parsed


# ---------------------------------------------------------------------------
# Visualization Tools - CreateDashboardTool
# ---------------------------------------------------------------------------


class TestCreateDashboardTool:
    """Tests for CreateDashboardTool."""

    @pytest.fixture
    def chart_configs(self):
        """Sample chart configs for dashboard."""
        return [
            {
                "data": {"x": [1, 2, 3], "y": [4, 5, 6]},
                "type": "line",
                "x_column": "x",
                "y_column": "y",
                "title": "Chart 1",
            },
            {
                "data": {"a": [1, 2], "b": [10, 20]},
                "type": "bar",
                "x_column": "a",
                "y_column": "b",
                "title": "Chart 2",
            },
        ]

    def test_create_dashboard(self, chart_configs):
        """CreateDashboardTool creates dashboard."""
        tool = CreateDashboardTool()
        result = tool.execute(charts=chart_configs, title="Test Dashboard", layout="2x2")
        assert result.success
        assert "dashboard_html" in result.output
        assert result.output["chart_count"] == 2
        assert result.output["layout"] == "2x2"

    def test_create_dashboard_empty_charts(self):
        """CreateDashboardTool rejects empty charts."""
        tool = CreateDashboardTool()
        result = tool.execute(charts=[], layout="2x2")
        assert result.success is False


# ---------------------------------------------------------------------------
# Visualization Tools - ExportVisualizationTool
# ---------------------------------------------------------------------------


class TestExportVisualizationTool:
    """Tests for ExportVisualizationTool."""

    def test_export_with_figure_json(self):
        """ExportVisualizationTool exports using figure_json when kaleido is installed."""
        try:
            import kaleido  # noqa: F401
        except ImportError:
            pytest.skip("kaleido not installed - run: pip install kaleido")

        # Create a chart first to get figure_json
        chart_tool = CreateChartTool()
        chart_result = chart_tool.execute(
            data={"x": [1, 2, 3], "y": [4, 5, 6]},
            chart_type="line",
            x_column="x",
            y_column="y",
        )
        assert chart_result.success
        figure_json = chart_result.output["figure_json"]

        export_tool = ExportVisualizationTool()
        result = export_tool.execute(
            figure_json=figure_json,
            format="png",
            filename="test_export.png",
        )
        assert result.success
        assert result.output["format"] == "png"
        assert "path" in result.output
        # Verify file was created
        path = Path(result.output["path"])
        assert path.exists()
        path.unlink()  # cleanup

    def test_export_requires_input(self):
        """ExportVisualizationTool requires chart_html or figure_json."""
        tool = ExportVisualizationTool()
        with pytest.raises(ValidationError):
            tool.validate_parameters(chart_html="", figure_json="", format="png")


# ---------------------------------------------------------------------------
# UDSToolRegistry
# ---------------------------------------------------------------------------


class TestUDSToolRegistry:
    """Tests for UDSToolRegistry (Task 012 schema + viz tools)."""

    def test_get_schema_tools(self):
        """UDSToolRegistry.get_schema_tools returns 4 schema tools."""
        tools = UDSToolRegistry.get_schema_tools()
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "list_tables" in names
        assert "describe_table" in names
        assert "get_table_relationships" in names
        assert "search_columns" in names

    def test_get_visualization_tools(self):
        """UDSToolRegistry.get_visualization_tools returns 3 viz tools."""
        tools = UDSToolRegistry.get_visualization_tools()
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "create_chart" in names
        assert "create_dashboard" in names
        assert "export_visualization" in names

    def test_schema_plus_viz_total_seven(self):
        """Task 012 delivers 7 tools: 4 schema + 3 visualization."""
        schema = UDSToolRegistry.get_schema_tools()
        viz = UDSToolRegistry.get_visualization_tools()
        assert len(schema) + len(viz) == 7
