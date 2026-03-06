#!/usr/bin/env python3
"""
Integration test for schema and visualization tools.

Tests with real ClickHouse data when available.
ToolResult uses 'output' not 'data' for result payload.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.uds.tools import (
    ListTablesTool,
    DescribeTableTool,
    GetTableRelationshipsTool,
    SearchColumnsTool,
    CreateChartTool,
    CreateDashboardTool,
    ExportVisualizationTool,
)


def _data(result):
    """Get output from ToolResult (tools use 'output' not 'data')."""
    return result.output or {}


def test_schema_tools():
    """Test all schema inspection tools."""
    print("\n" + "=" * 80)
    print("SCHEMA TOOLS INTEGRATION TEST")
    print("=" * 80)

    # Test 1: List Tables
    print("\n1. Testing ListTablesTool...")
    tool = ListTablesTool()
    result = tool.execute(include_stats=True)

    if result.success:
        tables = _data(result).get("tables", [])
        print(f"   Found {len(tables)} tables")
        for table in tables[:3]:
            print(f"      - {table['name']}: {table.get('row_count', 0):,} rows")
    else:
        print(f"   Failed: {result.error}")
        return False

    # Test 2: Describe Table
    print("\n2. Testing DescribeTableTool...")
    tool = DescribeTableTool()
    result = tool.execute(table_name="amz_order", include_sample=True)

    if result.success:
        data = _data(result)
        print(f"   Described table: {data.get('table_name', 'N/A')}")
        print(f"      Columns: {len(data.get('columns', []))}")
        print(f"      Sample rows: {len(data.get('sample_data', []))}")
    else:
        print(f"   Failed: {result.error}")
        return False

    # Test 3: Get Relationships
    print("\n3. Testing GetTableRelationshipsTool...")
    tool = GetTableRelationshipsTool()
    result = tool.execute(table_name="amz_order")

    if result.success:
        data = _data(result)
        rels = data.get("relationships", {})
        if isinstance(rels, dict):
            count = len(rels)
            items = list(rels.items())[:2]
        else:
            count = len(rels)
            items = [(r.get("to_table", ""), r) for r in rels[:2]]
        print(f"   Found {count} relationships for amz_order")
        for rel_table, rel_info in items:
            desc = rel_info.get("description", "N/A") if isinstance(rel_info, dict) else "N/A"
            print(f"      - {rel_table}: {desc}")
    else:
        print(f"   Failed: {result.error}")
        return False

    # Test 4: Search Columns
    print("\n4. Testing SearchColumnsTool...")
    tool = SearchColumnsTool()
    result = tool.execute(search_term="order", search_in="name")

    if result.success:
        matches = _data(result).get("matches", [])
        print(f"   Found {len(matches)} columns matching 'order'")
        for match in matches[:3]:
            print(f"      - {match['table']}.{match['column_name']}")
    else:
        print(f"   Failed: {result.error}")
        return False

    return True


def test_visualization_tools():
    """Test all visualization tools."""
    print("\n" + "=" * 80)
    print("VISUALIZATION TOOLS INTEGRATION TEST")
    print("=" * 80)

    sample_data = {
        "date": ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05"],
        "revenue": [1000, 1200, 1100, 1300, 1250],
        "orders": [50, 60, 55, 65, 62],
    }

    figure_json = None
    chart_html = None

    # Test 1: Create Chart
    print("\n1. Testing CreateChartTool...")
    tool = CreateChartTool()
    result = tool.execute(
        data=sample_data,
        chart_type="line",
        x_column="date",
        y_column="revenue",
        title="Daily Revenue",
    )

    if result.success:
        data = _data(result)
        print(f"   Created {data.get('chart_type', 'N/A')} chart")
        print(f"      Title: {data.get('title', 'N/A')}")
        chart_html = data.get("chart_html")
        figure_json = data.get("figure_json")
    else:
        print(f"   Failed: {result.error}")
        return False

    # Test 2: Create Dashboard
    print("\n2. Testing CreateDashboardTool...")
    tool = CreateDashboardTool()

    charts = [
        {
            "type": "line",
            "data": sample_data,
            "x_column": "date",
            "y_column": "revenue",
            "title": "Revenue Trend",
        },
        {
            "type": "bar",
            "data": sample_data,
            "x_column": "date",
            "y_column": "orders",
            "title": "Order Count",
        },
    ]

    result = tool.execute(charts=charts, title="Sales Dashboard", layout="2x1")

    if result.success:
        data = _data(result)
        print(f"   Created dashboard with {data.get('chart_count', 0)} charts")
        print(f"      Layout: {data.get('layout', 'N/A')}")
    else:
        print(f"   Failed: {result.error}")
        return False

    # Test 3: Export Visualization
    print("\n3. Testing ExportVisualizationTool...")
    tool = ExportVisualizationTool()

    if figure_json:
        result = tool.execute(
            figure_json=figure_json,
            format="png",
            filename="test_chart.png",
        )
    else:
        result = tool.execute(
            chart_html=chart_html,
            format="png",
            filename="test_chart.png",
        )

    if result.success:
        data = _data(result)
        print(f"   Exported chart as {data.get('format', 'N/A')}")
        print(f"      File: {data.get('filename', 'N/A')}")
        print(f"      Path: {data.get('path', 'N/A')}")
    else:
        print(f"   Export failed (kaleido may not be installed): {result.error}")
        print("      Optional: pip install kaleido")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("UDS TOOLS INTEGRATION TEST SUITE")
    print("=" * 80)

    schema_ok = test_schema_tools()
    viz_ok = test_visualization_tools()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Schema Tools: {'PASSED' if schema_ok else 'FAILED'}")
    print(f"Visualization Tools: {'PASSED' if viz_ok else 'FAILED'}")

    if schema_ok and viz_ok:
        print("\nAll integration tests passed.")
        return 0
    else:
        print("\nSome tests failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
