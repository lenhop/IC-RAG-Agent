"""Unit tests for analysis tools defined in src/uds/tools/analysis_tools.py."""

import pandas as pd
import pytest

from src.uds.tools.analysis_tools import (
    SalesTrendTool,
    InventoryAnalysisTool,
    ProductPerformanceTool,
    FinancialSummaryTool,
    ComparisonTool,
    ToolResult
)

# Bypass ABC abstract method requirements for testing by clearing
# the __abstractmethods__ sets.  This avoids needing to implement
# _get_parameters/validate_parameters which are provided by the
# ai_toolkit BaseTool in production.
for cls in [
    SalesTrendTool,
    InventoryAnalysisTool,
    ProductPerformanceTool,
    FinancialSummaryTool,
    ComparisonTool,
]:
    if hasattr(cls, "__abstractmethods__"):
        cls.__abstractmethods__ = set()


class DummyRegistry:
    def __init__(self):
        self.sales = self.Sales()
        self.inventory = self.Inventory()
        self.products = self.Products()
        self.financial = self.Financial()

    class Sales:
        def daily_sales_trend(self, start, end, marketplace=None):
            # create simple time series
            return pd.DataFrame({
                'date': ['2025-10-01', '2025-10-02'],
                'order_count': [10, 20],
                'total_revenue': [100.0, 200.0],
                'avg_order_value': [10.0, 10.0]
            })

        def sales_growth_rate(self, p1s, p1e, p2s, p2e):
            return pd.DataFrame([
                {'total_revenue': 100.0, 'order_count': 10},
                {'total_revenue': 150.0, 'order_count': 15}
            ])

    class Inventory:
        def current_inventory_levels(self, as_of_date=None):
            return pd.DataFrame({
                'sku': ['A', 'B'],
                'total_quantity': [5, 15]
            })

        def low_stock_alert(self, threshold=10, as_of_date=None):
            return pd.DataFrame({
                'sku': ['A'],
                'total_quantity': [5]
            })

    class Products:
        def top_selling_products(self, start, end, limit, metric):
            return pd.DataFrame({
                'sku': ['X'],
                'product_name': ['ProdX'],
                'total_revenue': [1000.0],
                'units_sold': [50],
                'avg_price': [20.0]
            })

    class Financial:
        def revenue_summary(self, start, end, group_by='day'):
            return pd.DataFrame({'gross_revenue': [500.0, 600.0]})

        def fee_analysis(self, start, end):
            return pd.DataFrame({'total_fees': [50.0, 60.0]})


def test_sales_trend_tool():
    # BaseTool is abstract; override required abstract methods for testing
    SalesTrendTool._get_parameters = lambda self: {}
    SalesTrendTool.validate_parameters = lambda self, params: None
    tool = SalesTrendTool()
    tool.registry = DummyRegistry()

    result: ToolResult = tool.execute('2025-10-01', '2025-10-02')
    assert result.success
    # ToolResult now stores returned values in `output` instead of `data`
    assert 'insights' in result.output
    assert result.output['insights']['total_revenue'] == 300.0
    assert 'summary' in result.output


def test_inventory_tool():
    InventoryAnalysisTool._get_parameters = lambda self: {}
    InventoryAnalysisTool.validate_parameters = lambda self, params: None
    tool = InventoryAnalysisTool()
    tool.registry = DummyRegistry()

    result = tool.execute(as_of_date='2025-10-02', low_stock_threshold=10)
    assert result.success
    assert result.output['inventory_summary']['total_skus'] == 2
    assert result.output['inventory_summary']['low_stock_items'] == 1
    assert 'alerts' in result.output


def test_product_performance_tool():
    ProductPerformanceTool._get_parameters = lambda self: {}
    ProductPerformanceTool.validate_parameters = lambda self, params: None
    tool = ProductPerformanceTool()
    tool.registry = DummyRegistry()

    result = tool.execute('2025-10-01', '2025-10-02')
    assert result.success
    assert result.output['insights']['total_products'] == 1
    assert result.output['insights']['revenue_concentration'] == 100.0


def test_financial_summary_tool():
    FinancialSummaryTool._get_parameters = lambda self: {}
    FinancialSummaryTool.validate_parameters = lambda self, params: None
    tool = FinancialSummaryTool()
    tool.registry = DummyRegistry()

    result = tool.execute('2025-10-01', '2025-10-02')
    assert result.success
    assert result.output['financial_metrics']['total_revenue'] == 1100.0
    assert 'profit_margin_pct' in result.output['financial_metrics']


def test_comparison_tool_period():
    ComparisonTool._get_parameters = lambda self: {}
    ComparisonTool.validate_parameters = lambda self, params: None
    tool = ComparisonTool()
    tool.registry = DummyRegistry()

    result = tool.execute(
        comparison_type='period',
        period1_start='2025-10-01',
        period1_end='2025-10-01',
        period2_start='2025-10-02',
        period2_end='2025-10-02'
    )
    assert result.success
    assert 'comparison' in result.output
    assert result.output['comparison']['growth']['revenue_pct'] == pytest.approx(50.0)


def test_comparison_tool_invalid():
    ComparisonTool._get_parameters = lambda self: {}
    ComparisonTool.validate_parameters = lambda self, params: None
    tool = ComparisonTool()
    tool.registry = DummyRegistry()
    result = tool.execute(comparison_type='unknown')
    assert not result.success
    assert 'not yet implemented' in result.error
