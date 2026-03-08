"""
Tests for UDS Task Planner.

Tests task planning logic including:
- Simple vs complex query identification
- Query decomposition
- Tool mapping
- Dependency resolution
- Parameter extraction
"""

import pytest
from unittest.mock import Mock

from src.uds.task_planner import UDSTaskPlanner, TaskPlan, Subtask
from src.uds.intent_classifier import IntentResult, IntentDomain


class TestSimpleQueryDetection:
    """Test simple vs complex query detection."""

    def test_simple_sales_query(self):
        """Test simple sales query is detected."""
        planner = UDSTaskPlanner()
        query = "What were total sales in October?"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.95,
            keywords=['sales', 'october'],
            suggested_tools=[],
            reasoning="Sales query with date"
        )

        is_simple = planner._is_simple_query(query, intent)
        assert is_simple is True

    def test_simple_inventory_query(self):
        """Test simple inventory query is detected."""
        planner = UDSTaskPlanner()
        query = "Show me current inventory levels"
        intent = IntentResult(
            primary_domain=IntentDomain.INVENTORY,
            secondary_domains=[],
            confidence=0.90,
            keywords=['inventory'],
            suggested_tools=[],
            reasoning="Inventory query"
        )

        is_simple = planner._is_simple_query(query, intent)
        assert is_simple is True

    def test_complex_query_with_and(self):
        """Test complex query with 'and' is detected."""
        planner = UDSTaskPlanner()
        query = "Show me sales and inventory"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[IntentDomain.INVENTORY],
            confidence=0.85,
            keywords=['sales', 'inventory'],
            suggested_tools=[],
            reasoning="Multi-domain query"
        )

        is_simple = planner._is_simple_query(query, intent)
        assert is_simple is False

    def test_complex_query_with_top_and_inventory(self):
        """Test complex query with 'top' and 'inventory'."""
        planner = UDSTaskPlanner()
        query = "Top 10 products with their inventory"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[IntentDomain.INVENTORY],
            confidence=0.88,
            keywords=['top', 'products', 'inventory'],
            suggested_tools=[],
            reasoning="Product and inventory query"
        )

        is_simple = planner._is_simple_query(query, intent)
        assert is_simple is False

    def test_simple_comparison_query(self):
        """Test simple comparison query (ComparisonTool handles it)."""
        planner = UDSTaskPlanner()
        query = "Compare Q3 vs Q4 sales"
        intent = IntentResult(
            primary_domain=IntentDomain.COMPARISON,
            secondary_domains=[],
            confidence=0.92,
            keywords=['compare', 'q3', 'q4'],
            suggested_tools=[],
            reasoning="Comparison query"
        )

        is_simple = planner._is_simple_query(query, intent)
        assert is_simple is True


class TestSimplePlanCreation:
    """Test creation of simple execution plans."""

    def test_create_simple_sales_plan(self):
        """Test creating simple sales plan."""
        planner = UDSTaskPlanner()
        query = "What were total sales in October?"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.95,
            keywords=['sales', 'october'],
            suggested_tools=[],
            reasoning="Sales query with date"
        )

        plan = planner.create_plan(query, intent)

        assert isinstance(plan, TaskPlan)
        assert plan.query == query
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].tool_name == 'SalesTrendTool'
        assert plan.execution_order == ['task_1']

    def test_create_simple_inventory_plan(self):
        """Test creating simple inventory plan."""
        planner = UDSTaskPlanner()
        query = "Show me low stock items"
        intent = IntentResult(
            primary_domain=IntentDomain.INVENTORY,
            secondary_domains=[],
            confidence=0.90,
            keywords=['low', 'stock'],
            suggested_tools=[],
            reasoning="Low stock query"
        )

        plan = planner.create_plan(query, intent)

        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].tool_name == 'InventoryAnalysisTool'

    def test_simple_plan_parameters(self):
        """Test parameter extraction in simple plan."""
        planner = UDSTaskPlanner()
        query = "Top 10 products by revenue"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[],
            confidence=0.88,
            keywords=['top', 'products'],
            suggested_tools=[],
            reasoning="Top products query"
        )

        plan = planner.create_plan(query, intent)

        params = plan.subtasks[0].parameters
        assert 'limit' in params
        assert params['limit'] == 10
        assert 'metric' in params
        assert params['metric'] == 'revenue'


class TestComplexPlanCreation:
    """Test creation of complex execution plans."""

    def test_create_complex_plan_top_inventory(self):
        """Test creating complex plan for top products with inventory."""
        planner = UDSTaskPlanner()
        query = "Top 10 products with their inventory"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[IntentDomain.INVENTORY],
            confidence=0.88,
            keywords=['top', 'products', 'inventory'],
            suggested_tools=[],
            reasoning="Product and inventory query"
        )

        plan = planner.create_plan(query, intent)

        assert isinstance(plan, TaskPlan)
        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].tool_name == 'ProductPerformanceTool'
        assert plan.subtasks[1].tool_name == 'InventoryAnalysisTool'
        assert plan.subtasks[2].tool_name == 'ExecuteQueryTool'

    def test_create_complex_plan_compare_show(self):
        """Test creating complex plan for compare and show."""
        planner = UDSTaskPlanner()
        query = "Compare Q3 vs Q4 and show chart"
        intent = IntentResult(
            primary_domain=IntentDomain.COMPARISON,
            secondary_domains=[],
            confidence=0.90,
            keywords=['compare', 'chart'],
            suggested_tools=[],
            reasoning="Compare and chart query"
        )

        plan = planner.create_plan(query, intent)

        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].tool_name == 'ComparisonTool'

    def test_create_complex_plan_sales_product(self):
        """Test creating complex plan for sales and products."""
        planner = UDSTaskPlanner()
        query = "Sales trends and product performance dashboard"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[IntentDomain.PRODUCT],
            confidence=0.85,
            keywords=['sales', 'product', 'dashboard'],
            suggested_tools=[],
            reasoning="Sales and product dashboard query"
        )

        plan = planner.create_plan(query, intent)

        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].tool_name == 'SalesTrendTool'
        assert plan.subtasks[1].tool_name == 'ProductPerformanceTool'
        assert plan.subtasks[2].tool_name == 'CreateChartTool'


class TestQueryDecomposition:
    """Test query decomposition into steps."""

    def test_decompose_top_inventory(self):
        """Test decomposing 'top products with inventory'."""
        planner = UDSTaskPlanner()
        query = "Top 10 products with their inventory"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[IntentDomain.INVENTORY],
            confidence=0.88,
            keywords=['top', 'products', 'inventory'],
            suggested_tools=[],
            reasoning="Product and inventory query"
        )

        steps = planner._decompose_query(query, intent)

        assert len(steps) == 3
        assert 'top products' in steps[0].lower()
        assert 'inventory' in steps[1].lower()
        assert 'combine' in steps[2].lower()

    def test_decompose_compare_show(self):
        """Test decomposing 'compare and show'."""
        planner = UDSTaskPlanner()
        query = "Compare Q3 vs Q4 and show chart"
        intent = IntentResult(
            primary_domain=IntentDomain.COMPARISON,
            secondary_domains=[],
            confidence=0.90,
            keywords=['compare', 'chart'],
            suggested_tools=[],
            reasoning="Compare and chart query"
        )

        steps = planner._decompose_query(query, intent)

        assert len(steps) == 2
        assert 'compare' in steps[0].lower()
        assert 'visualiz' in steps[1].lower() or 'chart' in steps[1].lower()

    def test_decompose_sales_product(self):
        """Test decomposing 'sales and products'."""
        planner = UDSTaskPlanner()
        query = "Sales trends and product performance dashboard"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[IntentDomain.PRODUCT],
            confidence=0.85,
            keywords=['sales', 'product', 'dashboard'],
            suggested_tools=[],
            reasoning="Sales and product dashboard query"
        )

        steps = planner._decompose_query(query, intent)

        assert len(steps) == 3
        assert 'sales' in steps[0].lower()
        assert 'product' in steps[1].lower()
        assert 'dashboard' in steps[2].lower()


class TestToolMapping:
    """Test mapping steps to tools."""

    def test_map_product_performance(self):
        """Test mapping to ProductPerformanceTool."""
        planner = UDSTaskPlanner()
        steps = ["Get top products by revenue"]
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[],
            confidence=0.88,
            keywords=['top', 'products'],
            suggested_tools=[],
            reasoning="Product query"
        )

        subtasks = planner._map_to_tools(steps, intent)

        assert len(subtasks) == 1
        assert subtasks[0].tool_name == 'ProductPerformanceTool'

    def test_map_inventory_analysis(self):
        """Test mapping to InventoryAnalysisTool."""
        planner = UDSTaskPlanner()
        steps = ["Get inventory levels"]
        intent = IntentResult(
            primary_domain=IntentDomain.INVENTORY,
            secondary_domains=[],
            confidence=0.90,
            keywords=['inventory'],
            suggested_tools=[],
            reasoning="Inventory query"
        )

        subtasks = planner._map_to_tools(steps, intent)

        assert len(subtasks) == 1
        assert subtasks[0].tool_name == 'InventoryAnalysisTool'

    def test_map_sales_trend(self):
        """Test mapping to SalesTrendTool."""
        planner = UDSTaskPlanner()
        steps = ["Analyze sales trends"]
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.85,
            keywords=['sales'],
            suggested_tools=[],
            reasoning="Sales query"
        )

        subtasks = planner._map_to_tools(steps, intent)

        assert len(subtasks) == 1
        assert subtasks[0].tool_name == 'SalesTrendTool'


class TestDependencyResolution:
    """Test dependency resolution and execution order."""

    def test_resolve_no_dependencies(self):
        """Test resolving tasks with no dependencies."""
        planner = UDSTaskPlanner()
        subtasks = [
            Subtask(id="task_1", description="Task 1", tool_name="Tool1", parameters={}, dependencies=[]),
            Subtask(id="task_2", description="Task 2", tool_name="Tool2", parameters={}, dependencies=[]),
        ]

        order = planner._resolve_dependencies(subtasks)

        assert len(order) == 2
        assert 'task_1' in order
        assert 'task_2' in order

    def test_resolve_sequential_dependencies(self):
        """Test resolving sequential dependencies."""
        planner = UDSTaskPlanner()
        subtasks = [
            Subtask(id="task_1", description="Task 1", tool_name="Tool1", parameters={}, dependencies=[]),
            Subtask(id="task_2", description="Task 2", tool_name="Tool2", parameters={}, dependencies=["task_1"]),
            Subtask(id="task_3", description="Task 3", tool_name="Tool3", parameters={}, dependencies=["task_2"]),
        ]

        order = planner._resolve_dependencies(subtasks)

        assert order == ['task_1', 'task_2', 'task_3']

    def test_resolve_complex_dependencies(self):
        """Test resolving complex dependency graph."""
        planner = UDSTaskPlanner()
        subtasks = [
            Subtask(id="task_1", description="Task 1", tool_name="Tool1", parameters={}, dependencies=[]),
            Subtask(id="task_2", description="Task 2", tool_name="Tool2", parameters={}, dependencies=["task_1"]),
            Subtask(id="task_3", description="Task 3", tool_name="Tool3", parameters={}, dependencies=["task_1"]),
            Subtask(id="task_4", description="Task 4", tool_name="Tool4", parameters={}, dependencies=["task_2", "task_3"]),
        ]

        order = planner._resolve_dependencies(subtasks)

        assert order.index('task_1') < order.index('task_2')
        assert order.index('task_1') < order.index('task_3')
        assert order.index('task_2') < order.index('task_4')
        assert order.index('task_3') < order.index('task_4')


class TestParameterExtraction:
    """Test parameter extraction from queries."""

    def test_extract_date_range(self):
        """Test extracting date range."""
        planner = UDSTaskPlanner()
        query = "Sales from 2025-10-01 to 2025-10-31"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.90,
            keywords=['sales'],
            suggested_tools=[],
            reasoning="Sales with date range"
        )

        params = planner._extract_parameters(query, intent)

        assert 'start_date' in params
        assert params['start_date'] == '2025-10-01'
        assert 'end_date' in params
        assert params['end_date'] == '2025-10-31'

    def test_extract_single_date(self):
        """Test extracting single date."""
        planner = UDSTaskPlanner()
        query = "Inventory as of 2025-10-15"
        intent = IntentResult(
            primary_domain=IntentDomain.INVENTORY,
            secondary_domains=[],
            confidence=0.90,
            keywords=['inventory'],
            suggested_tools=[],
            reasoning="Inventory with date"
        )

        params = planner._extract_parameters(query, intent)

        assert 'as_of_date' in params
        assert params['as_of_date'] == '2025-10-15'

    def test_extract_no_date_defaults(self):
        """Test default dates when no date in query."""
        planner = UDSTaskPlanner()
        query = "Total sales"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.85,
            keywords=['sales'],
            suggested_tools=[],
            reasoning="Sales query"
        )

        params = planner._extract_parameters(query, intent)

        assert 'start_date' in params
        assert params['start_date'] == '2025-10-01'
        assert 'end_date' in params
        assert params['end_date'] == '2025-10-31'

    def test_extract_limit(self):
        """Test extracting limit."""
        planner = UDSTaskPlanner()
        query = "Top 10 products"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[],
            confidence=0.88,
            keywords=['top', 'products'],
            suggested_tools=[],
            reasoning="Top products query"
        )

        params = planner._extract_parameters(query, intent)

        assert 'limit' in params
        assert params['limit'] == 10

    def test_extract_metric_revenue(self):
        """Test extracting revenue metric."""
        planner = UDSTaskPlanner()
        query = "Total revenue"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.90,
            keywords=['revenue'],
            suggested_tools=[],
            reasoning="Revenue query"
        )

        params = planner._extract_parameters(query, intent)

        assert 'metric' in params
        assert params['metric'] == 'revenue'

    def test_extract_metric_units(self):
        """Test extracting units metric."""
        planner = UDSTaskPlanner()
        query = "Total units sold"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.88,
            keywords=['units'],
            suggested_tools=[],
            reasoning="Units query"
        )

        params = planner._extract_parameters(query, intent)

        assert 'metric' in params
        assert params['metric'] == 'units'

    def test_extract_low_stock_threshold(self):
        """Test extracting low stock threshold."""
        planner = UDSTaskPlanner()
        query = "Show low stock items"
        intent = IntentResult(
            primary_domain=IntentDomain.INVENTORY,
            secondary_domains=[],
            confidence=0.90,
            keywords=['low', 'stock'],
            suggested_tools=[],
            reasoning="Low stock query"
        )

        params = planner._extract_parameters(query, intent)

        assert 'low_stock_threshold' in params
        assert params['low_stock_threshold'] == 10


class TestToolSelection:
    """Test tool selection for domains."""

    def test_get_tool_for_sales(self):
        """Test getting tool for sales domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.SALES)
        assert tool == 'SalesTrendTool'

    def test_get_tool_for_inventory(self):
        """Test getting tool for inventory domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.INVENTORY)
        assert tool == 'InventoryAnalysisTool'

    def test_get_tool_for_financial(self):
        """Test getting tool for financial domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.FINANCIAL)
        assert tool == 'FinancialSummaryTool'

    def test_get_tool_for_product(self):
        """Test getting tool for product domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.PRODUCT)
        assert tool == 'ProductPerformanceTool'

    def test_get_tool_for_comparison(self):
        """Test getting tool for comparison domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.COMPARISON)
        assert tool == 'ComparisonTool'

    def test_get_tool_for_general(self):
        """Test getting tool for general domain."""
        planner = UDSTaskPlanner()
        tool = planner._get_tool_for_domain(IntentDomain.GENERAL)
        assert tool == 'ListTablesTool'


class TestIntegration:
    """Integration tests for complete planning workflow."""

    def test_simple_query_end_to_end(self):
        """Test complete simple query planning."""
        planner = UDSTaskPlanner()
        query = "What were total sales in October?"
        intent = IntentResult(
            primary_domain=IntentDomain.SALES,
            secondary_domains=[],
            confidence=0.95,
            keywords=['sales', 'october'],
            suggested_tools=[],
            reasoning="Sales query with date"
        )

        plan = planner.create_plan(query, intent)

        assert plan.query == query
        assert len(plan.subtasks) == 1
        assert plan.execution_order == ['task_1']
        assert plan.subtasks[0].tool_name == 'SalesTrendTool'
        assert 'start_date' in plan.subtasks[0].parameters
        assert 'end_date' in plan.subtasks[0].parameters

    def test_complex_query_end_to_end(self):
        """Test complete complex query planning."""
        planner = UDSTaskPlanner()
        query = "Top 10 products with their inventory"
        intent = IntentResult(
            primary_domain=IntentDomain.PRODUCT,
            secondary_domains=[IntentDomain.INVENTORY],
            confidence=0.88,
            keywords=['top', 'products', 'inventory'],
            suggested_tools=[],
            reasoning="Product and inventory query"
        )

        plan = planner.create_plan(query, intent)

        assert plan.query == query
        assert len(plan.subtasks) == 3
        assert len(plan.execution_order) == 3
        assert plan.subtasks[0].tool_name == 'ProductPerformanceTool'
        assert plan.subtasks[1].tool_name == 'InventoryAnalysisTool'
        assert plan.subtasks[2].tool_name == 'ExecuteQueryTool'
        assert plan.subtasks[0].dependencies == []
        assert plan.subtasks[1].dependencies == ['task_1']
        assert plan.subtasks[2].dependencies == ['task_2']
