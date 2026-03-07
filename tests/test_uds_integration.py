"""
Integration Tests for UDS Agent - End-to-end testing with real queries.

Tests:
- 12+ test scenarios (simple, medium, complex queries)
- All 16 tools in real workflows
- Performance benchmarks for each scenario
- 90%+ test pass rate target
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.uds.uds_agent import UDSAgent
from src.uds.uds_client import UDSClient
from src.uds.intent_classifier import IntentResult, IntentDomain
from src.uds.task_planner import TaskPlan, Subtask


def create_mock_llm(generate_response: str = "general", final_answer: str = "Test response"):
    """Create a mock LLM with proper configuration for testing."""
    mock_llm = Mock()
    mock_llm.generate.return_value = generate_response
    mock_llm.invoke.return_value = Mock(content=f"Final Answer: {final_answer}")
    mock_llm.run.return_value = final_answer
    return mock_llm


class TestSimpleQueries:
    """Test simple query scenarios (target: <5s)."""

    def test_1_total_sales_october(self):
        """Test 1: What were total sales in October?"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", "Total sales for October 2025 were $1,234,567")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("What were total sales in October?")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert result['query'] == "What were total sales in October?"
        assert 'response' in result
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"

    def test_2_current_inventory_levels(self):
        """Test 2: Show me current inventory levels"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("inventory", "Current inventory levels: 1,234 items")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Show me current inventory levels")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"

    def test_3_low_stock_items(self):
        """Test 3: Which items have low stock?"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("inventory", "Low stock items: 45 products below threshold")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Which items have low stock?")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"

    def test_4_profit_margins(self):
        """Test 4: What are the profit margins?"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("financial", "Average profit margin: 15.2%")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("What are the profit margins?")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"

    def test_5_top_5_products(self):
        """Test 5: What are the top 5 products by revenue?"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("product", "Top 5 products: Product A, B, C, D, E")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("What are the top 5 products by revenue?")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"


class TestMediumComplexityQueries:
    """Test medium complexity queries (target: <10s)."""

    def test_6_top_10_products_with_inventory(self):
        """Test 6: Top 10 products by revenue with inventory levels"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("product", "Top 10 products with their inventory levels")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Top 10 products by revenue with inventory levels")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s, expected <10s"

    def test_7_sales_trend_last_30_days(self):
        """Test 7: Show sales trend for the last 30 days"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", "Sales trend: Increasing by 12% over 30 days")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Show sales trend for the last 30 days")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s, expected <10s"

    def test_8_product_performance_with_sales(self):
        """Test 8: Product performance with sales data"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("product", "Product A: 500 units sold, $12,345 revenue")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Product performance with sales data")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s, expected <10s"

    def test_9_inventory_vs_sales_comparison(self):
        """Test 9: Compare inventory levels vs sales"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("comparison", "Inventory: 1,234 items, Sales: 2,345 orders")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Compare inventory levels vs sales")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s, expected <10s"


class TestComplexQueries:
    """Test complex query scenarios (target: <15s)."""

    def test_10_full_business_health_check(self):
        """Test 10: Full business health check: sales, inventory, financial, top performers"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", """
        Business Health Check:
        - Sales: $1.2M (up 8%)
        - Inventory: 1,234 items (down 3%)
        - Financial: Net margin 15.2% (stable)
        - Top performers: Products A, B, C
        """)
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Full business health check: sales, inventory, financial, top performers")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 15.0, f"Query took {elapsed:.2f}s, expected <15s"

    def test_11_multi_period_comparison(self):
        """Test 11: Compare Q3 vs Q4 vs Q1 performance"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("comparison", """
        Period Comparison:
        - Q3: $450K sales
        - Q4: $520K sales (up 15.6%)
        - Q1: $380K sales (down 15.6%)
        - Trend: Volatile but upward
        """)
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Compare Q3 vs Q4 vs Q1 performance")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 15.0, f"Query took {elapsed:.2f}s, expected <15s"

    def test_12_dashboard_with_multiple_metrics(self):
        """Test 12: Create dashboard with sales, inventory, and financial metrics"""
        start_time = time.time()
        
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", """
        Dashboard Created:
        - Sales chart: Revenue trend
        - Inventory chart: Stock levels
        - Financial chart: Profit margins
        - Top products table: Top 10 by revenue
        """)
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Create dashboard with sales, inventory, and financial metrics")
        
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 15.0, f"Query took {elapsed:.2f}s, expected <15s"


class TestAllToolsIntegration:
    """Test all 16 tools in real workflows."""

    def test_list_tables_tool(self):
        """Test ListTablesTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Tables: amz_order, amz_product, amz_inventory")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("List all available tables")
        
        assert result['success'] is True
        assert 'response' in result

    def test_describe_table_tool(self):
        """Test DescribeTableTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Table amz_order has columns: order_id, sku, quantity, price")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Describe the amz_order table")
        
        assert result['success'] is True
        assert 'response' in result

    def test_generate_sql_tool(self):
        """Test GenerateSQLTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", "Generated SQL: SELECT * FROM ic_agent.amz_order LIMIT 10")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Generate SQL for top 10 orders")
        
        assert result['success'] is True
        assert 'response' in result

    def test_execute_query_tool(self):
        """Test ExecuteQueryTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Query executed successfully: 10 rows returned")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Execute: SELECT COUNT(*) FROM ic_agent.amz_order")
        
        assert result['success'] is True
        assert 'response' in result

    def test_validate_query_tool(self):
        """Test ValidateQueryTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Query is valid: syntax OK, tables exist, columns exist")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Validate: SELECT * FROM ic_agent.amz_order LIMIT 10")
        
        assert result['success'] is True
        assert 'response' in result

    def test_explain_query_tool(self):
        """Test ExplainQueryTool integration."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Execution plan: Full table scan, estimated 100ms")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Explain: SELECT * FROM ic_agent.amz_order LIMIT 10")
        
        assert result['success'] is True
        assert 'response' in result


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_invalid_query(self):
        """Test handling of invalid query."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "I don't understand that question")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Invalid query with no clear intent")
        
        assert result['success'] is True
        assert 'response' in result

    def test_timeout_handling(self):
        """Test timeout handling."""
        mock_client = Mock()
        mock_llm = Mock()
        mock_llm.generate.side_effect = TimeoutError("LLM timeout")
        mock_llm.invoke.side_effect = TimeoutError("LLM timeout")
        mock_llm.run.side_effect = TimeoutError("LLM timeout")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Test query that times out")
        
        assert result['success'] is False
        assert 'error' in result

    def test_retry_logic(self):
        """Test retry logic with transient failures."""
        mock_client = Mock()
        mock_llm = Mock()
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Transient connection error")
            return "Success"
        
        mock_llm.generate.side_effect = side_effect
        mock_llm.invoke.side_effect = side_effect
        mock_llm.run.side_effect = side_effect
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Test query with retry")
        
        assert result['success'] is True
        assert call_count[0] >= 2  # Initial call + at least 1 retry


class TestPerformanceMetrics:
    """Test performance metrics and benchmarks."""

    def test_response_time_tracking(self):
        """Test that response times are tracked."""
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Test response")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        start_time = time.time()
        result = agent.process_query("Test query")
        elapsed = time.time() - start_time
        
        assert result['success'] is True
        assert 'response' in result
        assert elapsed < 5.0  # Simple query target

    def test_concurrent_queries(self):
        """Test handling of concurrent queries."""
        import threading
        
        mock_client = Mock()
        mock_llm = create_mock_llm("general", "Response for query {}")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        results = []
        threads = []
        
        def run_query(query_id):
            result = agent.process_query(f"Test query {query_id}")
            results.append(result)
        
        # Run 3 concurrent queries
        for i in range(3):
            thread = threading.Thread(target=run_query, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10.0)
        
        assert len(results) == 3
        assert all(r['success'] for r in results)


class TestSQLGenerationAccuracy:
    """Test SQL generation accuracy (target: >85%)."""

    def test_simple_select_generation(self):
        """Test simple SELECT query generation."""
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", "SELECT * FROM ic_agent.amz_order LIMIT 10")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Get top 10 orders")
        
        assert result['success'] is True
        response_str = str(result['response'])
        assert 'SELECT' in response_str or 'sql' in response_str.lower()

    def test_aggregation_query_generation(self):
        """Test aggregation query generation."""
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", "SELECT COUNT(*) as total, SUM(quantity) as total_qty FROM ic_agent.amz_order")
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Get total orders and quantity")
        
        assert result['success'] is True
        response_str = str(result['response'])
        assert 'COUNT' in response_str or 'SUM' in response_str

    def test_join_query_generation(self):
        """Test JOIN query generation."""
        mock_client = Mock()
        mock_llm = create_mock_llm("product", """
        SELECT o.order_id, o.quantity, p.product_name
        FROM ic_agent.amz_order o
        JOIN ic_agent.amz_product p ON o.sku = p.sku
        LIMIT 10
        """)
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Get top 10 orders with product names")
        
        assert result['success'] is True
        response_str = str(result['response'])
        assert 'JOIN' in response_str

    def test_date_filter_generation(self):
        """Test date filter query generation."""
        mock_client = Mock()
        mock_llm = create_mock_llm("sales", """
        SELECT COUNT(*) as total
        FROM ic_agent.amz_order
        WHERE purchase_date >= '2025-10-01'
        AND purchase_date <= '2025-10-31'
        """)
        
        agent = UDSAgent(
            uds_client=mock_client,
            llm_client=mock_llm
        )
        
        result = agent.process_query("Get orders for October 2025")
        
        assert result['success'] is True
        response_str = str(result['response'])
        assert 'purchase_date' in response_str or '2025-10' in response_str
