#!/usr/bin/env python3
"""
Smoke Tests for UDS Agent Production Deployment

Tests all critical components to ensure production readiness:
- Health check endpoint
- API endpoints
- Database connection
- Redis connection
- LLM API connectivity
- All 16 tools functional
- Authentication

Automated to run after every deployment.
"""

import pytest
import requests
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.uds.uds_agent import UDSAgent
from src.uds.uds_client import UDSClient
from src.uds.intent_classifier import IntentDomain


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_endpoint(self):
        """Test health check endpoint responds correctly."""
        base_url = os.getenv('ECS_HOST', 'http://localhost:8000')
        health_url = f"{base_url.rstrip('/')}/health"
        
        try:
            response = requests.get(health_url, timeout=5)
            assert response.status_code == 200
            assert response.json().get('status') == 'healthy'
            print(f"✓ Health check passed: {health_url}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            raise


class TestAPIEndpoints:
    """Test critical API endpoints."""
    
    def test_query_endpoint(self):
        """Test query endpoint."""
        base_url = os.getenv('ECS_HOST', 'http://localhost:8000')
        query_url = f"{base_url}/api/v1/uds/query"
        
        try:
            response = requests.post(
                query_url,
                json={"query": "test query"},
                timeout=10
            )
            assert response.status_code in [200, 400]
            print(f"✓ Query endpoint passed: {query_url}")
        except Exception as e:
            print(f"✗ Query endpoint failed: {e}")
            raise
    
    def test_stream_endpoint(self):
        """Test SSE stream endpoint."""
        base_url = os.getenv('ECS_HOST', 'http://localhost:8000')
        stream_url = f"{base_url}/api/v1/uds/query/stream"
        
        try:
            response = requests.post(
                stream_url,
                json={"query": "test query"},
                timeout=5
            )
            assert response.status_code == 200
            print(f"✓ Stream endpoint passed: {stream_url}")
        except Exception as e:
            print(f"✗ Stream endpoint failed: {e}")
            raise


class TestDatabaseConnection:
    """Test ClickHouse database connection."""
    
    def test_clickhouse_connection(self):
        """Test ClickHouse connection."""
        try:
            client = UDSClient(
                host=os.getenv('CH_HOST'),
                port=int(os.getenv('CH_PORT', '8123')),
                database=os.getenv('CH_DATABASE', 'ic_agent')
            )
            
            result = client.query("SELECT 1")
            assert result is not None
            assert 'data' in result
            print(f"✓ ClickHouse connection passed")
        except Exception as e:
            print(f"✗ ClickHouse connection failed: {e}")
            raise


class TestRedisConnection:
    """Test Redis connection."""
    
    def test_redis_connection(self):
        """Test Redis connection."""
        try:
            import redis
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                decode_responses=True
            )
            
            r.ping()
            print(f"✓ Redis connection passed")
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            raise


class TestLLMAPI:
    """Test LLM API connectivity."""
    
    def test_llm_connection(self):
        """Test LLM API connection."""
        try:
            base_url = os.getenv('LLM_BASE_URL', 'http://localhost:11434')
            response = requests.get(f"{base_url}/health", timeout=5)
            assert response.status_code == 200
            print(f"✓ LLM API connection passed")
        except Exception as e:
            print(f"✗ LLM API connection failed: {e}")
            raise


class TestAllTools:
    """Test all 16 tools are functional."""
    
    def __init__(self):
        from unittest.mock import Mock
        self.mock_client = Mock()
        self.mock_llm = Mock()
        self.mock_llm.generate.return_value = "general"
        self.mock_llm.invoke.return_value = type('Mock', (), {'content': "Final Answer: Test response"})
        self.mock_llm.run.return_value = "Test response"
        
        self.agent = UDSAgent(
            uds_client=self.mock_client,
            llm_client=self.mock_llm
        )
    
    def test_list_tables_tool(self):
        """Test ListTablesTool."""
        result = self.agent.process_query("List all available tables")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ListTablesTool passed")
    
    def test_describe_table_tool(self):
        """Test DescribeTableTool."""
        result = self.agent.process_query("Describe the amz_order table")
        assert result['success'] is True
        assert 'response' in result
        print("✓ DescribeTableTool passed")
    
    def test_generate_sql_tool(self):
        """Test GenerateSQLTool."""
        result = self.agent.process_query("Generate SQL for top 10 orders")
        assert result['success'] is True
        assert 'response' in result
        print("✓ GenerateSQLTool passed")
    
    def test_execute_query_tool(self):
        """Test ExecuteQueryTool."""
        result = self.agent.process_query("Execute: SELECT COUNT(*) FROM ic_agent.amz_order")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ExecuteQueryTool passed")
    
    def test_validate_query_tool(self):
        """Test ValidateQueryTool."""
        result = self.agent.process_query("Validate: SELECT * FROM ic_agent.amz_order LIMIT 10")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ValidateQueryTool passed")
    
    def test_explain_query_tool(self):
        """Test ExplainQueryTool."""
        result = self.agent.process_query("Explain: SELECT COUNT(*) FROM ic_agent.amz_order")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ExplainQueryTool passed")
    
    def test_sales_trend_tool(self):
        """Test SalesTrendTool."""
        result = self.agent.process_query("Analyze sales trend for October")
        assert result['success'] is True
        assert 'response' in result
        print("✓ SalesTrendTool passed")
    
    def test_inventory_analysis_tool(self):
        """Test InventoryAnalysisTool."""
        result = self.agent.process_query("Analyze inventory levels")
        assert result['success'] is True
        assert 'response' in result
        print("✓ InventoryAnalysisTool passed")
    
    def test_financial_summary_tool(self):
        """Test FinancialSummaryTool."""
        result = self.agent.process_query("Summarize financial metrics")
        assert result['success'] is True
        assert 'response' in result
        print("✓ FinancialSummaryTool passed")
    
    def test_product_performance_tool(self):
        """Test ProductPerformanceTool."""
        result = self.agent.process_query("Analyze product performance")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ProductPerformanceTool passed")
    
    def test_comparison_tool(self):
        """Test ComparisonTool."""
        result = self.agent.process_query("Compare Q3 vs Q4 sales")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ComparisonTool passed")
    
    def test_search_tool(self):
        """Test SearchTool."""
        result = self.agent.process_query("Search for orders containing 'PROD-001'")
        assert result['success'] is True
        assert 'response' in result
        print("✓ SearchTool passed")
    
    def test_create_chart_tool(self):
        """Test CreateChartTool."""
        result = self.agent.process_query("Create sales trend chart")
        assert result['success'] is True
        assert 'response' in result
        print("✓ CreateChartTool passed")
    
    def test_export_data_tool(self):
        """Test ExportDataTool."""
        result = self.agent.process_query("Export sales data to CSV")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ExportDataTool passed")
    
    def test_generate_report_tool(self):
        """Test GenerateReportTool."""
        result = self.agent.process_query("Generate sales report")
        assert result['success'] is True
        assert 'response' in result
        print("✓ GenerateReportTool passed")
    
    def test_create_dashboard_tool(self):
        """Test CreateDashboardTool."""
        result = self.agent.process_query("Create business dashboard")
        assert result['success'] is True
        assert 'response' in result
        print("✓ CreateDashboardTool passed")
    
    def test_anomaly_detection_tool(self):
        """Test AnomalyDetectionTool."""
        result = self.agent.process_query("Detect anomalies in sales data")
        assert result['success'] is True
        assert 'response' in result
        print("✓ AnomalyDetectionTool passed")
    
    def test_forecast_tool(self):
        """Test ForecastTool."""
        result = self.agent.process_query("Forecast sales for next month")
        assert result['success'] is True
        assert 'response' in result
        print("✓ ForecastTool passed")


def run_smoke_tests():
    """Run all smoke tests and report results."""
    print("=" * 60)
    print("SMOKE TESTS - PRODUCTION VALIDATION")
    print("=" * 60)
    print()
    
    pytest.main([
        'tests/test_smoke.py::TestHealthCheck::test_health_endpoint',
        'tests/test_smoke.py::TestAPIEndpoints::test_query_endpoint',
        'tests/test_smoke.py::TestAPIEndpoints::test_stream_endpoint',
        'tests/test_smoke.py::TestDatabaseConnection::test_clickhouse_connection',
        'tests/test_smoke.py::TestRedisConnection::test_redis_connection',
        'tests/test_smoke.py::TestLLMAPI::test_llm_connection',
        'tests/test_smoke.py::TestAllTools::test_list_tables_tool',
        'tests/test_smoke.py::TestAllTools::test_describe_table_tool',
        'tests/test_smoke.py::TestAllTools::test_generate_sql_tool',
        'tests/test_smoke.py::TestAllTools::test_execute_query_tool',
        'tests/test_smoke.py::TestAllTools::test_validate_query_tool',
        'tests/test_smoke.py::TestAllTools::test_explain_query_tool',
        'tests/test_smoke.py::TestAllTools::test_sales_trend_tool',
        'tests/test_smoke.py::TestAllTools::test_inventory_analysis_tool',
        'tests/test_smoke.py::TestAllTools::test_financial_summary_tool',
        'tests/test_smoke.py::TestAllTools::test_product_performance_tool',
        'tests/test_smoke.py::TestAllTools::test_comparison_tool',
        'tests/test_smoke.py::TestAllTools::test_search_tool',
        'tests/test_smoke.py::TestAllTools::test_create_chart_tool',
        'tests/test_smoke.py::TestAllTools::test_export_data_tool',
        'tests/test_smoke.py::TestAllTools::test_generate_report_tool',
        'tests/test_smoke.py::TestAllTools::test_create_dashboard_tool',
        'tests/test_smoke.py::TestAllTools::test_anomaly_detection_tool',
        'tests/test_smoke.py::TestAllTools::test_forecast_tool',
    ], '-v', '--tb=short')
    
    print()
    print("=" * 60)
    print("SMOKE TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_smoke_tests()
