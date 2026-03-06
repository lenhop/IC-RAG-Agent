"""
Tests for UDS Query Generation Tools.
"""

import pytest
from unittest.mock import Mock, patch
from src.uds.tools.query_tools import (
    GenerateSQLTool,
    ExecuteQueryTool,
    ValidateQueryTool,
    ExplainQueryTool
)


class TestGenerateSQLTool:
    """Test GenerateSQLTool functionality."""

    def test_generate_sql_with_llm(self):
        """Test SQL generation from natural language with LLM."""
        tool = GenerateSQLTool()

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "SELECT * FROM ic_agent.amz_order LIMIT 10"
        tool.set_llm(mock_llm)

        result = tool.execute("What were the top 10 products by revenue in October?")

        assert result.success
        assert 'sql' in result.output
        assert 'SELECT' in result.output['sql'].upper()

    def test_generate_sql_without_llm(self):
        """Test SQL generation fails without LLM configured."""
        tool = GenerateSQLTool()
        result = tool.execute("Test question")

        assert not result.success
        assert "LLM not configured" in result.error

    def test_build_schema_context(self):
        """Test schema context building."""
        tool = GenerateSQLTool()

        # Test with specific tables
        context = tool._build_schema_context(['amz_order'])

        assert 'amz_order' in context
        assert 'Database Schema' in context

        # Test with all tables
        context_all = tool._build_schema_context()
        assert len(context) > 0

    def test_get_example_queries(self):
        """Test example queries retrieval."""
        tool = GenerateSQLTool()
        examples = tool._get_example_queries()

        assert 'Example Queries' in examples
        assert 'SELECT' in examples


class TestExecuteQueryTool:
    """Test ExecuteQueryTool functionality."""

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_execute_query_safe(self, mock_client_class):
        """Test query execution with safe query."""
        # Mock client
        mock_client = Mock()
        mock_client.query.return_value = Mock()
        mock_client.query.return_value.empty = False
        mock_client.query.return_value.__len__ = Mock(return_value=10)
        mock_client.query.return_value.columns = ['id', 'name']
        mock_client.query.return_value.to_dict = Mock(return_value=[{'id': 1, 'name': 'test'}])
        mock_client_class.return_value = mock_client

        tool = ExecuteQueryTool()
        sql = "SELECT COUNT(*) as count FROM ic_agent.amz_order LIMIT 1"
        result = tool.execute(sql)

        assert result.success
        assert 'results' in result.output
        assert result.output['row_count'] == 10

    def test_execute_query_dangerous(self):
        """Test execution blocks dangerous operations."""
        tool = ExecuteQueryTool()
        sql = "DROP TABLE ic_agent.amz_order"
        result = tool.execute(sql)

        assert not result.success
        assert "DROP" in result.error

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_execute_query_adds_limit(self, mock_client_class):
        """Test execution adds LIMIT if not present."""
        tool = ExecuteQueryTool()

        # Mock client
        with patch.object(tool, 'client') as mock_client:
            mock_client.query.return_value = Mock()
            mock_client.query.return_value.empty = False
            mock_client.query.return_value.__len__ = Mock(return_value=5)
            mock_client.query.return_value.columns = ['id']
            mock_client.query.return_value.to_dict = Mock(return_value=[{'id': 1}])

            sql = "SELECT * FROM ic_agent.amz_order"
            result = tool.execute(sql)

            # Check that LIMIT was added
            assert 'LIMIT' in result.output['sql']

    def test_is_safe_query(self):
        """Test safety check logic."""
        tool = ExecuteQueryTool()

        # Safe query
        is_safe, _ = tool._is_safe_query("SELECT * FROM ic_agent.amz_order LIMIT 10")
        assert is_safe

        # Dangerous query
        is_safe, error = tool._is_safe_query("DROP TABLE ic_agent.amz_order")
        assert not is_safe
        assert "DROP" in error

        # DELETE without WHERE - caught by dangerous keyword check
        is_safe, error = tool._is_safe_query("DELETE FROM ic_agent.amz_order")
        assert not is_safe
        assert "DELETE" in error


class TestValidateQueryTool:
    """Test ValidateQueryTool functionality."""

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_validate_query_safe(self, mock_client_class):
        """Test validation of safe query."""
        # Mock client
        mock_client = Mock()
        mock_client.query.return_value = Mock()
        mock_client_class.return_value = mock_client

        tool = ValidateQueryTool()
        sql = "SELECT * FROM ic_agent.amz_order LIMIT 10"
        result = tool.execute(sql)

        assert result.success
        assert result.output['is_valid']
        assert len(result.output['errors']) == 0

    def test_validate_query_dangerous(self):
        """Test validation catches dangerous operations."""
        tool = ValidateQueryTool()
        sql = "DROP TABLE ic_agent.amz_order"
        result = tool.execute(sql)

        assert result.success
        assert not result.output['is_valid']
        assert len(result.output['errors']) > 0

    def test_check_safety(self):
        """Test safety check logic."""
        tool = ValidateQueryTool()

        # Safe query
        is_safe, errors = tool._check_safety("SELECT * FROM ic_agent.amz_order")
        assert is_safe
        assert len(errors) == 0

        # Dangerous query
        is_safe, errors = tool._check_safety("DROP TABLE ic_agent.amz_order")
        assert not is_safe
        assert len(errors) > 0

    def test_check_tables_columns(self):
        """Test table/column name verification."""
        tool = ValidateQueryTool()

        # Valid table
        is_valid, errors = tool._check_tables_columns("SELECT * FROM ic_agent.amz_order")
        assert is_valid
        assert len(errors) == 0

        # Invalid table
        is_valid, errors = tool._check_tables_columns("SELECT * FROM ic_agent.nonexistent_table")
        assert not is_valid
        assert len(errors) > 0


class TestExplainQueryTool:
    """Test ExplainQueryTool functionality."""

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_explain_query(self, mock_client_class):
        """Test query explanation."""
        # Mock client
        mock_client = Mock()
        mock_df = Mock()
        mock_df.to_dict = Mock(return_value=[{'plan': 'test'}])
        mock_client.query.return_value = mock_df
        mock_client_class.return_value = mock_client

        tool = ExplainQueryTool()
        sql = "SELECT COUNT(*) FROM ic_agent.amz_order"
        result = tool.execute(sql)

        assert result.success
        assert 'execution_plan' in result.output
        assert 'suggestions' in result.output

    def test_explain_query_suggestions(self):
        """Test optimization suggestions generation."""
        tool = ExplainQueryTool()

        # Mock client
        with patch.object(tool, 'client') as mock_client:
            mock_df = Mock()
            mock_df.to_dict = Mock(return_value=[{'plan': 'test'}])
            mock_client.query.return_value = mock_df

            # Query without LIMIT
            sql = "SELECT * FROM ic_agent.amz_order"
            result = tool.execute(sql)

            # Check for LIMIT suggestion
            suggestions = result.output['suggestions']
            assert any('LIMIT' in s for s in suggestions)

            # Query with SELECT *
            sql = "SELECT * FROM ic_agent.amz_order"
            result = tool.execute(sql)

            # Check for column suggestion
            suggestions = result.output['suggestions']
            assert any('SELECT *' in s for s in suggestions)


class TestIntegration:
    """Integration tests for query tools."""

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_query_workflow(self, mock_client_class):
        """Test complete workflow: generate -> validate -> execute."""
        # Mock client
        mock_client = Mock()
        mock_df = Mock()
        mock_df.to_dict = Mock(return_value=[{'count': 100}])
        mock_df.empty = False
        mock_df.__len__ = Mock(return_value=1)
        mock_df.columns = ['count']
        mock_client.query.return_value = mock_df
        mock_client_class.return_value = mock_client

        # Step 1: Generate SQL
        gen_tool = GenerateSQLTool()
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "SELECT COUNT(*) FROM ic_agent.amz_order"
        gen_tool.set_llm(mock_llm)

        gen_result = gen_tool.execute("How many orders are there?")
        assert gen_result.success

        # Step 2: Validate SQL
        val_tool = ValidateQueryTool()
        val_result = val_tool.execute(gen_result.output['sql'])
        assert val_result.success
        assert val_result.output['is_valid']

        # Step 3: Execute SQL
        exec_tool = ExecuteQueryTool()
        exec_result = exec_tool.execute(gen_result.output['sql'])
        assert exec_result.success
        assert exec_result.output['row_count'] > 0

    @patch('src.uds.tools.query_tools.UDSClient')
    def test_error_handling(self, mock_client_class):
        """Test error handling across all tools."""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Database connection error")
        mock_client_class.return_value = mock_client

        # Test GenerateSQLTool
        gen_tool = GenerateSQLTool()
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        gen_tool.set_llm(mock_llm)

        gen_result = gen_tool.execute("Test question")
        assert not gen_result.success

        # Test ExecuteQueryTool
        exec_tool = ExecuteQueryTool()
        exec_result = exec_tool.execute("SELECT * FROM ic_agent.amz_order")
        assert not exec_result.success

        # Test ExplainQueryTool
        exp_tool = ExplainQueryTool()
        exp_result = exp_tool.execute("SELECT * FROM ic_agent.amz_order")
        assert not exp_result.success

        # Test ValidateQueryTool - validation itself succeeds but finds errors
        val_tool = ValidateQueryTool()
        val_result = val_tool.execute("SELECT * FROM ic_agent.amz_order")
        # Validation succeeds but query is invalid due to syntax error
        assert val_result.success
        assert not val_result.output['is_valid']
        assert len(val_result.output['errors']) > 0
