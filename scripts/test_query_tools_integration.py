"""
Integration Test Script for Query Generation Tools.

This script tests the query generation tools with real queries and ClickHouse data.
It verifies SQL generation, execution, validation, and explanation work correctly.
"""

import sys
import os
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uds.tools import (
    GenerateSQLTool,
    ExecuteQueryTool,
    ValidateQueryTool,
    ExplainQueryTool,
    UDSToolRegistry
)
from src.uds.config import UDSConfig


class IntegrationTestResults:
    """Track integration test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.errors = []
    
    def add_result(self, test_name, passed, error=None):
        """Add a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✓ {test_name}")
        else:
            self.failed_tests += 1
            print(f"✗ {test_name}")
            if error:
                print(f"  Error: {error}")
                self.errors.append((test_name, error))
    
    def summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        if self.errors:
            print("\nFailed Tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        print("="*60)
        return self.failed_tests == 0


def test_imports(results):
    """Test 1: Verify imports work from src.uds.tools."""
    print("\n" + "="*60)
    print("TEST 1: Verify Imports")
    print("="*60)
    
    try:
        # Test tool imports
        from src.uds.tools import (
            GenerateSQLTool,
            ExecuteQueryTool,
            ValidateQueryTool,
            ExplainQueryTool,
            UDSToolRegistry
        )
        
        # Test registry
        tools = UDSToolRegistry.get_query_tools()
        
        if len(tools) == 4:
            results.add_result("Import all query tools", True)
        else:
            results.add_result("Import all query tools", False, f"Expected 4 tools, got {len(tools)}")
        
        # Verify tool names
        expected_names = ['generate_sql', 'execute_query', 'validate_query', 'explain_query']
        actual_names = [tool.name for tool in tools]
        
        if set(expected_names) == set(actual_names):
            results.add_result("Tool names match expected", True)
        else:
            results.add_result("Tool names match expected", False, 
                           f"Expected {expected_names}, got {actual_names}")
        
    except Exception as e:
        results.add_result("Import all query tools", False, str(e))


def test_sql_generation(results):
    """Test 2: SQL Generation with mock LLM."""
    print("\n" + "="*60)
    print("TEST 2: SQL Generation")
    print("="*60)
    
    try:
        tool = GenerateSQLTool()
        
        # Test without LLM (should fail gracefully)
        result = tool.execute("What were the top 10 products by revenue?")
        if not result.success and "LLM not configured" in result.error:
            results.add_result("SQL generation without LLM fails gracefully", True)
        else:
            results.add_result("SQL generation without LLM fails gracefully", False,
                           "Expected error about LLM not configured")
        
        # Test with mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "SELECT * FROM ic_agent.amz_order LIMIT 10"
        tool.set_llm(mock_llm)
        
        result = tool.execute("What were the top 10 products by revenue?")
        if result.success and 'sql' in result.output:
            results.add_result("SQL generation with mock LLM succeeds", True)
            
            # Verify SQL contains expected elements
            sql = result.output['sql']
            if 'SELECT' in sql.upper() and 'LIMIT' in sql.upper():
                results.add_result("Generated SQL contains SELECT and LIMIT", True)
            else:
                results.add_result("Generated SQL contains SELECT and LIMIT", False,
                               f"SQL: {sql}")
        else:
            results.add_result("SQL generation with mock LLM succeeds", False,
                           result.error if not result.success else "No SQL in output")
        
        # Test schema context building
        context = tool._build_schema_context(['amz_order'])
        if 'amz_order' in context and 'Database Schema' in context:
            results.add_result("Schema context building works", True)
        else:
            results.add_result("Schema context building works", False,
                           "Schema context missing expected content")
        
        # Test example queries
        examples = tool._get_example_queries()
        if 'Example Queries' in examples and 'SELECT' in examples:
            results.add_result("Example queries retrieval works", True)
        else:
            results.add_result("Example queries retrieval works", False,
                           "Example queries missing expected content")
        
    except Exception as e:
        results.add_result("SQL generation tests", False, str(e))


def test_query_validation(results):
    """Test 3: Query Validation."""
    print("\n" + "="*60)
    print("TEST 3: Query Validation")
    print("="*60)
    
    try:
        tool = ValidateQueryTool()
        
        # Test safe query
        safe_sql = "SELECT * FROM ic_agent.amz_order LIMIT 10"
        result = tool.execute(safe_sql)
        
        if result.success and result.output['is_valid']:
            results.add_result("Valid query passes validation", True)
        else:
            results.add_result("Valid query passes validation", False,
                           f"Errors: {result.output['errors']}")
        
        # Test dangerous query
        dangerous_sql = "DROP TABLE ic_agent.amz_order"
        result = tool.execute(dangerous_sql)
        
        if result.success and not result.output['is_valid']:
            results.add_result("Dangerous query fails validation", True)
            
            # Check that dangerous operation was detected
            if any('DROP' in error for error in result.output['errors']):
                results.add_result("Dangerous operation detected", True)
            else:
                results.add_result("Dangerous operation detected", False,
                               "DROP not detected in errors")
        else:
            results.add_result("Dangerous query fails validation", False,
                           "Expected validation to fail")
        
        # Test invalid table
        invalid_sql = "SELECT * FROM ic_agent.nonexistent_table"
        result = tool.execute(invalid_sql)
        
        if result.success and not result.output['is_valid']:
            results.add_result("Invalid table detected", True)
        else:
            results.add_result("Invalid table detected", False,
                           "Expected validation to fail for invalid table")
        
        # Test safety check method directly
        is_safe, errors = tool._check_safety("SELECT * FROM ic_agent.amz_order")
        if is_safe and len(errors) == 0:
            results.add_result("Safety check for safe query", True)
        else:
            results.add_result("Safety check for safe query", False,
                           f"Expected safe, got: {errors}")
        
        is_safe, errors = tool._check_safety("DROP TABLE ic_agent.amz_order")
        if not is_safe and len(errors) > 0:
            results.add_result("Safety check for dangerous query", True)
        else:
            results.add_result("Safety check for dangerous query", False,
                           "Expected unsafe")
        
    except Exception as e:
        results.add_result("Query validation tests", False, str(e))


def test_query_execution_safety(results):
    """Test 4: Query Execution Safety Checks."""
    print("\n" + "="*60)
    print("TEST 4: Query Execution Safety Checks")
    print("="*60)
    
    try:
        tool = ExecuteQueryTool()
        
        # Test dangerous operations are blocked
        dangerous_queries = [
            "DROP TABLE ic_agent.amz_order",
            "DELETE FROM ic_agent.amz_order",
            "TRUNCATE TABLE ic_agent.amz_order",
            "ALTER TABLE ic_agent.amz_order",
            "CREATE TABLE test (id INT)",
            "INSERT INTO ic_agent.amz_order VALUES (1)",
            "UPDATE ic_agent.amz_order SET id = 1"
        ]
        
        all_blocked = True
        for sql in dangerous_queries:
            result = tool.execute(sql)
            if result.success:
                all_blocked = False
                print(f"  WARNING: Dangerous query not blocked: {sql[:50]}...")
        
        if all_blocked:
            results.add_result("All dangerous operations blocked", True)
        else:
            results.add_result("All dangerous operations blocked", False,
                           "Some dangerous queries were not blocked")
        
        # Test safety check method
        safe_sql = "SELECT * FROM ic_agent.amz_order LIMIT 10"
        is_safe, error = tool._is_safe_query(safe_sql)
        if is_safe and not error:
            results.add_result("Safety check passes safe query", True)
        else:
            results.add_result("Safety check passes safe query", False,
                           f"Expected safe, got: {error}")
        
        dangerous_sql = "DROP TABLE ic_agent.amz_order"
        is_safe, error = tool._is_safe_query(dangerous_sql)
        if not is_safe and error:
            results.add_result("Safety check blocks dangerous query", True)
        else:
            results.add_result("Safety check blocks dangerous query", False,
                           "Expected unsafe")
        
    except Exception as e:
        results.add_result("Query execution safety tests", False, str(e))


def test_query_explanation(results):
    """Test 5: Query Explanation."""
    print("\n" + "="*60)
    print("TEST 5: Query Explanation")
    print("="*60)
    
    try:
        tool = ExplainQueryTool()
        
        # Test with a simple query
        sql = "SELECT COUNT(*) FROM ic_agent.amz_order"
        
        # Mock the client query to avoid actual database connection
        from unittest.mock import patch
        
        with patch.object(tool, 'client') as mock_client:
            mock_df = Mock()
            mock_df.to_dict = Mock(return_value=[{'plan': 'test plan'}])
            mock_client.query.return_value = mock_df
            
            result = tool.execute(sql)
            
            if result.success:
                results.add_result("Query explanation succeeds", True)
                
                # Check output structure
                if 'execution_plan' in result.output:
                    results.add_result("Execution plan in output", True)
                else:
                    results.add_result("Execution plan in output", False,
                                   "Missing execution_plan")
                
                if 'suggestions' in result.output:
                    results.add_result("Suggestions in output", True)
                else:
                    results.add_result("Suggestions in output", False,
                                   "Missing suggestions")
            else:
                results.add_result("Query explanation succeeds", False,
                               result.error)
        
        # Test optimization suggestions
        sql_without_limit = "SELECT * FROM ic_agent.amz_order"
        
        with patch.object(tool, 'client') as mock_client:
            mock_df = Mock()
            mock_df.to_dict = Mock(return_value=[{'plan': 'test'}])
            mock_client.query.return_value = mock_df
            
            result = tool.execute(sql_without_limit)
            
            if result.success:
                suggestions = result.output['suggestions']
                
                # Check for LIMIT suggestion
                if any('LIMIT' in s for s in suggestions):
                    results.add_result("LIMIT suggestion generated", True)
                else:
                    results.add_result("LIMIT suggestion generated", False,
                                   "Expected LIMIT suggestion")
                
                # Check for SELECT * suggestion
                if any('SELECT *' in s for s in suggestions):
                    results.add_result("SELECT * suggestion generated", True)
                else:
                    results.add_result("SELECT * suggestion generated", False,
                                   "Expected SELECT * suggestion")
        
    except Exception as e:
        results.add_result("Query explanation tests", False, str(e))


def test_tool_parameters(results):
    """Test 6: Tool Parameters."""
    print("\n" + "="*60)
    print("TEST 6: Tool Parameters")
    print("="*60)
    
    try:
        # Test GenerateSQLTool parameters
        gen_tool = GenerateSQLTool()
        params = gen_tool._get_parameters()
        
        if len(params) >= 1 and params[0].name == 'question':
            results.add_result("GenerateSQLTool has question parameter", True)
        else:
            results.add_result("GenerateSQLTool has question parameter", False,
                           "Missing question parameter")
        
        # Test ExecuteQueryTool parameters
        exec_tool = ExecuteQueryTool()
        params = exec_tool._get_parameters()
        
        param_names = [p.name for p in params]
        if 'sql' in param_names and 'format' in param_names:
            results.add_result("ExecuteQueryTool has expected parameters", True)
        else:
            results.add_result("ExecuteQueryTool has expected parameters", False,
                           f"Got: {param_names}")
        
        # Test ValidateQueryTool parameters
        val_tool = ValidateQueryTool()
        params = val_tool._get_parameters()
        
        if len(params) >= 1 and params[0].name == 'sql':
            results.add_result("ValidateQueryTool has sql parameter", True)
        else:
            results.add_result("ValidateQueryTool has sql parameter", False,
                           "Missing sql parameter")
        
        # Test ExplainQueryTool parameters
        exp_tool = ExplainQueryTool()
        params = exp_tool._get_parameters()
        
        if len(params) >= 1 and params[0].name == 'sql':
            results.add_result("ExplainQueryTool has sql parameter", True)
        else:
            results.add_result("ExplainQueryTool has sql parameter", False,
                           "Missing sql parameter")
        
    except Exception as e:
        results.add_result("Tool parameters tests", False, str(e))


def test_parameter_validation(results):
    """Test 7: Parameter Validation."""
    print("\n" + "="*60)
    print("TEST 7: Parameter Validation")
    print("="*60)
    
    try:
        # Test GenerateSQLTool validation
        gen_tool = GenerateSQLTool()
        
        try:
            gen_tool.validate_parameters(question=None)
            results.add_result("GenerateSQLTool validates missing question", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("GenerateSQLTool validates missing question", True)
        
        try:
            gen_tool.validate_parameters(question="")
            results.add_result("GenerateSQLTool validates empty question", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("GenerateSQLTool validates empty question", True)
        
        # Test ExecuteQueryTool validation
        exec_tool = ExecuteQueryTool()
        
        try:
            exec_tool.validate_parameters(sql=None)
            results.add_result("ExecuteQueryTool validates missing sql", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("ExecuteQueryTool validates missing sql", True)
        
        try:
            exec_tool.validate_parameters(sql="SELECT *", format="invalid")
            results.add_result("ExecuteQueryTool validates invalid format", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("ExecuteQueryTool validates invalid format", True)
        
        # Test ValidateQueryTool validation
        val_tool = ValidateQueryTool()
        
        try:
            val_tool.validate_parameters(sql=None)
            results.add_result("ValidateQueryTool validates missing sql", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("ValidateQueryTool validates missing sql", True)
        
        # Test ExplainQueryTool validation
        exp_tool = ExplainQueryTool()
        
        try:
            exp_tool.validate_parameters(sql=None)
            results.add_result("ExplainQueryTool validates missing sql", False,
                           "Expected ValidationError")
        except Exception:
            results.add_result("ExplainQueryTool validates missing sql", True)
        
    except Exception as e:
        results.add_result("Parameter validation tests", False, str(e))


def test_tool_descriptions(results):
    """Test 8: Tool Descriptions."""
    print("\n" + "="*60)
    print("TEST 8: Tool Descriptions")
    print("="*60)
    
    try:
        tools = UDSToolRegistry.get_query_tools()
        
        for tool in tools:
            if tool.name and tool.description:
                results.add_result(f"{tool.name} has description", True)
            else:
                results.add_result(f"{tool.name} has description", False,
                               "Missing name or description")
        
    except Exception as e:
        results.add_result("Tool descriptions tests", False, str(e))


def main():
    """Run all integration tests."""
    print("="*60)
    print("QUERY TOOLS INTEGRATION TESTS")
    print("="*60)
    print(f"Testing Query Generation Tools")
    print(f"ClickHouse Host: {UDSConfig.CH_HOST}")
    print(f"ClickHouse Database: {UDSConfig.CH_DATABASE}")
    
    results = IntegrationTestResults()
    
    # Run all tests
    test_imports(results)
    test_sql_generation(results)
    test_query_validation(results)
    test_query_execution_safety(results)
    test_query_explanation(results)
    test_tool_parameters(results)
    test_parameter_validation(results)
    test_tool_descriptions(results)
    
    # Print summary
    success = results.summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
