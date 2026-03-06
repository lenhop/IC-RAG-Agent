"""
Real Query Testing Script for Query Generation Tools.

This script tests the query generation tools with real queries against
the actual ClickHouse database to verify end-to-end functionality.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uds.tools import (
    GenerateSQLTool,
    ExecuteQueryTool,
    ValidateQueryTool,
    ExplainQueryTool,
)
from src.uds.config import UDSConfig


class RealQueryTestResults:
    """Track real query test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.errors = []
        self.query_results = []
    
    def add_result(self, test_name, passed, error=None, query=None, result=None):
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
        
        if query and result:
            self.query_results.append({
                'test': test_name,
                'query': query,
                'result': result,
                'passed': passed
            })
    
    def summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("REAL QUERY TEST SUMMARY")
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


def test_real_query_validation(results):
    """Test 1: Real Query Validation."""
    print("\n" + "="*60)
    print("TEST 1: Real Query Validation")
    print("="*60)
    
    try:
        tool = ValidateQueryTool()
        
        # Test valid queries against real database
        valid_queries = [
            "SELECT COUNT(*) FROM ic_agent.amz_order",
            "SELECT * FROM ic_agent.amz_order LIMIT 10",
            "SELECT sku, SUM(quantity) FROM ic_agent.amz_inventory GROUP BY sku LIMIT 10",
            "SELECT * FROM ic_agent.amz_product LIMIT 5",
        ]
        
        for sql in valid_queries:
            result = tool.execute(sql)
            if result.success and result.output['is_valid']:
                results.add_result(f"Validate: {sql[:50]}...", True, query=sql, result=result)
            else:
                results.add_result(f"Validate: {sql[:50]}...", False,
                               result.output['errors'] if result.output else result.error,
                               query=sql, result=result)
        
        # Test invalid queries
        invalid_queries = [
            "SELECT * FROM ic_agent.nonexistent_table",
            "SELECT invalid_column FROM ic_agent.amz_order",
        ]
        
        for sql in invalid_queries:
            result = tool.execute(sql)
            if result.success and not result.output['is_valid']:
                results.add_result(f"Validate invalid: {sql[:50]}...", True, query=sql, result=result)
            else:
                results.add_result(f"Validate invalid: {sql[:50]}...", False,
                               "Expected validation to fail",
                               query=sql, result=result)
        
    except Exception as e:
        results.add_result("Real query validation tests", False, str(e))


def test_real_query_execution(results):
    """Test 2: Real Query Execution."""
    print("\n" + "="*60)
    print("TEST 2: Real Query Execution")
    print("="*60)
    
    try:
        tool = ExecuteQueryTool()
        
        # Test safe queries against real database
        safe_queries = [
            ("SELECT COUNT(*) as total FROM ic_agent.amz_order", "count"),
            ("SELECT * FROM ic_agent.amz_order LIMIT 5", "sample"),
            ("SELECT COUNT(*) as total FROM ic_agent.amz_product", "product_count"),
            ("SELECT COUNT(*) as total FROM ic_agent.amz_inventory", "inventory_count"),
        ]
        
        for sql, description in safe_queries:
            result = tool.execute(sql, format='dict')
            if result.success:
                results.add_result(f"Execute {description}: {sql[:50]}...", True, query=sql, result=result)
            else:
                results.add_result(f"Execute {description}: {sql[:50]}...", False,
                               result.error, query=sql, result=result)
        
        # Test dangerous queries are blocked
        dangerous_queries = [
            "DROP TABLE ic_agent.amz_order",
            "DELETE FROM ic_agent.amz_order",
            "TRUNCATE TABLE ic_agent.amz_order",
        ]
        
        for sql in dangerous_queries:
            result = tool.execute(sql)
            if not result.success and "Dangerous operation" in result.error:
                results.add_result(f"Block dangerous: {sql[:40]}...", True, query=sql, result=result)
            else:
                results.add_result(f"Block dangerous: {sql[:40]}...", False,
                               "Expected dangerous operation to be blocked",
                               query=sql, result=result)
        
    except Exception as e:
        results.add_result("Real query execution tests", False, str(e))


def test_real_query_explanation(results):
    """Test 3: Real Query Explanation."""
    print("\n" + "="*60)
    print("TEST 3: Real Query Explanation")
    print("="*60)
    
    try:
        tool = ExplainQueryTool()
        
        # Test query explanation on real queries
        queries = [
            "SELECT COUNT(*) FROM ic_agent.amz_order",
            "SELECT * FROM ic_agent.amz_order LIMIT 10",
            "SELECT sku, SUM(quantity) as total FROM ic_agent.amz_inventory GROUP BY sku LIMIT 10",
        ]
        
        for sql in queries:
            result = tool.execute(sql)
            if result.success:
                results.add_result(f"Explain: {sql[:50]}...", True, query=sql, result=result)
                
                # Verify output structure
                if 'execution_plan' in result.output and 'suggestions' in result.output:
                    results.add_result(f"Explain output structure: {sql[:50]}...", True)
                else:
                    results.add_result(f"Explain output structure: {sql[:50]}...", False,
                                   "Missing expected output fields")
            else:
                results.add_result(f"Explain: {sql[:50]}...", False,
                               result.error, query=sql, result=result)
        
    except Exception as e:
        results.add_result("Real query explanation tests", False, str(e))


def test_query_workflow(results):
    """Test 4: Complete Query Workflow."""
    print("\n" + "="*60)
    print("TEST 4: Complete Query Workflow")
    print("="*60)
    
    try:
        # Simulate a complete workflow: validate -> explain -> execute
        
        sql = "SELECT COUNT(*) as total_orders FROM ic_agent.amz_order"
        
        # Step 1: Validate
        val_tool = ValidateQueryTool()
        val_result = val_tool.execute(sql)
        
        if val_result.success and val_result.output['is_valid']:
            results.add_result("Workflow: Validate query", True, query=sql, result=val_result)
        else:
            results.add_result("Workflow: Validate query", False,
                           val_result.output['errors'] if val_result.output else val_result.error,
                           query=sql, result=val_result)
            return
        
        # Step 2: Explain
        exp_tool = ExplainQueryTool()
        exp_result = exp_tool.execute(sql)
        
        if exp_result.success:
            results.add_result("Workflow: Explain query", True, query=sql, result=exp_result)
        else:
            results.add_result("Workflow: Explain query", False,
                           exp_result.error, query=sql, result=exp_result)
            return
        
        # Step 3: Execute
        exec_tool = ExecuteQueryTool()
        exec_result = exec_tool.execute(sql, format='dict')
        
        if exec_result.success:
            results.add_result("Workflow: Execute query", True, query=sql, result=exec_result)
        else:
            results.add_result("Workflow: Execute query", False,
                           exec_result.error, query=sql, result=exec_result)
        
        # Complete workflow success
        if (val_result.success and exp_result.success and exec_result.success):
            results.add_result("Workflow: Complete workflow", True)
        else:
            results.add_result("Workflow: Complete workflow", False,
                           "One or more steps failed")
        
    except Exception as e:
        results.add_result("Query workflow tests", False, str(e))


def test_safety_checks_comprehensive(results):
    """Test 5: Comprehensive Safety Checks."""
    print("\n" + "="*60)
    print("TEST 5: Comprehensive Safety Checks")
    print("="*60)
    
    try:
        exec_tool = ExecuteQueryTool()
        val_tool = ValidateQueryTool()
        
        # Test all dangerous operations are blocked
        dangerous_operations = [
            ("DROP TABLE ic_agent.amz_order", "DROP"),
            ("DELETE FROM ic_agent.amz_order", "DELETE"),
            ("TRUNCATE TABLE ic_agent.amz_order", "TRUNCATE"),
            ("ALTER TABLE ic_agent.amz_order ADD COLUMN test INT", "ALTER"),
            ("CREATE TABLE test (id INT)", "CREATE"),
            ("INSERT INTO ic_agent.amz_order VALUES (1)", "INSERT"),
            ("UPDATE ic_agent.amz_order SET id = 1", "UPDATE"),
        ]
        
        all_blocked = True
        for sql, op_name in dangerous_operations:
            # Test execution blocking
            exec_result = exec_tool.execute(sql)
            if exec_result.success:
                all_blocked = False
                results.add_result(f"Block {op_name} in execution", False,
                               f"{op_name} not blocked", query=sql, result=exec_result)
            else:
                results.add_result(f"Block {op_name} in execution", True, query=sql, result=exec_result)
            
            # Test validation blocking
            val_result = val_tool.execute(sql)
            if val_result.success and val_result.output['is_valid']:
                all_blocked = False
                results.add_result(f"Block {op_name} in validation", False,
                               f"{op_name} not detected", query=sql, result=val_result)
            else:
                results.add_result(f"Block {op_name} in validation", True, query=sql, result=val_result)
        
        if all_blocked:
            results.add_result("All dangerous operations blocked", True)
        else:
            results.add_result("All dangerous operations blocked", False,
                           "Some dangerous operations were not blocked")
        
    except Exception as e:
        results.add_result("Safety checks tests", False, str(e))


def test_output_formats(results):
    """Test 6: Different Output Formats."""
    print("\n" + "="*60)
    print("TEST 6: Different Output Formats")
    print("="*60)
    
    try:
        tool = ExecuteQueryTool()
        
        sql = "SELECT COUNT(*) as total FROM ic_agent.amz_order"
        
        # Test different formats
        formats = ['dict', 'dataframe', 'json']
        
        for fmt in formats:
            result = tool.execute(sql, format=fmt)
            if result.success:
                results.add_result(f"Output format {fmt}", True, query=sql, result=result)
            else:
                results.add_result(f"Output format {fmt}", False,
                               result.error, query=sql, result=result)
        
    except Exception as e:
        results.add_result("Output format tests", False, str(e))


def test_complex_queries(results):
    """Test 7: Complex Queries."""
    print("\n" + "="*60)
    print("TEST 7: Complex Queries")
    print("="*60)
    
    try:
        val_tool = ValidateQueryTool()
        exec_tool = ExecuteQueryTool()
        
        # Test more complex queries
        complex_queries = [
            "SELECT sku, SUM(quantity) as total_qty FROM ic_agent.amz_inventory GROUP BY sku ORDER BY total_qty DESC LIMIT 10",
            "SELECT COUNT(*) as total, AVG(quantity) as avg_qty FROM ic_agent.amz_inventory",
            "SELECT * FROM ic_agent.amz_order WHERE purchase_date >= '2024-01-01' LIMIT 10",
        ]
        
        for sql in complex_queries:
            # Validate
            val_result = val_tool.execute(sql)
            if val_result.success and val_result.output['is_valid']:
                results.add_result(f"Validate complex: {sql[:50]}...", True, query=sql, result=val_result)
            else:
                results.add_result(f"Validate complex: {sql[:50]}...", False,
                               val_result.output['errors'] if val_result.output else val_result.error,
                               query=sql, result=val_result)
                continue
            
            # Execute
            exec_result = exec_tool.execute(sql, format='dict')
            if exec_result.success:
                results.add_result(f"Execute complex: {sql[:50]}...", True, query=sql, result=exec_result)
            else:
                results.add_result(f"Execute complex: {sql[:50]}...", False,
                               exec_result.error, query=sql, result=exec_result)
        
    except Exception as e:
        results.add_result("Complex query tests", False, str(e))


def main():
    """Run all real query tests."""
    print("="*60)
    print("REAL QUERY TESTING")
    print("="*60)
    print(f"Testing Query Generation Tools with Real ClickHouse Data")
    print(f"ClickHouse Host: {UDSConfig.CH_HOST}")
    print(f"ClickHouse Database: {UDSConfig.CH_DATABASE}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = RealQueryTestResults()
    
    # Run all tests
    test_real_query_validation(results)
    test_real_query_execution(results)
    test_real_query_explanation(results)
    test_query_workflow(results)
    test_safety_checks_comprehensive(results)
    test_output_formats(results)
    test_complex_queries(results)
    
    # Print summary
    success = results.summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
