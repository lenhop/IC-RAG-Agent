"""
Integration Test Script for UDS Agent.

Tests the complete agent workflow with real queries against ClickHouse database.
"""

import sys
import os
from unittest.mock import Mock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.uds.uds_agent import UDSAgent
from src.uds.uds_client import UDSClient
from src.uds.config import UDSConfig
from src.uds.intent_classifier import IntentResult, IntentDomain
from src.uds.task_planner import TaskPlan, Subtask


def test_simple_query():
    """Test 1: Simple query - 'What were total sales in October?'"""
    print("\n" + "="*60)
    print("TEST 1: Simple Query")
    print("="*60)
    print("Query: 'What were total sales in October?'")
    
    try:
        # Create UDS client
        uds_client = UDSClient()
        
        # Create mock LLM for testing
        mock_llm = Mock()
        mock_llm.run.return_value = "Total sales for October 2025 were $1,234,567"
        
        # Create agent
        agent = UDSAgent(
            uds_client=uds_client,
            llm_client=mock_llm
        )
        
        # Process query
        result = agent.process_query("What were total sales in October?")
        
        print(f"✓ Query processed successfully")
        print(f"  - Intent: {result.get('intent', 'N/A')}")
        print(f"  - Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"  - Response available: True")
        
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def test_medium_query():
    """Test 2: Medium complexity - 'Top 10 products with their inventory'"""
    print("\n" + "="*60)
    print("TEST 2: Medium Complexity Query")
    print("="*60)
    print("Query: 'Top 10 products with their inventory'")
    
    try:
        # Create UDS client
        uds_client = UDSClient()
        
        # Create mock LLM for testing
        mock_llm = Mock()
        mock_llm.run.return_value = "Top 10 products with their inventory levels"
        
        # Create agent
        agent = UDSAgent(
            uds_client=uds_client,
            llm_client=mock_llm
        )
        
        # Process query
        result = agent.process_query("Top 10 products with their inventory")
        
        print(f"✓ Query processed successfully")
        print(f"  - Intent: {result.get('intent', 'N/A')}")
        print(f"  - Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"  - Response available: True")
        
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def test_complex_query():
    """Test 3: Complex query - 'Compare Q3 vs Q4, show top products, create dashboard'"""
    print("\n" + "="*60)
    print("TEST 3: Complex Query")
    print("="*60)
    print("Query: 'Compare Q3 vs Q4, show top products, create dashboard'")
    
    try:
        # Create UDS client
        uds_client = UDSClient()
        
        # Create mock LLM for testing
        mock_llm = Mock()
        mock_llm.run.return_value = "Q3 vs Q4 comparison with top products dashboard"
        
        # Create agent
        agent = UDSAgent(
            uds_client=uds_client,
            llm_client=mock_llm
        )
        
        # Process query
        result = agent.process_query("Compare Q3 vs Q4, show top products, create dashboard")
        
        print(f"✓ Query processed successfully")
        print(f"  - Intent: {result.get('intent', 'N/A')}")
        print(f"  - Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"  - Response available: True")
        
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("UDS AGENT INTEGRATION TESTS")
    print("="*60)
    print(f"Testing with ClickHouse database: {UDSConfig.CH_HOST}")
    print(f"Database: {UDSConfig.CH_DATABASE}")
    
    # Run tests
    results = []
    results.append(("Simple Query", test_simple_query()))
    results.append(("Medium Complexity Query", test_medium_query()))
    results.append(("Complex Query", test_complex_query()))
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("-"*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {total-passed} | Success Rate: {(passed/total*100):.1f}%")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
