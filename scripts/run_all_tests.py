#!/usr/bin/env python3
"""
Comprehensive Test Runner for UDS Agent Production Validation

Runs all test suites and collects results:
- Smoke Tests
- Load Tests
- Security Tests
- User Acceptance Tests (UAT)
- Performance Benchmarks
- Disaster Recovery Tests

Collects results and generates comprehensive report.
"""

import subprocess
import sys
import os
import json
from datetime import datetime


def run_test_suite(test_name: str, test_file: str) -> dict:
    """Run a test suite and collect results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name}...")
    print(f"{'=' * 60}")
    
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', test_file, '-v', '--tb=short', '--json-report', f'/tmp/{test_name}-report.json'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        success = result.returncode == 0
        output = result.stdout
        
        # Parse JSON report if available
        report_file = f'/tmp/{test_name}-report.json'
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report_data = json.load(f)
                passed = report_data.get('summary', {}).get('passed', 0)
                failed = report_data.get('summary', {}).get('failed', 0)
                total = passed + failed
                pass_rate = (passed / total * 100) if total > 0 else 0
                
                print(f"\nResults:")
                print(f"  Total: {total}")
                print(f"  Passed: {passed}")
                print(f"  Failed: {failed}")
                print(f"  Pass Rate: {pass_rate:.1f}%")
                
                return {
                    'test_name': test_name,
                    'success': success,
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'pass_rate': pass_rate,
                    'output': output
                }
        else:
            print(f"\nResults:")
            print(f"  Success: {success}")
            print(f"  Output length: {len(output)}")
            
            return {
                'test_name': test_name,
                'success': success,
                'output': output
            }
    except subprocess.TimeoutExpired:
        print(f"\n✗ {test_name} timed out after 10 minutes")
        return {
            'test_name': test_name,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"\n✗ {test_name} failed: {e}")
        return {
            'test_name': test_name,
            'success': False,
            'error': str(e)
        }


def generate_comprehensive_report(results: list):
    """Generate comprehensive test report."""
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'=' * 60}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Summary
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    print("SUMMARY")
    print("-" * 60)
    print(f"Total Test Suites: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {successful_tests / total_tests * 100:.1f}%")
    print("-" * 60)
    
    # Detailed results
    print("DETAILED RESULTS")
    print("-" * 60)
    
    for result in results:
        print(f"\n{result['test_name'].upper()}")
        print("-" * 60)
        
        if result['success']:
            if 'total' in result:
                print(f"Total Tests: {result['total']}")
                print(f"Passed: {result['passed']}")
                print(f"Failed: {result['failed']}")
                print(f"Pass Rate: {result['pass_rate']:.1f}%")
            else:
                print(f"Status: ✓ PASSED")
        else:
            print(f"Status: ✗ FAILED")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("-" * 60)
    
    # Acceptance criteria
    print("ACCEPTANCE CRITERIA STATUS")
    print("-" * 60)
    
    # Smoke tests
    smoke_results = [r for r in results if 'smoke' in r['test_name'].lower()]
    smoke_success = sum(1 for r in smoke_results if r['success'])
    print(f"Smoke Tests: {smoke_success}/{len(smoke_results)} (100% passing)" if smoke_success == len(smoke_results) else f"Smoke Tests: {smoke_success}/{len(smoke_results)} (FAILED)")
    
    # Load tests
    load_results = [r for r in results if 'load' in r['test_name'].lower()]
    load_success = sum(1 for r in load_results if r['success'])
    print(f"Load Tests: {load_success}/{len(load_results)} (100% passing)" if load_success == len(load_results) else f"Load Tests: {load_success}/{len(load_results)} (FAILED)")
    
    # Security tests
    security_results = [r for r in results if 'security' in r['test_name'].lower()]
    security_success = sum(1 for r in security_results if r['success'])
    print(f"Security Tests: {security_success}/{len(security_results)} (100% passing)" if security_success == len(security_results) else f"Security Tests: {security_success}/{len(security_results)} (FAILED)")
    
    # UAT tests
    uat_results = [r for r in results if 'uat' in r['test_name'].lower()]
    uat_success = sum(1 for r in uat_results if r['success'])
    print(f"UAT Tests: {uat_success}/{len(uat_results)} (100% passing)" if uat_success == len(uat_results) else f"UAT Tests: {uat_success}/{len(uat_results)} (FAILED)")
    
    # Performance benchmarks
    perf_results = [r for r in results if 'perf' in r['test_name'].lower()]
    perf_success = sum(1 for r in perf_results if r['success'])
    print(f"Performance Benchmarks: {perf_success}/{len(perf_results)} (100% passing)" if perf_success == len(perf_results) else f"Performance Benchmarks: {perf_success}/{len(perf_results)} (FAILED)")
    
    # Disaster recovery tests
    dr_results = [r for r in results if 'dr' in r['test_name'].lower()]
    dr_success = sum(1 for r in dr_results if r['success'])
    print(f"Disaster Recovery Tests: {dr_success}/{len(dr_results)} (100% passing)" if dr_success == len(dr_results) else f"Disaster Recovery Tests: {dr_success}/{len(dr_results)} (FAILED)")
    
    print("-" * 60)
    
    # Overall assessment
    all_success = successful_tests == total_tests
    print(f"\nOVERALL ASSESSMENT: {'✓ PASSED' if all_success else '✗ FAILED'}")
    print("-" * 60)
    
    # Go-live recommendation
    print("GO-LIVE RECOMMENDATION")
    print("-" * 60)
    
    if all_success:
        print("✓ YES - All tests passed")
        print("✓ System is ready for production deployment")
        print("✓ Proceed with go-live decision")
    else:
        print("✗ NO - Some tests failed")
        print("✗ Review failed tests and fix issues")
        print("✗ Re-run tests after fixes")
        print("✗ Do not proceed with go-live")
    
    print("-" * 60)
    print(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def main():
    """Main function to run all test suites."""
    print("=" * 60)
    print("UDS AGENT COMPREHENSIVE TEST RUNNER")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all test suites
    results = []
    
    # Smoke tests
    print("\n" + "=" * 60)
    print("RUNNING SMOKE TESTS")
    print("=" * 60)
    result = run_test_suite(
        "Smoke Tests",
        "tests/test_smoke.py::TestHealthCheck::test_health_endpoint"
    )
    results.append(result)
    
    # Load tests
    print("\n" + "=" * 60)
    print("RUNNING LOAD TESTS")
    print("=" * 60)
    result = run_test_suite(
        "Load Tests",
        "tests/load_test.py::TestLoadScenarios"
    )
    results.append(result)
    
    # Security tests
    print("\n" + "=" * 60)
    print("RUNNING SECURITY TESTS")
    print("=" * 60)
    result = run_test_suite(
        "Security Tests",
        "tests/security_test.py"
    )
    results.append(result)
    
    # UAT tests
    print("\n" + "=" * 60)
    print("RUNNING USER ACCEPTANCE TESTS (UAT)")
    print("=" * 60)
    result = run_test_suite(
        "User Acceptance Tests (UAT)",
        "tests/uat_test.py"
    )
    results.append(result)
    
    # Performance benchmarks
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    result = run_test_suite(
        "Performance Benchmarks",
        "tests/perf_benchmark.py"
    )
    results.append(result)
    
    # Disaster recovery tests
    print("\n" + "=" * 60)
    print("RUNNING DISASTER RECOVERY TESTS")
    print("=" * 60)
    result = run_test_suite(
        "Disaster Recovery Tests",
        "tests/dr_test.py"
    )
    results.append(result)
    
    # Generate comprehensive report
    generate_comprehensive_report(results)


if __name__ == '__main__':
    main()
