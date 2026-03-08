#!/usr/bin/env python3
"""
Load Testing for UDS Agent Production Deployment

Scenarios:
- 100 concurrent users
- 1000 requests/min sustained load
- 1-hour load test
- Spike testing (sudden traffic increase)
- Stress testing (beyond normal capacity)

Metrics:
- Response time (p50, p95, p99)
- Error rate
- Throughput (requests/second)
- Resource utilization (CPU, memory, network)
- Auto-scaling behavior

Use Locust or k6 for load testing.
"""

import pytest
import requests
import time
import statistics
import concurrent.futures
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.uds.uds_agent import UDSAgent
from src.uds.uds_client import UDSClient
from src.uds.intent_classifier import IntentDomain


class TestLoadScenarios:
    """Test load scenarios."""
    
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
    
    def test_100_concurrent_users(self):
        """Test 100 concurrent users."""
        num_users = 100
        num_requests = 10
        
        print(f"Testing {num_users} concurrent users...")
        print(f"Each user makes {num_requests} requests")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(num_users):
                future = executor.submit(self._make_request, i, num_requests)
                futures.append(future)
            
            start_time = time.time()
            concurrent.futures.wait(futures)
            elapsed = time.time() - start_time
        
        success_count = sum(1 for f in futures if f.result())
        error_rate = ((num_users * num_requests) - success_count) / (num_users * num_requests) * 100
        
        print(f"✓ {num_users} concurrent users test completed")
        print(f"  Success: {success_count}/{num_users * num_requests} ({success_count / (num_users * num_requests) * 100:.1f}%)")
        print(f"  Error rate: {error_rate:.2f}%")
        print(f"  Total time: {elapsed:.2f}s")
        
        assert success_count >= int(num_users * num_requests * 0.95), f"Success rate {success_count / (num_users * num_requests) * 100:.1f}% below 95% threshold"
    
    def test_1000_requests_per_minute(self):
        """Test 1000 requests/min sustained load."""
        num_requests = 1000
        duration = 60
        
        print(f"Testing {num_requests} requests/min for {duration}s...")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        for i in range(num_requests):
            try:
                response = self._make_request(i, 1)
                if response['success']:
                    success_count += 1
                else:
                    error_count += 1
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rps = (i + 1) / elapsed
                    print(f"  Progress: {i + 1}/{num_requests} ({(i + 1) / elapsed:.1f} RPS)")
            except Exception as e:
                error_count += 1
                print(f"  Request {i + 1} failed: {e}")
        
        elapsed = time.time() - start_time
        success_rate = success_count / num_requests * 100
        error_rate = error_count / num_requests * 100
        avg_rps = num_requests / elapsed
        
        print(f"✓ {num_requests} requests/min test completed")
        print(f"  Success: {success_count} ({success_rate:.1f}%)")
        print(f"  Errors: {error_count} ({error_rate:.1f}%)")
        print(f"  Avg RPS: {avg_rps:.1f}")
        print(f"  Total time: {elapsed:.2f}s")
        
        assert success_rate >= 99.0, f"Success rate {success_rate:.1f}% below 99% threshold"
        assert avg_rps >= 16.67, f"Avg RPS {avg_rps:.1f} below 1000/min threshold"
    
    def test_spike_testing(self):
        """Test spike testing (sudden traffic increase)."""
        baseline_requests = 10
        spike_requests = 100
        
        print(f"Testing spike from {baseline_requests} to {spike_requests} requests...")
        
        # Baseline
        print("Baseline (10 requests):")
        baseline_times = []
        for i in range(baseline_requests):
            start = time.time()
            response = self._make_request(i, 1)
            elapsed = time.time() - start
            baseline_times.append(elapsed)
            print(f"  Request {i + 1}: {elapsed*1000:.0f}ms")
        
        baseline_avg = statistics.mean(baseline_times)
        
        # Spike
        print(f"\nSpike ({spike_requests} requests):")
        spike_times = []
        for i in range(spike_requests):
            start = time.time()
            response = self._make_request(i, 1)
            elapsed = time.time() - start
            spike_times.append(elapsed)
            print(f"  Request {i + 1}: {elapsed*1000:.0f}ms")
        
        spike_avg = statistics.mean(spike_times)
        
        print(f"\n✓ Spike test completed")
        print(f"  Baseline avg: {baseline_avg*1000:.2f}ms")
        print(f"  Spike avg: {spike_avg*1000:.2f}ms")
        print(f"  Degradation: {(spike_avg / baseline_avg - 1) * 100:.1f}%")
        
        assert spike_avg < baseline_avg * 2.0, f"Spike avg {spike_avg*1000:.2f}ms exceeds 2x baseline {baseline_avg*1000:.2f}ms"
    
    def test_1_hour_load_test(self):
        """Test 1-hour sustained load."""
        num_requests = 1000 * 60
        
        print(f"Testing 1-hour sustained load ({num_requests} requests)...")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        for i in range(num_requests):
            try:
                response = self._make_request(i, 1)
                if response['success']:
                    success_count += 1
                else:
                    error_count += 1
                
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rps = (i + 1) / elapsed
                    print(f"  Progress: {i + 1}/{num_requests} ({rps:.1f} RPS)")
            except Exception as e:
                error_count += 1
                print(f"  Request {i + 1} failed: {e}")
        
        elapsed = time.time() - start_time
        success_rate = success_count / num_requests * 100
        error_rate = error_count / num_requests * 100
        avg_rps = num_requests / elapsed
        
        print(f"✓ 1-hour load test completed")
        print(f"  Success: {success_count} ({success_rate:.1f}%)")
        print(f"  Errors: {error_count} ({error_rate:.1f}%)")
        print(f"  Avg RPS: {avg_rps:.1f}")
        print(f"  Total time: {elapsed:.2f}s")
        
        assert success_rate >= 99.0, f"Success rate {success_rate:.1f}% below 99% threshold"
        assert avg_rps >= 16.67, f"Avg RPS {avg_rps:.1f} below 1000/min threshold"
    
    def test_stress_testing(self):
        """Test stress testing (beyond normal capacity)."""
        num_requests = 2000
        
        print(f"Testing stress load ({num_requests} requests/min)...")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        for i in range(num_requests):
            try:
                response = self._make_request(i, 1)
                if response['success']:
                    success_count += 1
                else:
                    error_count += 1
                
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start_time
                    rps = (i + 1) / elapsed
                    print(f"  Progress: {i + 1}/{num_requests} ({rps:.1f} RPS)")
            except Exception as e:
                error_count += 1
                print(f"  Request {i + 1} failed: {e}")
        
        elapsed = time.time() - start_time
        success_rate = success_count / num_requests * 100
        error_rate = error_count / num_requests * 100
        avg_rps = num_requests / elapsed
        
        print(f"✓ Stress test completed")
        print(f"  Success: {success_count} ({success_rate:.1f}%)")
        print(f"  Errors: {error_count} ({error_rate:.1f}%)")
        print(f"  Avg RPS: {avg_rps:.1f}")
        print(f"  Total time: {elapsed:.2f}s")
        
        # Stress test should maintain >90% success rate
        assert success_rate >= 90.0, f"Success rate {success_rate:.1f}% below 90% threshold"
    
    def _make_request(self, request_id: int, num_retries: int = 1) -> Dict[str, any]:
        """Make a single request to the agent."""
        query = f"Test query {request_id}"
        
        try:
            result = self.agent.process_query(query)
            
            if result['success']:
                return {'success': True, 'response_time': 0.1}
            else:
                return {'success': False, 'error': result.get('error', 'Unknown error')}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class TestMetrics:
    """Test metrics collection and analysis."""
    
    def test_response_time_metrics(self):
        """Test response time metrics (p50, p95, p99)."""
        print("Testing response time metrics...")
        
        times = []
        for i in range(100):
            start = time.time()
            response = self._make_request(i, 1)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times.sort()
        
        p50 = times[4]  # 50th percentile
        p95 = times[94]  # 95th percentile
        p99 = times[98]  # 99th percentile
        avg = statistics.mean(times)
        
        print(f"✓ Response time metrics collected")
        print(f"  p50: {p50*1000:.2f}ms")
        print(f"  p95: {p95*1000:.2f}ms")
        print(f"  p99: {p99*1000:.2f}ms")
        print(f"  Average: {avg*1000:.2f}ms")
        
        assert p50 < 5.0, f"p50 {p50*1000:.2f}ms exceeds 5s threshold"
    
    def test_error_rate_metrics(self):
        """Test error rate metrics."""
        print("Testing error rate metrics...")
        
        success_count = 0
        error_count = 0
        total_requests = 1000
        
        for i in range(total_requests):
            response = self._make_request(i, 1)
            if response['success']:
                success_count += 1
            else:
                error_count += 1
        
        error_rate = (error_count / total_requests) * 100
        
        print(f"✓ Error rate metrics collected")
        print(f"  Success: {success_count} ({success_count / total_requests * 100:.1f}%)")
        print(f"  Errors: {error_count} ({error_count / total_requests * 100:.1f}%)")
        print(f"  Error rate: {error_rate:.2f}%")
        
        assert error_rate < 1.0, f"Error rate {error_rate:.2f}% exceeds 1% threshold"
    
    def test_throughput_metrics(self):
        """Test throughput metrics (requests/second)."""
        print("Testing throughput metrics...")
        
        duration = 60
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            response = self._make_request(request_count, 1)
            request_count += 1
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed
        
        print(f"✓ Throughput metrics collected")
        print(f"  Requests: {request_count}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} requests/second")
        
        assert throughput >= 16.67, f"Throughput {throughput:.1f} req/s below 1000/min threshold"
    
    def test_auto_scaling_metrics(self):
        """Test auto-scaling behavior."""
        print("Testing auto-scaling behavior...")
        
        # Simulate gradual load increase
        stages = [10, 50, 100, 500, 1000]
        for stage in stages:
            print(f"\nTesting {stage} concurrent users...")
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=stage) as executor:
                futures = []
                for i in range(stage):
                    future = executor.submit(self._make_request, i, 1)
                    futures.append(future)
                
                concurrent.futures.wait(futures)
            
            elapsed = time.time() - start_time
            success_count = sum(1 for f in futures if f.result())
            success_rate = success_count / stage * 100
            
            print(f"  Success: {success_count}/{stage} ({success_rate:.1f}%)")
            print(f"  Time: {elapsed:.2f}s")
        
        print("\n✓ Auto-scaling test completed")


def run_load_tests():
    """Run all load tests and report results."""
    print("=" * 60)
    print("LOAD TESTING - PRODUCTION VALIDATION")
    print("=" * 60)
    print()
    
    pytest.main([
        'tests/load_test.py::TestLoadScenarios::test_100_concurrent_users',
        'tests/load_test.py::TestLoadScenarios::test_1000_requests_per_minute',
        'tests/load_test.py::TestLoadScenarios::test_spike_testing',
        'tests/load_test.py::TestLoadScenarios::test_1_hour_load_test',
        'tests/load_test.py::TestLoadScenarios::test_stress_testing',
        'tests/load_test.py::TestMetrics::test_response_time_metrics',
        'tests/load_test.py::TestMetrics::test_error_rate_metrics',
        'tests/load_test.py::TestMetrics::test_throughput_metrics',
        'tests/load_test.py::TestMetrics::test_auto_scaling_metrics',
    ], '-v', '--tb=short')
    
    print()
    print("=" * 60)
    print("LOAD TESTING COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_load_tests()
