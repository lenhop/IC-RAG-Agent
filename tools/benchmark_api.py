"""
API Performance Benchmarks.

Measures response times for UDS REST API endpoints.
Requires: API server running (uvicorn src.uds.api:app --port 8000)

Usage:
  python tools/benchmark_api.py
  UDS_API_URL=http://localhost:8000 python tools/benchmark_api.py

Output: Mean, median, min, max, stdev for each query.
"""

import os
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("Error: requests required. pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("UDS_API_URL", "http://localhost:8000")
ITERATIONS = int(os.getenv("UDS_BENCHMARK_ITERATIONS", "5"))
TIMEOUT = int(os.getenv("UDS_API_TIMEOUT", "60"))


def check_api_available() -> bool:
    """Verify API is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


def benchmark_query(query: str, iterations: int = ITERATIONS) -> dict:
    """
    Benchmark a single query via POST /api/v1/uds/query.

    Returns:
        Dict with mean, median, min, max, stdev, success_count
    """
    times = []
    errors = 0

    for _ in range(iterations):
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/uds/query",
                json={"query": query},
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "completed":
                    times.append(elapsed)
                else:
                    errors += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    if not times:
        return {
            "query": query,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "stdev": 0,
            "success_count": 0,
            "error_count": errors,
        }

    return {
        "query": query,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "success_count": len(times),
        "error_count": errors,
    }


def main():
    """Run benchmarks and print results."""
    print("UDS API Performance Benchmarks")
    print("=" * 80)
    print(f"Base URL: {BASE_URL}")
    print(f"Iterations per query: {ITERATIONS}")
    print()

    if not check_api_available():
        print(f"Error: API not available at {BASE_URL}")
        print("Start with: uvicorn src.uds.api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    queries = [
        ("Simple", "What were total sales in October?"),
        ("Simple", "Top 10 products by revenue"),
        ("Simple", "Show me current inventory levels"),
        ("Medium", "Compare first half vs second half of October sales"),
        ("Medium", "Financial summary for October"),
        ("Complex", "Analyze October sales trends, identify top products, check inventory"),
    ]

    results = []
    for category, query in queries:
        print(f"Benchmarking [{category}]: {query[:50]}...")
        result = benchmark_query(query)
        result["category"] = category
        results.append(result)

        if result["success_count"] > 0:
            print(
                f"  Mean: {result['mean']:.3f}s "
                f"(+/- {result['stdev']:.3f}s) "
                f"Range: {result['min']:.3f}s - {result['max']:.3f}s"
            )
        else:
            print(f"  No successful runs (errors: {result['error_count']})")
        print()

    print("=" * 80)
    print("Summary")
    print("-" * 80)

    simple_results = [item for item in results if item["category"] == "Simple" and item["success_count"] > 0]
    if simple_results:
        simple_mean = statistics.mean(item["mean"] for item in simple_results)
        print(f"Simple queries avg: {simple_mean:.3f}s (target: <5s)")
        if simple_mean > 5:
            print("  WARNING: Simple queries exceed 5s target")

    complex_results = [item for item in results if item["category"] == "Complex" and item["success_count"] > 0]
    if complex_results:
        complex_mean = statistics.mean(item["mean"] for item in complex_results)
        print(f"Complex queries avg: {complex_mean:.3f}s (target: <15s)")
        if complex_mean > 15:
            print("  WARNING: Complex queries exceed 15s target")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
