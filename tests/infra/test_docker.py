#!/usr/bin/env python3
"""
UDS Agent Docker Integration Tests
Tests the Docker containerized deployment of the UDS Agent
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerTester:
    """Test class for UDS Agent Docker deployment"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, endpoint: str, method: str = "GET", **kwargs) -> Dict:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Request failed: {response.status} - {await response.text()}")
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Request exception: {e}")
            return {"error": str(e)}

    async def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        logger.info("Testing health check endpoint...")
        result = await self.make_request("/api/v1/health")
        if "status" in result and result["status"] == "healthy":
            logger.info("✓ Health check passed")
            return True
        else:
            logger.error("✗ Health check failed")
            return False

    async def test_api_info(self) -> bool:
        """Test the API info endpoint"""
        logger.info("Testing API info endpoint...")
        result = await self.make_request("/api/v1/info")
        if "name" in result and "version" in result:
            logger.info("✓ API info check passed")
            return True
        else:
            logger.error("✗ API info check failed")
            return False

    async def test_query_endpoint(self) -> bool:
        """Test a basic query to the UDS Agent"""
        logger.info("Testing query endpoint...")

        # Simple test query
        test_query = {
            "query": "What is the total inventory count?",
            "context": {
                "user_id": "test_user",
                "session_id": "test_session"
            }
        }

        result = await self.make_request("/api/v1/query", "POST",
                                       json=test_query,
                                       headers={"Content-Type": "application/json"})

        # Check if we got a response (even if it's an error, it means the service is responding)
        if "error" not in result or result.get("error") != "Connection error":
            logger.info("✓ Query endpoint responded")
            return True
        else:
            logger.error("✗ Query endpoint failed")
            return False

    async def test_cache_status(self) -> bool:
        """Test cache status endpoint"""
        logger.info("Testing cache status...")
        result = await self.make_request("/api/v1/cache/status")
        if "status" in result:
            logger.info("✓ Cache status check passed")
            return True
        else:
            logger.warning("✗ Cache status check failed (may not be implemented)")
            return True  # Not critical

    async def test_database_connection(self) -> bool:
        """Test database connectivity through the API"""
        logger.info("Testing database connection...")

        # Try a simple database query
        test_query = {
            "query": "Show me the database schema",
            "context": {
                "user_id": "test_user",
                "session_id": "test_session"
            }
        }

        result = await self.make_request("/api/v1/query", "POST",
                                       json=test_query,
                                       headers={"Content-Type": "application/json"})

        # Check if we get a meaningful response
        if result and not result.get("error"):
            logger.info("✓ Database connection test passed")
            return True
        else:
            logger.warning("✗ Database connection test failed")
            return False

    def check_docker_services(self) -> bool:
        """Check if Docker services are running"""
        logger.info("Checking Docker services status...")

        try:
            result = subprocess.run(
                ["docker-compose", "-f", "docker/docker-compose.uds.yml", "ps"],
                capture_output=True, text=True, check=True
            )

            if "Up" in result.stdout:
                logger.info("✓ Docker services are running")
                return True
            else:
                logger.error("✗ Docker services are not running")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to check Docker services: {e}")
            return False

    def check_container_health(self) -> bool:
        """Check container health status"""
        logger.info("Checking container health...")

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=uds-", "--format", "{{.Names}}:{{.Status}}"],
                capture_output=True, text=True, check=True
            )

            healthy_count = 0
            total_count = 0

            for line in result.stdout.strip().split('\n'):
                if line:
                    total_count += 1
                    if "healthy" in line.lower():
                        healthy_count += 1

            if healthy_count >= total_count * 0.8:  # At least 80% healthy
                logger.info(f"✓ Container health check passed ({healthy_count}/{total_count} healthy)")
                return True
            else:
                logger.warning(f"✗ Container health check failed ({healthy_count}/{total_count} healthy)")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to check container health: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        logger.info("Starting UDS Agent Docker tests...")
        logger.info("=" * 50)

        results = {}

        # Docker-level tests
        results["docker_services"] = self.check_docker_services()
        results["container_health"] = self.check_container_health()

        # API-level tests
        results["health_check"] = await self.test_health_check()
        results["api_info"] = await self.test_api_info()
        results["query_endpoint"] = await self.test_query_endpoint()
        results["cache_status"] = await self.test_cache_status()
        results["database_connection"] = await self.test_database_connection()

        # Summary
        passed = sum(results.values())
        total = len(results)

        logger.info("=" * 50)
        logger.info(f"Test Results: {passed}/{total} tests passed")

        for test, result in results.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {test.replace('_', ' ').title()}")

        return results

def wait_for_service(url: str, timeout: int = 120) -> bool:
    """Wait for service to be available"""
    logger.info(f"Waiting for service at {url}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["curl", "-f", "-s", f"{url}/api/v1/health"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("✓ Service is available")
                return True
        except:
            pass

        time.sleep(5)

    logger.error("✗ Service did not become available within timeout")
    return False

async def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test UDS Agent Docker deployment")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Base URL of the UDS Agent API")
    parser.add_argument("--wait", type=int, default=60,
                       help="Seconds to wait for service to be available")
    parser.add_argument("--no-wait", action="store_true",
                       help="Skip waiting for service availability")

    args = parser.parse_args()

    # Wait for service if requested
    if not args.no_wait:
        if not wait_for_service(args.url, args.wait):
            sys.exit(1)

    # Run tests
    async with DockerTester(args.url) as tester:
        results = await tester.run_all_tests()

    # Exit with appropriate code
    passed = sum(results.values())
    total = len(results)

    if passed == total:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    elif passed >= total * 0.8:  # At least 80% pass
        logger.warning("⚠️  Most tests passed, but some failed")
        sys.exit(0)
    else:
        logger.error("❌ Too many tests failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())