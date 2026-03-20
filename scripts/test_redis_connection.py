#!/usr/bin/env python3
"""
Test Redis connection for gateway short-term memory.

Uses GATEWAY_REDIS_URL from .env (default redis://localhost:6379/0).
Writes a short-lived list key, reads it back, then deletes it.

Usage:
  python scripts/test_redis_connection.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    redis_url = os.getenv("GATEWAY_REDIS_URL", "redis://localhost:6379/0")
    print(f"Connecting to Redis: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")

    try:
        import redis
    except ImportError:
        print("ERROR: redis package not installed. Run: pip install redis")
        return 1

    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        print("  Ping: OK")
    except redis.ConnectionError as e:
        print(f"  ERROR: Connection failed - {e}")
        return 1

    # Test write: same format as gateway short-term memory
    test_user_id = "test_user_redis_check"
    test_session_id = "test_session_001"
    test_key = f"gateway:user:{test_user_id}:history"

    test_event = {
        "ts_utc": "2026-03-13T12:00:00.000Z",
        "user_id": test_user_id,
        "session_id": test_session_id,
        "request_id": "test-req-001",
        "event_type": "user_query",
        "event_content": "Redis connection test message",
        "status": "ok",
        "note": "test_redis_connection.py",
    }

    try:
        client.rpush(test_key, json.dumps(test_event))
        client.expire(test_key, 60)  # 1 min TTL for test key
        print("  Write: OK")

        # Read back
        items = client.lrange(test_key, -1, -1)
        if not items:
            print("  Read: FAILED (empty)")
            return 1

        parsed = json.loads(items[0])
        if parsed.get("event_content") == test_event["event_content"]:
            print("  Read: OK")
        else:
            print(f"  Read: mismatch - got {parsed}")

        # Cleanup
        client.delete(test_key)
        print("  Cleanup: OK")

    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    print("\nSUCCESS: Redis connection and message send verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
