"""
Redis-based caching layer for UDS Agent.

Provides simple get/set methods with TTLs, as well as helpers for
intent classification, query results, and schema metadata. Keeps
cache statistics (hits/misses) and allows configuration via
`cache_config.CacheConfig`.
"""

import redis
import json
import threading
from typing import Any, Optional
from .cache_config import CacheConfig


class CacheStats:
    """Simple cache statistics tracker."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def record_hit(self):
        with self.lock:
            self.hits += 1

    def record_miss(self):
        with self.lock:
            self.misses += 1

    def summary(self) -> dict:
        with self.lock:
            return {"hits": self.hits, "misses": self.misses}


class UDSCache:
    """Redis cache wrapper."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", config: CacheConfig = CacheConfig()):
        self._client = redis.from_url(redis_url, decode_responses=True)
        self.config = config
        self.stats = CacheStats()

    def _key(self, prefix: str, key: str) -> str:
        return prefix + key

    def get(self, prefix: str, key: str) -> Optional[Any]:
        k = self._key(prefix, key)
        val = self._client.get(k)
        if val is None:
            self.stats.record_miss()
            return None
        self.stats.record_hit()
        try:
            return json.loads(val)
        except Exception:
            return val

    def set(self, prefix: str, key: str, value: Any, ttl: int):
        k = self._key(prefix, key)
        try:
            payload = json.dumps(value)
        except Exception:
            payload = str(value)
        self._client.set(k, payload, ex=ttl)

    # convenience methods
    def get_query(self, query_str: str) -> Optional[Any]:
        return self.get(self.config.PREFIX_QUERY, query_str)

    def set_query(self, query_str: str, result: Any):
        self.set(self.config.PREFIX_QUERY, query_str, result, self.config.QUERY_TTL)

    def get_intent(self, text: str) -> Optional[Any]:
        return self.get(self.config.PREFIX_INTENT, text)

    def set_intent(self, text: str, intent: Any):
        self.set(self.config.PREFIX_INTENT, text, intent, self.config.INTENT_TTL)

    def get_schema(self, name: str) -> Optional[Any]:
        return self.get(self.config.PREFIX_SCHEMA, name)

    def set_schema(self, name: str, schema: Any):
        self.set(self.config.PREFIX_SCHEMA, name, schema, self.config.SCHEMA_TTL)

    def invalidate(self, pattern: str):
        """Invalidate keys matching pattern."""
        # pattern should include prefix
        for key in self._client.scan_iter(match=pattern):
            self._client.delete(key)

    def stats_summary(self) -> dict:
        return self.stats.summary()
