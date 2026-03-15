"""
Cache configuration settings.

Defines TTLs, prefixes and invalidation rules for different cache types.
"""

from dataclasses import dataclass


@dataclass
class CacheConfig:
    # default time-to-live values (seconds)
    QUERY_TTL: int = 3600           # 1 hour for query results
    INTENT_TTL: int = 86400         # 1 day for intent classification
    SCHEMA_TTL: int = 86400         # 1 day for schema metadata

    # cache key prefixes
    PREFIX_QUERY: str = "query:"
    PREFIX_INTENT: str = "intent:"
    PREFIX_SCHEMA: str = "schema:"

    # additional prefixes used by the UDS agent
    PREFIX_USER_CONTEXT: str = "user:context:"

    # invalidation patterns
    INVALIDATE_ON_SCHEMA_CHANGE: bool = True
    INVALIDATE_ON_DATA_LOAD: bool = True

    # default prefix for all keys (prepended automatically if set)
    GLOBAL_PREFIX: str = "uds:"
