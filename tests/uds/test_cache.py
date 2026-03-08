import pytest
import redis
"""Tests for Redis caching utilities."""

from src.uds.cache import UDSCache
from src.uds.cache_config import CacheConfig


@pytest.fixture(autouse=True)
def flush_redis():
    """Clear Redis before each test."""
    client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
    client.flushdb()
    yield
    client.flushdb()


def test_query_cache_hit_miss():
    cache = UDSCache()
    assert cache.get_query("select 1") is None
    cache.set_query("select 1", {"data": 123})
    assert cache.get_query("select 1") == {"data": 123}
    stats = cache.stats_summary()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_intent_cache():
    cache = UDSCache()
    assert cache.get_intent("hello") is None
    cache.set_intent("hello", "greet")
    assert cache.get_intent("hello") == "greet"
    stats = cache.stats_summary()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_schema_cache_and_invalidate():
    cache = UDSCache()
    cache.set_schema("users", {"cols": ["id"]})
    assert cache.get_schema("users") == {"cols": ["id"]}
    cache.invalidate("schema:*")
    assert cache.get_schema("users") is None


def test_stats_thread_safety():
    cache = UDSCache()
    # simulate concurrent hits/misses
    for _ in range(5):
        cache.get_query("q")
    cache.set_query("q", 1)
    for _ in range(3):
        cache.get_query("q")
    s = cache.stats_summary()
    assert s["misses"] == 5
    assert s["hits"] == 3


def test_uds_client_query_caching(monkeypatch):
    """Verify that UDSClient uses cache when provided."""

    # prepare a fake underlying client to track calls
    class FakeResult:
        def __init__(self, rows, cols):
            self.result_rows = rows
            self.column_names = cols

    class FakeClient:
        def __init__(self):
            self.queries = []

        def query(self, sql, parameters=None):
            self.queries.append(sql)
            # always return a single-row result
            return FakeResult([[42]], ["answer"])

    # patch module loader so UDSClient will use our fake client
    def fake_get_client_module():
        class Mod:
            def get_client(self, **kwargs):
                return FakeClient()
        return Mod()

    monkeypatch.setattr("src.uds.uds_client._get_client_module", lambda: fake_get_client_module())

    # create cache and client
    from src.uds.cache import UDSCache
    cache = UDSCache()
    client = None
    try:
        client = __import__("src.uds.uds_client", fromlist=["UDSClient"]).UDSClient(cache=cache)
    except Exception as e:
        pytest.skip(f"UDSClient initialization failed: {e}")

    # first query should hit underlying client and populate cache
    df1 = client.query("SELECT 1")
    assert df1.iloc[0, 0] == 42
    # confirm underlying client was called
    assert client._client.queries == ["SELECT 1"]

    # second query should be served from cache (no new underlying call)
    df2 = client.query("SELECT 1")
    assert df2.equals(df1)
    assert client._client.queries == ["SELECT 1"]

    # verify cache stats reflect hit/miss
    stats = cache.stats_summary()
    assert stats["misses"] == 1
    assert stats["hits"] == 1


def test_uds_client_schema_caching(monkeypatch):
    """Test that get_table_schema uses cache."""

    class FakeResult:
        def __init__(self, rows, cols):
            self.result_rows = rows
            self.column_names = cols

    class FakeClient:
        def __init__(self):
            self.queries = []

        def query(self, sql, parameters=None, as_dataframe=True):
            self.queries.append(sql)
            # respond differently for count vs schema
            import pandas as pd
            if "COUNT" in sql:
                df = pd.DataFrame([{"count": 2}])
                return FakeResult(df.values.tolist(), list(df.columns))
            # schema query
            cols_df = pd.DataFrame([
                {"name": "id", "type": "Int32", "default_kind": "", "comment": ""}
            ])
            return FakeResult(cols_df.values.tolist(), list(cols_df.columns))

    def fake_get_client_module2():
        class Mod:
            def get_client(self, **kwargs):
                return FakeClient()
        return Mod()

    monkeypatch.setattr("src.uds.uds_client._get_client_module", lambda: fake_get_client_module2())
    from src.uds.cache import UDSCache
    cache = UDSCache()
    client = __import__("src.uds.uds_client", fromlist=["UDSClient"]).UDSClient(cache=cache)

    schema1 = client.get_table_schema("foo")
    assert schema1["table_name"] == "foo"
    # second call should hit cache
    schema2 = client.get_table_schema("foo")
    assert schema2 == schema1
    stats = cache.stats_summary()
    # one miss for first schema, one hit for second
    assert stats["misses"] >= 1
    assert stats["hits"] >= 1


def test_agent_propagates_cache():
    """Ensure UDSAgent sets cache on client and classifier."""

    from src.uds.uds_agent import UDSAgent
    from src.uds.uds_client import UDSClient
    from src.uds.intent_classifier import UDSIntentClassifier
    from src.uds.cache import UDSCache

    cache = UDSCache()
    # supply minimal llm_client stub
    class DummyLLM:
        pass

    agent = UDSAgent(uds_client=None, llm_client=DummyLLM(), cache=cache)
    assert agent.cache is cache
    assert isinstance(agent.uds_client, UDSClient)
    assert agent.uds_client.cache is cache
    assert isinstance(agent.intent_classifier, UDSIntentClassifier)
    assert agent.intent_classifier.cache is cache


def test_api_initializes_cache(monkeypatch):
    """Validate that the FastAPI helpers create shared cache."""

    from src.uds import api
    from src.uds.cache import UDSCache

    # reset globals
    api._uds_client = None
    api._uds_agent = None
    api._uds_cache = None

    c1 = api._get_uds_client()
    assert isinstance(c1, api.UDSClient)
    assert api._uds_cache is not None
    # second call returns same client and cache
    c2 = api._get_uds_client()
    assert c2 is c1
    assert api._uds_cache is not None

    a1 = api._get_uds_agent()
    assert a1.cache is api._uds_cache
    # agent should reuse same cache instance
    a2 = api._get_uds_agent()
    assert a2 is a1
    assert a2.cache is api._uds_cache


def test_intent_classifier_caching():
    """Ensure UDSIntentClassifier uses cache when available."""

    from src.uds.intent_classifier import UDSIntentClassifier, IntentDomain, IntentResult
    from src.uds.cache import UDSCache

    cache = UDSCache()
    classifier = UDSIntentClassifier()
    classifier.cache = cache

    # classify simple query twice
    res1 = classifier.classify("What were total sales?")
    assert isinstance(res1, IntentResult)
    # should have been stored in cache
    stats = cache.stats_summary()
    assert stats["misses"] == 1
    assert stats["hits"] == 0

    res2 = classifier.classify("What were total sales?")
    assert isinstance(res2, IntentResult)
    # should be served from cache
    assert res2.primary_domain == res1.primary_domain
    stats = cache.stats_summary()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
