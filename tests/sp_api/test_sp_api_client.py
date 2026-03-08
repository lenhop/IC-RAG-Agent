"""Tests for SPAPIClient — token refresh, rate limiter, cache."""
import time
from unittest.mock import MagicMock, patch

import pytest

from sp_api.sp_api_client import (
    SPAPIClient,
    SPAPICredentials,
    _RateLimiter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def creds():
    return SPAPICredentials(
        refresh_token="test_refresh",
        client_id="test_client_id",
        client_secret="test_client_secret",
    )


@pytest.fixture
def client(creds):
    return SPAPIClient(creds)


def _mock_lwa_response(access_token="test_access_token", expires_in=3600):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"access_token": access_token, "expires_in": expires_in}
    return resp


def _mock_api_response(data: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# SPAPICredentials
# ---------------------------------------------------------------------------

def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("SP_API_REFRESH_TOKEN", "rt")
    monkeypatch.setenv("SP_API_CLIENT_ID", "cid")
    monkeypatch.setenv("SP_API_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SP_API_MARKETPLACE_ID", "ATVPDKIKX0DER")
    monkeypatch.setenv("SP_API_REGION", "us-east-1")
    monkeypatch.setenv("SP_API_ROLE_ARN", "")
    monkeypatch.setenv("SP_API_AWS_ACCESS_KEY", "")
    monkeypatch.setenv("SP_API_AWS_SECRET_KEY", "")
    monkeypatch.setenv("SP_API_APP_ID", "")
    c = SPAPICredentials.from_env()
    assert c.refresh_token == "rt"
    assert c.client_id == "cid"


def test_credentials_missing_required_raises():
    with pytest.raises(ValueError):
        SPAPICredentials(refresh_token="", client_id="cid", client_secret="cs")


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def test_token_refresh_on_first_call(client):
    with patch.object(client._http, "post", return_value=_mock_lwa_response()) as mock_post, \
         patch.object(client._http, "get", return_value=_mock_api_response({"payload": {}})):
        client.get("/orders/v0/orders")
        mock_post.assert_called_once()
        assert client._access_token == "test_access_token"


def test_token_not_refreshed_when_valid(client):
    # Pre-set a valid token
    client._access_token = "existing_token"
    client._token_expiry = time.monotonic() + 3600

    with patch.object(client._http, "post") as mock_post, \
         patch.object(client._http, "get", return_value=_mock_api_response({})):
        client.get("/orders/v0/orders")
        mock_post.assert_not_called()


def test_token_refreshed_when_near_expiry(client):
    # Token expires in 30s (within 60s threshold)
    client._access_token = "old_token"
    client._token_expiry = time.monotonic() + 30

    with patch.object(client._http, "post", return_value=_mock_lwa_response("new_token")) as mock_post, \
         patch.object(client._http, "get", return_value=_mock_api_response({})):
        client.get("/orders/v0/orders")
        mock_post.assert_called_once()
        assert client._access_token == "new_token"


def test_get_auth_header_returns_bearer(client):
    with patch.object(client._http, "post", return_value=_mock_lwa_response("tok123")):
        header = client._get_auth_header()
    assert header["Authorization"] == "Bearer tok123"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_burst():
    rl = _RateLimiter(rate=10.0, burst=5)
    start = time.monotonic()
    for _ in range(5):
        rl.acquire()
    elapsed = time.monotonic() - start
    # 5 burst tokens should be consumed near-instantly
    assert elapsed < 1.0


def test_rate_limiter_blocks_after_burst():
    rl = _RateLimiter(rate=10.0, burst=2)
    # Drain burst
    rl.acquire()
    rl.acquire()
    # Third call must wait
    start = time.monotonic()
    rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05  # at 10/s, ~0.1s per token


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_hit_skips_http(client):
    redis_mock = MagicMock()
    import json
    cached_data = {"payload": {"cached": True}}
    redis_mock.get.return_value = json.dumps(cached_data)
    client._redis = redis_mock

    with patch.object(client._http, "get") as mock_get, \
         patch.object(client._http, "post", return_value=_mock_lwa_response()):
        client._access_token = "tok"
        client._token_expiry = time.monotonic() + 3600
        result = client.get("/catalog/2022-04-01/items/B001")

    mock_get.assert_not_called()
    assert result == cached_data


def test_cache_miss_calls_api_and_stores(client):
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    client._redis = redis_mock
    client._access_token = "tok"
    client._token_expiry = time.monotonic() + 3600

    api_data = {"payload": {"fresh": True}}
    with patch.object(client._http, "get", return_value=_mock_api_response(api_data)):
        result = client.get("/catalog/2022-04-01/items/B001")

    assert result == api_data
    redis_mock.setex.assert_called_once()


def test_cache_disabled_when_no_redis(client):
    client._redis = None
    assert client._cache_get("any_key") is None
    # Should not raise
    client._cache_set("any_key", {"data": 1})
