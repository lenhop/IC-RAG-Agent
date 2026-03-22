"""
SP-API Client — LWA OAuth2 + rate limiting + Redis cache.
"""
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .exceptions import SPAPIAuthError


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@dataclass
class SPAPICredentials:
    refresh_token: str
    client_id: str
    client_secret: str
    marketplace_id: str = "ATVPDKIKX0DER"
    role_arn: str = ""
    aws_access_key: str = ""
    aws_secret_key: str = ""
    region: str = "us-east-1"
    app_id: str = ""
    seller_id: str = ""

    def __post_init__(self):
        for fname in ("refresh_token", "client_id", "client_secret"):
            if not getattr(self, fname):
                raise ValueError(f"SPAPICredentials: '{fname}' is required and cannot be empty")

    @classmethod
    def from_env(cls) -> "SPAPICredentials":
        """
        Load credentials from environment variables.

        Refresh token resolution (first non-empty wins):
            SP_API_REFRESH_TOKEN, then SP_API_SNB_NA_REFRESH_TOKEN (per-account .env layout).

        Seller id resolution:
            SP_API_SELLER_ID, then SNB_NA_SELLER_ID (optional; listings need one of these).
        """
        def _get(key: str, required: bool = True) -> str:
            val = os.environ.get(key, "")
            if required and not val:
                raise ValueError(f"Missing required env var: {key}")
            return val

        def _refresh_token_from_env() -> str:
            """Pick LWA refresh token from primary or snb_na-specific env names."""
            for key in ("SP_API_REFRESH_TOKEN", "SP_API_SNB_NA_REFRESH_TOKEN"):
                raw = (os.environ.get(key) or "").strip()
                if raw:
                    return raw
            raise ValueError(
                "Missing LWA refresh token: set SP_API_REFRESH_TOKEN or "
                "SP_API_SNB_NA_REFRESH_TOKEN in the environment."
            )

        seller_primary = (os.environ.get("SP_API_SELLER_ID") or "").strip()
        seller_fallback = (os.environ.get("SNB_NA_SELLER_ID") or "").strip()
        seller_id = seller_primary or seller_fallback

        return cls(
            refresh_token=_refresh_token_from_env(),
            client_id=_get("SP_API_CLIENT_ID"),
            client_secret=_get("SP_API_CLIENT_SECRET"),
            marketplace_id=_get("SP_API_MARKETPLACE_ID", required=False) or "ATVPDKIKX0DER",
            role_arn=_get("SP_API_ROLE_ARN", required=False),
            aws_access_key=_get("SP_API_AWS_ACCESS_KEY", required=False),
            aws_secret_key=_get("SP_API_AWS_SECRET_KEY", required=False),
            region=_get("SP_API_REGION", required=False) or "us-east-1",
            app_id=_get("SP_API_APP_ID", required=False),
            seller_id=seller_id,
        )


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token bucket rate limiter — blocks until a token is available."""

    def __init__(self, rate: float, burst: int):
        self._rate = rate          # tokens per second
        self._burst = burst        # max bucket size
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()  # Fix #5: thread safety

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
            # sleep outside the lock so other threads can proceed
            time.sleep(deficit / self._rate)


# Per-endpoint rate limit config: (rate/s, burst). Match SP-API default usage plans;
# actual limits may appear in x-amzn-RateLimit-Limit response headers.
# Orders getOrder: ~0.5 r/s, burst 30. Listings getListingsItem: ~5 r/s, burst 10.
_RATE_LIMITS: Dict[str, tuple] = {
    "/listings/": (5.0, 10),
    "/catalog/": (2.0, 2),
    "/fba/inventory/": (2.0, 2),
    "/orders/": (0.5, 30),
    "/finances/": (0.5, 30),
    "/reports/": (0.0222, 10),
}
_DEFAULT_RATE = (1.0, 5)


# ---------------------------------------------------------------------------
# SP-API Client
# ---------------------------------------------------------------------------

class SPAPIClient:
    """Authenticated Amazon SP-API HTTP client."""

    _LWA_URL = "https://api.amazon.com/auth/o2/token"

    def __init__(self, credentials: SPAPICredentials, redis_client=None):
        self._creds = credentials
        self._redis = redis_client
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._rate_limiters: Dict[str, _RateLimiter] = {}
        self._http = httpx.Client(timeout=30.0)
        # Base URL without trailing slash; override with SP_API_ENDPOINT for EU/FE etc.
        raw_base = (os.environ.get("SP_API_ENDPOINT") or "").strip().rstrip("/")
        self._sp_api_base = raw_base or "https://sellingpartnerapi-na.amazon.com"

    @property
    def marketplace_id(self) -> str:
        """Marketplace ID from credentials (e.g. ATVPDKIKX0DER for US)."""
        return self._creds.marketplace_id

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _refresh_token(self) -> None:
        """Exchange refresh token for a new LWA access token."""
        resp = self._http.post(
            self._LWA_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._creds.refresh_token,
                "client_id": self._creds.client_id,
                "client_secret": self._creds.client_secret,
            },
        )
        if resp.status_code != 200:
            raise SPAPIAuthError(
                f"LWA token refresh failed: {resp.status_code} {resp.text}",
                auth_type="lwa_oauth2",
            )
        data = resp.json()
        # Fix #2: guard missing access_token key
        if "access_token" not in data:
            raise SPAPIAuthError(
                "LWA response missing access_token",
                auth_type="lwa_oauth2",
            )
        self._access_token = data["access_token"]
        # Fix #3: guard invalid expires_in
        try:
            expires_in = int(data.get("expires_in", 3600))
        except (ValueError, TypeError):
            expires_in = 3600
        self._token_expiry = time.monotonic() + expires_in

    def _get_auth_header(self) -> Dict[str, str]:
        """Return Authorization header, refreshing token if within 60s of expiry."""
        if self._access_token is None or time.monotonic() >= self._token_expiry - 60:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _get_rate_limiter(self, path: str) -> _RateLimiter:
        for prefix, (rate, burst) in _RATE_LIMITS.items():
            if path.startswith(prefix):
                if prefix not in self._rate_limiters:
                    self._rate_limiters[prefix] = _RateLimiter(rate, burst)
                return self._rate_limiters[prefix]
        if "_default" not in self._rate_limiters:
            self._rate_limiters["_default"] = _RateLimiter(*_DEFAULT_RATE)
        return self._rate_limiters["_default"]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, path: str, params: Optional[Dict]) -> str:
        params_str = json.dumps(sorted((params or {}).items()))
        digest = hashlib.sha256(params_str.encode()).hexdigest()[:16]  # Fix #6: 16 chars
        return f"spapi:{path}:{digest}"

    def _cache_get(self, key: str) -> Optional[Dict]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _cache_set(self, key: str, value: Dict, ttl: int = 3600) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, ttl, json.dumps(value))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Authenticated GET — checks cache first, then calls SP-API."""
        cache_key = self._cache_key(path, params)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        self._get_rate_limiter(path).acquire()
        headers = self._get_auth_header()
        headers["x-amz-access-token"] = self._access_token or ""

        resp = self._http.get(
            self._sp_api_base + path,
            params=params,
            headers=headers,
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            # Fix #4: wrap HTTP errors in a consistent exception
            status = getattr(getattr(e, "response", None), "status_code", None)
            raise httpx.HTTPStatusError(
                f"SP-API error {status}: {resp.text}",
                request=resp.request,
                response=resp,
            ) from e
        data = resp.json()
        self._cache_set(cache_key, data)
        return data

    def post(self, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        """POST is disabled — this agent is read-only.

        All SP-API write operations (create, update, delete) are prohibited.
        Only GET requests are permitted to prevent accidental mutations.
        """
        raise PermissionError(
            f"POST to '{path}' is not allowed. "
            "This SP-API agent is read-only. Only GET requests are permitted."
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
