"""
Redis-backed fast path for unified query rewrite (L1 / L2) before the L3 LLM.

Reads the same hashes as ``external/IC-Self-Study/redis/redis_clear_intent_sentence_ops.py``:
- L1: ``clear_intent_sentence:data`` (clear_sentence dataset; field = sentence, value = JSON row).
- L2: ``regular_patterns:data`` (field = regex pattern, value = JSON row).

Connection URL precedence: ``GATEWAY_REWRITE_REDIS_URL``, then ``GATEWAY_REDIS_URL``,
then ``REDIS_URL``, else ``redis://localhost:6379/0``.

If Redis is unreachable or hashes are empty, all matchers return miss (caller falls through to L3).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Redis key names (must match redis_clear_intent_sentence_ops._DATASET_SPECS).
_HASH_CLEAR_SENTENCE = "clear_intent_sentence:data"
_HASH_REGULAR_PATTERNS = "regular_patterns:data"


def normalize_clear_sentence_key(query: str) -> str:
    """Same normalization as ``keyword_regular_match`` CSV L1 (strip + casefold)."""
    return (query or "").strip().casefold()


def _parse_json_row(raw: str, fallback: Dict[str, str]) -> Dict[str, Any]:
    """Parse Redis hash value JSON; on failure return a minimal dict for logging."""
    if not raw:
        return dict(fallback)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        logger.debug("rewrite_redis_fastpath: invalid JSON row, using fallback keys")
    return dict(fallback)


def build_clear_sentence_index(hgetall: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Map casefold sentence key -> parsed row dict.

    Hash fields are the canonical sentence strings from import (CSV ``sentence`` column).
    """
    by_norm: Dict[str, Dict[str, Any]] = {}
    for sentence, payload in (hgetall or {}).items():
        key = normalize_clear_sentence_key(sentence)
        if not key:
            continue
        row = _parse_json_row(payload, {"sentence": sentence})
        # First wins to mirror CSV loader de-dupe by norm key.
        if key not in by_norm:
            by_norm[key] = row
    return by_norm


def build_regular_pattern_list(hgetall: Dict[str, str]) -> List[Tuple[re.Pattern[str], Dict[str, Any]]]:
    """
    Compile regex patterns from hash; deterministic order by pattern string for stable first-hit.
    """
    items: List[Tuple[str, str]] = []
    for pattern_key, payload in (hgetall or {}).items():
        pat_s = (pattern_key or "").strip()
        if not pat_s:
            continue
        items.append((pat_s, payload))
    items.sort(key=lambda x: x[0])
    out: List[Tuple[re.Pattern[str], Dict[str, Any]]] = []
    for pat_s, payload in items:
        try:
            cre = re.compile(pat_s)
        except re.error as exc:
            logger.warning(
                "rewrite_redis_fastpath: skip invalid regex %r: %s",
                pat_s,
                exc,
            )
            continue
        row = _parse_json_row(payload, {"pattern": pat_s})
        out.append((cre, row))
    return out


def match_clear_sentence_from_index(
    by_norm: Dict[str, Dict[str, Any]],
    query: str,
) -> Optional[Dict[str, Any]]:
    """Return parsed row when ``query`` matches a clear_sentence entry (after norm)."""
    key = normalize_clear_sentence_key(query)
    if not key:
        return None
    return by_norm.get(key)


def match_first_regular_pattern_from_list(
    compiled: List[Tuple[re.Pattern[str], Dict[str, Any]]],
    text: str,
) -> Optional[Dict[str, Any]]:
    """First regex that matches ``text`` (``search``); returns the row dict plus ``_matched_pattern``."""
    if not (text or "").strip():
        return None
    for cre, row in compiled:
        m = cre.search(text)
        if m:
            out = dict(row)
            out["_matched_pattern"] = row.get("pattern", cre.pattern)
            return out
    return None


def _cache_ttl_seconds() -> float:
    raw = (os.getenv("GATEWAY_REWRITE_REDIS_CACHE_SECONDS") or "60").strip()
    try:
        v = float(raw)
        return max(5.0, min(v, 3600.0))
    except (ValueError, TypeError):
        return 60.0


class RewriteRedisFastpath:
    """
    Process-local TTL cache over Redis HGETALL for L1/L2 rewrite skips.

    Thread-safe refresh; failures leave previous cache or empty miss.
    """

    _lock = threading.Lock()
    _clear_index: Dict[str, Dict[str, Any]] = {}
    _regular_list: List[Tuple[re.Pattern[str], Dict[str, Any]]] = []
    _cache_expires_at: float = 0.0
    _last_error_log: float = 0.0

    @classmethod
    def reset_cache_for_tests(cls) -> None:
        """Clear cached indexes (pytest)."""
        with cls._lock:
            cls._clear_index = {}
            cls._regular_list = []
            cls._cache_expires_at = 0.0

    @classmethod
    def _connect(cls):
        """Return a Redis client or None if unavailable."""
        try:
            import redis  # type: ignore
        except ImportError:
            logger.debug("rewrite_redis_fastpath: redis package not installed")
            return None
        url = (
            (os.getenv("GATEWAY_REWRITE_REDIS_URL") or "").strip()
            or (os.getenv("GATEWAY_REDIS_URL") or "").strip()
            or (os.getenv("REDIS_URL") or "").strip()
            or "redis://localhost:6379/0"
        )
        try:
            client = redis.from_url(url, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:
            now = time.monotonic()
            if now - cls._last_error_log > 30.0:
                cls._last_error_log = now
                logger.warning(
                    "rewrite_redis_fastpath: Redis unavailable (%s); L1/L2 rewrite skip disabled",
                    exc,
                )
            return None

    @classmethod
    def _bump_expiry_short(cls) -> None:
        """After failed Redis ops, wait briefly before retrying (avoid tight loops)."""
        cls._cache_expires_at = time.monotonic() + min(_cache_ttl_seconds(), 30.0)

    @classmethod
    def _refresh_locked(cls) -> None:
        """Reload both hashes from Redis; caller must hold ``_lock``."""
        client = cls._connect()
        if client is None:
            cls._bump_expiry_short()
            return
        try:
            clear_raw = client.hgetall(_HASH_CLEAR_SENTENCE)
            reg_raw = client.hgetall(_HASH_REGULAR_PATTERNS)
        except Exception as exc:
            logger.warning("rewrite_redis_fastpath: HGETALL failed: %s", exc)
            cls._bump_expiry_short()
            return
        cls._clear_index = build_clear_sentence_index(clear_raw)
        cls._regular_list = build_regular_pattern_list(reg_raw)
        cls._cache_expires_at = time.monotonic() + _cache_ttl_seconds()
        logger.debug(
            "rewrite_redis_fastpath: cache refreshed clear=%d regular=%d",
            len(cls._clear_index),
            len(cls._regular_list),
        )

    @classmethod
    def _ensure_cache(cls) -> None:
        """Load or refresh TTL cache when expired (including initial ``expires_at == 0``)."""
        now = time.monotonic()
        with cls._lock:
            if cls._cache_expires_at > now:
                return
            cls._refresh_locked()

    @classmethod
    def match_clear_sentence(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        L1: exact match on normalized sentence against Redis clear_sentence hash.

        Returns:
            Parsed row dict (``sentence``, ``workflow``, …) or None.
        """
        cls._ensure_cache()
        with cls._lock:
            return match_clear_sentence_from_index(cls._clear_index, query)

    @classmethod
    def match_first_regular_pattern(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        L2: first matching regex from Redis ``regular_patterns`` hash.

        Returns:
            Row dict with optional ``_matched_pattern`` or None.
        """
        cls._ensure_cache()
        with cls._lock:
            return match_first_regular_pattern_from_list(cls._regular_list, text)


__all__ = [
    "RewriteRedisFastpath",
    "build_clear_sentence_index",
    "build_regular_pattern_list",
    "match_clear_sentence_from_index",
    "match_first_regular_pattern_from_list",
    "normalize_clear_sentence_key",
]
