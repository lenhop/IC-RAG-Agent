"""
Orders v0 — getOrder (single and batch).

See SP-API reference (getOrder). Path: GET /orders/v0/orders/{orderId}
Note: Orders v0 getOrder is deprecated; plan migration to Orders v2026-01-01 when needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Union
from urllib.parse import quote

import httpx

from .sp_api_client import SPAPIClient

logger = logging.getLogger(__name__)


def _normalize_id_list(ids: Union[str, Sequence[str], None]) -> List[str]:
    """Flatten user input into a de-duplicated list of non-empty order IDs (stable order)."""
    if ids is None:
        return []
    if isinstance(ids, str):
        raw = [x.strip() for x in ids.replace(",", " ").split() if x.strip()]
    else:
        raw = [str(x).strip() for x in ids if str(x).strip()]
    seen: set[str] = set()
    out: List[str] = []
    for oid in raw:
        if oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


def get_order(client: SPAPIClient, order_id: str) -> Dict[str, Any]:
    """
    Fetch one order by Amazon order ID.

    Args:
        client: Authenticated SP-API client.
        order_id: Order identifier (e.g. 3-1234-5678901).

    Returns:
        Parsed JSON payload (typically includes ``payload`` with order fields).

    Raises:
        httpx.HTTPStatusError: On non-success HTTP status.
        ValueError: If order_id is empty.
    """
    oid = (order_id or "").strip()
    if not oid:
        raise ValueError("order_id must be non-empty")
    # Hyphens are common in Amazon order IDs and are safe in path segments.
    enc = quote(oid, safe="-")
    path = f"/orders/v0/orders/{enc}"
    return client.get(path)


def get_orders_batch(
    client: SPAPIClient,
    order_ids: Union[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    """
    Fetch multiple orders sequentially (respects client rate limits). One failure does not stop others.

    Args:
        client: Authenticated SP-API client.
        order_ids: One ID, comma/space-separated string, or sequence of IDs.

    Returns:
        List of dicts: ``{"order_id", "ok", "payload"|"error", "status_code"?}``.
    """
    results: List[Dict[str, Any]] = []
    for oid in _normalize_id_list(order_ids):
        try:
            data = get_order(client, oid)
            results.append({"order_id": oid, "ok": True, "payload": data})
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = ""
            try:
                if exc.response is not None:
                    body = exc.response.text[:2000]
            except Exception:
                body = ""
            logger.warning("get_order failed order_id=%s status=%s", oid, status)
            err: Dict[str, Any] = {
                "order_id": oid,
                "ok": False,
                "error": f"HTTP {status}: {body}",
            }
            if status is not None:
                err["status_code"] = status
            results.append(err)
        except Exception as exc:
            logger.exception("get_order unexpected error order_id=%s", oid)
            results.append(
                {
                    "order_id": oid,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return results
