"""
Format SP-API getOrder batch results as human-readable YAML.

Amazon returns JSON; we keep every field (order id, status, totals when present, dates, etc.)
and dump a stable structure for chat / logs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - PyYAML is in requirements.txt
    yaml = None  # type: ignore[assignment]


def _json_safe(obj: Any) -> Any:
    """
    Recursively normalize values so YAML/JSON serialization never fails on odd types.

    Args:
        obj: Arbitrary nested structure from httpx/SP-API JSON.

    Returns:
        Structure using only dict, list, str, int, float, bool, None.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return str(obj)


def format_orders_batch_as_yaml(results: List[Dict[str, Any]]) -> str:
    """
    Build a multi-order YAML document with full SP-API JSON per success row.

    Each successful row mirrors ``scripts/test_get_amazon_order.py`` output shape:
    ``ok``, ``order_id``, ``sp_api_response`` (full getOrder JSON body).

    Args:
        results: Output of ``get_orders_batch`` (``order_id``, ``ok``, ``payload`` or ``error``).

    Returns:
        UTF-8 YAML string (falls back to indented JSON if PyYAML is unavailable).
    """
    orders_out: List[Dict[str, Any]] = []
    for r in results:
        oid = str(r.get("order_id") or "").strip()
        if r.get("ok"):
            raw = r.get("payload")
            # Same envelope as test_get_amazon_order.py / live getOrder (no invented fields).
            orders_out.append(
                {
                    "ok": True,
                    "order_id": oid,
                    "sp_api_response": _json_safe(raw) if raw is not None else {},
                }
            )
        else:
            err_entry: Dict[str, Any] = {
                "ok": False,
                "order_id": oid,
                "error": str(r.get("error") or ""),
            }
            if r.get("status_code") is not None:
                err_entry["http_status"] = r["status_code"]
            orders_out.append(err_entry)

    wrapper = {
        "orders": orders_out,
        "order_count": len(orders_out),
    }

    if yaml is not None:
        try:
            return yaml.safe_dump(
                wrapper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=1000,
            )
        except Exception as exc:
            logger.warning("YAML dump failed, using JSON fallback: %s", exc)

    return json.dumps(wrapper, indent=2, ensure_ascii=False, default=str)
