"""
Format SP-API getListingsItem batch results as human-readable YAML.

Amazon returns JSON; we keep every field (sku, attributes, summaries, offers, issues, etc.)
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


def format_listings_batch_as_yaml(results: List[Dict[str, Any]]) -> str:
    """
    Build a multi-SKU YAML document with full SP-API JSON per success row.

    Args:
        results: Output of ``get_listings_items_batch`` (``sku``, ``ok``, ``payload`` or ``error``).

    Returns:
        UTF-8 YAML string (falls back to indented JSON if PyYAML is unavailable).
    """
    listings_out: List[Dict[str, Any]] = []
    for row in results:
        sku = str(row.get("sku") or "").strip()
        if row.get("ok"):
            raw = row.get("payload")
            listings_out.append(
                {
                    "ok": True,
                    "sku": sku,
                    "sp_api_response": _json_safe(raw) if raw is not None else {},
                }
            )
        else:
            err_row: Dict[str, Any] = {
                "ok": False,
                "sku": sku,
                "error": str(row.get("error") or ""),
            }
            if row.get("status_code") is not None:
                err_row["http_status"] = row["status_code"]
            listings_out.append(err_row)

    wrapper = {
        "listings": listings_out,
        "listing_count": len(listings_out),
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
