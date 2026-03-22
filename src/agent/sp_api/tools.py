"""
ReAct tools wrapping read-only SP-API order and listing operations.

Includes batch YAML formatters (merged from former ``order_yaml`` / ``listing_yaml`` modules)
as class-based facades for downstream use.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ai_toolkit.errors import ValidationError
from ai_toolkit.tools import BaseTool, ToolParameter

from .listing import get_listings_items_batch
from .order import get_orders_batch
from .sp_api_client import SPAPIClient, SPAPICredentials

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - PyYAML is in requirements.txt
    yaml = None  # type: ignore[assignment]


def _sp_api_yaml_json_safe(obj: Any) -> Any:
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
        return {str(k): _sp_api_yaml_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sp_api_yaml_json_safe(v) for v in obj]
    return str(obj)


class SpApiOrderBatchYamlFormatter:
    """
    Format SP-API getOrder batch results as human-readable YAML for chat / logs.

    Each successful row uses shape: ``ok``, ``order_id``, ``sp_api_response`` (full JSON).
    """

    @classmethod
    def format_batch(cls, results: List[Dict[str, Any]]) -> str:
        """
        Build a multi-order YAML document with full SP-API JSON per success row.

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
                orders_out.append(
                    {
                        "ok": True,
                        "order_id": oid,
                        "sp_api_response": _sp_api_yaml_json_safe(raw) if raw is not None else {},
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


class SpApiListingBatchYamlFormatter:
    """
    Format SP-API getListingsItem batch results as human-readable YAML for chat / logs.

    Each successful row uses shape: ``ok``, ``sku``, ``sp_api_response`` (full JSON).
    """

    @classmethod
    def format_batch(cls, results: List[Dict[str, Any]]) -> str:
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
                        "sp_api_response": _sp_api_yaml_json_safe(raw) if raw is not None else {},
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


def _coerce_id_list(value: Union[str, List[Any], None]) -> List[str]:
    """Accept JSON array, comma-separated string, or single string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in s.replace(",", " ").split() if x.strip()]
    return [str(value).strip()] if str(value).strip() else []


class SpApiGetOrdersTool(BaseTool):
    """
    Retrieve Amazon orders by order ID via Orders v0 getOrder (supports multiple IDs, rate-limited).
    """

    def __init__(self, client: SPAPIClient) -> None:
        super().__init__(
            name="sp_api_get_orders",
            description=(
                "Fetch full order details from Amazon SP-API (Orders v0 getOrder): all fields "
                "Amazon returns (order id, status, dates, totals when present, channel, etc.). "
                "The tool response includes orders_yaml with the complete payload formatted as YAML. "
                "Pass order_ids as a JSON array, e.g. [\"111-2222222-3333333\"]."
            ),
        )
        self._client = client

    def validate_parameters(self, order_ids: Any = None, **kwargs: Any) -> None:
        ids = _coerce_id_list(order_ids)
        if not ids:
            raise ValidationError(
                message="order_ids is required (non-empty list or comma-separated string)",
                field_name="order_ids",
            )

    def execute(self, order_ids: Any = None, **kwargs: Any) -> dict:
        self.validate_parameters(order_ids=order_ids, **kwargs)
        ids = _coerce_id_list(order_ids)
        results = get_orders_batch(self._client, ids)
        ok_n = sum(1 for r in results if r.get("ok"))
        # Human-readable YAML: full SP-API JSON per order (not only OrderStatus).
        orders_yaml = SpApiOrderBatchYamlFormatter.format_batch(results)
        return {
            "orders_yaml": orders_yaml,
            "results": results,
            "summary": (
                f"Retrieved {ok_n} of {len(results)} order(s). "
                "Full data is in orders_yaml; pass that block to the user."
            ),
        }

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="order_ids",
                type="array",
                description="Amazon order ID(s): JSON array of strings, e.g. [\"111-2222222-3333333\"]",
                required=True,
            ),
        ]


class SpApiGetListingsTool(BaseTool):
    """
    Retrieve listing item details via Listings Items 2021-08-01 getListingsItem (batch SKUs).
    """

    def __init__(self, client: SPAPIClient, credentials: SPAPICredentials) -> None:
        super().__init__(
            name="sp_api_get_listings",
            description=(
                "Fetch listing item details (SKU level) from Amazon SP-API getListingsItem. "
                "Default seller ID comes from configuration (SP_API_SELLER_ID). "
                "Pass skus as a JSON array of seller SKUs."
            ),
        )
        self._client = client
        self._credentials = credentials

    def validate_parameters(self, skus: Any = None, **kwargs: Any) -> None:
        ids = _coerce_id_list(skus)
        if not ids:
            raise ValidationError(
                message="skus is required (non-empty list or comma-separated string)",
                field_name="skus",
            )

    def execute(self, skus: Any = None, seller_id: Optional[str] = None, **kwargs: Any) -> dict:
        self.validate_parameters(skus=skus, **kwargs)
        sku_list = _coerce_id_list(skus)
        sid = (seller_id or "").strip() or None
        results = get_listings_items_batch(
            self._client,
            sku_list,
            seller_id=sid,
            credentials=self._credentials,
        )
        ok_n = sum(1 for r in results if r.get("ok"))
        listings_yaml = SpApiListingBatchYamlFormatter.format_batch(results)
        return {
            "listings_yaml": listings_yaml,
            "results": results,
            "summary": (
                f"Retrieved {ok_n} of {len(results)} SKU(s). "
                "Full data is in listings_yaml; pass that block to the user."
            ),
        }

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="skus",
                type="array",
                description="Seller SKU(s): JSON array of strings",
                required=True,
            ),
            ToolParameter(
                name="seller_id",
                type="string",
                description="Optional selling partner ID; defaults to SP_API_SELLER_ID from environment",
                required=False,
            ),
        ]
