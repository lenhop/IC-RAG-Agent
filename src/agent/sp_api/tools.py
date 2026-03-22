"""
ReAct tools wrapping read-only SP-API order and listing operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from ai_toolkit.errors import ValidationError
from ai_toolkit.tools import BaseTool, ToolParameter

from .listing import get_listings_items_batch
from .listing_yaml import format_listings_batch_as_yaml
from .order import get_orders_batch
from .order_yaml import format_orders_batch_as_yaml
from .sp_api_client import SPAPIClient, SPAPICredentials

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SP_API_RESULTS_DIR = _PROJECT_ROOT / "tests" / "sp_api_results"
_SP_API_RESULTS_LATEST = _SP_API_RESULTS_DIR / "latest_get_orders_result.json"
_SP_API_LISTINGS_RESULTS_LATEST = _SP_API_RESULTS_DIR / "latest_get_listings_result.json"


def _persist_get_orders_result(results: List[dict], order_ids: List[str]) -> Optional[Path]:
    """
    Save raw getOrder batch output under tests/ for debugging and comparison.

    Args:
        results: Raw batch rows from ``get_orders_batch``.
        order_ids: Input Amazon order IDs for traceability.

    Returns:
        Path to the timestamped JSON file when persisted, otherwise None.
    """
    try:
        _SP_API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "order_ids": order_ids,
            "results": results,
        }
        out_file = _SP_API_RESULTS_DIR / f"get_orders_result_{stamp}.json"
        out_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        out_file.write_text(out_text, encoding="utf-8")
        _SP_API_RESULTS_LATEST.write_text(out_text, encoding="utf-8")
        return out_file
    except Exception as exc:
        # Persistence must not break user-facing SP-API flow.
        logger.warning("Failed to persist SP-API get_orders result: %s", exc, exc_info=True)
        return None


def _persist_get_listings_result(results: List[dict], skus: List[str]) -> Optional[Path]:
    """
    Save raw getListingsItem batch output under tests/ for debugging and comparison.

    Args:
        results: Raw batch rows from ``get_listings_items_batch``.
        skus: Input seller SKUs for traceability.

    Returns:
        Path to the timestamped JSON file when persisted, otherwise None.
    """
    try:
        _SP_API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "skus": skus,
            "results": results,
        }
        out_file = _SP_API_RESULTS_DIR / f"get_listings_result_{stamp}.json"
        out_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        out_file.write_text(out_text, encoding="utf-8")
        _SP_API_LISTINGS_RESULTS_LATEST.write_text(out_text, encoding="utf-8")
        return out_file
    except Exception as exc:
        # Persistence must not break user-facing SP-API flow.
        logger.warning("Failed to persist SP-API get_listings result: %s", exc, exc_info=True)
        return None


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
        saved_file = _persist_get_orders_result(results, ids)
        ok_n = sum(1 for r in results if r.get("ok"))
        # Human-readable YAML: full SP-API JSON per order (not only OrderStatus).
        orders_yaml = format_orders_batch_as_yaml(results)
        response = {
            "orders_yaml": orders_yaml,
            "results": results,
            "summary": (
                f"Retrieved {ok_n} of {len(results)} order(s). "
                "Full data is in orders_yaml; pass that block to the user."
            ),
        }
        if saved_file is not None:
            response["saved_result_file"] = str(saved_file)
            response["saved_result_latest_file"] = str(_SP_API_RESULTS_LATEST)
        return response

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
        saved_file = _persist_get_listings_result(results, sku_list)
        ok_n = sum(1 for r in results if r.get("ok"))
        listings_yaml = format_listings_batch_as_yaml(results)
        response = {
            "listings_yaml": listings_yaml,
            "results": results,
            "summary": (
                f"Retrieved {ok_n} of {len(results)} SKU(s). "
                "Full data is in listings_yaml; pass that block to the user."
            ),
        }
        if saved_file is not None:
            response["saved_result_file"] = str(saved_file)
            response["saved_result_latest_file"] = str(_SP_API_LISTINGS_RESULTS_LATEST)
        return response

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
