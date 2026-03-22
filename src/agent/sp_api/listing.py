"""
Listings Items 2021-08-01 — getListingsItem (single and batch).

See SP-API reference (getListingsItem).
Path: GET /listings/2021-08-01/items/{sellerId}/{sku}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote

import httpx

from .sp_api_client import SPAPIClient, SPAPICredentials

logger = logging.getLogger(__name__)

LISTINGS_API_PREFIX = "/listings/2021-08-01/items"


def _normalize_sku_list(skus: Union[str, Sequence[str], None]) -> List[str]:
    """De-duplicate SKUs while preserving order."""
    if skus is None:
        return []
    if isinstance(skus, str):
        raw = [x.strip() for x in skus.replace(",", " ").split() if x.strip()]
    else:
        raw = [str(x).strip() for x in skus if str(x).strip()]
    seen: set[str] = set()
    out: List[str] = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _marketplace_params(
    marketplace_ids: Optional[Sequence[str]],
    default_marketplace_id: str,
) -> List[tuple[str, str]]:
    """Build query pairs for marketplaceIds (SP-API allows repeated keys)."""
    ids: List[str] = []
    if marketplace_ids:
        for m in marketplace_ids:
            t = str(m).strip()
            if t:
                ids.append(t)
    if not ids:
        dm = (default_marketplace_id or "").strip()
        if not dm:
            raise ValueError(
                "marketplaceIds is required: set SP_API_MARKETPLACE_ID or pass marketplace_ids"
            )
        ids = [dm]
    return [("marketplaceIds", mid) for mid in ids]


def get_listings_item(
    client: SPAPIClient,
    sku: str,
    *,
    seller_id: Optional[str] = None,
    credentials: Optional[SPAPICredentials] = None,
    marketplace_ids: Optional[Sequence[str]] = None,
    included_data: Optional[str] = None,
    issue_locale: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch listings item details for one seller SKU.

    Args:
        client: Authenticated SP-API client.
        sku: Seller SKU (path segment; will be URL-encoded).
        seller_id: Selling partner ID; defaults to ``credentials.seller_id`` or env SP_API_SELLER_ID.
        credentials: Used when ``seller_id`` is omitted to read ``seller_id`` and default marketplace.
        marketplace_ids: Optional marketplace id list; defaults to single marketplace from credentials.
        included_data: Optional getListingsItem ``includedData`` query (comma-separated).
        issue_locale: Optional ``issueLocale`` query.

    Returns:
        Parsed JSON from SP-API.

    Raises:
        ValueError: If seller_id or marketplace cannot be resolved.
        httpx.HTTPStatusError: On HTTP error.
    """
    sk = (sku or "").strip()
    if not sk:
        raise ValueError("sku must be non-empty")

    sid = (seller_id or "").strip()
    if not sid and credentials is not None:
        sid = (credentials.seller_id or "").strip()
    if not sid:
        raise ValueError(
            "seller_id is required for getListingsItem: set SP_API_SELLER_ID or pass seller_id"
        )

    default_mp = ""
    if credentials is not None:
        default_mp = (credentials.marketplace_id or "").strip()

    pairs = _marketplace_params(marketplace_ids, default_mp)
    if included_data:
        pairs = pairs + [("includedData", included_data)]
    if issue_locale:
        pairs = pairs + [("issueLocale", issue_locale)]

    enc_seller = quote(sid, safe="-")
    enc_sku = quote(sk, safe="-_.~")
    path = f"{LISTINGS_API_PREFIX}/{enc_seller}/{enc_sku}"
    return client.get(path, params=pairs)


def get_listings_items_batch(
    client: SPAPIClient,
    skus: Union[str, Sequence[str]],
    *,
    seller_id: Optional[str] = None,
    credentials: Optional[SPAPICredentials] = None,
    marketplace_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch multiple SKUs sequentially (rate-limited by client). Failures are isolated per SKU.

    Args:
        client: Authenticated SP-API client.
        skus: One or more seller SKUs.
        seller_id: Optional override; else from credentials / env.
        credentials: For default seller_id and marketplace.
        marketplace_ids: Optional list of marketplace ids.

    Returns:
        List of ``{"sku", "ok", "payload"|"error", "status_code"?}``.
    """
    results: List[Dict[str, Any]] = []
    for sk in _normalize_sku_list(skus):
        try:
            data = get_listings_item(
                client,
                sk,
                seller_id=seller_id,
                credentials=credentials,
                marketplace_ids=marketplace_ids,
            )
            results.append({"sku": sk, "ok": True, "payload": data})
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = ""
            try:
                if exc.response is not None:
                    body = exc.response.text[:2000]
            except Exception:
                body = ""
            logger.warning("get_listings_item failed sku=%s status=%s", sk, status)
            row: Dict[str, Any] = {
                "sku": sk,
                "ok": False,
                "error": f"HTTP {status}: {body}",
            }
            if status is not None:
                row["status_code"] = status
            results.append(row)
        except Exception as exc:
            logger.exception("get_listings_item unexpected error sku=%s", sk)
            results.append({"sku": sk, "ok": False, "error": str(exc)})
    return results
