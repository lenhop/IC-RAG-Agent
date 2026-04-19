"""
Catalog Items API v2022-04-01 — getCatalogItem by ASIN.

See SP-API reference (getCatalogItem). Full JSON is preserved for YAML formatting downstream.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote

import httpx

from .listing import _marketplace_params
from .sp_api_client import SPAPIClient, SPAPICredentials

logger = logging.getLogger(__name__)

CATALOG_ITEMS_PREFIX = "/catalog/2022-04-01/items"

# Request all public datasets so the YAML mirrors the API (no placeholder summaries).
# Override with SP_API_CATALOG_INCLUDED_DATA (comma-separated).
_DEFAULT_CATALOG_INCLUDED_DATA = (
    "attributes,classifications,dimensions,identifiers,images,productTypes,"
    "relationships,salesRanks,summaries"
)


def _resolve_included_data() -> str:
    raw = (os.environ.get("SP_API_CATALOG_INCLUDED_DATA") or "").strip()
    if raw:
        return raw
    return _DEFAULT_CATALOG_INCLUDED_DATA


def _normalize_asin_list(asins: Union[str, Sequence[str], None]) -> List[str]:
    """De-duplicate ASINs while preserving order."""
    if asins is None:
        return []
    if isinstance(asins, str):
        raw = [x.strip().upper() for x in asins.replace(",", " ").split() if x.strip()]
    else:
        raw = [str(x).strip().upper() for x in asins if str(x).strip()]
    seen: set[str] = set()
    out: List[str] = []
    for a in raw:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def get_catalog_item(
    client: SPAPIClient,
    asin: str,
    *,
    credentials: Optional[SPAPICredentials] = None,
    marketplace_ids: Optional[Sequence[str]] = None,
    included_data: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GET getCatalogItem for one ASIN (full JSON body).

    Args:
        client: Authenticated SP-API client.
        asin: Amazon Standard Identification Number.
        credentials: Used for default marketplace id when ``marketplace_ids`` is omitted.
        marketplace_ids: Optional marketplace id list.
        included_data: Comma-separated includedData; defaults to env or broad defaults.

    Returns:
        Parsed JSON from SP-API.

    Raises:
        ValueError: If ASIN or marketplace cannot be resolved.
        httpx.HTTPStatusError: On HTTP error from Amazon.
    """
    a = (asin or "").strip().upper()
    if not a:
        raise ValueError("asin must be non-empty")

    default_mp = ""
    if credentials is not None:
        default_mp = (credentials.marketplace_id or "").strip()

    enc = quote(a, safe="")
    path = f"{CATALOG_ITEMS_PREFIX}/{enc}"
    mp_params = _marketplace_params(marketplace_ids, default_mp)
    inc = (included_data or "").strip() or _resolve_included_data()
    params: Dict[str, Any] = {**mp_params, "includedData": inc}
    return client.get(path, params=params)


def get_catalog_items_batch(
    client: SPAPIClient,
    asins: Union[str, Sequence[str]],
    *,
    credentials: Optional[SPAPICredentials] = None,
    marketplace_ids: Optional[Sequence[str]] = None,
    included_data: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch multiple ASINs sequentially (rate-limited by client). Failures are isolated per ASIN.

    Returns:
        List of ``{"asin", "ok", "payload"|"error", "status_code"?}``.
    """
    results: List[Dict[str, Any]] = []
    for asin in _normalize_asin_list(asins):
        try:
            data = get_catalog_item(
                client,
                asin,
                credentials=credentials,
                marketplace_ids=marketplace_ids,
                included_data=included_data,
            )
            results.append({"asin": asin, "ok": True, "payload": data})
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body = ""
            try:
                if exc.response is not None:
                    body = exc.response.text[:2000]
            except Exception:
                body = ""
            logger.warning("get_catalog_item failed asin=%s status=%s", asin, status)
            row: Dict[str, Any] = {
                "asin": asin,
                "ok": False,
                "error": f"HTTP {status}: {body}",
            }
            if status is not None:
                row["status_code"] = status
            results.append(row)
        except Exception as exc:
            logger.exception("get_catalog_item unexpected error asin=%s", asin)
            results.append({"asin": asin, "ok": False, "error": str(exc)})
    return results
