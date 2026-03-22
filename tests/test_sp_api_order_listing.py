"""
Unit tests for SP-API order/listing helpers and FastAPI health (no live Amazon calls).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.sp_api.listing import get_listings_items_batch, get_listings_item
from src.agent.sp_api.order import get_order, get_orders_batch
from src.agent.sp_api.order_yaml import format_orders_batch_as_yaml
from src.agent.sp_api.sp_api_client import SPAPICredentials


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.com/test")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError("err", request=req, response=resp)


def test_get_order_encodes_path() -> None:
    client = MagicMock()
    client.get.return_value = {"payload": {"AmazonOrderId": "111-1"}}
    out = get_order(client, "111-222-333")
    client.get.assert_called_once()
    path_arg = client.get.call_args[0][0]
    assert path_arg == "/orders/v0/orders/111-222-333"
    assert out["payload"]["AmazonOrderId"] == "111-1"


def test_format_orders_batch_as_yaml_includes_full_payload() -> None:
    """YAML output must embed the full SP-API response, not only status."""
    results = [
        {
            "order_id": "112-1111111-2222222",
            "ok": True,
            "payload": {
                "payload": {
                    "AmazonOrderId": "112-1111111-2222222",
                    "OrderStatus": "Shipped",
                    "OrderTotal": {"CurrencyCode": "USD", "Amount": "29.99"},
                }
            },
        },
        {"order_id": "999", "ok": False, "error": "not found", "status_code": 404},
    ]
    text = format_orders_batch_as_yaml(results)
    assert "AmazonOrderId" in text
    assert "OrderTotal" in text or "Amount" in text
    assert "112-1111111-2222222" in text
    assert "404" in text or "not found" in text
    assert "orders:" in text.lower() or "order_count" in text
    assert "ok:" in text or '"ok"' in text
    assert "sp_api_response" in text


def test_get_orders_batch_dedupes_and_partial_failure() -> None:
    client = MagicMock()

    def _get(path: str, params=None):
        if "222" in path:
            raise _http_error(404)
        return {"ok": True}

    client.get.side_effect = _get
    results = get_orders_batch(client, ["111-1", "222-2", "111-1"])
    assert len(results) == 2
    assert results[0]["order_id"] == "111-1" and results[0]["ok"] is True
    assert results[1]["order_id"] == "222-2" and results[1]["ok"] is False


def test_get_listings_item_requires_seller() -> None:
    creds = SPAPICredentials(
        refresh_token="r",
        client_id="i",
        client_secret="s",
        seller_id="",
    )
    client = MagicMock()
    with pytest.raises(ValueError, match="seller_id"):
        get_listings_item(client, "SKU1", credentials=creds)


def test_get_listings_items_batch_uses_seller_from_creds() -> None:
    creds = SPAPICredentials(
        refresh_token="r",
        client_id="i",
        client_secret="s",
        seller_id="SELLER123",
        marketplace_id="ATVPDKIKX0DER",
    )
    client = MagicMock()
    client.get.return_value = {"sku": "SKU1"}
    out = get_listings_items_batch(client, ["SKU1", "SKU1"], credentials=creds)
    assert len(out) == 1
    assert out[0]["ok"] is True
    path = client.get.call_args[0][0]
    assert "SELLER123" in path
    assert "SKU1" in path


def test_sp_api_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    from src.agent.sp_api.app import app

    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_sp_api_seller_query_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SP_API_TEST_MODE", "true")
    # Package __init__ re-exports the FastAPI instance as ``app``.
    from src.agent.sp_api import app as fastapi_app

    client = TestClient(fastapi_app)
    r = client.post("/api/v1/seller/query", json={"query": "hello"})
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert "SP_API_TEST_MODE" in data["response"]
    assert "hello" in data["response"]
