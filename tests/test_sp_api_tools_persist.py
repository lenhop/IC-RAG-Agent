"""
Tests for SP-API order tool raw-result persistence under tests directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_sp_api_get_orders_tool_persists_raw_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """sp_api_get_orders should save API result JSON and latest pointer file."""
    from src.agent.sp_api import tools as sp_tools

    results_dir = tmp_path / "sp_api_results"
    latest_file = results_dir / "latest_get_orders_result.json"
    monkeypatch.setattr(sp_tools, "_SP_API_RESULTS_DIR", results_dir)
    monkeypatch.setattr(sp_tools, "_SP_API_RESULTS_LATEST", latest_file)

    fake_results = [
        {
            "order_id": "111-2886487-4917844",
            "ok": True,
            "payload": {"payload": {"AmazonOrderId": "111-2886487-4917844", "OrderStatus": "Shipped"}},
        }
    ]

    def fake_get_orders_batch(_client, _ids):
        return fake_results

    monkeypatch.setattr(sp_tools, "get_orders_batch", fake_get_orders_batch)

    tool = sp_tools.SpApiGetOrdersTool(client=object())
    out = tool.execute(order_ids=["111-2886487-4917844"])

    assert "saved_result_file" in out
    assert "saved_result_latest_file" in out

    saved_path = Path(out["saved_result_file"])
    latest_path = Path(out["saved_result_latest_file"])
    assert saved_path.is_file()
    assert latest_path.is_file()

    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["order_ids"] == ["111-2886487-4917844"]
    assert payload["results"] == fake_results
    assert "saved_at_utc" in payload
