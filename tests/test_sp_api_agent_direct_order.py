"""
Tests for deterministic direct SP-API paths in ReAct agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sp_api_agent_direct_order_id_calls_tool_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When query includes order id, agent should bypass LLM and return API YAML."""
    from src.agent.sp_api import tools as sp_tools
    from src.agent.sp_api.sp_api_agent import SpApiReActAgent

    # Redirect persistence to temp directory for deterministic test isolation.
    results_dir = tmp_path / "sp_api_results"
    latest_file = results_dir / "latest_get_orders_result.json"
    monkeypatch.setattr(sp_tools, "_SP_API_RESULTS_DIR", results_dir)
    monkeypatch.setattr(sp_tools, "_SP_API_RESULTS_LATEST", latest_file)

    def fake_get_orders_batch(_client, _ids):
        return [
            {
                "order_id": "111-2886487-4917844",
                "ok": True,
                "payload": {
                    "payload": {
                        "AmazonOrderId": "111-2886487-4917844",
                        "OrderStatus": "Shipped",
                    }
                },
            }
        ]

    monkeypatch.setattr(sp_tools, "get_orders_batch", fake_get_orders_batch)

    tool = sp_tools.SpApiGetOrdersTool(client=object())
    # LLM should not be reached in this path.
    agent = SpApiReActAgent(llm=lambda _prompt: "unused", tools=[tool], max_iterations=2)

    answer = agent.run("what is order status of 111-2886487-4917844")

    assert "Amazon Selling Partner API getOrder response" in answer
    assert "OrderStatus: Shipped" in answer
    assert latest_file.is_file()


def test_sp_api_agent_direct_sku_calls_listings_tool_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When query includes SKU, agent should bypass LLM and return listings YAML."""
    from src.agent.sp_api import tools as sp_tools
    from src.agent.sp_api.sp_api_agent import SpApiReActAgent

    results_dir = tmp_path / "sp_api_results"
    latest_listings_file = results_dir / "latest_get_listings_result.json"
    monkeypatch.setattr(sp_tools, "_SP_API_RESULTS_DIR", results_dir)
    monkeypatch.setattr(sp_tools, "_SP_API_LISTINGS_RESULTS_LATEST", latest_listings_file)

    def fake_get_listings_items_batch(_client, _skus, **_kwargs):
        return [
            {
                "sku": "LJ-CHGYI-100818-04-1_SNB",
                "ok": True,
                "payload": {
                    "sku": "LJ-CHGYI-100818-04-1_SNB",
                    "summaries": [{"status": "BUYABLE"}],
                },
            }
        ]

    monkeypatch.setattr(sp_tools, "get_listings_items_batch", fake_get_listings_items_batch)

    listing_tool = sp_tools.SpApiGetListingsTool(client=object(), credentials=object())  # type: ignore[arg-type]
    agent = SpApiReActAgent(llm=lambda _prompt: "unused", tools=[listing_tool], max_iterations=2)

    answer = agent.run("Get listing item details for SKU LJ-CHGYI-100818-04-1_SNB as yaml")

    assert "Amazon Selling Partner API getListingsItem response" in answer
    assert "LJ-CHGYI-100818-04-1_SNB" in answer
    assert "sp_api_response" in answer
    assert latest_listings_file.is_file()
