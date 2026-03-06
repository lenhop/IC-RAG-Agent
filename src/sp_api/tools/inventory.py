"""Inventory Tool - FBA inventory summary."""
from typing import List
from ai_toolkit.tools import BaseTool, ToolParameter
from ..sp_api_client import SPAPIClient


class InventoryTool(BaseTool):
    """Retrieve FBA inventory summaries."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(name="inventory_summary", description="Retrieves inventory levels by SKU.")
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("sku", "string", "Filter by SKU", required=False),
            ToolParameter("next_token", "string", "Pagination token", required=False),
        ]

    def validate_parameters(self, **kwargs) -> None:
        pass

    def execute(self, sku=None, next_token=None, **kwargs) -> dict:
        # SP-API FBA inventory requires marketplaceIds, details, granularityType, granularityId
        mid = self._client.marketplace_id
        params = {
            "marketplaceIds": mid,
            "details": "true",
            "granularityType": "Marketplace",
            "granularityId": mid,
        }
        if sku:
            params["QueryType"] = "INVENTORY_SKU"
            params["QueryValue"] = sku
        if next_token:
            params["nextToken"] = next_token
        data = self._client.get("/fba/inventory/v1/summaries", params=params)
        items = data.get("payload", {}).get("inventorySummaries", [])
        return {"items": items, "next_token": data.get("payload", {}).get("nextToken")}
