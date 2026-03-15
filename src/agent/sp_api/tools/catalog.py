"""
Product Catalog Tool - Look up product by ASIN or SKU.
"""
from typing import Any, List

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError

from ..sp_api_client import SPAPIClient


class ProductCatalogTool(BaseTool):
    """Look up a product by ASIN or SKU. Returns title, price, category, status."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="product_catalog",
            description="Look up a product by ASIN or SKU. Returns title, price, category, status.",
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("identifier", "string", "ASIN or SKU", required=True),
            ToolParameter(
                "identifier_type",
                "string",
                "Type: asin or sku",
                required=True,
            ),
        ]

    def validate_parameters(self, identifier: str = None, identifier_type: str = None, **kwargs) -> None:
        if not identifier or not str(identifier).strip():
            raise ValidationError(message="identifier is required", field_name="identifier")
        it = (identifier_type or "").strip().lower()
        if it not in ("asin", "sku"):
            raise ValidationError(
                message="identifier_type must be 'asin' or 'sku'",
                field_name="identifier_type",
            )

    def execute(self, identifier: str, identifier_type: str, **kwargs) -> dict:
        self.validate_parameters(identifier=identifier, identifier_type=identifier_type, **kwargs)
        path = f"/catalog/2022-04-01/items/{identifier.strip()}"
        params = {
            "identifiersType": identifier_type.strip().lower(),
            "marketplaceIds": self._client.marketplace_id,
        }
        data = self._client.get(path, params=params)
        summaries = data.get("summaries", [])
        if not summaries:
            return {"identifier": identifier, "status": "not_found"}
        s = summaries[0]
        # Fix #7: correct nested array access for buyBoxPrice
        inner_summaries = s.get("summaries", [])
        price = inner_summaries[0].get("buyBoxPrice", {}).get("amount") if inner_summaries else None
        return {
            "asin": s.get("asin"),
            "sku": s.get("sku"),
            "title": s.get("itemName"),
            "price": price,
            "category": s.get("productType"),
            "status": s.get("lifecycleState", "unknown"),
        }
