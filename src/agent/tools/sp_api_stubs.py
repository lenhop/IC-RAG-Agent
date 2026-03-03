"""
SP-API Stub Tools

Stub implementations of Seller Partner API tools for testing the ReAct Agent
before real SP-API integrations are available.

Classes:
    ProductCatalogToolStub: Mock product catalog lookup
    InventorySummaryToolStub: Mock inventory summary
    OrderDetailsToolStub: Mock order details

Feature: react-agent-core, Req 8.1-8.3
"""

from typing import Any, List

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError


class ProductCatalogToolStub(BaseTool):
    """
    Stub for product catalog lookup by ASIN or SKU.

    Returns realistic mock product data.
    """

    _is_stub = True

    def __init__(self):
        super().__init__(
            name="product_catalog",
            description="Retrieves product details by ASIN or SKU (stub)",
        )

    def execute(self, asin: str = None, sku: str = None, **kwargs) -> dict:
        """Return mock product data."""
        self.validate_parameters(asin=asin, sku=sku, **kwargs)
        identifier = asin or sku or "UNKNOWN"
        return {
            "asin": identifier if identifier.startswith("B0") else f"B0{identifier}",
            "sku": identifier,
            "title": "Mock Product - Test Item",
            "price": 29.99,
            "category": "Electronics",
            "brand": "MockBrand",
        }

    def validate_parameters(self, asin: str = None, sku: str = None, **kwargs) -> None:
        """Require either asin or sku."""
        if not asin and not sku:
            raise ValidationError(
                message="Either asin or sku is required",
                field_name="asin",
            )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="asin",
                type="string",
                description="Amazon ASIN",
                required=False,
            ),
            ToolParameter(
                name="sku",
                type="string",
                description="Seller SKU",
                required=False,
            ),
        ]


class InventorySummaryToolStub(BaseTool):
    """
    Stub for inventory summary by SKU.

    Returns realistic mock inventory data.
    """

    _is_stub = True

    def __init__(self):
        super().__init__(
            name="inventory_summary",
            description="Retrieves inventory levels by SKU (stub)",
        )

    def execute(self, sku: str = None, **kwargs) -> dict:
        """Return mock inventory data."""
        self.validate_parameters(sku=sku, **kwargs)
        return {
            "sku": sku,
            "quantity": 150,
            "fulfillment_center": "AMZN_FC_NA",
            "reserved": 5,
            "inbound": 20,
        }

    def validate_parameters(self, sku: str = None, **kwargs) -> None:
        """Require sku."""
        if not sku or not str(sku).strip():
            raise ValidationError(
                message="sku is required",
                field_name="sku",
            )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="sku",
                type="string",
                description="Seller SKU",
                required=True,
            ),
        ]


class OrderDetailsToolStub(BaseTool):
    """
    Stub for order details by order ID.

    Returns realistic mock order data.
    """

    _is_stub = True

    def __init__(self):
        super().__init__(
            name="order_details",
            description="Retrieves order details by order ID (stub)",
        )

    def execute(self, order_id: str = None, **kwargs) -> dict:
        """Return mock order data."""
        self.validate_parameters(order_id=order_id, **kwargs)
        return {
            "order_id": order_id,
            "status": "Shipped",
            "items": [
                {"sku": "ITEM-001", "quantity": 2, "price": 19.99},
            ],
            "shipping": {
                "carrier": "USPS",
                "tracking": "9400111899223344556677",
            },
        }

    def validate_parameters(self, order_id: str = None, **kwargs) -> None:
        """Require order_id."""
        if not order_id or not str(order_id).strip():
            raise ValidationError(
                message="order_id is required",
                field_name="order_id",
            )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="order_id",
                type="string",
                description="Amazon order ID",
                required=True,
            ),
        ]
