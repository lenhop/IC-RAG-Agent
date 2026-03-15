"""
Orders Tools - List orders and get order details.
"""
from typing import Any, List, Optional

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError

from ..sp_api_client import SPAPIClient


class ListOrdersTool(BaseTool):
    """List orders with optional filters."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="list_orders",
            description="List orders. Params: created_after (required), created_before, order_statuses.",
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("created_after", "string", "ISO date (required)", required=True),
            ToolParameter("created_before", "string", "ISO date (optional)", required=False),
            ToolParameter("order_statuses", "array", "Filter by status (optional)", required=False),
        ]

    def validate_parameters(self, created_after: str = None, **kwargs) -> None:
        if not created_after or not str(created_after).strip():
            raise ValidationError(message="created_after is required", field_name="created_after")

    def execute(
        self,
        created_after: str,
        created_before: str = None,
        order_statuses: List[str] = None,
        **kwargs,
    ) -> dict:
        self.validate_parameters(created_after=created_after, **kwargs)
        params: dict = {"CreatedAfter": created_after.strip(), "MarketplaceIds": ["ATVPDKIKX0DER"]}
        if created_before:
            params["CreatedBefore"] = created_before.strip()
        if order_statuses:
            params["OrderStatuses"] = order_statuses
        data = self._client.get("/orders/v0/orders", params=params)
        orders = data.get("payload", {}).get("Orders", [])
        return {
            "orders": [
                {
                    "order_id": o.get("AmazonOrderId"),
                    "purchase_date": o.get("PurchaseDate"),
                    "status": o.get("OrderStatus"),
                    "total": o.get("OrderTotal", {}).get("Amount"),
                    "item_count": o.get("NumberOfItemsShipped", 0) + o.get("NumberOfItemsUnshipped", 0),
                }
                for o in orders
            ],
            "next_token": data.get("payload", {}).get("NextToken"),
        }


class OrderDetailsTool(BaseTool):
    """Get order details and line items."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="order_details",
            description="Get order details by order_id.",
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("order_id", "string", "Amazon order ID", required=True),
        ]

    def validate_parameters(self, order_id: str = None, **kwargs) -> None:
        if not order_id or not str(order_id).strip():
            raise ValidationError(message="order_id is required", field_name="order_id")

    def execute(self, order_id: str, **kwargs) -> dict:
        self.validate_parameters(order_id=order_id, **kwargs)
        oid = order_id.strip()
        path = f"/orders/v0/orders/{oid}"
        order_data = self._client.get(path)
        items_data = self._client.get(f"{path}/orderItems")
        order = order_data.get("payload", {})
        items = items_data.get("payload", {}).get("OrderItems", [])
        return {
            "order_id": order.get("AmazonOrderId"),
            "status": order.get("OrderStatus"),
            "purchase_date": order.get("PurchaseDate"),
            "total": order.get("OrderTotal", {}).get("Amount"),
            "items": [
                {
                    "order_item_id": i.get("OrderItemId"),
                    "sku": i.get("SellerSKU"),
                    "quantity": i.get("QuantityOrdered"),
                    "price": i.get("ItemPrice", {}).get("Amount"),
                }
                for i in items
            ],
        }
