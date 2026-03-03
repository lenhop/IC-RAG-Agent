"""Shipments Tools - List and create FBA inbound shipments."""
from typing import List
from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError
from ..sp_api_client import SPAPIClient


class ListShipmentsTool(BaseTool):
    """List FBA inbound shipments."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(name="list_shipments", description="List FBA inbound shipments.")
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("shipment_status_list", "array", "Filter by status", required=False)]

    def validate_parameters(self, **kwargs) -> None:
        pass

    def execute(self, shipment_status_list=None, **kwargs) -> dict:
        params = {"ShipmentStatusList": shipment_status_list} if shipment_status_list else None
        data = self._client.get("/fba/inbound/v0/shipments", params=params)
        items = data.get("payload", {}).get("ShipmentData", [])
        return {"shipments": items, "next_token": data.get("payload", {}).get("NextToken")}


class CreateShipmentTool(BaseTool):
    """Look up existing FBA inbound shipment details by shipment ID (read-only).

    NOTE: Creating shipments (POST) is disabled — this agent is read-only.
    Use list_shipments to find existing shipments, then this tool to get details.
    """

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="create_shipment",
            description="Get FBA inbound shipment details by shipment ID (read-only). "
                        "Creating shipments is not permitted.",
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("shipment_id", "string", "Shipment ID to look up", required=True),
        ]

    def validate_parameters(self, shipment_id=None, **kwargs) -> None:
        if not shipment_id or not str(shipment_id).strip():
            raise ValidationError(message="shipment_id is required", field_name="shipment_id")

    def execute(self, shipment_id=None, **kwargs) -> dict:
        self.validate_parameters(shipment_id=shipment_id, **kwargs)
        data = self._client.get(
            "/fba/inbound/v0/shipments",
            params={"ShipmentIdList": [shipment_id.strip()]},
        )
        items = data.get("payload", {}).get("ShipmentData", [])
        s = items[0] if items else {}
        return {
            "shipment_id": s.get("ShipmentId"),
            "name": s.get("ShipmentName"),
            "destination_fc": s.get("DestinationFulfillmentCenterId"),
            "status": s.get("ShipmentStatus"),
        }
