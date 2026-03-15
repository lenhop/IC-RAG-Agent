"""Financials Tool - Financial events."""
from typing import List
from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError
from ..sp_api_client import SPAPIClient


class FinancialsTool(BaseTool):
    """Retrieve financial events."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(name="financials", description="Retrieve financial events. Params: posted_after.")
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("posted_after", "string", "ISO date", required=True),
            ToolParameter("posted_before", "string", "ISO date", required=False),
        ]

    def validate_parameters(self, posted_after=None, **kwargs) -> None:
        if not posted_after or not str(posted_after).strip():
            raise ValidationError(message="posted_after is required", field_name="posted_after")

    def execute(self, posted_after=None, posted_before=None, **kwargs) -> dict:
        self.validate_parameters(posted_after=posted_after, **kwargs)
        params = {"PostedAfter": posted_after.strip(), "MarketplaceIds": ["ATVPDKIKX0DER"]}
        if posted_before:
            params["PostedBefore"] = posted_before.strip()
        data = self._client.get("/finances/v0/financialEvents", params=params)
        payload = data.get("payload", {})
        return {"financial_events": payload.get("FinancialEvents", {}), "next_token": payload.get("NextToken")}
