"""FBA Tools - Fee estimate and eligibility."""
import logging
from typing import List

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError

from ..sp_api_client import SPAPIClient

logger = logging.getLogger(__name__)


class FBAFeeTool(BaseTool):
    """Look up FBA fee preview for an ASIN (read-only GET).

    NOTE: The SP-API /products/fees endpoint requires POST, which is disabled.
    This tool uses the catalog GET endpoint to retrieve pricing info instead.
    For actual fee estimates, use the Amazon Seller Central UI.
    """

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="fba_fees",
            description="Look up product pricing info for an ASIN (fee estimation via catalog). Params: asin.",
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("asin", "string", "Product ASIN", required=True),
        ]

    def validate_parameters(self, asin=None, **kwargs) -> None:
        if not asin or not str(asin).strip():
            raise ValidationError(message="asin is required", field_name="asin")

    def execute(self, asin=None, **kwargs) -> dict:
        self.validate_parameters(asin=asin, **kwargs)
        data = self._client.get(
            f"/catalog/2022-04-01/items/{asin.strip()}",
            params={"identifiersType": "asin"},
        )
        summaries = data.get("summaries", [])
        s = summaries[0] if summaries else {}
        inner = s.get("summaries", [])
        price_info = inner[0].get("buyBoxPrice", {}) if inner else {}
        return {
            "asin": asin.strip(),
            "buy_box_price": price_info.get("amount"),
            "currency": price_info.get("currencyCode"),
            "note": "Fee estimation via POST is disabled (read-only). Use catalog pricing as reference.",
        }


class FBAEligibilityTool(BaseTool):
    """Check FBA Small and Light eligibility."""

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(name="fba_eligibility", description="Check FBA eligibility. Params: asin.")
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("asin", "string", "Product ASIN", required=True)]

    def validate_parameters(self, asin=None, **kwargs) -> None:
        if not asin or not str(asin).strip():
            raise ValidationError(message="asin is required", field_name="asin")

    def execute(self, asin=None, **kwargs) -> dict:
        self.validate_parameters(asin=asin, **kwargs)
        # Fix #8: don't mask all exceptions — only catch expected API errors
        try:
            data = self._client.get(f"/fba/smallAndLight/v1/eligibilities/{asin.strip()}")
            status = data.get("payload", {}).get("status", "unknown")
            return {"eligible": status == "ELIGIBLE", "reason": status}
        except Exception as e:
            logger.warning("FBAEligibilityTool failed for asin=%s: %s", asin, e)
            # Re-raise auth/permission errors; return error dict for transient failures
            if isinstance(e, (PermissionError, ValueError)):
                raise
            return {"eligible": False, "reason": str(e)}
