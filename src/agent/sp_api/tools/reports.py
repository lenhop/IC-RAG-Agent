"""
Reports Tool - List and retrieve existing reports (read-only).
"""
from typing import List

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError

from ..sp_api_client import SPAPIClient


class ReportRequestTool(BaseTool):
    """List existing SP-API reports or retrieve a report document URL (read-only GET).

    NOTE: Creating new reports (POST) is disabled — this agent is read-only.
    This tool lists existing reports and retrieves document URLs for completed ones.
    """

    def __init__(self, sp_api_client: SPAPIClient):
        super().__init__(
            name="list_reports",
            description=(
                "List existing SP-API reports or get a report document URL. "
                "Params: report_type (optional filter), report_id (optional, to get document URL)."
            ),
        )
        self._client = sp_api_client

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("report_type", "string", "Filter by report type (optional)", required=False),
            ToolParameter("report_id", "string", "Specific report ID to get document URL (optional)", required=False),
        ]

    def validate_parameters(self, report_type=None, report_id=None, **kwargs) -> None:
        # Both params are optional — nothing required
        pass

    def execute(self, report_type=None, report_id=None, **kwargs) -> dict:
        self.validate_parameters(report_type=report_type, report_id=report_id, **kwargs)

        # If a specific report_id is given, fetch its document URL
        if report_id and str(report_id).strip():
            rid = str(report_id).strip()
            status_resp = self._client.get(f"/reports/2021-06-30/reports/{rid}")
            status = status_resp.get("processingStatus", "UNKNOWN")
            doc_url = None
            if status == "DONE":
                doc_id = status_resp.get("reportDocumentId")
                if doc_id:
                    doc_resp = self._client.get(f"/reports/2021-06-30/documents/{doc_id}")
                    doc_url = doc_resp.get("url")
            return {
                "report_id": rid,
                "status": status,
                "document_url": doc_url,
            }

        # Otherwise list reports; reportTypes is required (min 1, max 10)
        report_type_val = str(report_type).strip() if report_type else "GET_FBA_INVENTORY_SUMMARY_DATA"
        params = {"reportTypes": report_type_val}

        data = self._client.get("/reports/2021-06-30/reports", params=params)
        reports = data.get("reports", [])
        return {
            "reports": [
                {
                    "report_id": r.get("reportId"),
                    "report_type": r.get("reportType"),
                    "status": r.get("processingStatus"),
                    "created_time": r.get("createdTime"),
                }
                for r in reports
            ],
            "note": "Creating new reports (POST) is disabled. This lists existing reports only.",
        }
