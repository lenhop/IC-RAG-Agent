"""
UDS Stub Tools

Stub implementations of Unified Data Service tools for testing the ReAct Agent
before real UDS integrations are available.

Classes:
    UDSQueryToolStub: Mock UDS query execution
    UDSReportGeneratorToolStub: Mock report generation

Feature: react-agent-core, Req 8.4-8.5
"""

from typing import Any, List

from ai_toolkit.tools import BaseTool, ToolParameter
from ai_toolkit.errors import ValidationError


class UDSQueryToolStub(BaseTool):
    """
    Stub for UDS query execution.

    Returns mock query result rows.
    """

    _is_stub = True

    def __init__(self):
        super().__init__(
            name="uds_query",
            description="Executes a query against UDS and returns results (stub)",
        )

    def execute(self, query: str = None, **kwargs) -> List[dict]:
        """Return mock query results."""
        self.validate_parameters(query=query, **kwargs)
        return [
            {"id": 1, "name": "Mock Result 1", "value": 100},
            {"id": 2, "name": "Mock Result 2", "value": 200},
        ]

    def validate_parameters(self, query: str = None, **kwargs) -> None:
        """Require query."""
        if not query or not str(query).strip():
            raise ValidationError(
                message="query is required",
                field_name="query",
            )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="UDS query string",
                required=True,
            ),
        ]


class UDSReportGeneratorToolStub(BaseTool):
    """
    Stub for UDS report generation.

    Returns mock markdown report.
    """

    _is_stub = True

    def __init__(self):
        super().__init__(
            name="uds_report",
            description="Generates a report of the specified type (stub)",
        )

    def execute(self, report_type: str = None, **kwargs) -> str:
        """Return mock report as markdown."""
        self.validate_parameters(report_type=report_type, **kwargs)
        return f"""# Mock Report: {report_type}

## Summary
This is a stub report for testing.

## Data
- Metric A: 100
- Metric B: 200
- Metric C: 300
"""

    def validate_parameters(self, report_type: str = None, **kwargs) -> None:
        """Require report_type."""
        if not report_type or not str(report_type).strip():
            raise ValidationError(
                message="report_type is required",
                field_name="report_type",
            )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="report_type",
                type="string",
                description="Type of report to generate",
                required=True,
            ),
        ]
