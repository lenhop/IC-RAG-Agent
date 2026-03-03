"""
Agent Tools

This module provides stub tool implementations for SP-API and UDS integrations.
Used for testing the ReAct Agent before real integrations are available.

Feature: react-agent-core
"""

from .sp_api_stubs import (
    ProductCatalogToolStub,
    InventorySummaryToolStub,
    OrderDetailsToolStub,
)
from .uds_stubs import UDSQueryToolStub, UDSReportGeneratorToolStub

__all__ = [
    "ProductCatalogToolStub",
    "InventorySummaryToolStub",
    "OrderDetailsToolStub",
    "UDSQueryToolStub",
    "UDSReportGeneratorToolStub",
]
