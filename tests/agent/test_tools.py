"""
Tests for stub tools.

Feature: react-agent-core
"""

import pytest

from ai_toolkit.tools import ToolExecutor
from ai_toolkit.errors import ValidationError

from src.agent.tools import (
    ProductCatalogToolStub,
    InventorySummaryToolStub,
    OrderDetailsToolStub,
    UDSQueryToolStub,
    UDSReportGeneratorToolStub,
)


class TestProductCatalogToolStub:
    """Tests for ProductCatalogToolStub."""

    def test_execute_with_asin(self):
        tool = ProductCatalogToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, asin="B001TEST")
        assert result.success is True
        assert result.output is not None
        assert "title" in result.output
        assert "price" in result.output

    def test_execute_with_sku(self):
        tool = ProductCatalogToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, sku="SKU-001")
        assert result.success is True

    def test_invalid_params_raises(self):
        tool = ProductCatalogToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool)
        assert result.success is False
        assert "validation" in result.error.lower() or "required" in result.error.lower()


class TestInventorySummaryToolStub:
    """Tests for InventorySummaryToolStub."""

    def test_execute_valid(self):
        tool = InventorySummaryToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, sku="SKU-001")
        assert result.success is True
        assert "quantity" in result.output
        assert "fulfillment_center" in result.output

    def test_missing_sku_fails(self):
        tool = InventorySummaryToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool)
        assert result.success is False


class TestOrderDetailsToolStub:
    """Tests for OrderDetailsToolStub."""

    def test_execute_valid(self):
        tool = OrderDetailsToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, order_id="ORDER-123")
        assert result.success is True
        assert "status" in result.output
        assert "items" in result.output


class TestUDSQueryToolStub:
    """Tests for UDSQueryToolStub."""

    def test_execute_valid(self):
        tool = UDSQueryToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, query="SELECT * FROM table")
        assert result.success is True
        assert isinstance(result.output, list)
        assert len(result.output) > 0


class TestUDSReportGeneratorToolStub:
    """Tests for UDSReportGeneratorToolStub."""

    def test_execute_valid(self):
        tool = UDSReportGeneratorToolStub()
        executor = ToolExecutor()
        result = executor.execute(tool, report_type="sales")
        assert result.success is True
        assert isinstance(result.output, str)
        assert "Mock Report" in result.output or "sales" in result.output


class TestStubToolsHaveIsStub:
    """Verify all stubs have _is_stub=True."""

    def test_all_stubs_marked(self):
        stubs = [
            ProductCatalogToolStub(),
            InventorySummaryToolStub(),
            OrderDetailsToolStub(),
            UDSQueryToolStub(),
            UDSReportGeneratorToolStub(),
        ]
        for stub in stubs:
            assert getattr(stub, "_is_stub", False) is True
