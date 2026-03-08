"""
Unit tests for UDS Result Formatter.
"""

import pytest

from src.uds.result_formatter import UDSResultFormatter, FormattedResponse
from src.uds.intent_classifier import IntentResult, IntentDomain


@pytest.fixture
def formatter():
    """Result formatter instance."""
    return UDSResultFormatter()


@pytest.fixture
def sales_intent():
    """Sales domain intent."""
    return IntentResult(
        primary_domain=IntentDomain.SALES,
        secondary_domains=[],
        confidence=0.9,
        keywords=["sales", "revenue"],
        suggested_tools=["SalesTrendTool"],
        reasoning="Test",
    )


@pytest.fixture
def inventory_intent():
    """Inventory domain intent."""
    return IntentResult(
        primary_domain=IntentDomain.INVENTORY,
        secondary_domains=[],
        confidence=0.9,
        keywords=[],
        suggested_tools=["InventoryAnalysisTool"],
        reasoning="Test",
    )


@pytest.fixture
def financial_intent():
    """Financial domain intent."""
    return IntentResult(
        primary_domain=IntentDomain.FINANCIAL,
        secondary_domains=[],
        confidence=0.9,
        keywords=[],
        suggested_tools=["FinancialSummaryTool"],
        reasoning="Test",
    )


class TestResultFormatter:
    """Tests for UDSResultFormatter."""

    def test_format_returns_formatted_response(self, formatter, sales_intent):
        """Format returns FormattedResponse."""
        agent_result = {"output": {"insights": {"total_revenue": 1000}}}
        result = formatter.format(agent_result, sales_intent)
        assert isinstance(result, FormattedResponse)
        assert hasattr(result, "summary")
        assert hasattr(result, "insights")
        assert hasattr(result, "data")
        assert hasattr(result, "charts")
        assert hasattr(result, "recommendations")
        assert hasattr(result, "metadata")

    def test_extract_data_from_output(self, formatter, sales_intent):
        """Extracts data from output key."""
        agent_result = {"output": {"insights": {"total_revenue": 5000}}}
        result = formatter.format(agent_result, sales_intent)
        assert result.data["insights"]["total_revenue"] == 5000

    def test_extract_data_from_data_key(self, formatter, sales_intent):
        """Extracts data from data key."""
        agent_result = {"data": {"total": 100}}
        result = formatter.format(agent_result, sales_intent)
        assert result.data["total"] == 100

    def test_summarize_sales(self, formatter, sales_intent):
        """Sales summary generated from insights."""
        agent_result = {
            "output": {
                "insights": {
                    "total_revenue": 10000,
                    "total_orders": 500,
                    "avg_order_value": 20,
                    "growth_rate_pct": 15.5,
                    "trend": "increasing",
                }
            }
        }
        result = formatter.format(agent_result, sales_intent)
        assert "10000" in result.summary or "10,000" in result.summary
        assert "500" in result.summary or "increasing" in result.summary

    def test_summarize_inventory(self, formatter, inventory_intent):
        """Inventory summary generated."""
        agent_result = {
            "output": {
                "inventory_summary": {
                    "total_skus": 100,
                    "total_units": 5000,
                    "low_stock_items": 5,
                    "stockout_risk": 2,
                }
            }
        }
        result = formatter.format(agent_result, inventory_intent)
        assert "100" in result.summary
        assert "5" in result.summary or "low" in result.summary.lower()

    def test_summarize_financial(self, formatter, financial_intent):
        """Financial summary generated."""
        agent_result = {
            "output": {
                "financial_metrics": {
                    "total_revenue": 50000,
                    "total_fees": 5000,
                    "net_revenue": 45000,
                    "profit_margin_pct": 90,
                }
            }
        }
        result = formatter.format(agent_result, financial_intent)
        assert "50000" in result.summary or "50,000" in result.summary
        assert "90" in result.summary

    def test_extract_insights_growth(self, formatter, sales_intent):
        """Growth insight extracted when growth > 10%."""
        agent_result = {
            "output": {"insights": {"growth_rate_pct": 25}}
        }
        result = formatter.format(agent_result, sales_intent)
        assert any("25" in i for i in result.insights) or len(result.insights) >= 1

    def test_extract_insights_low_stock(self, formatter, inventory_intent):
        """Low stock insight extracted."""
        agent_result = {
            "output": {
                "inventory_summary": {"low_stock_items": 10},
                "alerts": ["5 items below threshold"],
            }
        }
        result = formatter.format(agent_result, inventory_intent)
        assert len(result.insights) >= 1

    def test_recommendations_inventory(self, formatter, inventory_intent):
        """Recommendations for low stock."""
        agent_result = {
            "output": {"inventory_summary": {"low_stock_items": 3}}
        }
        result = formatter.format(agent_result, inventory_intent)
        assert any("reorder" in r.lower() for r in result.recommendations)

    def test_metadata_includes_intent(self, formatter, sales_intent):
        """Metadata includes intent and confidence."""
        agent_result = {"output": {}}
        result = formatter.format(agent_result, sales_intent)
        assert result.metadata["intent"] == "sales"
        assert result.metadata["confidence"] == 0.9

    def test_summarize_uses_precomputed_summary(self, formatter, sales_intent):
        """Uses precomputed summary when present."""
        agent_result = {
            "output": {
                "summary": "Custom summary text",
                "insights": {},
            }
        }
        result = formatter.format(agent_result, sales_intent)
        assert result.summary == "Custom summary text"
