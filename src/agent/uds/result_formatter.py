"""
Result Formatter for UDS Agent.

Formats agent results into user-friendly output: summaries, insights,
visualizations, and actionable recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .intent_classifier import IntentDomain, IntentResult

logger = logging.getLogger(__name__)


@dataclass
class FormattedResponse:
    """Formatted response for user."""

    summary: str
    insights: List[str]
    data: Optional[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class UDSResultFormatter:
    """
    Formats agent results into user-friendly output.

    Generates natural language summaries, extracts insights,
    creates visualizations, and provides recommendations.
    """

    def __init__(self):
        """Initialize result formatter."""
        self._chart_tool = None
        self._dashboard_tool = None

    def _get_chart_tool(self):
        """Lazy load CreateChartTool to avoid circular imports."""
        if self._chart_tool is None:
            from .tools import CreateChartTool
            self._chart_tool = CreateChartTool()
        return self._chart_tool

    def format(
        self,
        agent_result: Dict[str, Any],
        intent: IntentResult,
    ) -> FormattedResponse:
        """
        Format agent result based on intent.

        Args:
            agent_result: Raw agent output (dict with output/data or ToolResult.output)
            intent: Query intent

        Returns:
            FormattedResponse
        """
        data = self._extract_data(agent_result)
        summary = self._generate_summary(data, intent)
        insights = self._extract_insights(data, intent)
        charts = self._create_visualizations(data, intent)
        recommendations = self._generate_recommendations(data, intent)

        return FormattedResponse(
            summary=summary,
            insights=insights,
            data=data,
            charts=charts,
            recommendations=recommendations,
            metadata={
                "intent": intent.primary_domain.value,
                "confidence": intent.confidence,
                "tools_used": agent_result.get("tools_used", []),
            },
        )

    def _extract_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from agent result."""
        if "output" in result:
            out = result["output"]
            return out if isinstance(out, dict) else {"raw": out}
        if "data" in result:
            data = result["data"]
            return data if isinstance(data, dict) else {"raw": data}
        return result if isinstance(result, dict) else {"raw": result}

    def _generate_summary(self, data: Dict[str, Any], intent: IntentResult) -> str:
        """Generate text summary based on domain."""
        domain = intent.primary_domain

        if domain == IntentDomain.SALES:
            return self._summarize_sales(data)
        if domain == IntentDomain.INVENTORY:
            return self._summarize_inventory(data)
        if domain == IntentDomain.FINANCIAL:
            return self._summarize_financial(data)
        if domain == IntentDomain.PRODUCT:
            return self._summarize_product(data)
        if domain == IntentDomain.COMPARISON:
            return self._summarize_comparison(data)
        return self._summarize_general(data)

    def _summarize_sales(self, data: Dict[str, Any]) -> str:
        """Generate sales summary."""
        if "summary" in data and data["summary"]:
            return data["summary"]
        if "insights" in data:
            insights = data["insights"]
            return f"""Sales Analysis:
- Total Revenue: ${insights.get('total_revenue', 0):,.2f}
- Total Orders: {insights.get('total_orders', 0):,}
- Average Order Value: ${insights.get('avg_order_value', 0):.2f}
- Growth Rate: {insights.get('growth_rate_pct', 0):.2f}%
- Trend: {str(insights.get('trend', 'N/A')).capitalize()}"""
        return "Sales data retrieved successfully."

    def _summarize_inventory(self, data: Dict[str, Any]) -> str:
        """Generate inventory summary."""
        if "summary" in data and data["summary"]:
            return data["summary"]
        if "inventory_summary" in data:
            summary = data["inventory_summary"]
            return f"""Inventory Analysis:
- Total SKUs: {summary.get('total_skus', 0):,}
- Total Units: {summary.get('total_units', 0):,}
- Low Stock Items: {summary.get('low_stock_items', 0)}
- Stockout Risk: {summary.get('stockout_risk', 0)} items"""
        return "Inventory data retrieved successfully."

    def _summarize_financial(self, data: Dict[str, Any]) -> str:
        """Generate financial summary."""
        if "summary" in data and data["summary"]:
            return data["summary"]
        if "financial_metrics" in data:
            metrics = data["financial_metrics"]
            return f"""Financial Summary:
- Gross Revenue: ${metrics.get('total_revenue', 0):,.2f}
- Total Fees: ${metrics.get('total_fees', 0):,.2f}
- Net Revenue: ${metrics.get('net_revenue', 0):,.2f}
- Profit Margin: {metrics.get('profit_margin_pct', 0):.2f}%"""
        return "Financial data retrieved successfully."

    def _summarize_product(self, data: Dict[str, Any]) -> str:
        """Generate product summary."""
        if "summary" in data and data["summary"]:
            return data["summary"]
        if "insights" in data:
            insights = data["insights"]
            top = insights.get("top_product", {})
            return f"""Product Performance:
- Products Analyzed: {insights.get('total_products', 0)}
- Total Revenue: ${insights.get('total_revenue', 0):,.2f}
- Top Product: {top.get('name', 'N/A')}
- Top Product Revenue: ${top.get('revenue', 0):,.2f}"""
        return "Product data retrieved successfully."

    def _summarize_comparison(self, data: Dict[str, Any]) -> str:
        """Generate comparison summary."""
        if "summary" in data and data["summary"]:
            return data["summary"]
        if "comparison" in data:
            comp = data["comparison"]
            growth = comp.get("growth", {})
            return f"""Period Comparison:
- Revenue Growth: {growth.get('revenue_pct', 0):.2f}%
- Order Growth: {growth.get('orders_pct', 0):.2f}%"""
        return "Comparison completed successfully."

    def _summarize_general(self, data: Dict[str, Any]) -> str:
        """Generate general summary."""
        if "tables" in data:
            return f"Found {len(data['tables'])} tables in the database."
        if "columns" in data:
            return f"Table has {len(data['columns'])} columns."
        return "Query executed successfully."

    def _extract_insights(self, data: Dict[str, Any], intent: IntentResult) -> List[str]:
        """Extract key insights from data."""
        insights: List[str] = []

        if "insights" in data and isinstance(data["insights"], dict):
            insight_data = data["insights"]
            if "growth_rate_pct" in insight_data:
                rate = insight_data["growth_rate_pct"]
                if rate > 10:
                    insights.append(f"Strong growth of {rate:.1f}%")
                elif rate < -10:
                    insights.append(f"Declining by {abs(rate):.1f}%")

        inv_summary = data.get("inventory_summary", {})
        if isinstance(inv_summary, dict) and "low_stock_items" in inv_summary:
            count = inv_summary["low_stock_items"]
            if count > 0:
                insights.append(f"{count} items need attention")

        fin_metrics = data.get("financial_metrics", {})
        if isinstance(fin_metrics, dict) and "profit_margin_pct" in fin_metrics:
            margin = fin_metrics["profit_margin_pct"]
            if margin < 10:
                insights.append(f"Low profit margin: {margin:.1f}%")

        if "alerts" in data and isinstance(data["alerts"], list):
            insights.extend(data["alerts"])

        return insights if insights else ["Analysis completed successfully"]

    def _create_visualizations(self, data: Dict[str, Any], intent: IntentResult) -> List[Dict[str, Any]]:
        """Create appropriate visualizations."""
        charts: List[Dict[str, Any]] = []

        try:
            chart_tool = self._get_chart_tool()
        except Exception as e:
            logger.warning("Could not load CreateChartTool: %s", e)
            return charts

        if "sales_data" in data:
            sales_data = data["sales_data"]
            if isinstance(sales_data, list) and len(sales_data) > 0:
                result = chart_tool.execute(
                    data=sales_data,
                    chart_type="line",
                    x_column="date",
                    y_column="total_revenue",
                    title="Revenue Trend",
                )
                if hasattr(result, "success") and result.success and hasattr(result, "output"):
                    charts.append(result.output)

        if "top_products" in data:
            products = data["top_products"]
            if isinstance(products, list) and len(products) > 0:
                result = chart_tool.execute(
                    data=products[:10],
                    chart_type="bar",
                    x_column="product_name",
                    y_column="total_revenue",
                    title="Top Products",
                )
                if hasattr(result, "success") and result.success and hasattr(result, "output"):
                    charts.append(result.output)

        return charts

    def _generate_recommendations(self, data: Dict[str, Any], intent: IntentResult) -> List[str]:
        """Generate actionable recommendations."""
        recommendations: List[str] = []
        domain = intent.primary_domain

        inv_summary = data.get("inventory_summary", {})
        if domain == IntentDomain.INVENTORY and isinstance(inv_summary, dict):
            count = inv_summary.get("low_stock_items", 0)
            if count > 0:
                recommendations.append(f"Review and reorder {count} low stock items")

        fin_metrics = data.get("financial_metrics", {})
        if domain == IntentDomain.FINANCIAL and isinstance(fin_metrics, dict):
            margin = fin_metrics.get("profit_margin_pct", 0)
            if margin < 15:
                recommendations.append("Consider reviewing pricing strategy to improve margins")

        insights = data.get("insights", {})
        if domain == IntentDomain.SALES and isinstance(insights, dict):
            rate = insights.get("growth_rate_pct", 0)
            if rate < 0:
                recommendations.append("Investigate causes of declining sales")

        return recommendations
