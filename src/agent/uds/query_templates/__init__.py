"""
Query template registry.
Provides easy access to all query templates.
"""

from .sales import SalesQueries
from .inventory import InventoryQueries
from .financial import FinancialQueries
from .products import ProductQueries
from ..uds_client import UDSClient


class QueryTemplateRegistry:
    """
    Central registry for all query templates.
    """

    def __init__(self, client: UDSClient):
        self.client = client

        # Initialize all query categories
        self.sales = SalesQueries(client)
        self.inventory = InventoryQueries(client)
        self.financial = FinancialQueries(client)
        self.products = ProductQueries(client)

    def list_templates(self) -> dict:
        """
        List all available query templates.

        Returns:
            Dictionary of categories and their templates
        """
        return {
            "sales": [
                "daily_sales_trend",
                "top_products_by_revenue",
                "sales_by_marketplace",
                "hourly_sales_pattern",
                "sales_growth_rate",
                "monthly_sales_comparison",
                "sales_by_category",
                "sales_by_region",
                "repeat_purchase_rate",
                "sales_velocity",
                "sales_forecast_accuracy",
                "discount_impact_analysis",
                "sales_channel_performance",
                "sales_by_fulfillment_type",
                "sales_peak_hours",
                "sales_seasonality"
            ],
            "inventory": [
                "current_inventory_levels",
                "low_stock_alert",
                "inventory_turnover",
                "inventory_aging",
                "stockout_analysis",
                "inventory_by_warehouse",
                "reorder_point_analysis",
                "inventory_valuation",
                "inventory_movement",
                "safety_stock_analysis"
            ],
            "financial": [
                "revenue_summary",
                "fee_analysis",
                "profitability_by_product",
                "transaction_summary",
                "cash_flow_analysis",
                "profit_margin_trends",
                "cost_breakdown",
                "revenue_by_payment_method",
                "refund_analysis",
                "tax_summary"
            ],
            "products": [
                "top_selling_products",
                "product_search_performance",
                "product_listing_quality",
                "product_lifecycle_analysis",
                "product_performance_comparison",
                "product_category_analysis",
                "product_price_analysis",
                "product_review_analysis",
                "product_return_analysis",
                "product_cross_sell_analysis"
            ]
        }

    def get_template_info(self, category: str, template_name: str) -> dict:
        """
        Get information about a specific template.

        Args:
            category: Template category (sales, inventory, financial, products, customers)
            template_name: Template function name

        Returns:
            Dictionary with template metadata
        """
        category_obj = getattr(self, category)
        template_func = getattr(category_obj, template_name)

        return {
            "name": template_name,
            "category": category,
            "description": template_func.__doc__,
            "parameters": template_func.__annotations__
        }


# Example usage:
# from src.uds.query_templates import QueryTemplateRegistry
# from src.uds.uds_client import UDSClient
#
# client = UDSClient(...)
# registry = QueryTemplateRegistry(client)
#
# # Use templates
# df = registry.sales.daily_sales_trend('2025-10-01', '2025-10-31')
# df = registry.inventory.low_stock_alert(threshold=5)
