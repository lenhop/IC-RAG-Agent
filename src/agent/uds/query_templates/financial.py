"""
Financial analysis query templates.
"""

from typing import Optional
import pandas as pd
from ..uds_client import UDSClient


class FinancialQueries:
    """Financial analysis query templates."""

    def __init__(self, client: UDSClient):
        self.client = client

    def revenue_summary(
        self,
        start_date: str,
        end_date: str,
        group_by: str = 'day'
    ) -> pd.DataFrame:
        """
        Get revenue summary with breakdown.

        Args:
            start_date: Start date
            end_date: End date
            group_by: Grouping level (day, week, month)

        Returns:
            DataFrame with revenue breakdown
        """
        if group_by == 'day':
            date_expr = 'start_date'
        elif group_by == 'week':
            date_expr = 'toStartOfWeek(start_date)'
        else:  # month
            date_expr = 'toStartOfMonth(start_date)'

        query = f"""
        SELECT
            {date_expr} as period,
            SUM(item_price) as gross_revenue,
            COUNT(DISTINCT amazon_order_id) as order_count,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY period
        ORDER BY period
        """
        return self.client.query(query)

    def fee_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Analyze fees. amz_fee uses estimated_fee_total, not fee_type/fee_amount."""
        query = f"""
        SELECT
            'total' as fee_type,
            COUNT(*) as fee_count,
            SUM(estimated_fee_total) as total_fees,
            AVG(estimated_fee_total) as avg_fee
        FROM ic_agent.amz_fee
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        """
        return self.client.query(query)

    def profitability_by_product(
        self,
        start_date: str,
        end_date: str,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Calculate profitability by product.

        Profit = Revenue - Fees
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            SUM(o.item_price) as revenue,
            SUM(f.estimated_fee_total) as total_fees,
            (SUM(o.item_price) - SUM(f.estimated_fee_total)) as profit,
            ((SUM(o.item_price) - SUM(f.estimated_fee_total)) / SUM(o.item_price) * 100) as profit_margin_pct
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_fee f ON o.asin = f.asin AND o.start_date = f.start_date
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        HAVING revenue > 0
        ORDER BY profit DESC
        LIMIT {limit}
        """
        return self.client.query(query)

    def transaction_summary(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Summarize transactions by type."""
        query = f"""
        SELECT
            transaction_type,
            COUNT(*) as transaction_count,
            SUM(amount) as item_price,
            AVG(amount) as avg_amount
        FROM ic_agent.amz_transaction
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY transaction_type
        ORDER BY item_price DESC
        """
        return self.client.query(query)

    def cash_flow_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze cash flow from transactions.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with cash flow metrics
        """
        query = f"""
        SELECT
            toStartOfMonth(start_date) as month,
            SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as cash_inflow,
            SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as cash_outflow,
            SUM(amount) as net_cash_flow,
            COUNT(*) as transaction_count
        FROM ic_agent.amz_transaction
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY month
        ORDER BY month
        """
        return self.client.query(query)

    def profit_margin_trends(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze profit margin trends over time.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with profit margin trends
        """
        query = f"""
        SELECT
            m.month,
            m.revenue,
            COALESCE(f.fee_sum, 0) as total_fees,
            (m.revenue - COALESCE(f.fee_sum, 0)) as profit,
            ((m.revenue - COALESCE(f.fee_sum, 0)) / m.revenue * 100) as profit_margin_pct
        FROM (
            SELECT toStartOfMonth(start_date) as month, SUM(item_price) as revenue
            FROM ic_agent.amz_order
            WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY month
        ) m
        LEFT JOIN (
            SELECT toStartOfMonth(start_date) as month, SUM(estimated_fee_total) as fee_sum
            FROM ic_agent.amz_fee
            WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY month
        ) f ON m.month = f.month
        ORDER BY m.month
        """
        return self.client.query(query)

    def cost_breakdown(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Break down costs by category.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with cost breakdown
        """
        query = f"""
        SELECT
            'total' as cost_category,
            COUNT(*) as count,
            SUM(estimated_fee_total) as total_cost,
            AVG(estimated_fee_total) as avg_cost,
            (SUM(estimated_fee_total) * 100.0 / (SELECT SUM(estimated_fee_total) FROM ic_agent.amz_fee WHERE start_date BETWEEN '{start_date}' AND '{end_date}')) as cost_share_pct
        FROM ic_agent.amz_fee
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY cost_category
        ORDER BY total_cost DESC
        """
        return self.client.query(query)

    def revenue_by_payment_method(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze revenue by payment method.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with payment method breakdown
        """
        query = f"""
        SELECT
            payment_method,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY payment_method
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def refund_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze refund patterns and impact.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with refund analysis
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            COUNT(DISTINCT CASE WHEN t.transaction_type = 'Refund' THEN t.transaction_id END) as refund_count,
            (COUNT(DISTINCT CASE WHEN t.transaction_type = 'Refund' THEN t.transaction_id END) * 100.0 / COUNT(DISTINCT o.amazon_order_id)) as refund_rate_pct,
            SUM(CASE WHEN t.transaction_type = 'Refund' THEN ABS(t.amount) ELSE 0 END) as total_refund_amount
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_transaction t ON o.amazon_order_id = t.order_id
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY refund_rate_pct DESC
        """
        return self.client.query(query)

    def tax_summary(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Summarize tax obligations.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with tax summary
        """
        query = f"""
        SELECT
            toStartOfMonth(start_date) as month,
            SUM(item_price) as gross_revenue,
            SUM(tax_amount) as total_tax,
            (SUM(tax_amount) * 100.0 / SUM(item_price)) as tax_rate_pct,
            COUNT(DISTINCT amazon_order_id) as order_count
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY month
        ORDER BY month
        """
        return self.client.query(query)
