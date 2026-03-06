"""
Sales analysis query templates.
Provides common sales analytics queries.
"""

from typing import Optional
import pandas as pd
from ..uds_client import UDSClient


class SalesQueries:
    """Sales analysis query templates."""

    def __init__(self, client: UDSClient):
        self.client = client

    def daily_sales_trend(
        self,
        start_date: str,
        end_date: str,
        marketplace: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get daily sales trend with order count and revenue.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            marketplace: Optional marketplace filter (US, UK, DE, FR, JP)

        Returns:
            DataFrame with columns: date, order_count, total_revenue, avg_order_value

        Example:
            >>> sales = SalesQueries(client)
            >>> df = sales.daily_sales_trend('2025-10-01', '2025-10-31')
        """
        marketplace_filter = f"AND marketplace = '{marketplace}'" if marketplace else ""

        query = f"""
        SELECT
            start_date as date,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        {marketplace_filter}
        GROUP BY start_date
        ORDER BY start_date
        """
        return self.client.query(query)

    def top_products_by_revenue(
        self,
        start_date: str,
        end_date: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get top products by revenue.

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of products to return

        Returns:
            DataFrame with: sku, product_name, order_count, total_revenue
        """
        query = f"""
        SELECT
            o.sku,
            p.title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as order_count,
            SUM(o.item_price) as total_revenue,
            AVG(o.item_price) as avg_order_value
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.title
        ORDER BY total_revenue DESC
        LIMIT {limit}
        """
        return self.client.query(query)

    def sales_by_marketplace(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get sales breakdown by marketplace."""
        query = f"""
        SELECT
            marketplace,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value,
            (SUM(item_price) * 100.0 / (SELECT SUM(item_price) FROM ic_agent.amz_order WHERE start_date BETWEEN '{start_date}' AND '{end_date}')) as revenue_share_pct
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY marketplace
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def hourly_sales_pattern(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Analyze sales by hour of day."""
        query = f"""
        SELECT
            toHour(purchase_date) as hour,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY hour
        ORDER BY hour
        """
        return self.client.query(query)

    def sales_growth_rate(
        self,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str
    ) -> pd.DataFrame:
        """
        Compare sales between two periods.

        Returns:
            DataFrame with period comparison and growth rate
        """
        query = f"""
        SELECT
            'Period 1' as period,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{period1_start}' AND '{period1_end}'
        UNION ALL
        SELECT
            'Period 2' as period,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{period2_start}' AND '{period2_end}'
        """
        df = self.client.query(query)

        # Calculate growth rate
        if len(df) == 2:
            growth_rate = ((df.iloc[1]['total_revenue'] - df.iloc[0]['total_revenue']) /
                          df.iloc[0]['total_revenue'] * 100)
            df['growth_rate_pct'] = [0, growth_rate]

        return df

    def monthly_sales_comparison(
        self,
        year: int,
        months: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Compare sales across months.

        Args:
            year: Year to analyze
            months: List of months (1-12), defaults to all months

        Returns:
            DataFrame with monthly comparison
        """
        month_filter = f"AND toMonth(start_date) IN ({','.join(map(str, months))})" if months else ""

        query = f"""
        SELECT
            toMonth(start_date) as month,
            toYear(start_date) as year,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE toYear(start_date) = {year}
        {month_filter}
        GROUP BY month, year
        ORDER BY month
        """
        return self.client.query(query)

    def sales_by_category(
        self,
        start_date: str,
        end_date: str,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Analyze sales by product category.

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of categories to return

        Returns:
            DataFrame with category breakdown
        """
        query = f"""
        SELECT
            p.product_type as category,
            COUNT(DISTINCT o.amazon_order_id) as order_count,
            SUM(o.item_price) as total_revenue,
            AVG(o.item_price) as avg_order_value
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY p.product_type
        ORDER BY total_revenue DESC
        LIMIT {limit}
        """
        return self.client.query(query)

    def sales_by_region(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze sales by geographic region.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with regional breakdown
        """
        query = f"""
        SELECT
            marketplace,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY marketplace
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def repeat_purchase_rate(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate repeat purchase rate.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with repeat purchase metrics
        """
        query = f"""
        SELECT
            COUNT(DISTINCT CASE WHEN order_count > 1 THEN buyer_email END) as repeat_customers,
            COUNT(DISTINCT buyer_email) as total_customers,
            (COUNT(DISTINCT CASE WHEN order_count > 1 THEN buyer_email END) * 100.0 / COUNT(DISTINCT buyer_email)) as repeat_rate_pct
        FROM (
            SELECT buyer_email, COUNT(DISTINCT amazon_order_id) as order_count
            FROM ic_agent.amz_order
            WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY buyer_email
        )
        """
        return self.client.query(query)

    def sales_velocity(
        self,
        start_date: str,
        end_date: str,
        product_limit: int = 20
    ) -> pd.DataFrame:
        """
        Calculate sales velocity (sales per day) for products.

        Args:
            start_date: Start date
            end_date: End date
            product_limit: Number of products to return

        Returns:
            DataFrame with sales velocity metrics
        """
        query = f"""
        SELECT
            o.sku,
            p.title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            SUM(o.item_price) as total_revenue,
            COUNT(DISTINCT o.start_date) as days_with_sales,
            (COUNT(DISTINCT o.amazon_order_id) * 1.0 / COUNT(DISTINCT o.start_date)) as orders_per_day,
            (SUM(o.item_price) * 1.0 / COUNT(DISTINCT o.start_date)) as revenue_per_day
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.title
        ORDER BY revenue_per_day DESC
        LIMIT {product_limit}
        """
        return self.client.query(query)

    def sales_forecast_accuracy(
        self,
        forecast_start: str,
        forecast_end: str,
        actual_start: str,
        actual_end: str
    ) -> pd.DataFrame:
        """
        Compare forecasted vs actual sales.

        Args:
            forecast_start: Forecast period start
            forecast_end: Forecast period end
            actual_start: Actual period start
            actual_end: Actual period end

        Returns:
            DataFrame with forecast accuracy metrics
        """
        forecast_query = f"""
        SELECT
            toStartOfWeek(start_date) as week,
            SUM(item_price) as forecast_revenue
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{forecast_start}' AND '{forecast_end}'
        GROUP BY week
        ORDER BY week
        """

        actual_query = f"""
        SELECT
            toStartOfWeek(start_date) as week,
            SUM(item_price) as actual_revenue
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{actual_start}' AND '{actual_end}'
        GROUP BY week
        ORDER BY week
        """

        forecast_df = self.client.query(forecast_query)
        actual_df = self.client.query(actual_query)

        # Merge and calculate accuracy
        merged = forecast_df.merge(actual_df, on='week', how='inner', suffixes=('_forecast', '_actual'))
        if not merged.empty:
            merged['accuracy_pct'] = (1 - abs(merged['forecast_revenue'] - merged['actual_revenue']) / merged['actual_revenue']) * 100

        return merged

    def discount_impact_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze impact of discounts on sales.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with discount impact metrics
        """
        query = f"""
        SELECT
            CASE
                WHEN item_promotion_discount > 0 THEN 'With Discount'
                ELSE 'No Discount'
            END as discount_status,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value,
            AVG(item_promotion_discount) as avg_discount_amount
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY discount_status
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def sales_channel_performance(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze sales by channel (FBA vs FBM).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with channel performance metrics
        """
        query = f"""
        SELECT
            fulfillment_channel,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY fulfillment_channel
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def sales_peak_hours(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Identify peak sales hours.

        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of top hours to return

        Returns:
            DataFrame with peak hours
        """
        query = f"""
        SELECT
            toHour(purchase_date) as hour,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY hour
        ORDER BY total_revenue DESC
        LIMIT {top_n}
        """
        return self.client.query(query)

    def sales_seasonality(
        self,
        year: int
    ) -> pd.DataFrame:
        """
        Analyze sales seasonality throughout the year.

        Args:
            year: Year to analyze

        Returns:
            DataFrame with seasonal patterns
        """
        query = f"""
        SELECT
            toMonth(start_date) as month,
            toQuarter(start_date) as quarter,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(item_price) as total_revenue,
            AVG(item_price) as avg_order_value,
            (SUM(item_price) * 100.0 / (SELECT SUM(item_price) FROM ic_agent.amz_order WHERE toYear(start_date) = {year})) as revenue_share_pct
        FROM ic_agent.amz_order
        WHERE toYear(start_date) = {year}
        GROUP BY month, quarter
        ORDER BY month
        """
        return self.client.query(query)
