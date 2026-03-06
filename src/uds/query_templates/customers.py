"""
Customer analytics query templates.
"""

import pandas as pd
from ..uds_client import UDSClient


class CustomerQueries:
    """Customer analytics query templates."""

    def __init__(self, client: UDSClient):
        self.client = client

    def customer_segmentation(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Segment customers based on purchase behavior.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with customer segments
        """
        query = f"""
        SELECT
            buyer_email,
            COUNT(DISTINCT amazon_order_id) as order_count,
            SUM(total_amount) as total_spent,
            AVG(total_amount) as avg_order_value,
            MIN(start_date) as first_purchase_date,
            MAX(start_date) as last_purchase_date,
            dateDiff('day', MIN(start_date), MAX(start_date)) as days_active,
            CASE
                WHEN COUNT(DISTINCT amazon_order_id) >= 10 THEN 'VIP'
                WHEN COUNT(DISTINCT amazon_order_id) >= 5 THEN 'Loyal'
                WHEN COUNT(DISTINCT amazon_order_id) >= 2 THEN 'Regular'
                ELSE 'One-time'
            END as segment
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY buyer_email
        ORDER BY total_spent DESC
        """
        return self.client.query(query)

    def customer_lifetime_value(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Calculate customer lifetime value (CLV).

        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of top customers to return

        Returns:
            DataFrame with CLV metrics
        """
        query = f"""
        SELECT
            buyer_email,
            COUNT(DISTINCT amazon_order_id) as total_orders,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_order_value,
            dateDiff('day', MIN(start_date), MAX(start_date)) as days_active,
            (SUM(total_amount) / NULLIF(dateDiff('day', MIN(start_date), MAX(start_date)), 0)) as daily_value,
            (SUM(total_amount) / NULLIF(dateDiff('day', MIN(start_date), MAX(start_date)), 0) * 365) as annual_clv
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY buyer_email
        HAVING total_orders > 0
        ORDER BY total_revenue DESC
        LIMIT {top_n}
        """
        return self.client.query(query)

    def customer_purchase_frequency(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze customer purchase frequency patterns.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with frequency analysis
        """
        query = f"""
        SELECT
            CASE
                WHEN COUNT(DISTINCT amazon_order_id) = 1 THEN 'One-time'
                WHEN COUNT(DISTINCT amazon_order_id) BETWEEN 2 AND 3 THEN 'Occasional'
                WHEN COUNT(DISTINCT amazon_order_id) BETWEEN 4 AND 9 THEN 'Frequent'
                ELSE 'Very Frequent'
            END as frequency_category,
            COUNT(DISTINCT buyer_email) as customer_count,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_revenue_per_customer,
            (COUNT(DISTINCT buyer_email) * 100.0 / (SELECT COUNT(DISTINCT buyer_email) FROM ic_agent.amz_order WHERE start_date BETWEEN '{start_date}' AND '{end_date}')) as customer_share_pct
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY frequency_category
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def customer_geographic_distribution(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze customer distribution by geographic region.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with geographic distribution
        """
        query = f"""
        SELECT
            marketplace,
            COUNT(DISTINCT buyer_email) as unique_customers,
            COUNT(DISTINCT amazon_order_id) as total_orders,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_order_value,
            (SUM(total_amount) * 100.0 / (SELECT SUM(total_amount) FROM ic_agent.amz_order WHERE start_date BETWEEN '{start_date}' AND '{end_date}')) as revenue_share_pct
        FROM ic_agent.amz_order
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY marketplace
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def customer_retention_analysis(
        self,
        cohort_start: str,
        cohort_end: str,
        analysis_start: str,
        analysis_end: str
    ) -> pd.DataFrame:
        """
        Analyze customer retention rates over time.

        Args:
            cohort_start: Cohort start date
            cohort_end: Cohort end date
            analysis_start: Analysis period start
            analysis_end: Analysis period end

        Returns:
            DataFrame with retention analysis
        """
        query = f"""
        WITH cohort_customers AS (
            SELECT DISTINCT buyer_email
            FROM ic_agent.amz_order
            WHERE start_date BETWEEN '{cohort_start}' AND '{cohort_end}'
        ),
        retained_customers AS (
            SELECT DISTINCT o.buyer_email
            FROM ic_agent.amz_order o
            JOIN cohort_customers c ON o.buyer_email = c.buyer_email
            WHERE o.start_date BETWEEN '{analysis_start}' AND '{analysis_end}'
        )
        SELECT
            (SELECT COUNT(*) FROM cohort_customers) as cohort_size,
            (SELECT COUNT(*) FROM retained_customers) as retained_count,
            ((SELECT COUNT(*) FROM retained_customers) * 100.0 / (SELECT COUNT(*) FROM cohort_customers)) as retention_rate_pct,
            dateDiff('day', '{cohort_end}', '{analysis_start}') as days_since_cohort
        """
        return self.client.query(query)
