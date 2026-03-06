"""
Product performance query templates.
"""

import pandas as pd
from ..uds_client import UDSClient


class ProductQueries:
    """Product performance query templates."""

    def __init__(self, client: UDSClient):
        self.client = client

    def top_selling_products(
        self,
        start_date: str,
        end_date: str,
        limit: int = 10,
        metric: str = 'revenue'
    ) -> pd.DataFrame:
        """
        Get top selling products.

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of products
            metric: Sort by 'revenue' or 'units'

        Returns:
            DataFrame with top products
        """
        order_by = 'total_revenue' if metric == 'revenue' else 'units_sold'

        query = f"""
        SELECT
            o.sku,
            o.asin,
            p.Title as product_name,
            p.Brand,
            COUNT(DISTINCT o.amazon_order_id) as units_sold,
            SUM(o.item_price) as total_revenue,
            AVG(o.item_price) as avg_price
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, o.asin, p.Title, p.Brand
        ORDER BY {order_by} DESC
        LIMIT {limit}
        """
        return self.client.query(query)

    def product_search_performance(
        self,
        month: str
    ) -> pd.DataFrame:
        """
        Analyze product search performance.

        Args:
            month: Month in YYYYMM format (e.g., '202510')

        Returns:
            DataFrame with search metrics
        """
        query = f"""
        SELECT
            search_term,
            search_frequency_rank,
            click_count,
            click_share,
            conversion_rate
        FROM ic_agent.amz_monthly_search_term
        WHERE month = '{month}'
        ORDER BY search_frequency_rank ASC
        LIMIT 100
        """
        return self.client.query(query)

    def product_listing_quality(self) -> pd.DataFrame:
        """Analyze product listing completeness."""
        query = """
        SELECT
            COUNT(*) as total_products,
            SUM(CASE WHEN title IS NOT NULL AND title != '' THEN 1 ELSE 0 END) as has_title,
            SUM(CASE WHEN brand IS NOT NULL AND brand != '' THEN 1 ELSE 0 END) as has_brand,
            SUM(CASE WHEN description IS NOT NULL AND description != '' THEN 1 ELSE 0 END) as has_description,
            SUM(CASE WHEN image_url IS NOT NULL AND image_url != '' THEN 1 ELSE 0 END) as has_image,
            (SUM(CASE WHEN title IS NOT NULL AND brand IS NOT NULL AND description IS NOT NULL AND image_url IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as completeness_pct
        FROM ic_agent.amz_product
        """
        return self.client.query(query)

    def product_lifecycle_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze product lifecycle stages.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with lifecycle analysis
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            MIN(o.start_date) as first_sale_date,
            MAX(o.start_date) as last_sale_date,
            dateDiff('day', MIN(o.start_date), MAX(o.start_date)) as days_active,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            SUM(o.item_price) as total_revenue,
            CASE
                WHEN dateDiff('day', MIN(o.start_date), '{end_date}') < 30 THEN 'New'
                WHEN dateDiff('day', MAX(o.start_date), '{end_date}') > 30 THEN 'Declining'
                ELSE 'Mature'
            END as lifecycle_stage
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def product_performance_comparison(
        self,
        start_date: str,
        end_date: str,
        skus: list
    ) -> pd.DataFrame:
        """
        Compare performance of specific products.

        Args:
            start_date: Start date
            end_date: End date
            skus: List of SKUs to compare

        Returns:
            DataFrame with product comparison
        """
        sku_list = "', '".join(skus)
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as orders,
            SUM(o.item_price) as revenue,
            AVG(o.item_price) as avg_price,
            COUNT(DISTINCT o.buyer_email) as unique_customers
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        AND o.sku IN ('{sku_list}')
        GROUP BY o.sku, p.Title
        ORDER BY revenue DESC
        """
        return self.client.query(query)

    def product_category_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze performance by product category.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with category analysis
        """
        query = f"""
        SELECT
            p.product_type as category,
            COUNT(DISTINCT o.sku) as unique_products,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            SUM(o.item_price) as total_revenue,
            AVG(o.item_price) as avg_order_value,
            (SUM(o.item_price) / COUNT(DISTINCT o.sku)) as revenue_per_product
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY p.product_type
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def product_price_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze product pricing and its impact on sales.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price analysis
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            MIN(o.item_price) as min_price,
            MAX(o.item_price) as max_price,
            AVG(o.item_price) as avg_price,
            MEDIAN(o.item_price) as median_price,
            COUNT(DISTINCT o.amazon_order_id) as order_count,
            SUM(o.item_price) as total_revenue,
            (MAX(o.item_price) - MIN(o.item_price)) as price_range
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY total_revenue DESC
        """
        return self.client.query(query)

    def product_review_analysis(
        self,
        start_date: str,
        end_date: str,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Analyze product reviews and ratings.

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of products to return

        Returns:
            DataFrame with review analysis
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            COUNT(DISTINCT CASE WHEN o.is_prime = 1 THEN o.amazon_order_id END) as prime_orders,
            (COUNT(DISTINCT CASE WHEN o.is_prime = 1 THEN o.amazon_order_id END) * 100.0 / COUNT(DISTINCT o.amazon_order_id)) as prime_rate_pct
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY total_orders DESC
        LIMIT {limit}
        """
        return self.client.query(query)

    def product_return_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze product return rates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with return analysis
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
        HAVING total_orders > 0
        ORDER BY refund_rate_pct DESC
        """
        return self.client.query(query)

    def product_cross_sell_analysis(
        self,
        start_date: str,
        end_date: str,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Analyze products frequently bought together.

        Args:
            start_date: Start date
            end_date: End date
            limit: Number of product pairs to return

        Returns:
            DataFrame with cross-sell analysis
        """
        query = f"""
        SELECT
            o1.sku as sku1,
            p1.title as product1_name,
            o2.sku as sku2,
            p2.title as product2_name,
            COUNT(DISTINCT o1.amazon_order_id) as co_occurrence_count,
            (COUNT(DISTINCT o1.amazon_order_id) * 100.0 / COUNT(DISTINCT o1.amazon_order_id)) as co_occurrence_rate
        FROM ic_agent.amz_order o1
        JOIN ic_agent.amz_order o2 ON o1.amazon_order_id = o2.amazon_order_id AND o1.sku < o2.sku
        LEFT JOIN ic_agent.amz_product p1 ON o1.asin = p1.ASIN
        LEFT JOIN ic_agent.amz_product p2 ON o2.asin = p2.ASIN
        WHERE o1.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o1.sku, p1.title, o2.sku, p2.title
        ORDER BY co_occurrence_count DESC
        LIMIT {limit}
        """
        return self.client.query(query)
