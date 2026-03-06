"""
Inventory analysis query templates.
"""

from typing import Optional
import pandas as pd
from ..uds_client import UDSClient


class InventoryQueries:
    """Inventory analysis query templates."""

    def __init__(self, client: UDSClient):
        self.client = client

    def current_inventory_levels(
        self,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get current inventory levels by SKU.

        Args:
            as_of_date: Date to check inventory (defaults to latest)

        Returns:
            DataFrame with: sku, fnsku, quantity, warehouse
        """
        date_filter = f"WHERE start_date = '{as_of_date}'" if as_of_date else ""

        query = f"""
        SELECT
            sku,
            fnsku,
            SUM(afn_fulfillable_quantity) as total_quantity
        FROM ic_agent.amz_fba_inventory_all
        {date_filter}
        GROUP BY sku, fnsku
        ORDER BY total_quantity DESC
        """
        return self.client.query(query)

    def low_stock_alert(
        self,
        threshold: int = 10,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Identify products with low stock levels.

        Args:
            threshold: Minimum quantity threshold
            as_of_date: Date to check

        Returns:
            DataFrame with low stock products
        """
        date_filter = f"AND start_date = '{as_of_date}'" if as_of_date else ""

        query = f"""
        SELECT
            i.sku,
            i.fnsku,
            p.Title as product_name,
            SUM(i.afn_fulfillable_quantity) as total_quantity,
            COUNT(*) as warehouse_count
        FROM ic_agent.amz_fba_inventory_all i
        LEFT JOIN ic_agent.amz_product p ON i.asin = p.ASIN
        WHERE 1=1 {date_filter}
        GROUP BY i.sku, i.fnsku, p.Title
        HAVING total_quantity < {threshold}
        ORDER BY total_quantity ASC
        """
        return self.client.query(query)

    def inventory_turnover(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate inventory turnover rate.

        Formula: Units Sold / Average Inventory
        """
        query = f"""
        SELECT
            o.sku,
            COUNT(DISTINCT o.amazon_order_id) as units_sold,
            AVG(i.afn_fulfillable_quantity) as avg_inventory,
            (COUNT(DISTINCT o.amazon_order_id) / AVG(i.afn_fulfillable_quantity)) as turnover_rate
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_fba_inventory_all i ON o.sku = i.sku
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        AND i.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku
        HAVING avg_inventory > 0
        ORDER BY turnover_rate DESC
        """
        return self.client.query(query)

    def inventory_aging(self, as_of_date: str) -> pd.DataFrame:
        """Analyze inventory age and identify slow-moving items."""
        query = f"""
        SELECT
            i.sku,
            i.fnsku,
            p.Title as title,
            SUM(i.afn_fulfillable_quantity) as quantity,
            MIN(i.start_date) as first_received_date,
            dateDiff('day', MIN(i.start_date), '{as_of_date}') as days_in_inventory,
            CASE
                WHEN dateDiff('day', MIN(i.start_date), '{as_of_date}') > 90 THEN 'Slow Moving'
                WHEN dateDiff('day', MIN(i.start_date), '{as_of_date}') > 60 THEN 'Moderate'
                ELSE 'Fast Moving'
            END as movement_category
        FROM ic_agent.amz_fba_inventory_all i
        LEFT JOIN ic_agent.amz_product p ON i.fnsku = p.fnsku
        WHERE i.start_date <= '{as_of_date}'
        GROUP BY i.sku, i.fnsku, p.Title
        ORDER BY days_in_inventory DESC
        """
        return self.client.query(query)

    def stockout_analysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze stockout events and their impact.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with stockout analysis
        """
        query = f"""
        SELECT
            i.sku,
            p.Title as product_name,
            COUNT(DISTINCT i.start_date) as stockout_days,
            SUM(i.afn_fulfillable_quantity) as total_stockout_quantity,
            COUNT(DISTINCT o.amazon_order_id) as lost_orders
        FROM ic_agent.amz_fba_inventory_all i
        LEFT JOIN ic_agent.amz_product p ON i.fnsku = p.fnsku
        LEFT JOIN ic_agent.amz_order o ON i.sku = o.sku AND o.start_date = i.start_date
        WHERE i.start_date BETWEEN '{start_date}' AND '{end_date}'
        AND i.afn_fulfillable_quantity = 0
        GROUP BY i.sku, p.Title
        ORDER BY lost_orders DESC
        """
        return self.client.query(query)

    def inventory_by_warehouse(
        self,
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Analyze inventory distribution across warehouses.

        Args:
            as_of_date: Date to analyze

        Returns:
            DataFrame with warehouse breakdown
        """
        # amz_fba_inventory_all has no fulfillment_center_id; aggregate by sku
        query = f"""
        SELECT
            'all' as warehouse,
            COUNT(DISTINCT sku) as unique_skus,
            SUM(afn_fulfillable_quantity) as total_quantity,
            AVG(afn_fulfillable_quantity) as avg_quantity_per_sku
        FROM ic_agent.amz_fba_inventory_all
        WHERE start_date = '{as_of_date}'
        GROUP BY warehouse
        ORDER BY total_quantity DESC
        """
        return self.client.query(query)

    def reorder_point_analysis(
        self,
        start_date: str,
        end_date: str,
        lead_time_days: int = 7
    ) -> pd.DataFrame:
        """
        Calculate reorder points based on historical demand.

        Args:
            start_date: Start date for demand analysis
            end_date: End date for demand analysis
            lead_time_days: Average lead time in days

        Returns:
            DataFrame with reorder point recommendations
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            (COUNT(DISTINCT o.amazon_order_id) * 1.0 / dateDiff('day', '{start_date}', '{end_date}')) as daily_demand,
            (COUNT(DISTINCT o.amazon_order_id) * 1.0 / dateDiff('day', '{start_date}', '{end_date}') * {lead_time_days}) as reorder_point,
            AVG(i.afn_fulfillable_quantity) as current_inventory,
            CASE
                WHEN AVG(i.afn_fulfillable_quantity) < (COUNT(DISTINCT o.amazon_order_id) * 1.0 / dateDiff('day', '{start_date}', '{end_date}') * {lead_time_days}) THEN 'Reorder Recommended'
                ELSE 'Sufficient Stock'
            END as status
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_fba_inventory_all i ON o.sku = i.sku AND i.start_date BETWEEN '{start_date}' AND '{end_date}'
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY daily_demand DESC
        """
        return self.client.query(query)

    def inventory_valuation(
        self,
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Calculate total inventory valuation.

        Args:
            as_of_date: Date to calculate valuation

        Returns:
            DataFrame with inventory valuation
        """
        query = f"""
        SELECT
            SUM(i.afn_fulfillable_quantity * COALESCE(o.unit_price, 0)) as total_value,
            COUNT(DISTINCT i.sku) as unique_skus,
            SUM(i.afn_fulfillable_quantity) as total_units,
            AVG(i.afn_fulfillable_quantity * COALESCE(o.unit_price, 0)) as avg_value_per_sku
        FROM ic_agent.amz_fba_inventory_all i
        LEFT JOIN (
            SELECT asin, AVG(item_price / NULLIF(quantity, 0)) as unit_price
            FROM ic_agent.amz_order
            WHERE start_date = '{as_of_date}' AND quantity > 0
            GROUP BY asin
        ) o ON i.asin = o.asin
        WHERE i.start_date = '{as_of_date}'
        """
        return self.client.query(query)

    def inventory_movement(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Track inventory movements (inflows and outflows).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with movement analysis
        """
        query = f"""
        SELECT
            i.sku,
            p.Title as product_name,
            SUM(i.afn_fulfillable_quantity) as ending_inventory,
            COUNT(DISTINCT o.amazon_order_id) as units_sold,
            (SUM(i.afn_fulfillable_quantity) - COUNT(DISTINCT o.amazon_order_id)) as net_change,
            ((COUNT(DISTINCT o.amazon_order_id) * 100.0) / NULLIF(SUM(i.afn_fulfillable_quantity), 0)) as sell_through_rate_pct
        FROM ic_agent.amz_fba_inventory_all i
        LEFT JOIN ic_agent.amz_product p ON i.fnsku = p.fnsku
        LEFT JOIN ic_agent.amz_order o ON i.sku = o.sku AND o.start_date BETWEEN '{start_date}' AND '{end_date}'
        WHERE i.start_date = '{end_date}'
        GROUP BY i.sku, p.Title
        ORDER BY units_sold DESC
        """
        return self.client.query(query)

    def safety_stock_analysis(
        self,
        start_date: str,
        end_date: str,
        service_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calculate safety stock requirements.

        Args:
            start_date: Start date for demand analysis
            end_date: End date for demand analysis
            service_level: Desired service level (0.0-1.0)

        Returns:
            DataFrame with safety stock recommendations
        """
        query = f"""
        SELECT
            o.sku,
            p.Title as product_name,
            COUNT(DISTINCT o.amazon_order_id) as total_orders,
            (COUNT(DISTINCT o.amazon_order_id) * 1.0 / dateDiff('day', '{start_date}', '{end_date}')) as avg_daily_demand,
            STDDEV(COUNT(DISTINCT o.amazon_order_id)) OVER () as demand_stddev,
            (1.65 * STDDEV(COUNT(DISTINCT o.amazon_order_id)) OVER ()) as safety_stock,
            AVG(i.afn_fulfillable_quantity) as current_inventory
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_fba_inventory_all i ON o.sku = i.sku AND i.start_date BETWEEN '{start_date}' AND '{end_date}'
        LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
        WHERE o.start_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY o.sku, p.Title
        ORDER BY avg_daily_demand DESC
        """
        return self.client.query(query)
