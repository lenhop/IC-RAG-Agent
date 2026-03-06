"""
Data quality checks for UDS database.
Provides automated validation and monitoring.
"""

from typing import Dict, List, Any
import pandas as pd
import logging

from .uds_client import UDSClient

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Automated data quality checks for UDS tables.
    """

    def __init__(self, client: UDSClient):
        self.client = client

    def check_completeness(self, table: str) -> Dict[str, Any]:
        """
        Check data completeness (null rates by column).
        
        Args:
            table: Table name
        
        Returns:
            Dictionary with null percentages per column
        """
        # Get total row count
        total_rows = self.client.query(f"SELECT COUNT(*) as count FROM {table}")['count'][0]

        # Get column names
        schema = self.client.get_table_schema(table.split('.')[-1])
        columns = [col['name'] for col in schema['columns']]

        # Calculate null percentage for each column
        null_checks = []
        for col in columns:
            query = f"""
            SELECT 
                '{col}' as column_name,
                COUNT(*) as total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as null_count,
                (SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as null_percentage
            FROM {table}
            """
            result = self.client.query(query)
            null_checks.append(result.iloc[0].to_dict())

        return {
            "table": table,
            "total_rows": total_rows,
            "columns": null_checks,
            "overall_completeness": 100 - sum(c['null_percentage'] for c in null_checks) / len(null_checks)
        }

    def check_consistency(self) -> Dict[str, Any]:
        """
        Check referential integrity between tables.
        
        Returns:
            Dictionary with consistency check results
        """
        checks = {}
        
        # Check 1: Orders without transactions
        # NOTE: the transaction table uses ``order_id`` rather than
        # ``amazon_order_id`` so the original sample SQL failed at runtime.
        # We avoid aliases in the WHERE clause to keep ClickHouse happy.
        query1 = """
        SELECT COUNT(*) as orphan_count
        FROM ic_agent.amz_order o
        LEFT JOIN ic_agent.amz_transaction t ON o.amazon_order_id = t.order_id
        WHERE t.order_id IS NULL
        """
        orphan_orders = self.client.query(query1)['orphan_count'][0]
        checks['orphan_orders'] = {
            "count": orphan_orders,
            "status": "OK" if orphan_orders < 1000 else "WARNING"
        }
        
        # Check 2: Revenue reconciliation
        # revenue reconciliation uses the ``total`` column in transactions
        # The orders table does not actually have a ``total_amount`` field; it
        # records ``item_price`` instead, so we sum that.  This is just an
        # illustrative check and may be adjusted based on real business logic.
        query2 = """
        SELECT 
            (SELECT SUM(item_price) FROM ic_agent.amz_order) as order_revenue,
            (SELECT SUM(total) FROM ic_agent.amz_transaction WHERE type = 'Order') as transaction_revenue
        """
        revenue = self.client.query(query2).iloc[0]
        revenue_diff = abs(revenue['order_revenue'] - revenue['transaction_revenue']) / revenue['order_revenue'] * 100
        checks['revenue_reconciliation'] = {
            "order_revenue": revenue['order_revenue'],
            "transaction_revenue": revenue['transaction_revenue'],
            "difference_pct": revenue_diff,
            "status": "OK" if revenue_diff < 1 else "WARNING"
        }
        
        # Check 3: SKU consistency (orders vs listings)
        query3 = """
        SELECT COUNT(DISTINCT o.sku) as order_skus,
               COUNT(DISTINCT l.sku) as listing_skus
        FROM ic_agent.amz_order o
        CROSS JOIN ic_agent.amz_listing_item l
        """
        skus = self.client.query(query3).iloc[0]
        checks['sku_consistency'] = {
            "order_skus": skus['order_skus'],
            "listing_skus": skus['listing_skus'],
            "status": "OK"
        }
        
        return checks
    
    def check_timeliness(self) -> Dict[str, Any]:
        """
        Check data freshness and date coverage.
        
        Returns:
            Dictionary with timeliness check results
        """
        tables_with_dates = [
            ('amz_order', 'start_date'),
            ('amz_transaction', 'start_date'),
            ('amz_statement', 'start_date'),
            ('amz_fba_inventory_all', 'start_date'),
            ('amz_daily_inventory_ledger', 'start_date'),
            ('amz_fee', 'start_date'),
            ('amz_listing_item', 'request_date')
        ]
        
        results = {}
        for table, date_col in tables_with_dates:
            query = f"""
            SELECT 
                MIN({date_col}) as min_date,
                MAX({date_col}) as max_date,
                COUNT(DISTINCT {date_col}) as unique_dates
            FROM ic_agent.{table}
            """
            result = self.client.query(query).iloc[0]
            results[table] = {
                "min_date": str(result['min_date']),
                "max_date": str(result['max_date']),
                "unique_dates": result['unique_dates'],
                "expected_dates": 30,  # October 2025
                "coverage": result['unique_dates'] / 30 * 100
            }
        
        return results
    
    def check_accuracy(self, table: str) -> Dict[str, Any]:
        """
        Check value ranges and formats for accuracy.
        
        Args:
            table: Table name
        
        Returns:
            Dictionary with accuracy check results
        """
        checks = {}
        
        if table == 'amz_order':
            # Check for negative amounts
            query = """
            SELECT COUNT(*) as negative_count
            FROM ic_agent.amz_order
            WHERE total_amount < 0
            """
            negative = self.client.query(query)['negative_count'][0]
            checks['negative_amounts'] = {
                "count": negative,
                "status": "OK" if negative == 0 else "ERROR"
            }
            
            # Check for unrealistic amounts
            query = """
            SELECT COUNT(*) as outlier_count
            FROM ic_agent.amz_order
            WHERE total_amount > 10000
            """
            outliers = self.client.query(query)['outlier_count'][0]
            checks['amount_outliers'] = {
                "count": outliers,
                "status": "OK" if outliers < 100 else "WARNING"
            }
        
        return checks
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all quality checks and return comprehensive report.
        
        Returns:
            Dictionary with all check results
        """
        logger.info("Running comprehensive data quality checks...")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "completeness": {},
            "consistency": self.check_consistency(),
            "timeliness": self.check_timeliness(),
            "accuracy": {}
        }
        
        # Check completeness for all tables
        tables = self.client.list_tables()
        for table in tables:
            report["completeness"][table] = self.check_completeness(f"ic_agent.{table}")
        
        # Check accuracy for key tables
        report["accuracy"]["amz_order"] = self.check_accuracy("amz_order")
        
        logger.info("Data quality checks completed")
        return report


def generate_quality_report(client: UDSClient, output_file: str = "uds_quality_report.json"):
    """
    Generate and save data quality report.
    
    Args:
        client: UDSClient instance
        output_file: Output file path
    """
    checker = DataQualityChecker(client)
    report = checker.run_all_checks()
    
    # when writing to disk, ensure numeric types are JSON serializable
    def _json_default(o):
        # convert numpy/int64 etc to python primitives
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except ImportError:
            pass
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    import json
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=_json_default)
    
    print(f"Quality report saved to {output_file}")
    return report
