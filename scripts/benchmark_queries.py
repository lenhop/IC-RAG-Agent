#!/usr/bin/env python3
"""
Benchmark query template performance.
"""

import sys
import os
import time
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath('.'))

from src.uds.uds_client import UDSClient
from src.uds.query_templates import QueryTemplateRegistry


class QueryBenchmark:
    """Benchmark query template performance."""

    def __init__(self, client: UDSClient):
        self.client = client
        self.registry = QueryTemplateRegistry(client)
        self.results: List[Dict[str, Any]] = []

    def benchmark_template(
        self,
        category: str,
        template_name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a single query template.

        Args:
            category: Template category
            template_name: Template function name
            *args: Positional arguments for the template
            **kwargs: Keyword arguments for the template

        Returns:
            Dictionary with benchmark results
        """
        try:
            category_obj = getattr(self.registry, category)
            template_func = getattr(category_obj, template_name)

            # Warm-up run
            try:
                template_func(*args, **kwargs)
            except Exception:
                pass

            # Benchmark run
            start_time = time.time()
            result = template_func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            row_count = len(result) if hasattr(result, '__len__') else 0

            benchmark_result = {
                "category": category,
                "template": template_name,
                "execution_time": elapsed_time,
                "row_count": row_count,
                "success": True,
                "error": None
            }

            self.results.append(benchmark_result)
            return benchmark_result

        except Exception as e:
            benchmark_result = {
                "category": category,
                "template": template_name,
                "execution_time": 0.0,
                "row_count": 0,
                "success": False,
                "error": str(e)
            }

            self.results.append(benchmark_result)
            return benchmark_result

    def benchmark_all(
        self,
        start_date: str = "2025-10-01",
        end_date: str = "2025-10-31"
    ) -> List[Dict[str, Any]]:
        """
        Benchmark all query templates.

        Args:
            start_date: Start date for queries
            end_date: End date for queries

        Returns:
            List of benchmark results
        """
        print(f"Benchmarking query templates from {start_date} to {end_date}...")
        print("=" * 80)

        # Sales queries
        print("\n📊 Sales Analysis Queries")
        print("-" * 80)
        self.benchmark_template("sales", "daily_sales_trend", start_date, end_date)
        self.benchmark_template("sales", "top_products_by_revenue", start_date, end_date)
        self.benchmark_template("sales", "sales_by_marketplace", start_date, end_date)
        self.benchmark_template("sales", "hourly_sales_pattern", start_date, end_date)
        self.benchmark_template("sales", "monthly_sales_comparison", 2025)
        self.benchmark_template("sales", "sales_by_category", start_date, end_date)
        self.benchmark_template("sales", "sales_by_region", start_date, end_date)
        self.benchmark_template("sales", "repeat_purchase_rate", start_date, end_date)
        self.benchmark_template("sales", "sales_velocity", start_date, end_date)
        self.benchmark_template("sales", "discount_impact_analysis", start_date, end_date)
        self.benchmark_template("sales", "sales_channel_performance", start_date, end_date)
        self.benchmark_template("sales", "sales_peak_hours", start_date, end_date)
        self.benchmark_template("sales", "sales_seasonality", 2025)

        # Inventory queries
        print("\n📦 Inventory Analysis Queries")
        print("-" * 80)
        self.benchmark_template("inventory", "current_inventory_levels")
        self.benchmark_template("inventory", "low_stock_alert", threshold=10)
        self.benchmark_template("inventory", "inventory_turnover", start_date, end_date)
        self.benchmark_template("inventory", "inventory_aging", end_date)
        self.benchmark_template("inventory", "stockout_analysis", start_date, end_date)
        self.benchmark_template("inventory", "inventory_by_warehouse", end_date)
        self.benchmark_template("inventory", "reorder_point_analysis", start_date, end_date)
        self.benchmark_template("inventory", "inventory_valuation", end_date)
        self.benchmark_template("inventory", "inventory_movement", start_date, end_date)
        self.benchmark_template("inventory", "safety_stock_analysis", start_date, end_date)

        # Financial queries
        print("\n💰 Financial Analysis Queries")
        print("-" * 80)
        self.benchmark_template("financial", "revenue_summary", start_date, end_date, "day")
        self.benchmark_template("financial", "fee_analysis", start_date, end_date)
        self.benchmark_template("financial", "profitability_by_product", start_date, end_date)
        self.benchmark_template("financial", "transaction_summary", start_date, end_date)
        self.benchmark_template("financial", "cash_flow_analysis", start_date, end_date)
        self.benchmark_template("financial", "profit_margin_trends", start_date, end_date)
        self.benchmark_template("financial", "cost_breakdown", start_date, end_date)
        self.benchmark_template("financial", "revenue_by_payment_method", start_date, end_date)
        self.benchmark_template("financial", "refund_analysis", start_date, end_date)
        self.benchmark_template("financial", "tax_summary", start_date, end_date)

        # Product queries
        print("\n🏷️  Product Performance Queries")
        print("-" * 80)
        self.benchmark_template("products", "top_selling_products", start_date, end_date)
        self.benchmark_template("products", "product_search_performance", "202510")
        self.benchmark_template("products", "product_listing_quality")
        self.benchmark_template("products", "product_lifecycle_analysis", start_date, end_date)
        self.benchmark_template("products", "product_category_analysis", start_date, end_date)
        self.benchmark_template("products", "product_price_analysis", start_date, end_date)
        self.benchmark_template("products", "product_review_analysis", start_date, end_date)
        self.benchmark_template("products", "product_return_analysis", start_date, end_date)
        self.benchmark_template("products", "product_cross_sell_analysis", start_date, end_date)

        # Customer queries
        print("\n👥 Customer Analytics Queries")
        print("-" * 80)
        self.benchmark_template("customers", "customer_segmentation", start_date, end_date)
        self.benchmark_template("customers", "customer_lifetime_value", start_date, end_date)
        self.benchmark_template("customers", "customer_purchase_frequency", start_date, end_date)
        self.benchmark_template("customers", "customer_geographic_distribution", start_date, end_date)
        self.benchmark_template("customers", "customer_retention_analysis",
                              start_date, end_date, start_date, end_date)

        print("\n" + "=" * 80)
        print("Benchmark completed!")
        print("=" * 80)

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n📈 Benchmark Summary")
        print("=" * 80)

        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        print(f"\nTotal Templates: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            avg_time = sum(r["execution_time"] for r in successful) / len(successful)
            total_rows = sum(r["row_count"] for r in successful)
            print(f"\nAverage Execution Time: {avg_time:.3f}s")
            print(f"Total Rows Returned: {total_rows}")

            print("\n⚡ Top 10 Fastest Queries:")
            fastest = sorted(successful, key=lambda x: x["execution_time"])[:10]
            for i, result in enumerate(fastest, 1):
                print(f"  {i}. {result['category']}.{result['template']}: {result['execution_time']:.3f}s")

            print("\n🐌 Top 10 Slowest Queries:")
            slowest = sorted(successful, key=lambda x: x["execution_time"], reverse=True)[:10]
            for i, result in enumerate(slowest, 1):
                print(f"  {i}. {result['category']}.{result['template']}: {result['execution_time']:.3f}s")

        if failed:
            print("\n❌ Failed Queries:")
            for result in failed:
                print(f"  - {result['category']}.{result['template']}: {result['error']}")

    def export_results(self, filename: str = "benchmark_results.csv"):
        """
        Export benchmark results to CSV.

        Args:
            filename: Output filename
        """
        import pandas as pd

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark UDS query templates")
    parser.add_argument("--start-date", default="2025-10-01", help="Start date for queries")
    parser.add_argument("--end-date", default="2025-10-31", help="End date for queries")
    parser.add_argument("--export", help="Export results to CSV file")
    args = parser.parse_args()

    # Initialize client
    print("Connecting to ClickHouse...")
    client = UDSClient()

    # Run benchmarks
    benchmark = QueryBenchmark(client)
    benchmark.benchmark_all(args.start_date, args.end_date)

    # Print summary
    benchmark.print_summary()

    # Export results if requested
    if args.export:
        benchmark.export_results(args.export)


if __name__ == "__main__":
    main()
