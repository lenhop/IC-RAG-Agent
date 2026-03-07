#!/usr/bin/env python3
"""
Query Optimization Script

Analyzes query execution plans, measures performance, and identifies
opportunities for improvement. Intended to be run against a ClickHouse
instance with realistic data.

Usage:
    python tools/optimize_queries.py
"""

import time

from clickhouse_connect import get_client

QUERIES = {
    "sales_trend": (
        "SELECT start_date, COUNT(*) FROM ic_agent.amz_order "
        "WHERE start_date BETWEEN '2025-10-01' AND '2025-10-31' "
        "GROUP BY start_date ORDER BY start_date"
    ),
    "product_top_revenue": (
        "SELECT sku, SUM(item_price) AS revenue FROM ic_agent.amz_order "
        "GROUP BY sku ORDER BY revenue DESC LIMIT 10"
    ),
    "inventory_levels": (
        "SELECT sku, SUM(afn_fulfillable_quantity) AS qty FROM ic_agent.amz_fba_inventory_all "
        "GROUP BY sku ORDER BY qty DESC LIMIT 20"
    ),
}


def profile_query(client, sql: str) -> dict:
    """Run EXPLAIN and measure execution time for a query."""
    info = {}
    try:
        start = time.time()
        plan = client.command(f"EXPLAIN {sql}")
        elapsed = time.time() - start
        info["plan"] = plan
        info["exec_time"] = elapsed

        start2 = time.time()
        client.command(sql)
        info["run_time"] = time.time() - start2
    except Exception as exc:
        info["error"] = str(exc)
    return info


def main():
    """Run optimization profile for predefined SQL statements."""
    client = get_client(host="localhost", port=8123, username="default", password="")
    results = {}
    for name, sql in QUERIES.items():
        print(f"\nProfiling {name}...")
        info = profile_query(client, sql)
        results[name] = info
        print(f"Execution plan:\n{info.get('plan')}\n")
        print(f"Explain time: {info.get('exec_time'):.3f}s, run time: {info.get('run_time', 0):.3f}s")
        if "error" in info:
            print(f"Error: {info['error']}")
    return results


if __name__ == "__main__":
    main()
