#!/usr/bin/env python3
"""
Generate comprehensive data quality report for UDS database.

Usage:
    python tools/generate_quality_report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uds.config import UDSConfig
from src.uds.maintenance.quality_checks import generate_quality_report
from src.uds.maintenance.statistics import StatisticalAnalyzer
from src.uds.uds_client import UDSClient


def main() -> None:
    """Run data quality checks and export summary files."""
    print("=" * 80)
    print("UDS Data Quality Report Generator")
    print("=" * 80)

    print("\n1. Connecting to ClickHouse...")
    client = UDSClient(
        host=UDSConfig.CH_HOST,
        port=UDSConfig.CH_PORT,
        user=UDSConfig.CH_USER,
        password=UDSConfig.CH_PASSWORD,
        database=UDSConfig.CH_DATABASE,
    )

    if not client.ping():
        print("Failed to connect to ClickHouse")
        return

    print("Connected successfully")

    print("\n2. Running data quality checks...")
    generate_quality_report(client, "uds_quality_report.json")
    print("Quality checks completed")

    print("\n3. Generating statistical summaries...")
    analyzer = StatisticalAnalyzer(client)
    tables = client.list_tables()
    stats = {}

    for table in tables:
        print(f"   Analyzing {table}...")
        stats[table] = analyzer.get_table_statistics(table)

    stats_path = UDSConfig.project_path(UDSConfig.STATISTICS_PATH)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, indent=2)

    print("Statistics generated")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Tables: {len(tables)}")
    print(f"Total Rows: {sum(item['row_count'] for item in stats.values()):,}")
    print("\nReports generated:")
    print("  - uds_quality_report.json")
    print(f"  - {stats_path}")
    print("\nAll done")


if __name__ == "__main__":
    main()
