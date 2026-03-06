#!/usr/bin/env python3
"""
Generate comprehensive data quality report for UDS database.

Usage:
    python scripts/generate_quality_report.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.uds.uds_client import UDSClient
from src.uds.quality_checks import DataQualityChecker, generate_quality_report
from src.uds.statistics import StatisticalAnalyzer
from src.uds.config import UDSConfig
import json


def main():
    print("=" * 80)
    print("UDS Data Quality Report Generator")
    print("=" * 80)
    
    # Initialize client
    print("\n1. Connecting to ClickHouse...")
    client = UDSClient(
        host=UDSConfig.CH_HOST,
        port=UDSConfig.CH_PORT,
        user=UDSConfig.CH_USER,
        password=UDSConfig.CH_PASSWORD,
        database=UDSConfig.CH_DATABASE
    )
    
    if not client.ping():
        print("❌ Failed to connect to ClickHouse")
        return
    
    print("✅ Connected successfully")
    
    # Run quality checks
    print("\n2. Running data quality checks...")
    report = generate_quality_report(client, "uds_quality_report.json")
    print("✅ Quality checks completed")
    
    # Generate statistics
    print("\n3. Generating statistical summaries...")
    analyzer = StatisticalAnalyzer(client)
    
    tables = client.list_tables()
    stats = {}
    for table in tables:
        print(f"   Analyzing {table}...")
        stats[table] = analyzer.get_table_statistics(table)
    
    with open("uds_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✅ Statistics generated")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Tables: {len(tables)}")
    print(f"Total Rows: {sum(s['row_count'] for s in stats.values()):,}")
    print(f"\nReports generated:")
    print("  - uds_quality_report.json")
    print("  - uds_statistics.json")
    print("\n✅ All done!")


if __name__ == "__main__":
    main()
