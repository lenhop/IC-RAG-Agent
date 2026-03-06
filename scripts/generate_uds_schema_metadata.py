#!/usr/bin/env python3
"""
Generate uds_schema_metadata.json from schema CSVs and ClickHouse.
Run from IC-RAG-Agent root: python scripts/generate_uds_schema_metadata.py
"""

import csv
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uds.uds_client import UDSClient
from src.uds.config import UDSConfig

# Table descriptions and use cases (curated)
TABLE_META = {
    "amz_order": {
        "description": "Customer orders from Amazon marketplace",
        "business_use_cases": [
            "Sales performance analysis",
            "Order fulfillment tracking",
            "Customer behavior analysis",
            "Revenue forecasting",
        ],
        "date_column": "start_date",
    },
    "amz_transaction": {
        "description": "Financial transactions (payments, refunds, adjustments)",
        "business_use_cases": [
            "Revenue reconciliation",
            "Fee analysis",
            "Settlement tracking",
        ],
        "date_column": "start_date",
    },
    "amz_statement": {
        "description": "Settlement statements",
        "business_use_cases": [
            "Financial reporting",
            "Settlement reconciliation",
        ],
        "date_column": "settlement_id",
    },
    "amz_fba_inventory_all": {
        "description": "FBA inventory snapshots",
        "business_use_cases": [
            "Inventory levels",
            "Stock planning",
        ],
        "date_column": "start_date",
    },
    "amz_daily_inventory_ledger": {
        "description": "Daily inventory ledger (in/out movements)",
        "business_use_cases": [
            "Inventory turnover",
            "Stock movement analysis",
        ],
        "date_column": "start_date",
    },
    "amz_fee": {
        "description": "Amazon fees (FBA, referral, etc.)",
        "business_use_cases": [
            "Fee analysis",
            "Cost optimization",
        ],
        "date_column": "start_date",
    },
    "amz_listing_item": {
        "description": "Product listing catalog with snapshots",
        "business_use_cases": [
            "Catalog management",
            "Listing health",
            "Product attribute analysis",
        ],
        "date_column": "request_date",
    },
    "amz_product": {
        "description": "Product master data (ASIN-level)",
        "business_use_cases": [
            "Product catalog",
            "ASIN lookup",
        ],
        "date_column": None,
    },
    "amz_monthly_search_term": {
        "description": "Monthly search term performance",
        "business_use_cases": [
            "Search analytics",
            "Keyword performance",
        ],
        "date_column": "start_date",
    },
}


def load_schema_csv(schema_dir: Path, table_name: str) -> list:
    """Load column definitions from schema CSV."""
    schema_path = schema_dir / f"{table_name}.csv"
    if not schema_path.exists():
        return []
    columns = []
    with open(schema_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["field_name", "data_type", "nullable", "description"]
        for row in reader:
            name = row.get("field_name", "").strip()
            if not name:
                continue
            data_type = row.get("data_type", "String").strip()
            nullable_str = row.get("nullable", "yes").strip().lower()
            nullable = nullable_str in ("yes", "y", "true", "1", "")
            desc = row.get("description", "").strip()
            index_type = row.get("index_type", "").strip()
            is_pk = "pk" in index_type.lower() or (row.get("constraint", "") or "").lower() == "pk"
            columns.append({
                "name": name,
                "type": data_type,
                "nullable": nullable,
                "description": desc or name,
                "business_meaning": desc or name,
                "is_primary_key": is_pk,
            })
    return columns


def main():
    schema_dir = Path(UDSConfig.SCHEMA_DIR)
    if not schema_dir.exists():
        print(f"Schema dir not found: {schema_dir}")
        sys.exit(1)

    client = UDSClient()
    tables = [t for t in client.list_tables() if t.startswith("amz_") and t != "agent_queries"]
    tables.sort()

    metadata = {
        "metadata": {
            "database": UDSConfig.CH_DATABASE,
            "total_tables": len(tables),
            "total_rows": 0,
            "date_range": "2025-10-01 to 2025-10-30",
            "last_updated": "2026-03-05",
        },
        "tables": {},
    }

    for table in tables:
        meta = TABLE_META.get(table, {"description": table, "business_use_cases": [], "date_column": None})
        cols = load_schema_csv(schema_dir, table)

        # Get row count and date range from ClickHouse
        try:
            count_df = client.query(f"SELECT COUNT(*) as cnt FROM {UDSConfig.CH_DATABASE}.{table}")
            row_count = int(count_df["cnt"].iloc[0])
        except Exception:
            row_count = 0

        date_range = None
        if meta.get("date_column") and cols:
            col_names = [c["name"] for c in cols]
            if meta["date_column"] in col_names:
                try:
                    dr_df = client.query(
                        f"SELECT min({meta['date_column']}) as mn, max({meta['date_column']}) as mx "
                        f"FROM {UDSConfig.CH_DATABASE}.{table}"
                    )
                    if not dr_df.empty and dr_df["mn"].iloc[0] and dr_df["mx"].iloc[0]:
                        date_range = f"{str(dr_df['mn'].iloc[0])[:10]} to {str(dr_df['mx'].iloc[0])[:10]}"
                except Exception:
                    pass

        # Get sample values for first few string columns only (avoids type conversion errors)
        for col in cols[:8]:
            try:
                # Use toString() to avoid type issues; only for String-like columns
                if "String" in col.get("type", "") or "LowCardinality" in col.get("type", ""):
                    sample_df = client.query(
                        f"SELECT toString({col['name']}) as v FROM {UDSConfig.CH_DATABASE}.{table} "
                        f"WHERE {col['name']} IS NOT NULL AND length(toString({col['name']})) > 0 LIMIT 3"
                    )
                    col["sample_values"] = sample_df["v"].astype(str).tolist()
                else:
                    col["sample_values"] = []
            except Exception:
                col["sample_values"] = []

        # Order by from IC-Data-Loader schema config
        ORDER_BY_MAP = {
            "amz_order": ["start_date", "amazon_order_id"],
            "amz_transaction": ["start_date", "settlement_id"],
            "amz_monthly_search_term": ["start_date", "search_frequency_rank"],
            "amz_daily_inventory_ledger": ["start_date", "fnsku"],
            "amz_fba_inventory_all": ["start_date", "sku"],
            "amz_fee": ["start_date", "asin"],
            "amz_listing_item": ["channel_id", "sku", "request_date"],
            "amz_product": ["ASIN"],
            "amz_statement": ["settlement_id"],
        }
        order_by = ORDER_BY_MAP.get(table, ["start_date"])

        metadata["tables"][table] = {
            "description": meta["description"],
            "row_count": row_count,
            "date_range": date_range or "N/A",
            "primary_key": order_by,
            "order_by": order_by,
            "columns": cols,
            "business_use_cases": meta.get("business_use_cases", []),
            "common_joins": {},
            "key_metrics": {},
            "data_quality": {"completeness": "N/A", "null_columns": [], "notes": ""},
        }
        metadata["metadata"]["total_rows"] += row_count

    client.close()

    out_path = PROJECT_ROOT / UDSConfig.SCHEMA_METADATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Generated {out_path}")
    print(f"Total rows: {metadata['metadata']['total_rows']}")


if __name__ == "__main__":
    main()
