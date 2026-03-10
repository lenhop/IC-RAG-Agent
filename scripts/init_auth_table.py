#!/usr/bin/env python3
"""
Create ic_rag_agent_user table in ClickHouse.

Uses UDSConfig (CH_HOST, CH_DATABASE, etc.). Run from project root:
  python scripts/init_auth_table.py
"""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uds.config import UDSConfig


def main() -> int:
    """Create database and table if not exist."""
    try:
        import clickhouse_connect
    except ImportError:
        print("Install clickhouse-connect: pip install clickhouse-connect")
        return 1

    db = UDSConfig.CH_DATABASE
    client = clickhouse_connect.get_client(
        host=UDSConfig.CH_HOST,
        port=UDSConfig.CH_PORT,
        username=UDSConfig.CH_USER,
        password=UDSConfig.CH_PASSWORD,
    )

    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    client.command(f"""
        CREATE TABLE IF NOT EXISTS {db}.ic_rag_agent_user (
            user_id UUID,
            user_name String,
            email String DEFAULT '',
            password_hash String,
            role LowCardinality(String) DEFAULT 'general',
            status LowCardinality(String) DEFAULT 'active',
            created_time DateTime64(3),
            updated_time DateTime64(3),
            last_login_time Nullable(DateTime64(3)),
            last_login_ip Nullable(String),
            metadata String DEFAULT '{{}}'
        ) ENGINE = ReplacingMergeTree(updated_time)
        ORDER BY (user_id)
    """)
    print(f"Created {db}.ic_rag_agent_user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
