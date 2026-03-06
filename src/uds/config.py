"""
UDS Agent configuration.
Reads connection parameters from environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class UDSConfig:
    """UDS database configuration."""

    # ClickHouse connection (UDS/ic_agent database)
    CH_HOST = os.getenv("UDS_CH_HOST", os.getenv("CH_HOST", "8.163.3.40"))
    CH_PORT = int(os.getenv("UDS_CH_PORT", os.getenv("CH_PORT", "8123")))
    CH_USER = os.getenv("UDS_CH_USER", os.getenv("CH_USER", "ic_agent"))
    CH_PASSWORD = os.getenv("UDS_CH_PASSWORD", os.getenv("CH_PASSWORD", "ic_agent_2026"))
    CH_DATABASE = os.getenv("UDS_CH_DATABASE", os.getenv("CH_DATABASE", "ic_agent"))

    # Query settings
    QUERY_TIMEOUT = int(os.getenv("UDS_QUERY_TIMEOUT", "300"))  # 5 minutes
    STREAM_CHUNK_SIZE = int(os.getenv("UDS_STREAM_CHUNK_SIZE", "10000"))

    # Schema metadata path (relative to project root)
    SCHEMA_METADATA_PATH = "src/uds/uds_schema_metadata.json"

    # Path to IC-Data-Loader schema CSVs (sibling project)
    SCHEMA_DIR = os.getenv(
        "UDS_SCHEMA_DIR",
        str(_PROJECT_ROOT.parent / "IC-Data-Loader" / "schema"),
    )
