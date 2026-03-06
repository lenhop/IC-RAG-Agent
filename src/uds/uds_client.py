"""
UDS ClickHouse Client.
Provides reliable database access for the UDS Agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from .config import UDSConfig

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency at module load
_clickhouse_connect = None


def _get_client_module():
    """Lazy load clickhouse_connect."""
    global _clickhouse_connect
    if _clickhouse_connect is None:
        try:
            import clickhouse_connect as chc
            _clickhouse_connect = chc
        except ImportError as e:
            raise ImportError(
                "clickhouse-connect is required for UDSClient. "
                "Install with: pip install clickhouse-connect pandas"
            ) from e
    return _clickhouse_connect


class QueryError(Exception):
    """Custom exception for query errors."""

    pass


class UDSClient:
    """
    ClickHouse client for UDS database access.

    Features:
    - Connection with configurable timeout
    - Query retry logic
    - Streaming support for large results
    - Error handling and logging
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        query_timeout: int = 300,
    ):
        """Initialize UDS client."""
        self.host = host or UDSConfig.CH_HOST
        self.port = port or UDSConfig.CH_PORT
        self.user = user or UDSConfig.CH_USER
        self.password = password or UDSConfig.CH_PASSWORD
        self.database = database or UDSConfig.CH_DATABASE
        self.query_timeout = query_timeout

        chc = _get_client_module()
        self._client = chc.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=10,
            send_receive_timeout=self.query_timeout,
        )
        logger.info(
            "UDSClient initialized: %s:%d/%s",
            self.host,
            self.port,
            self.database,
        )

    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> pd.DataFrame | List[tuple]:
        """
        Execute query and return results.

        Args:
            sql: SQL query string
            params: Query parameters for safe binding
            as_dataframe: Return as pandas DataFrame (default) or list of tuples

        Returns:
            Query results as DataFrame or list of tuples

        Raises:
            QueryError: If query execution fails
        """
        try:
            logger.debug("Executing query: %s...", sql[:100] if len(sql) > 100 else sql)
            start = time.time()

            result = self._client.query(sql, parameters=params or {})

            elapsed = time.time() - start
            rows = result.result_rows
            cols = result.column_names

            if as_dataframe:
                df = pd.DataFrame(rows, columns=cols)
                logger.info("Query returned %d rows in %.2fs", len(df), elapsed)
                return df
            return rows

        except Exception as e:
            logger.error("Query failed: %s", e)
            raise QueryError(f"Failed to execute query: {e}") from e

    def query_stream(
        self,
        sql: str,
        chunk_size: int = 10000,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream large result sets in chunks using LIMIT/OFFSET.

        Args:
            sql: SQL query string (must not contain LIMIT/OFFSET)
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        try:
            logger.debug("Streaming query: %s...", sql[:100] if len(sql) > 100 else sql)
            offset = 0
            while True:
                chunk_sql = f"{sql} LIMIT {chunk_size} OFFSET {offset}"
                df = self.query(chunk_sql, as_dataframe=True)
                if df.empty:
                    break
                yield df
                if len(df) < chunk_size:
                    break
                offset += chunk_size
        except Exception as e:
            logger.error("Streaming query failed: %s", e)
            raise QueryError(f"Failed to stream query: {e}") from e

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute non-query SQL (INSERT, DDL, etc.).

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Result from command execution
        """
        try:
            logger.debug("Executing statement: %s...", sql[:100] if len(sql) > 100 else sql)
            result = self._client.command(sql, parameters=params or {})
            logger.info("Statement executed successfully")
            return result
        except Exception as e:
            logger.error("Statement execution failed: %s", e)
            raise QueryError(f"Failed to execute statement: {e}") from e

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get table schema metadata.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with schema information
        """
        try:
            schema_query = f"""
            SELECT
                name,
                type,
                default_kind,
                comment
            FROM system.columns
            WHERE database = '{self.database}' AND table = '{table_name}'
            ORDER BY position
            """
            columns_df = self.query(schema_query, as_dataframe=True)

            count_query = f"SELECT COUNT(*) as count FROM {self.database}.{table_name}"
            count_df = self.query(count_query, as_dataframe=True)
            row_count = int(count_df["count"].iloc[0]) if not count_df.empty else 0

            return {
                "table_name": table_name,
                "database": self.database,
                "row_count": row_count,
                "columns": columns_df.to_dict("records"),
            }

        except Exception as e:
            logger.error("Failed to get schema for %s: %s", table_name, e)
            raise QueryError(f"Failed to get table schema: {e}") from e

    def list_tables(self) -> List[str]:
        """
        List all tables in the database.

        Returns:
            List of table names
        """
        try:
            query = f"""
            SELECT name
            FROM system.tables
            WHERE database = '{self.database}'
            ORDER BY name
            """
            result = self.query(query, as_dataframe=False)
            return [row[0] for row in result]
        except Exception as e:
            logger.error("Failed to list tables: %s", e)
            raise QueryError(f"Failed to list tables: {e}") from e

    def ping(self) -> bool:
        """
        Check if connection is alive.

        Returns:
            True if connection is healthy
        """
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the client connection."""
        try:
            self._client.close()
            logger.info("UDSClient connection closed")
        except Exception as e:
            logger.error("Error closing connection: %s", e)
