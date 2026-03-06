"""
Query builder utilities for UDS Agent.
Provides safe SQL generation with parameter binding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def safe_identifier(name: str) -> str:
    """
    Sanitize SQL identifier to prevent injection.

    Args:
        name: Table or column name

    Returns:
        Sanitized identifier

    Raises:
        ValueError: If identifier contains invalid characters
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid identifier: {name}")
    if not name.replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"Invalid identifier: {name}")
    return name


def build_date_filter(
    column: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Build SQL date filter clause.

    Args:
        column: Date column name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        SQL WHERE clause fragment
    """
    safe_identifier(column)
    conditions = []
    if start_date:
        conditions.append(f"{column} >= '{start_date}'")
    if end_date:
        conditions.append(f"{column} <= '{end_date}'")
    return " AND ".join(conditions) if conditions else "1=1"


class QueryBuilder:
    """
    Safe SQL query builder with parameter binding.
    """

    def __init__(self, table: str, database: str = "ic_agent"):
        """
        Initialize query builder.

        Args:
            table: Table name
            database: Database name (default: ic_agent)
        """
        self.table = f"{safe_identifier(database)}.{safe_identifier(table)}"
        self.select_cols = ["*"]
        self.where_conditions: List[str] = []
        self.group_by_cols: List[str] = []
        self.order_by_cols: List[str] = []
        self.limit_value: Optional[int] = None
        self.params: Dict[str, Any] = {}

    def select(self, *columns: str) -> "QueryBuilder":
        """Select specific columns or expressions (e.g. COUNT(*) as cnt)."""
        validated = []
        for c in columns:
            if " " in c or "(" in c:
                validated.append(c)  # Expression, skip strict validation
            else:
                validated.append(safe_identifier(c))
        self.select_cols = validated
        return self

    def where(self, condition: str, **params: Any) -> "QueryBuilder":
        """Add WHERE condition with parameters."""
        self.where_conditions.append(condition)
        self.params.update(params)
        return self

    def group_by(self, *columns: str) -> "QueryBuilder":
        """Add GROUP BY clause."""
        self.group_by_cols = [safe_identifier(c) for c in columns]
        return self

    def order_by(self, *columns: str) -> "QueryBuilder":
        """Add ORDER BY clause."""
        self.order_by_cols = [safe_identifier(c) for c in columns]
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        if n < 0:
            raise ValueError("LIMIT must be non-negative")
        self.limit_value = n
        return self

    def build(self) -> tuple[str, Dict[str, Any]]:
        """
        Build the SQL query.

        Returns:
            Tuple of (sql_string, parameters)
        """
        sql = f"SELECT {', '.join(self.select_cols)}"
        sql += f" FROM {self.table}"
        if self.where_conditions:
            sql += f" WHERE {' AND '.join(self.where_conditions)}"
        if self.group_by_cols:
            sql += f" GROUP BY {', '.join(self.group_by_cols)}"
        if self.order_by_cols:
            sql += f" ORDER BY {', '.join(self.order_by_cols)}"
        if self.limit_value is not None:
            sql += f" LIMIT {self.limit_value}"
        return sql, self.params
