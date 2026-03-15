"""
Schema Inspection Tools for UDS Agent.

Provides 4 tools for exploring database schema: ListTablesTool,
DescribeTableTool, GetTableRelationshipsTool, SearchColumnsTool.

All tools inherit from BaseTool (ai-toolkit) and return ToolResult.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_toolkit.tools import BaseTool, ToolParameter, ToolResult
from ai_toolkit.errors import ValidationError

from ..uds_client import UDSClient
from ..config import UDSConfig

logger = logging.getLogger(__name__)

# Resolve schema metadata path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCHEMA_METADATA_PATH = _PROJECT_ROOT / UDSConfig.SCHEMA_METADATA_PATH


def _load_schema_metadata() -> dict:
    """Load schema metadata from JSON file."""
    with open(_SCHEMA_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class ListTablesTool(BaseTool):
    """
    List all available tables in the UDS database.

    Returns table names with descriptions and row counts.
    """

    def __init__(self):
        super().__init__(
            name="list_tables",
            description="List all available tables in the UDS database with descriptions and row counts",
        )
        self._schema_metadata = _load_schema_metadata()

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="include_stats",
                type="boolean",
                description="Include row counts and date ranges (default: true)",
                required=False,
                default=True,
            )
        ]

    def validate_parameters(self, include_stats: bool = True, **kwargs) -> None:
        """Validate parameters."""
        if include_stats is not None and not isinstance(include_stats, bool):
            raise ValidationError(
                message="include_stats must be a boolean",
                field_name="include_stats",
                field_value=include_stats,
            )

    def execute(self, include_stats: bool = True, **kwargs) -> ToolResult:
        """
        List all tables with metadata.

        Args:
            include_stats: Include row counts and date ranges

        Returns:
            ToolResult with table list
        """
        try:
            tables = []
            for table_name, table_info in self._schema_metadata.get("tables", {}).items():
                table_data = {
                    "name": table_name,
                    "description": table_info.get("description", ""),
                }
                if include_stats:
                    table_data.update(
                        {
                            "row_count": table_info.get("row_count", 0),
                            "date_range": table_info.get("date_range", ""),
                            "primary_key": table_info.get("primary_key", []),
                        }
                    )
                tables.append(table_data)

            return ToolResult(
                success=True,
                output={
                    "tables": tables,
                    "total_tables": len(tables),
                },
                metadata={"message": f"Found {len(tables)} tables"},
            )

        except Exception as e:
            logger.exception("Failed to list tables")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to list tables"},
            )


class DescribeTableTool(BaseTool):
    """
    Get detailed schema information for a specific table.

    Returns columns, types, relationships, and sample data.
    """

    def __init__(self):
        super().__init__(
            name="describe_table",
            description="Get detailed schema information for a specific table including columns, types, relationships, and sample data",
        )
        self._client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE,
        )
        self._schema_metadata = _load_schema_metadata()

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="table_name",
                type="string",
                description="Name of the table to describe (e.g., 'amz_order')",
                required=True,
            ),
            ToolParameter(
                name="include_sample",
                type="boolean",
                description="Include sample data rows (default: true)",
                required=False,
                default=True,
            ),
        ]

    def validate_parameters(
        self, table_name: str = None, include_sample: bool = True, **kwargs
    ) -> None:
        """Validate parameters."""
        if not table_name or not str(table_name).strip():
            raise ValidationError(
                message="table_name is required",
                field_name="table_name",
            )
        if include_sample is not None and not isinstance(include_sample, bool):
            raise ValidationError(
                message="include_sample must be a boolean",
                field_name="include_sample",
                field_value=include_sample,
            )

    def execute(
        self, table_name: str, include_sample: bool = True, **kwargs
    ) -> ToolResult:
        """
        Describe a table in detail.

        Args:
            table_name: Name of the table
            include_sample: Include sample data

        Returns:
            ToolResult with table schema
        """
        try:
            tables = self._schema_metadata.get("tables", {})
            table_name = str(table_name).strip()
            if table_name not in tables:
                return ToolResult(
                    success=False,
                    error=f"Table '{table_name}' not found",
                    metadata={"message": f"Available tables: {', '.join(tables.keys())}"},
                )

            table_info = tables[table_name]
            result = {
                "table_name": table_name,
                "description": table_info.get("description", ""),
                "row_count": table_info.get("row_count", 0),
                "date_range": table_info.get("date_range", ""),
                "primary_key": table_info.get("primary_key", []),
                "order_by": table_info.get("order_by", []),
                "columns": table_info.get("columns", []),
                "business_use_cases": table_info.get("business_use_cases", []),
                "common_joins": table_info.get("common_joins", {}),
                "key_metrics": table_info.get("key_metrics", {}),
            }

            if include_sample:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_df = self._client.query(sample_query)
                result["sample_data"] = sample_df.to_dict("records")

            return ToolResult(
                success=True,
                output=result,
                metadata={"message": f"Table '{table_name}' described successfully"},
            )

        except Exception as e:
            logger.exception("Failed to describe table %s", table_name)
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": f"Failed to describe table '{table_name}'"},
            )


class GetTableRelationshipsTool(BaseTool):
    """
    Find relationships between tables.

    Shows how tables can be joined together.
    """

    def __init__(self):
        super().__init__(
            name="get_table_relationships",
            description="Find relationships between tables and show how they can be joined",
        )
        self._schema_metadata = _load_schema_metadata()

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="table_name",
                type="string",
                description="Specific table to find relationships for (optional, shows all if not provided)",
                required=False,
            )
        ]

    def validate_parameters(self, table_name: str = None, **kwargs) -> None:
        """Validate parameters."""
        pass  # table_name is optional

    def execute(self, table_name: str = None, **kwargs) -> ToolResult:
        """
        Get table relationships.

        Args:
            table_name: Optional specific table

        Returns:
            ToolResult with relationships
        """
        try:
            tables = self._schema_metadata.get("tables", {})

            if table_name:
                table_name = str(table_name).strip()
                if table_name not in tables:
                    return ToolResult(
                        success=False,
                        error=f"Table '{table_name}' not found",
                        metadata={"message": f"Available tables: {', '.join(tables.keys())}"},
                    )

                table_info = tables[table_name]
                relationships = table_info.get("common_joins", {})
                result = {
                    "table": table_name,
                    "relationships": relationships,
                }
            else:
                all_relationships = []
                for tbl_name, tbl_info in tables.items():
                    joins = tbl_info.get("common_joins", {})
                    for related_table, join_info in joins.items():
                        all_relationships.append(
                            {
                                "from_table": tbl_name,
                                "to_table": related_table,
                                "join_key": join_info.get("join_key", ""),
                                "relationship": join_info.get("relationship", ""),
                                "description": join_info.get("description", ""),
                            }
                        )
                result = {
                    "relationships": all_relationships,
                    "total_relationships": len(all_relationships),
                }

            return ToolResult(
                success=True,
                output=result,
                metadata={"message": "Relationships retrieved successfully"},
            )

        except Exception as e:
            logger.exception("Failed to get relationships")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to get relationships"},
            )


class SearchColumnsTool(BaseTool):
    """
    Search for columns across all tables by name or description.

    Useful for finding where specific data is stored.
    """

    def __init__(self):
        super().__init__(
            name="search_columns",
            description="Search for columns across all tables by name or description",
        )
        self._schema_metadata = _load_schema_metadata()

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="search_term",
                type="string",
                description="Term to search for in column names or descriptions",
                required=True,
            ),
            ToolParameter(
                name="search_in",
                type="string",
                description="Where to search: 'name', 'description', or 'both' (default: 'both')",
                required=False,
                default="both",
            ),
        ]

    def validate_parameters(
        self, search_term: str = None, search_in: str = "both", **kwargs
    ) -> None:
        """Validate parameters."""
        if not search_term or not str(search_term).strip():
            raise ValidationError(
                message="search_term is required",
                field_name="search_term",
            )
        valid_search_in = ("name", "description", "both")
        if search_in and search_in.lower() not in valid_search_in:
            raise ValidationError(
                message=f"search_in must be one of: {', '.join(valid_search_in)}",
                field_name="search_in",
                field_value=search_in,
            )

    def execute(self, search_term: str, search_in: str = "both", **kwargs) -> ToolResult:
        """
        Search for columns.

        Args:
            search_term: Term to search for
            search_in: Where to search (name, description, both)

        Returns:
            ToolResult with matching columns
        """
        try:
            search_term_lower = str(search_term).lower()
            search_in = (search_in or "both").lower()
            if search_in not in ("name", "description", "both"):
                search_in = "both"

            matches = []
            tables = self._schema_metadata.get("tables", {})
            for tbl_name, table_info in tables.items():
                columns = table_info.get("columns", [])
                for column in columns:
                    col_name = column.get("name", "").lower()
                    col_desc = column.get("description", "").lower()

                    match = False
                    if search_in in ("name", "both") and search_term_lower in col_name:
                        match = True
                    if search_in in ("description", "both") and search_term_lower in col_desc:
                        match = True

                    if match:
                        matches.append(
                            {
                                "table": tbl_name,
                                "column_name": column.get("name", ""),
                                "type": column.get("type", ""),
                                "description": column.get("description", ""),
                                "business_meaning": column.get("business_meaning", ""),
                            }
                        )

            return ToolResult(
                success=True,
                output={
                    "matches": matches,
                    "total_matches": len(matches),
                    "search_term": search_term,
                },
                metadata={"message": f"Found {len(matches)} matching columns"},
            )

        except Exception as e:
            logger.exception("Failed to search columns")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to search columns"},
            )
