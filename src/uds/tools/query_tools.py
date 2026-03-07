"""
Query Generation Tools for UDS Agent.

Provides tools for generating, executing, validating, and explaining SQL queries.
"""

from ai_toolkit.tools import BaseTool, ToolParameter, ToolResult
from ai_toolkit.errors import ValidationError
from ..uds_client import UDSClient
from ..config import UDSConfig
import json
import os
import re


class GenerateSQLTool(BaseTool):
    """
    Generate SQL query from natural language question.
    Uses LLM with schema context and example queries.
    """

    name = "generate_sql"
    description = "Generate SQL query from natural language question about data"

    parameters = [
        ToolParameter(
            name="question",
            type="string",
            required=True,
            description="Natural language question (e.g., 'What were the top 10 products by revenue in October?')"
        ),
        ToolParameter(
            name="tables",
            type="array",
            required=False,
            description="Specific tables to query (optional, will auto-detect if not provided)"
        )
    ]

    def __init__(self):
        super().__init__(self.name, self.description)
        self.client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE
        )

        # Load schema metadata
        schema_path = UDSConfig.project_path(UDSConfig.SCHEMA_METADATA_PATH)
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema_metadata = json.load(f)

        # Initialize LLM (will be configured later)
        self.llm = None

    def set_llm(self, llm):
        """Set the LLM for SQL generation."""
        self.llm = llm

    def _get_parameters(self) -> list:
        """Get parameter definitions."""
        return self.parameters

    def validate_parameters(self, question: str = None, tables: list = None, **kwargs) -> None:
        """Validate parameters."""
        if question is None or not str(question).strip():
            raise ValidationError(
                message="question is required",
                field_name="question"
            )

    def _build_schema_context(self, tables: list = None) -> str:
        """Build schema context for LLM."""
        context = "# Database Schema\n\n"

        tables_dict = self.schema_metadata.get('tables', {})

        # If specific tables requested, use those; otherwise use all
        table_names = tables if tables else list(tables_dict.keys())

        for table_name in table_names:
            if table_name not in tables_dict:
                continue

            table_info = tables_dict[table_name]
            context += f"## Table: {table_name}\n"
            context += f"Description: {table_info.get('description', '')}\n"
            context += f"Row Count: {table_info.get('row_count', 0):,}\n"
            context += f"Primary Key: {', '.join(table_info.get('primary_key', []))}\n\n"

            context += "### Columns:\n"
            for col in table_info.get('columns', [])[:10]:  # Limit to first 10 columns
                context += f"- {col['name']} ({col['type']}): {col.get('description', '')}\n"

            context += "\n"

        return context

    def _get_example_queries(self) -> str:
        """Get example queries from templates."""
        examples = """
# Example Queries

## Sales Analysis
```sql
-- Daily sales trend
SELECT
    start_date as date,
    COUNT(DISTINCT amazon_order_id) as order_count,
    SUM(total_amount) as total_revenue
FROM ic_agent.amz_order
WHERE start_date BETWEEN '2025-10-01' AND '2025-10-31'
GROUP BY start_date
ORDER BY start_date;
```

## Top Products
```sql
-- Top 10 products by revenue
SELECT
    o.sku,
    p.title as product_name,
    COUNT(DISTINCT o.amazon_order_id) as order_count,
    SUM(o.total_amount) as total_revenue
FROM ic_agent.amz_order o
LEFT JOIN ic_agent.amz_product p ON o.asin = p.ASIN
WHERE o.start_date BETWEEN '2025-10-01' AND '2025-10-31'
GROUP BY o.sku, p.title
ORDER BY total_revenue DESC
LIMIT 10;
```

## Inventory Analysis
```sql
-- Current inventory levels
SELECT
    sku,
    fnsku,
    SUM(quantity) as total_quantity
FROM ic_agent.amz_fba_inventory_all
WHERE start_date = '2025-10-30'
GROUP BY sku, fnsku
ORDER BY total_quantity DESC;
```
"""
        return examples

    def execute(self, question: str, tables: list = None, **kwargs) -> ToolResult:
        """
        Generate SQL from natural language.

        Args:
            question: Natural language question
            tables: Optional specific tables

        Returns:
            ToolResult with generated SQL
        """
        try:
            # Check if LLM is configured
            if self.llm is None:
                return ToolResult(
                    success=False,
                    error="LLM not configured. Use set_llm() to configure."
                )

            # Build context
            schema_context = self._build_schema_context(tables)
            example_queries = self._get_example_queries()

            # Create prompt
            prompt = f"""You are a SQL expert. Generate a ClickHouse SQL query to answer the user's question.

{schema_context}

{example_queries}

# Rules:
1. Use proper ClickHouse syntax
2. Always use fully qualified table names (ic_agent.table_name)
3. Use appropriate aggregations and GROUP BY
4. Add ORDER BY for better results
5. Use LIMIT to prevent huge result sets
6. Return ONLY the SQL query, no explanations or markdown formatting

# User Question:
{question}

# SQL Query:
"""

            # Generate SQL using LLM
            response = self.llm.invoke(prompt)
            sql = response.content.strip()

            # Clean up SQL (remove markdown code blocks if present)
            if sql.startswith('```sql'):
                sql = sql.split('```sql')[1].split('```')[0].strip()
            elif sql.startswith('```'):
                sql = sql.split('```')[1].split('```')[0].strip()

            return ToolResult(
                success=True,
                output={
                    "sql": sql,
                    "question": question,
                    "tables_used": tables or "auto-detected"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class ExecuteQueryTool(BaseTool):
    """
    Execute SQL query and return results.
    Includes safety checks and result formatting.
    """

    name = "execute_query"
    description = "Execute SQL query and return results as DataFrame or JSON"

    parameters = [
        ToolParameter(
            name="sql",
            type="string",
            required=True,
            description="SQL query to execute"
        ),
        ToolParameter(
            name="format",
            type="string",
            required=False,
            description="Result format: 'dataframe' or 'json' (default: 'json')",
            default="json"
        ),
        ToolParameter(
            name="limit",
            type="integer",
            required=False,
            description="Maximum rows to return (default: 1000)",
            default=1000
        )
    ]

    def __init__(self):
        super().__init__(self.name, self.description)
        self.client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE
        )

    def _get_parameters(self) -> list:
        """Get parameter definitions."""
        return self.parameters

    def validate_parameters(self, sql: str = None, format: str = "json", limit: int = 1000, **kwargs) -> None:
        """Validate parameters."""
        if sql is None or not str(sql).strip():
            raise ValidationError(
                message="sql is required",
                field_name="sql"
            )

        if format not in ["json", "dataframe"]:
            raise ValidationError(
                message="format must be 'json' or 'dataframe'",
                field_name="format"
            )

        if limit is not None and limit <= 0:
            raise ValidationError(
                message="limit must be positive",
                field_name="limit"
            )

    def _is_safe_query(self, sql: str) -> tuple:
        """
        Check if query is safe to execute.

        Returns:
            (is_safe, error_message)
        """
        sql_upper = sql.upper()

        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Dangerous operation detected: {keyword}"

        # Check for DELETE without WHERE (even though we block DELETE above)
        if 'DELETE' in sql_upper and 'WHERE' not in sql_upper:
            return False, "DELETE without WHERE clause is not allowed"

        return True, ""

    def execute(self, sql: str, format: str = "json", limit: int = 1000, **kwargs) -> ToolResult:
        """
        Execute SQL query.

        Args:
            sql: SQL query
            format: Result format (dataframe, json)
            limit: Max rows

        Returns:
            ToolResult with query results
        """
        try:
            # Safety check
            is_safe, error_msg = self._is_safe_query(sql)
            if not is_safe:
                return ToolResult(
                    success=False,
                    error=error_msg
                )

            # Add LIMIT if not present
            if 'LIMIT' not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {limit}"

            # Execute query
            df = self.client.query(sql)

            # Format results
            if format == "json":
                results = df.to_dict('records')
            else:
                results = df

            return ToolResult(
                success=True,
                output={
                    "results": results,
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "sql": sql
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class ValidateQueryTool(BaseTool):
    """
    Validate SQL query for syntax and safety.
    Checks for dangerous operations and verifies table/column names.
    """

    name = "validate_query"
    description = "Validate SQL query for syntax, safety, and correctness"

    parameters = [
        ToolParameter(
            name="sql",
            type="string",
            required=True,
            description="SQL query to validate"
        )
    ]

    def __init__(self):
        super().__init__(self.name, self.description)
        self.client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE
        )

        schema_path = UDSConfig.project_path(UDSConfig.SCHEMA_METADATA_PATH)
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema_metadata = json.load(f)

    def _get_parameters(self) -> list:
        """Get parameter definitions."""
        return self.parameters

    def validate_parameters(self, sql: str = None, **kwargs) -> None:
        """Validate parameters."""
        if sql is None or not str(sql).strip():
            raise ValidationError(
                message="sql is required",
                field_name="sql"
            )

    def _check_syntax(self, sql: str) -> tuple:
        """Check SQL syntax using ClickHouse."""
        errors = []

        try:
            # Use ClickHouse's EXPLAIN to check syntax
            explain_query = f"EXPLAIN SYNTAX {sql}"
            self.client.query(explain_query)
            return True, []
        except Exception as e:
            errors.append(f"Syntax error: {str(e)}")
            return False, errors

    def _check_safety(self, sql: str) -> tuple:
        """Check for dangerous operations."""
        errors = []
        sql_upper = sql.upper()

        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                errors.append(f"Dangerous operation: {keyword}")

        return len(errors) == 0, errors

    def _check_tables_columns(self, sql: str) -> tuple:
        """Verify table and column names exist."""
        errors = []

        # Extract table names (simplified - you might want a proper SQL parser)
        table_pattern = r'FROM\s+(\w+\.)?(\w+)'
        tables = re.findall(table_pattern, sql, re.IGNORECASE)

        available_tables = list(self.schema_metadata.get('tables', {}).keys())

        for _, table_name in tables:
            if table_name not in available_tables:
                errors.append(f"Table '{table_name}' does not exist")

        return len(errors) == 0, errors

    def execute(self, sql: str, **kwargs) -> ToolResult:
        """
        Validate SQL query.

        Args:
            sql: SQL query to validate

        Returns:
            ToolResult with validation results
        """
        try:
            all_errors = []
            checks = {}

            # Run all checks
            syntax_ok, syntax_errors = self._check_syntax(sql)
            checks['syntax'] = {'passed': syntax_ok, 'errors': syntax_errors}
            all_errors.extend(syntax_errors)

            safety_ok, safety_errors = self._check_safety(sql)
            checks['safety'] = {'passed': safety_ok, 'errors': safety_errors}
            all_errors.extend(safety_errors)

            tables_ok, table_errors = self._check_tables_columns(sql)
            checks['tables_columns'] = {'passed': tables_ok, 'errors': table_errors}
            all_errors.extend(table_errors)

            is_valid = len(all_errors) == 0

            return ToolResult(
                success=True,
                output={
                    "is_valid": is_valid,
                    "checks": checks,
                    "errors": all_errors,
                    "sql": sql
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class ExplainQueryTool(BaseTool):
    """
    Get query execution plan and performance estimates.
    Provides optimization suggestions.
    """

    name = "explain_query"
    description = "Get query execution plan and optimization suggestions"

    parameters = [
        ToolParameter(
            name="sql",
            type="string",
            required=True,
            description="SQL query to explain"
        )
    ]

    def __init__(self):
        super().__init__(self.name, self.description)
        self.client = UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE
        )

    def _get_parameters(self) -> list:
        """Get parameter definitions."""
        return self.parameters

    def validate_parameters(self, sql: str = None, **kwargs) -> None:
        """Validate parameters."""
        if sql is None or not str(sql).strip():
            raise ValidationError(
                message="sql is required",
                field_name="sql"
            )

    def execute(self, sql: str, **kwargs) -> ToolResult:
        """
        Explain query execution plan.

        Args:
            sql: SQL query

        Returns:
            ToolResult with execution plan
        """
        try:
            # Get execution plan
            explain_query = f"EXPLAIN {sql}"
            plan_df = self.client.query(explain_query)
            plan = plan_df.to_dict('records')

            # Get estimated rows
            explain_estimate = f"EXPLAIN ESTIMATE {sql}"
            try:
                estimate_df = self.client.query(explain_estimate)
                estimate = estimate_df.to_dict('records')
            except:
                estimate = []

            # Generate optimization suggestions
            suggestions = []
            sql_upper = sql.upper()

            if 'SELECT *' in sql_upper:
                suggestions.append("Consider selecting only needed columns instead of SELECT *")

            if 'LIMIT' not in sql_upper:
                suggestions.append("Add LIMIT clause to prevent large result sets")

            if 'WHERE' not in sql_upper and 'GROUP BY' in sql_upper:
                suggestions.append("Consider adding WHERE clause to filter data before aggregation")

            return ToolResult(
                success=True,
                output={
                    "execution_plan": plan,
                    "estimate": estimate,
                    "suggestions": suggestions,
                    "sql": sql
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
