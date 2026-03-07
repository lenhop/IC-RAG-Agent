# UDS Agent Developer Guide

Technical guide for developers extending and maintaining the UDS Agent.

---

## 1. Architecture Overview

### High-Level Architecture

```
+------------------+     +------------------+     +------------------+
|   REST API       |     |   UDS Agent     |     |   ClickHouse     |
|   (FastAPI)      |---->|   (ReAct)       |---->|   Database       |
+------------------+     +------------------+     +------------------+
        |                         |                         |
        v                         v                         v
+------------------+     +------------------+     +------------------+
|   Query Request  |     | Intent Classifier|     |   UDS Client     |
|   QueryResponse  |     | Task Planner     |     |   (Cache)        |
+------------------+     +------------------+     +------------------+
                                 |
                                 v
                        +------------------+
                        |   Tool Registry  |
                        |   (16 tools)     |
                        +------------------+
```

### Component Diagram

```
                    +-------------------+
                    |   UDS Agent API   |
                    |   (api.py)        |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +-----------------+           +-----------------+
    |  UDS Agent      |           |  UDS Client      |
    |  (uds_agent.py) |           |  (uds_client.py)|
    +--------+--------+           +--------+--------+
             |                             |
             |  +-------------------------+
             |  |
             v  v
    +-----------------+           +-----------------+
    | Intent Classifier|           | ClickHouse      |
    | Task Planner    |           | Database        |
    | Result Formatter|           +-----------------+
    +--------+--------+
             |
             v
    +-----------------+
    | Tool Registry   |
    | - Schema (4)    |
    | - Query (4)     |
    | - Analysis (5)  |
    | - Viz (3)       |
    +-----------------+
```

### Request Flow

1. **API Layer** – FastAPI receives POST /api/v1/uds/query
2. **Agent** – UDSAgent.process_query() orchestrates the flow
3. **Intent Classification** – LLM classifies domain (sales, inventory, financial, product, comparison, general)
4. **Task Planning** – LLM creates subtasks with tool assignments
5. **Context Building** – Relevant tables, schema, and plan injected into prompt
6. **ReAct Loop** – Agent iterates: Thought -> Action -> Observation until done
7. **Result Formatting** – UDSResultFormatter produces summary, insights, data, charts
8. **Response** – JSON returned to client

### Key Directories

| Path | Purpose |
|------|---------|
| `src/uds/` | UDS Agent core |
| `src/uds/api.py` | REST API endpoints |
| `src/uds/uds_agent.py` | Main agent orchestration |
| `src/uds/tools/` | All 16 tools |
| `src/uds/uds_client.py` | ClickHouse client |
| `src/uds/intent_classifier.py` | Intent classification |
| `src/uds/task_planner.py` | Task planning |
| `src/uds/result_formatter.py` | Response formatting |
| `src/uds/cache.py` | Query/SQL caching |
| `src/uds/error_handler.py` | Error handling, circuit breaker |

---

## 2. Tool Reference (16 Tools)

All tools inherit from `ai_toolkit.tools.BaseTool` and return `ToolResult`.

### Schema Tools (4)

| Tool | Name | Purpose | Key Parameters |
|------|------|---------|----------------|
| ListTablesTool | list_tables | List all tables in database | include_stats (bool) |
| DescribeTableTool | describe_table | Get table schema and optional sample | table_name, include_sample |
| GetTableRelationshipsTool | get_table_relationships | Get FK relationships for a table | table_name |
| SearchColumnsTool | search_columns | Search columns by name/description | search_term, search_in |

### Query Tools (4)

| Tool | Name | Purpose | Key Parameters |
|------|------|---------|----------------|
| GenerateSQLTool | generate_sql | Generate SQL from natural language | question, tables |
| ExecuteQueryTool | execute_query | Run SQL and return results | sql, format, limit |
| ValidateQueryTool | validate_query | Validate SQL syntax and safety | sql |
| ExplainQueryTool | explain_query | Explain what SQL does | sql |

### Analysis Tools (5)

| Tool | Name | Purpose | Key Parameters |
|------|------|---------|----------------|
| SalesTrendTool | analyze_sales_trend | Sales trends over time | start_date, end_date, granularity, marketplace |
| InventoryAnalysisTool | analyze_inventory | Inventory levels and low-stock | as_of_date, low_stock_threshold |
| ProductPerformanceTool | analyze_product_performance | Product metrics and rankings | start_date, end_date, metric, limit |
| FinancialSummaryTool | financial_summary | Financial summary for period | start_date, end_date |
| ComparisonTool | compare_metrics | Compare two periods | comparison_type, period1_*, period2_* |

### Visualization Tools (3)

| Tool | Name | Purpose | Key Parameters |
|------|------|---------|----------------|
| CreateChartTool | create_chart | Create a single chart | data, chart_type, x_column, y_column, title |
| CreateDashboardTool | create_dashboard | Create multi-chart dashboard | charts, title, layout |
| ExportVisualizationTool | export_visualization | Export chart to file | chart_html, figure_json, format, filename |

### Tool Registration

The UDS Agent registers only **query tools** by default (used in ReAct loop):

```python
# uds_agent.py
def _register_tools(self):
    for tool in UDSToolRegistry.get_query_tools():
        self.register_tool(tool)
```

Schema, analysis, and visualization tools are available via `UDSToolRegistry.get_all_tools()` for extension or direct use.

---

## 3. Step-by-Step Tool Creation Tutorial

### Prerequisites

- Python 3.10+
- ai-toolkit installed
- UDS Agent codebase

### Step 1: Create Tool File

Create `src/uds/tools/my_custom_tool.py`:

```python
"""
Custom UDS tool example.
"""
from typing import Any, Dict, List

from ai_toolkit.tools import BaseTool, ToolParameter, ToolResult


class MyCustomTool(BaseTool):
    """Example custom tool for UDS Agent."""

    name = "my_custom_tool"
    description = "Performs a custom analysis on the data"

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input_param",
                type="string",
                description="Input parameter for the tool",
                required=True,
            ),
            ToolParameter(
                name="optional_param",
                type="integer",
                description="Optional numeric parameter",
                required=False,
            ),
        ]

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool logic."""
        try:
            input_param = str(kwargs.get("input_param", "")).strip()
            optional_param = kwargs.get("optional_param")
            if optional_param is not None:
                optional_param = int(optional_param)

            # Validate
            if not input_param:
                return ToolResult(success=False, error="input_param is required")

            # Your logic here
            result = {"processed": input_param, "optional": optional_param}

            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### Step 2: Register in UDSToolRegistry

Edit `src/uds/tools/__init__.py`:

```python
from .my_custom_tool import MyCustomTool

# In UDSToolRegistry.get_all_tools():
return [
    # ... existing tools ...
    MyCustomTool(),
]
```

### Step 3: Register in Agent (if needed in ReAct loop)

Edit `src/uds/uds_agent.py`:

```python
def _register_tools(self):
    for tool in UDSToolRegistry.get_query_tools():
        self.register_tool(tool)
    # Add custom tool if it should participate in ReAct
    self.register_tool(MyCustomTool())
```

### Step 4: Update Task Planner (optional)

If the tool should be suggested by the task planner, update `task_planner.py` or the LLM prompt to include `my_custom_tool` in the available tools list.

### Step 5: Add Tests

Create `tests/uds/test_my_custom_tool.py`:

```python
import pytest
from src.uds.tools.my_custom_tool import MyCustomTool


def test_my_custom_tool_success():
    tool = MyCustomTool()
    result = tool.execute(input_param="test")
    assert result.success
    assert result.data["processed"] == "test"


def test_my_custom_tool_validation():
    tool = MyCustomTool()
    result = tool.execute(input_param="")
    assert not result.success
    assert "required" in result.error.lower()
```

### Best Practices

- Use `ToolResult(success=True, data=...)` for success
- Use `ToolResult(success=False, error="...")` for failures
- Validate all parameters in `execute()`
- Handle exceptions and return structured errors
- Add docstrings and type hints
- Follow PEP8 and project style

---

## 4. Testing Guidelines

### Unit Tests

- **Location**: `tests/` or `tests/uds/`
- **Naming**: `test_*.py`, `test_<module>_<behavior>`
- **Framework**: pytest

```bash
pytest tests/ -v
pytest tests/uds/ -v
```

### Integration Tests

- **API tests**: `tests/test_api_integration.py`, `tests/test_api_streaming.py`
- **Agent tests**: `tests/test_uds_agent.py`
- **Database**: Use test ClickHouse or mocks

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/uds

# Specific file
pytest tests/test_api_integration.py -v

# Specific test
pytest tests/test_api_integration.py::test_submit_query -v
```

### Test Structure

```python
def test_submit_query():
    """Test synchronous query submission."""
    # Arrange
    client = TestClient(app)
    payload = {"query": "What were total sales in October?"}

    # Act
    response = client.post("/api/v1/uds/query", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert data["status"] in ("completed", "failed")
```

### Mocking

- Mock `UDSClient` for tests without ClickHouse
- Mock LLM for deterministic intent/plan
- Use `pytest.fixture` for shared setup

---

## 5. Contributing Guide

### Code Style

- **Python**: PEP8, black, isort
- **Comments**: English, >=10% of code volume
- **Docstrings**: All public functions/classes

### Pull Request Process

1. Create a feature branch from `main`
2. Implement changes with tests
3. Run `pytest` and fix failures
4. Run `black src/` and `isort src/`
5. Submit PR with description and test coverage
6. Address review feedback

### Commit Messages

- Use present tense: "Add tool X", "Fix circuit breaker"
- Reference issue/task if applicable

### Documentation

- Update relevant docs when adding features
- Add docstrings for new tools and APIs
- Update UDS_USER_GUIDE.md for user-facing changes

### Dependencies

- Add new dependencies to `requirements.txt` with versions
- Prefer existing packages over new ones

---

## 6. Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| UDS_LLM_PROVIDER | LLM provider (ollama, etc.) | ollama |
| UDS_LLM_MODEL | Model name | qwen3:1.7b |
| RAG_LLM_PROVIDER | Fallback provider | ollama |
| RAG_LLM_MODEL | Fallback model | qwen3:1.7b |
| CH_HOST | ClickHouse host | localhost |
| CH_PORT | ClickHouse port | 9000 |
| CH_USER | ClickHouse user | default |
| CH_PASSWORD | ClickHouse password | (empty) |
| CH_DATABASE | Database name | uds |

### Config Module

`src/uds/config.py` centralizes configuration. Override via environment or config file.

---

## 7. Extending the Agent

### Adding a New Intent Domain

1. Add domain to `IntentDomain` enum in `intent_classifier.py`
2. Update classifier prompt/examples
3. Add domain tables in `_get_relevant_tables()` in `uds_agent.py`
4. Add domain-specific tools if needed

### Custom Result Formatting

Override `UDSResultFormatter.format()` or create a subclass for custom output structure.

### Caching

- `UDSCache` caches SQL and query results
- Use cache keys based on query + params
- Consider TTL for production

---

## Quick Reference

| Task | File / Command |
|------|----------------|
| Add tool | `src/uds/tools/`, `__init__.py` |
| Add endpoint | `src/uds/api.py` |
| Change intent | `src/uds/intent_classifier.py` |
| Change plan | `src/uds/task_planner.py` |
| Run tests | `pytest tests/ -v` |
| Run API | `uvicorn src.uds.api:app --reload` |
