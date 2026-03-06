# UDS Data Foundation Plan - Supporting IC UDS Agent

**Project:** IC-Agent / UDS Agent  
**Phase:** Data Foundation (Post Data Loading)  
**Status:** Planning  
**Owner:** Kiro (Project Manager)  
**Created:** 2026-03-05

---

## Overview

With 40.3M rows of Amazon business data successfully loaded into ClickHouse, we now need to build the data foundation layer that will enable the IC UDS Agent to perform intelligent business analytics.

**Data Loaded:** ✅ Complete
- 9 tables, 40.3M rows
- October 2025 data (primary period)
- ClickHouse database: `ic_agent`
- Server: 8.163.3.40:8123

---

## Current State: What We Have

### ✅ Completed: Data Loading Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Schema definitions | ✅ | `IC-Data-Loader/schema/*.csv` |
| Data loader | ✅ | `IC-Data-Loader/src/` |
| Verification tools | ✅ | `IC-Data-Loader/scripts/verify_data_load.py` |
| Quality reports | ✅ | `IC-Data-Loader/src/quality_report.py` |
| Documentation | ✅ | `IC-Data-Loader/docs/` |

### ✅ Data Assets Available

| Table | Rows | Description | Key Columns |
|-------|------|-------------|-------------|
| amz_order | 4.2M | Customer orders | amazon_order_id, start_date, total_amount |
| amz_transaction | 8.1M | Financial transactions | settlement_id, transaction_type, amount |
| amz_statement | 7.2M | Settlement statements | settlement_id, start_date, total_amount |
| amz_fba_inventory_all | 7.1M | FBA inventory snapshots | sku, fnsku, quantity, start_date |
| amz_daily_inventory_ledger | 4.9M | Daily inventory changes | fnsku, event_type, quantity, start_date |
| amz_fee | 4.2M | Fee details | asin, fee_type, fee_amount, start_date |
| amz_listing_item | 2.6M | Product listings | channel_id, sku, request_date |
| amz_product | 1.1M | Product catalog | ASIN, title, brand, dimensions |
| amz_monthly_search_term | 1.1M | Search analytics | search_term, search_frequency_rank |

**Total:** 40.3M rows across 9 tables

---

## Phase 1: Data Foundation Layer (Week 1-2)

### 1.1 Database Schema Documentation

**Goal:** Create comprehensive schema documentation for the UDS Agent to understand the data structure.

**Deliverables:**
- [ ] **Schema metadata file** (`uds_schema_metadata.json`)
  - Table descriptions, relationships, business meanings
  - Column data types, constraints, indexes
  - Sample queries for common use cases
  - Data quality notes (nullable columns, date ranges)

- [ ] **Entity Relationship Diagram** (`uds_erd.md`)
  - Visual representation of table relationships
  - Primary/foreign key relationships
  - Common join patterns

- [ ] **Business glossary** (`uds_business_glossary.md`)
  - Amazon-specific terminology (ASIN, SKU, FNSKU, FBA)
  - Metric definitions (GMV, conversion rate, inventory turnover)
  - Common business questions mapped to tables

**Example Schema Metadata:**
```json
{
  "tables": {
    "amz_order": {
      "description": "Customer orders from Amazon marketplace",
      "row_count": 4175232,
      "date_range": "2025-10-01 to 2025-10-30",
      "primary_key": ["start_date", "amazon_order_id"],
      "business_use_cases": [
        "Sales performance analysis",
        "Order fulfillment tracking",
        "Customer behavior analysis"
      ],
      "common_joins": {
        "amz_transaction": "amazon_order_id",
        "amz_listing_item": "sku"
      },
      "key_metrics": {
        "total_sales": "SUM(total_amount)",
        "order_count": "COUNT(DISTINCT amazon_order_id)",
        "avg_order_value": "AVG(total_amount)"
      }
    }
  }
}
```

**Timeline:** 2-3 days  
**Assigned to:** Claude Code

---

### 1.2 Data Quality & Statistics

**Goal:** Understand data quality, distributions, and patterns to help the agent make informed decisions.

**Deliverables:**
- [ ] **Data profiling report** (`uds_data_profile.md`)
  - Row counts, null percentages, unique values
  - Date coverage and gaps
  - Value distributions (min, max, avg, percentiles)
  - Outlier detection

- [ ] **Data quality dashboard** (SQL queries)
  - Completeness checks (null rates by column)
  - Consistency checks (referential integrity)
  - Timeliness checks (data freshness)
  - Accuracy checks (value ranges, formats)

- [ ] **Statistical summaries** (`uds_statistics.json`)
  - Per-table statistics (for agent context)
  - Time-series patterns (daily/weekly trends)
  - Correlation analysis (which metrics move together)

**Example Profiling Query:**
```sql
-- Data completeness by table
SELECT 
    'amz_order' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT amazon_order_id) as unique_orders,
    SUM(CASE WHEN total_amount IS NULL THEN 1 ELSE 0 END) as null_amounts,
    MIN(start_date) as min_date,
    MAX(start_date) as max_date
FROM ic_agent.amz_order;
```

**Timeline:** 2-3 days  
**Assigned to:** VSCode

---

### 1.3 Common Query Patterns Library

**Goal:** Pre-build and test common analytical queries that the agent can use as templates.

**Deliverables:**
- [ ] **Query template library** (`uds_query_templates/`)
  - Sales analysis queries (daily sales, top products, trends)
  - Inventory queries (stock levels, turnover, aging)
  - Financial queries (revenue, fees, profitability)
  - Performance queries (conversion rates, fulfillment speed)

- [ ] **Parameterized query functions** (Python)
  - Reusable query builders with parameters
  - Date range handling, filtering, aggregation
  - Performance-optimized (proper indexes, partitions)

- [ ] **Query performance benchmarks**
  - Execution time for each template
  - Optimization recommendations
  - Index suggestions

**Example Query Template:**
```python
def get_daily_sales_trend(start_date: str, end_date: str, marketplace: str = None):
    """
    Get daily sales trend with order count and revenue.
    
    Args:
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        marketplace: Optional marketplace filter
    
    Returns:
        DataFrame with columns: date, order_count, total_revenue, avg_order_value
    """
    query = f"""
    SELECT 
        start_date as date,
        COUNT(DISTINCT amazon_order_id) as order_count,
        SUM(total_amount) as total_revenue,
        AVG(total_amount) as avg_order_value
    FROM ic_agent.amz_order
    WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
    {f"AND marketplace = '{marketplace}'" if marketplace else ""}
    GROUP BY start_date
    ORDER BY start_date
    """
    return execute_query(query)
```

**Timeline:** 3-4 days  
**Assigned to:** TRAE

---

### 1.4 ClickHouse Client Library

**Goal:** Create a robust, reusable ClickHouse client for the UDS Agent.

**Deliverables:**
- [ ] **UDSClient class** (`src/uds/uds_client.py`)
  - Connection pooling (reuse connections)
  - Query execution with retry logic
  - Streaming support for large result sets
  - Query timeout and cancellation
  - Error handling and logging

- [ ] **Query builder utilities**
  - Safe SQL generation (prevent injection)
  - Parameter binding
  - Date range helpers
  - Aggregation helpers

- [ ] **Result formatters**
  - DataFrame conversion (pandas)
  - JSON serialization
  - CSV export
  - Markdown table formatting

**Example Client:**
```python
class UDSClient:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.client = clickhouse_connect.get_client(
            host=host, port=port, username=user, 
            password=password, database=database
        )
    
    def query(self, sql: str, params: dict = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        result = self.client.query(sql, parameters=params)
        return result.result_rows_as_dataframe()
    
    def query_stream(self, sql: str, chunk_size: int = 10000):
        """Stream large result sets."""
        # Implementation for streaming
        pass
    
    def get_table_schema(self, table_name: str) -> dict:
        """Get table schema metadata."""
        # Implementation
        pass
```

**Timeline:** 2-3 days  
**Assigned to:** Cursor

---

## Phase 2: UDS Agent Tools (Week 3-4)

### 2.1 Schema Inspection Tools

**Goal:** Enable the agent to understand and explore the database schema dynamically.

**Tools to Build:**
- [ ] **ListTablesTool** - List all available tables
- [ ] **DescribeTableTool** - Get table schema, row count, date range
- [ ] **GetTableRelationshipsTool** - Find relationships between tables
- [ ] **SearchColumnsTool** - Find columns by name or description

**Example Tool:**
```python
class DescribeTableTool(BaseTool):
    name = "describe_table"
    description = "Get detailed schema information about a UDS table"
    
    parameters = [
        ToolParameter(name="table_name", type="string", required=True,
                     description="Name of the table to describe")
    ]
    
    def execute(self, table_name: str) -> ToolResult:
        # Get schema, row count, sample data
        schema = uds_client.get_table_schema(table_name)
        row_count = uds_client.query(f"SELECT COUNT(*) FROM {table_name}")
        sample = uds_client.query(f"SELECT * FROM {table_name} LIMIT 5")
        
        return ToolResult(
            success=True,
            data={
                "schema": schema,
                "row_count": row_count,
                "sample_data": sample
            }
        )
```

**Timeline:** 3-4 days  
**Assigned to:** Claude Code

---

### 2.2 Query Generation Tools

**Goal:** Enable the agent to generate and execute SQL queries based on natural language.

**Tools to Build:**
- [ ] **GenerateSQLTool** - Convert natural language to SQL
- [ ] **ExecuteQueryTool** - Run SQL and return results
- [ ] **ValidateQueryTool** - Check SQL syntax and safety
- [ ] **ExplainQueryTool** - Get query execution plan

**Example Tool:**
```python
class GenerateSQLTool(BaseTool):
    name = "generate_sql"
    description = "Generate SQL query from natural language question"
    
    parameters = [
        ToolParameter(name="question", type="string", required=True,
                     description="Natural language question about the data"),
        ToolParameter(name="tables", type="array", required=False,
                     description="Specific tables to query (optional)")
    ]
    
    def execute(self, question: str, tables: list = None) -> ToolResult:
        # Use LLM to generate SQL from question
        # Include schema context, example queries
        sql = llm_generate_sql(question, tables, schema_context)
        
        # Validate the generated SQL
        is_valid, errors = validate_sql(sql)
        
        return ToolResult(
            success=is_valid,
            data={"sql": sql, "validation_errors": errors}
        )
```

**Timeline:** 4-5 days  
**Assigned to:** TRAE

---

### 2.3 Analysis Tools

**Goal:** Pre-built analytical functions for common business questions.

**Tools to Build:**
- [ ] **SalesTrendTool** - Analyze sales trends over time
- [ ] **InventoryAnalysisTool** - Inventory levels, turnover, aging
- [ ] **ProductPerformanceTool** - Top/bottom products by metrics
- [ ] **FinancialSummaryTool** - Revenue, costs, profitability
- [ ] **ComparisonTool** - Compare periods, products, marketplaces

**Example Tool:**
```python
class SalesTrendTool(BaseTool):
    name = "analyze_sales_trend"
    description = "Analyze sales trends with automatic insights"
    
    parameters = [
        ToolParameter(name="start_date", type="string", required=True),
        ToolParameter(name="end_date", type="string", required=True),
        ToolParameter(name="granularity", type="string", required=False,
                     description="daily, weekly, or monthly", default="daily")
    ]
    
    def execute(self, start_date: str, end_date: str, 
                granularity: str = "daily") -> ToolResult:
        # Get sales data
        df = get_daily_sales_trend(start_date, end_date)
        
        # Calculate insights
        insights = {
            "total_revenue": df['total_revenue'].sum(),
            "avg_daily_revenue": df['total_revenue'].mean(),
            "trend": "increasing" if df['total_revenue'].iloc[-1] > df['total_revenue'].iloc[0] else "decreasing",
            "peak_day": df.loc[df['total_revenue'].idxmax(), 'date'],
            "growth_rate": calculate_growth_rate(df)
        }
        
        return ToolResult(
            success=True,
            data={"sales_data": df.to_dict(), "insights": insights}
        )
```

**Timeline:** 5-6 days  
**Assigned to:** VSCode

---

### 2.4 Visualization Tools

**Goal:** Generate charts and visualizations for data insights.

**Tools to Build:**
- [ ] **CreateChartTool** - Generate charts (line, bar, pie)
- [ ] **CreateDashboardTool** - Multi-chart dashboard
- [ ] **ExportVisualizationTool** - Save as PNG, PDF

**Libraries:**
- Plotly (interactive charts)
- Matplotlib (static charts)
- Seaborn (statistical visualizations)

**Timeline:** 3-4 days  
**Assigned to:** Cursor

---

## Phase 3: UDS Agent Core (Week 5-6)

### 3.1 Task Planner

**Goal:** Decompose complex analytical questions into subtasks.

**Components:**
- [ ] **Query decomposer** - Break complex questions into steps
- [ ] **Dependency resolver** - Determine execution order
- [ ] **Context manager** - Track intermediate results

**Example:**
```
User: "Show me the top 10 products by revenue and their inventory levels"

Task Plan:
1. Query amz_order to get revenue by product (SKU)
2. Aggregate and rank top 10 products
3. Query amz_fba_inventory_all for current inventory of those SKUs
4. Join results and format output
```

**Timeline:** 4-5 days  
**Assigned to:** Claude Code

---

### 3.2 UDS Agent Implementation

**Goal:** Build the main UDS Agent that orchestrates all tools.

**Components:**
- [ ] **UDSAgent class** (extends ReActAgent)
- [ ] **Tool registration** (all UDS tools)
- [ ] **Intent classification** (what type of analysis?)
- [ ] **Context enrichment** (add schema info, examples)
- [ ] **Result formatting** (tables, charts, insights)

**Example:**
```python
class UDSAgent(ReActAgent):
    def __init__(self, uds_client: UDSClient):
        super().__init__(name="UDS Agent")
        self.uds_client = uds_client
        
        # Register all tools
        self.register_tool(ListTablesTool())
        self.register_tool(DescribeTableTool())
        self.register_tool(GenerateSQLTool())
        self.register_tool(ExecuteQueryTool())
        self.register_tool(SalesTrendTool())
        # ... more tools
    
    def process_query(self, query: str) -> AgentResponse:
        # Classify intent
        intent = self.classify_intent(query)
        
        # Enrich with schema context
        context = self.build_context(intent)
        
        # Execute ReAct loop
        result = self.run(query, context)
        
        # Format response
        return self.format_response(result)
```

**Timeline:** 5-6 days  
**Assigned to:** TRAE

---

## Phase 4: Integration & Testing (Week 7-8)

### 4.1 Integration with IC-RAG

**Goal:** Enable UDS Agent to query IC-RAG for documentation and examples.

**Components:**
- [ ] RAG query tool for UDS Agent
- [ ] Documentation retrieval (SQL syntax, ClickHouse features)
- [ ] Example query retrieval (similar questions)

**Timeline:** 2-3 days

---

### 4.2 API Development

**Goal:** Expose UDS Agent via REST API.

**Endpoints:**
- `POST /uds/query` - Submit analytical question
- `GET /uds/query/{query_id}` - Get query status/results
- `POST /uds/query/stream` - Streaming response (SSE)
- `GET /uds/tables` - List available tables
- `GET /uds/tables/{table_name}` - Get table schema

**Timeline:** 3-4 days  
**Assigned to:** Cursor

---

### 4.3 Testing & Validation

**Goal:** Comprehensive testing of all components.

**Test Types:**
- [ ] Unit tests (tools, client, utilities)
- [ ] Integration tests (agent workflows)
- [ ] Performance tests (query speed, large datasets)
- [ ] Accuracy tests (SQL generation quality)

**Timeline:** 4-5 days  
**Assigned to:** VSCode

---

## Phase 5: Documentation & Deployment (Week 9-10)

### 5.1 Documentation

- [ ] API documentation (OpenAPI/Swagger)
- [ ] User guide (how to ask questions)
- [ ] Developer guide (how to add tools)
- [ ] Deployment guide (AWS ECS setup)

**Timeline:** 3-4 days

---

### 5.2 Deployment

- [ ] Docker containerization
- [ ] AWS ECS deployment
- [ ] Monitoring setup (Prometheus, CloudWatch)
- [ ] Alerting configuration

**Timeline:** 3-4 days  
**Assigned to:** Cursor

---

## Summary Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Data Foundation | Week 1-2 | Schema docs, data quality, query library, client |
| Phase 2: UDS Tools | Week 3-4 | 15+ tools for schema, query, analysis, viz |
| Phase 3: UDS Agent Core | Week 5-6 | Task planner, UDS Agent implementation |
| Phase 4: Integration & Testing | Week 7-8 | RAG integration, API, comprehensive tests |
| Phase 5: Documentation & Deployment | Week 9-10 | Docs, Docker, AWS ECS deployment |

**Total:** 10 weeks (2.5 months)

---

## Success Criteria

✅ **Data Foundation:**
- Complete schema documentation with business context
- Data quality report showing >95% completeness
- 50+ query templates covering common use cases
- Robust ClickHouse client with connection pooling

✅ **UDS Agent:**
- 15+ tools for data analysis
- Accurate SQL generation (>80% success rate)
- Sub-5-second response time for simple queries
- Support for complex multi-step analysis

✅ **Integration:**
- Seamless RAG integration for documentation
- REST API with streaming support
- Comprehensive test coverage (>80%)

✅ **Deployment:**
- Production-ready Docker container
- AWS ECS deployment with auto-scaling
- Monitoring and alerting configured

---

## Next Steps

1. **Review this plan** with the team
2. **Create detailed specs** for Phase 1 components
3. **Assign tasks** to team members (Cursor, VSCode, Claude Code, TRAE)
4. **Start Phase 1** - Data Foundation Layer

---

**Document Owner:** Kiro (AI Project Manager)  
**Last Updated:** 2026-03-05
