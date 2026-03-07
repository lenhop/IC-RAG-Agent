# UDS Agent User Guide

Comprehensive guide for using the UDS (Unified Data Service) Agent to analyze Amazon seller data through natural language queries.

---

## 1. Introduction

### What is UDS Agent?

UDS Agent is a Business Intelligence system that answers natural language questions about your Amazon seller data. It connects to ClickHouse databases containing order, inventory, financial, and product data, and returns structured insights, summaries, and visualizations.

### Key Capabilities

- **Natural language queries** – Ask questions in plain English
- **Multi-domain analysis** – Sales, inventory, financial, product, and BI
- **Structured responses** – Summary, insights, data tables, charts
- **REST API** – Integrate with dashboards, scripts, or applications
- **Streaming support** – Real-time progress for long-running queries

### When to Use UDS Agent

- Ad-hoc business questions (e.g., "What were total sales in October?")
- Quick data exploration without writing SQL
- Building dashboards or reports that need flexible querying
- Comparing periods, products, or metrics
- Identifying trends, anomalies, or opportunities

### Quick Start Example

```bash
curl -X POST http://localhost:8000/api/v1/uds/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were total sales in October?"}'
```

---

## 2. How to Ask Questions

### Best Practices

- **Be specific** – Include time ranges (October, last 30 days), metrics, or filters
- **Use business terms** – Revenue, orders, inventory, fees, SKU, AOV
- **One main question** – For complex analysis, break into separate queries
- **Include context** – "Compare first half vs second half of October" is clearer than "Compare periods"

### Query Structure Tips

- Start with the metric: "What were...", "Show me...", "How many..."
- Add time scope: "in October", "for the last 30 days", "Q3 vs Q4"
- Specify granularity: "daily", "by product", "by category"

### What Makes a Good Question

| Good | Why |
|------|-----|
| "What were total sales in October?" | Clear metric, clear period |
| "Top 10 products by revenue with inventory levels" | Specific count, metric, and scope |
| "Compare sales between first and second half of October" | Clear comparison, defined periods |

### Common Pitfalls to Avoid

| Avoid | Instead |
|-------|---------|
| "Data" | "What are total orders and revenue?" |
| "Show me everything" | "Show me sales summary for October" |
| "Sales and inventory and fees" | Ask three separate questions |
| "Last month" without year | "October 2025" or "Last 30 days" |

---

## 3. Query Examples by Domain

### Sales Analysis (12 examples)

| # | Query | Expected Output |
|---|-------|-----------------|
| 1 | What were total sales in October? | Total revenue, order count, AOV |
| 2 | Show me sales trends for the last 30 days | Daily/weekly trend, growth rate |
| 3 | Top 10 products by revenue | Ranked list with revenue per product |
| 4 | Compare sales between first and second half of October | Period comparison, growth % |
| 5 | Which products have the highest profit margins? | Product list with margin % |
| 6 | Show me daily sales breakdown | Daily revenue and order counts |
| 7 | What's the average order value? | AOV metric |
| 8 | Sales by product category | Category-level revenue breakdown |
| 9 | Revenue growth rate | MoM or YoY growth percentage |
| 10 | Peak sales days in October | Top days by revenue |
| 11 | Which day had the highest sales? | Single day with revenue |
| 12 | Daily sales for last week | 7-day breakdown |

### Inventory Management (12 examples)

| # | Query | Expected Output |
|---|-------|-----------------|
| 1 | Show me current inventory levels | SKU-level or aggregate inventory |
| 2 | Which items are low in stock? | Low-stock items, threshold used |
| 3 | Inventory turnover rate | Turnover metric |
| 4 | Products with excess inventory | Overstocked items |
| 5 | Inventory value by category | Category-level inventory value |
| 6 | Stock-out risk analysis | At-risk items |
| 7 | Reorder recommendations | Items to reorder |
| 8 | Inventory aging analysis | Age of inventory |
| 9 | Compare inventory vs sales | Inventory-to-sales ratio |
| 10 | Warehouse utilization | Utilization metrics |
| 11 | List all available tables | Table names (schema exploration) |
| 12 | Describe the amz_order table | Column definitions, row count |

### Financial Analysis (12 examples)

| # | Query | Expected Output |
|---|-------|-----------------|
| 1 | Financial summary for October | Gross, fees, net revenue |
| 2 | Total fees breakdown | Fee types and amounts |
| 3 | Profit and loss statement | P&L summary |
| 4 | Fee analysis by type | ReferralFee, FBAFee, etc. |
| 5 | Cost of goods sold | COGS metrics |
| 6 | Gross margin analysis | Margin % and trends |
| 7 | Operating expenses | Expense breakdown |
| 8 | Net profit calculation | Net profit figure |
| 9 | Fee trends over time | Fee trend data |
| 10 | Cost optimization opportunities | Recommendations |
| 11 | What's the profit margin for October? | Margin percentage |
| 12 | What fees did we pay last month? | Fee total and breakdown |

### Product Performance (12 examples)

| # | Query | Expected Output |
|---|-------|-----------------|
| 1 | Top performing products | Best sellers by revenue/units |
| 2 | Underperforming products | Low performers |
| 3 | Product profitability analysis | Profit per product |
| 4 | Sales velocity by product | Units sold per period |
| 5 | Product lifecycle analysis | Product stage metrics |
| 6 | New product performance | New SKU metrics |
| 7 | Product comparison | Side-by-side comparison |
| 8 | Category performance | Category-level metrics |
| 9 | SKU rationalization analysis | SKU efficiency |
| 10 | Product portfolio optimization | Recommendations |
| 11 | What are the top 5 products by revenue? | Top 5 list |
| 12 | Analyze product performance | Product metrics summary |

### Business Intelligence (12 examples)

| # | Query | Expected Output |
|---|-------|-----------------|
| 1 | Full business health check | Sales, inventory, financial summary |
| 2 | Create a sales dashboard | Dashboard-ready data/charts |
| 3 | Executive summary for October | High-level summary |
| 4 | Key performance indicators | KPI metrics |
| 5 | Business trends analysis | Trend insights |
| 6 | Competitive positioning | Positioning metrics |
| 7 | Market share analysis | Share metrics |
| 8 | Customer segmentation | Segment breakdown |
| 9 | Seasonal patterns | Seasonal trends |
| 10 | Forecast next month's sales | Forecast data |
| 11 | Compare Q3 vs Q4 vs Q1 performance | Multi-period comparison |
| 12 | Create dashboard with sales, inventory, and financial metrics | Combined dashboard |

**Total: 60 query examples across 5 domains**

---

## 4. Understanding Responses

### Response Structure

Each successful query returns:

```json
{
  "summary": "Natural language summary of the answer",
  "insights": ["Key insight 1", "Key insight 2"],
  "data": { "structured": "metrics and tables" },
  "charts": [{ "type": "line", "title": "...", "data": [...] }],
  "recommendations": ["Actionable next step 1"],
  "metadata": {
    "intent": "sales",
    "confidence": 0.95,
    "tools_used": ["analyze_sales_trend"],
    "execution_time": 1.23
  }
}
```

### Summary

A short text answer to your question. Example:

> "Total sales in October: $125,000. 4,175 orders. Average order value: $29.95."

### Insights

Automatically detected patterns or anomalies:

- "Strong growth of 12.5%"
- "15 items need attention"
- "Low profit margin: 8.2%"

### Data

Structured metrics (numbers, arrays, objects) for programmatic use or display in tables.

### Charts

Visualization suggestions (line charts for trends, bar charts for rankings). Use with charting libraries.

### Recommendations

Suggested next steps based on the data:

- "Review and reorder 15 low stock items"
- "Consider reviewing pricing strategy to improve margins"

### Confidence Scores

- **High (0.8+)** – Query well understood, reliable result
- **Medium (0.5–0.8)** – May need clarification
- **Low (<0.5)** – Rephrase or simplify

### Metadata

- **execution_time** – Query duration in seconds
- **tools_used** – Tools invoked
- **intent** – Classified domain (sales, inventory, financial, product, comparison, general)

---

## 5. Troubleshooting

### Query Not Understood

- Rephrase using business terms (revenue, orders, inventory, fees)
- Add time range (October, last 30 days)
- Break complex questions into simpler ones

### No Data Returned

- Check date range (data may be October 2025 only)
- Verify table exists: `GET /api/v1/uds/tables`
- Try a simpler query first

### Timeout Errors

- Use streaming endpoint: `POST /api/v1/uds/query/stream`
- Narrow date range or add filters
- Increase `max_execution_time` in options

### Incorrect Results

- Confirm date format (YYYY-MM-DD)
- Check if question matches available tables
- Review schema: `GET /api/v1/uds/tables/{name}`

### Performance Issues

- Use streaming for long queries
- Cache repeated queries
- Check `GET /health` and `GET /api/v1/uds/status`

### Common Error Messages

| Error | Cause | Action |
|-------|-------|--------|
| Query not found | Query ID expired | Submit new query |
| Table not found | Invalid table name | List tables, use correct name |
| Database connection failed | ClickHouse down | Check DB, network, credentials |
| Circuit breaker is open | Too many failures | Wait, retry later |
| Rate limit exceeded | Too many requests | Wait, reduce request rate |

---

## 6. FAQ

### How accurate are the results?

Results are based on your ClickHouse data. Accuracy depends on data quality and query clarity. The agent uses SQL generated from your question; verify critical numbers in production.

### Can I export data?

Yes. Use the `data` field in the response for programmatic export. For CSV/Excel, process the response in your application. The `ExportVisualizationTool` supports chart export.

### What time period is covered?

Default data is typically October 2025. Specify dates in your query (e.g., "October 2025", "last 30 days"). Check `GET /api/v1/uds/statistics` for available ranges.

### How do I ask complex questions?

Break into steps: (1) Get sales summary, (2) Get top products, (3) Get inventory for those products. Or use multi-part queries: "Top 10 products by revenue with their inventory levels."

### Are there rate limits?

The API does not enforce rate limits by default. For production, add rate limiting (e.g., per client) to protect the LLM and database.

### What about data privacy and security?

- Data stays in your infrastructure (ClickHouse, API server)
- LLM calls can be local (Ollama) or remote; choose based on sensitivity
- Use HTTPS, authentication, and network isolation in production

### Which tables are available?

Call `GET /api/v1/uds/tables` for the list. Common tables: amz_order, amz_transaction, amz_fee, amz_fba_inventory_all, amz_product, amz_listing_item, amz_statement, amz_daily_inventory_ledger, amz_monthly_search_term.

### Can I use the API from JavaScript?

Yes. Use `fetch` or axios. Example:

```javascript
const res = await fetch('http://localhost:8000/api/v1/uds/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'What were total sales in October?' })
});
const data = await res.json();
```

### Where is the API documentation?

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- API Guide: [UDS_API_GUIDE.md](UDS_API_GUIDE.md)
- API Reference: [UDS_API_REFERENCE.md](UDS_API_REFERENCE.md)

---

## Quick Reference

| Need | Endpoint |
|------|----------|
| Ask a question | POST /api/v1/uds/query |
| Stream long query | POST /api/v1/uds/query/stream |
| Get result later | GET /api/v1/uds/query/{id} |
| List tables | GET /api/v1/uds/tables |
| Table schema | GET /api/v1/uds/tables/{name} |
| Sample data | GET /api/v1/uds/tables/{name}/sample |
| Statistics | GET /api/v1/uds/statistics |
| Check health | GET /health |
| Agent status | GET /api/v1/uds/status |
