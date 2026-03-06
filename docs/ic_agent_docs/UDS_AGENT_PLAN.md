# UDS Agent - Implementation Plan

**Phase:** Part 3 of IC-Agent  
**Duration:** 8 weeks  
**Goal:** Build the UDS (Unified Data System) database and the Data Analysis Agent that queries it for business intelligence

**Dependencies:** Part 1 (ReAct Agent Core) must be complete before agent development begins. Database setup (Week 0) can start immediately in parallel.

---

## Overview

UDS is the analytics database for Amazon seller business intelligence. It stores orders, inventory, products, shipments, and financials at scale (200M+ rows), hosted on Alibaba Cloud. The UDS Agent provides natural language querying, trend analysis, and automated report generation on top of this data.

---

## Infrastructure: Alibaba Cloud (China Region)

**Region:** cn-hangzhou (China mainland)

### Architecture: Single ECS + Docker Compose

All services run as Docker containers on one ECS instance. Simple to manage, cost-effective, sufficient for this workload.

```
Alibaba Cloud ECS (cn-hangzhou)
┌─────────────────────────────────────────┐
│  16 vCPU / 64 GB RAM / 1 TB SSD        │
│                                         │
│  Docker Compose                         │
│  ┌─────────────────┐                   │
│  │   ClickHouse    │  10C / 48GB / 800GB│
│  │   port: 8123    │                   │
│  │   port: 9000    │                   │
│  └─────────────────┘                   │
│  ┌─────────────────┐                   │
│  │     Redis       │  2C / 8GB / 50GB  │
│  │   port: 6379    │                   │
│  └─────────────────┘                   │
│  ┌─────────────────┐                   │
│  │    ChromaDB     │  2C / 6GB / 150GB │
│  │   port: 8000    │                   │
│  └─────────────────┘                   │
│                                         │
│  OSS Mount (backups + reports)          │
└─────────────────────────────────────────┘
```

### ECS Instance Spec

| Resource | Value |
|----------|-------|
| CPU | 16 vCPU |
| RAM | 64 GB |
| SSD | 1 TB |
| OS | Ubuntu 22.04 LTS |
| Region | cn-hangzhou |

### Container Resource Allocation

| Container | CPU | RAM | Disk Volume |
|-----------|-----|-----|-------------|
| ClickHouse | 10 vCPU | 48 GB | 800 GB |
| Redis | 2 vCPU | 8 GB | 50 GB |
| ChromaDB | 2 vCPU | 6 GB | 150 GB |

### Additional Services

| Service | Purpose |
|---------|---------|
| Alibaba Cloud OSS | Backups, report exports, seed data archives (~100GB to start) |

**Estimated monthly cost:** ¥3,000–4,500 RMB (ECS + OSS)

**Config:** `IC-RAG-Agent/docker/docker-compose.yml`

### Why ClickHouse

- Columnar storage — 10-100x faster than PostgreSQL for analytics aggregations
- Handles 200M+ rows comfortably with proper partitioning
- Native time-series support (orders by date, inventory snapshots)
- Docker image `clickhouse/clickhouse-server` is production-grade
- Compatible with standard ClickHouse Python client (`clickhouse-driver`, `clickhouse-connect`)

---

## UDS Database Schema

### Design Principles

- All tables use `MergeTree` family engines
- Partition by month (`toYYYYMM(date_column)`) for query performance
- Sort keys tuned for the most common query patterns
- Single seller account, US marketplace (Amazon.com)
- Dates stored as `DateTime` (UTC)

---

### Table 1: `orders`

Core order-level data. One row per order.

```sql
CREATE TABLE uds.orders (
    order_id          String,
    purchase_date     DateTime,
    last_updated_date DateTime,
    order_status      LowCardinality(String),  -- PENDING, SHIPPED, DELIVERED, CANCELED, RETURNED
    fulfillment_channel LowCardinality(String), -- AFN (FBA), MFN (FBM)
    marketplace_id    LowCardinality(String),   -- ATVPDKIKX0DER (US)
    buyer_email       String,                   -- hashed for privacy
    order_total       Decimal(12, 2),
    currency_code     LowCardinality(String),   -- USD
    number_of_items   UInt16,
    is_business_order UInt8,                    -- 0/1
    is_prime          UInt8,                    -- 0/1
    ship_service_level LowCardinality(String),  -- Standard, Expedited, etc.
    created_at        DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(purchase_date)
ORDER BY (purchase_date, order_id)
SETTINGS index_granularity = 8192;
```

---

### Table 2: `order_items`

Line items per order. One row per SKU per order.

```sql
CREATE TABLE uds.order_items (
    order_item_id     String,
    order_id          String,
    purchase_date     DateTime,
    asin              String,
    seller_sku        String,
    title             String,
    quantity_ordered  UInt16,
    quantity_shipped  UInt16,
    item_price        Decimal(12, 2),
    item_tax          Decimal(12, 2),
    shipping_price    Decimal(12, 2),
    shipping_tax      Decimal(12, 2),
    promotion_discount Decimal(12, 2),
    condition_id      LowCardinality(String),   -- New, Used, etc.
    created_at        DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(purchase_date)
ORDER BY (purchase_date, order_id, order_item_id)
SETTINGS index_granularity = 8192;
```

---

### Table 3: `inventory`

Daily inventory snapshots. One row per SKU per day.

```sql
CREATE TABLE uds.inventory (
    snapshot_date         Date,
    seller_sku            String,
    asin                  String,
    fnsku                 String,                    -- FBA fulfillment network SKU
    condition             LowCardinality(String),    -- NewItem, UsedGood, etc.
    warehouse_condition   LowCardinality(String),
    fulfillable_quantity  UInt32,
    inbound_working       UInt32,                    -- Being prepped
    inbound_shipped       UInt32,                    -- In transit to FBA
    inbound_receiving     UInt32,                    -- At FBA, being received
    reserved_quantity     UInt32,                    -- Reserved for orders
    unfulfillable_quantity UInt32,                   -- Damaged/defective
    researching_quantity  UInt32,
    total_quantity        UInt32,
    days_of_supply        UInt16,                    -- Estimated days until stockout
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (snapshot_date, seller_sku)
SETTINGS index_granularity = 8192;
```

---

### Table 4: `products`

Product catalog. One row per SKU (slowly changing).

```sql
CREATE TABLE uds.products (
    seller_sku            String,
    asin                  String,
    fnsku                 String,
    title                 String,
    brand                 String,
    category              LowCardinality(String),
    subcategory           String,
    product_type          LowCardinality(String),
    condition             LowCardinality(String),
    status                LowCardinality(String),    -- Active, Inactive, Suppressed
    price                 Decimal(12, 2),
    sale_price            Decimal(12, 2),
    currency              LowCardinality(String),
    -- Physical dimensions (for FBA fee calculation)
    item_weight_kg        Float32,
    item_length_cm        Float32,
    item_width_cm         Float32,
    item_height_cm        Float32,
    -- FBA fee estimates
    fba_fulfillment_fee   Decimal(8, 2),
    referral_fee_rate     Float32,                   -- e.g. 0.15 for 15%
    -- Metadata
    listing_date          Date,
    last_updated          DateTime,
    created_at            DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (seller_sku)
SETTINGS index_granularity = 8192;
```

---

### Table 5: `shipments`

Inbound FBA shipments. One row per shipment.

```sql
CREATE TABLE uds.shipments (
    shipment_id           String,
    shipment_name         String,
    destination_fc        String,                    -- FBA fulfillment center code (e.g. PHX7)
    shipment_status       LowCardinality(String),    -- WORKING, SHIPPED, IN_TRANSIT, RECEIVING, CLOSED, CANCELLED
    label_prep_type       LowCardinality(String),    -- SELLER_LABEL, AMAZON_LABEL
    are_cases_required    UInt8,
    created_date          DateTime,
    last_updated_date     DateTime,
    confirmed_need_by_date Date,
    estimated_arrival_date Date,
    total_units           UInt32,
    total_skus            UInt16,
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_date)
ORDER BY (created_date, shipment_id)
SETTINGS index_granularity = 8192;
```

### Table 6: `shipment_items`

Items within each inbound shipment.

```sql
CREATE TABLE uds.shipment_items (
    shipment_id           String,
    seller_sku            String,
    fnsku                 String,
    asin                  String,
    condition             LowCardinality(String),
    quantity_shipped      UInt32,
    quantity_received     UInt32,
    quantity_in_case      UInt16,
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (shipment_id, seller_sku)
SETTINGS index_granularity = 8192;
```

---

### Table 7: `financials`

Settlement-level P&L. One row per transaction/settlement line.

```sql
CREATE TABLE uds.financials (
    settlement_id         String,
    transaction_type      LowCardinality(String),    -- Order, Refund, FBAFee, ReferralFee, StorageFee, etc.
    order_id              String,
    order_item_id         String,
    seller_sku            String,
    asin                  String,
    transaction_date      DateTime,
    settlement_start_date DateTime,
    settlement_end_date   DateTime,
    -- Revenue
    product_sales         Decimal(12, 2),
    product_sales_tax     Decimal(12, 2),
    shipping_credits      Decimal(12, 2),
    promotional_rebates   Decimal(12, 2),
    -- Fees
    selling_fees          Decimal(12, 2),            -- Referral fee
    fba_fees              Decimal(12, 2),            -- Fulfillment fee
    other_transaction_fees Decimal(12, 2),
    other                 Decimal(12, 2),
    -- Net
    total                 Decimal(12, 2),            -- Net amount for this line
    currency              LowCardinality(String),
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(transaction_date)
ORDER BY (transaction_date, settlement_id, order_id)
SETTINGS index_granularity = 8192;
```

---

## Common Analytics Queries (UDS Agent will generate these)

```sql
-- Daily revenue last 30 days
SELECT toDate(purchase_date) AS date, sum(order_total) AS revenue, count() AS orders
FROM uds.orders
WHERE purchase_date >= today() - 30 AND order_status NOT IN ('CANCELED')
GROUP BY date ORDER BY date;

-- Top 20 SKUs by revenue this month
SELECT oi.seller_sku, sum(oi.item_price * oi.quantity_ordered) AS revenue
FROM uds.order_items oi
WHERE toYYYYMM(oi.purchase_date) = toYYYYMM(today())
GROUP BY oi.seller_sku ORDER BY revenue DESC LIMIT 20;

-- Inventory health — SKUs at risk of stockout (< 14 days supply)
SELECT seller_sku, fulfillable_quantity, days_of_supply
FROM uds.inventory
WHERE snapshot_date = today() - 1 AND days_of_supply < 14
ORDER BY days_of_supply ASC;

-- Net profit by SKU (revenue - all fees)
SELECT seller_sku, sum(total) AS net_profit
FROM uds.financials
WHERE transaction_date >= today() - 90
GROUP BY seller_sku ORDER BY net_profit DESC;
```

---

## Implementation Plan

### Week 0 (Pre-work, can start NOW — parallel with Part 1)

**Goal:** Provision cloud infrastructure and create database schema

- [ ] Provision ECS instance on Alibaba Cloud cn-hangzhou (16C/64GB/1TB)
- [ ] Install Docker + Docker Compose on ECS
- [ ] Deploy `docker/docker-compose.yml` — starts ClickHouse, Redis, ChromaDB
- [ ] Create OSS bucket for backups and report storage
- [ ] Create `uds` database and all 7 tables (DDL scripts)
- [ ] Configure VPC, security groups (open ports 8123, 9000, 6379, 8000 to app only)
- [ ] Store credentials in `.env` (never commit to git)
- [ ] Write Python connection test script
- [ ] Generate seed data (1M rows per table for dev testing)

**Deliverables:**
- `IC-RAG-Agent/docker/docker-compose.yml` — all 3 services
- `IC-RAG-Agent/scripts/uds/create_tables.sql` — DDL for all 7 tables
- `IC-RAG-Agent/scripts/uds/seed_data.py` — mock data generator
- `IC-RAG-Agent/scripts/uds/test_connection.py` — connection verification
- `IC-RAG-Agent/config/cloud.env.example` — connection config template

---

### Week 1–2: UDS Database Client

**Goal:** Python client for ClickHouse with connection pooling, auth, streaming

- [ ] Implement `UDSClient` class (`src/uds/uds_client.py`)
  - Connection pooling (clickhouse-connect or clickhouse-driver)
  - Authentication from environment variables
  - Query execution with timeout (default 30s)
  - Streaming results for large queries
  - Schema introspection (list tables, describe columns)
  - Parameterized queries (prevent SQL injection)
  - Retry with exponential backoff on connection failure
- [ ] Unit tests with mock ClickHouse responses
- [ ] Integration test against real Alibaba Cloud instance

---

### Week 3–4: UDS Query Tools

**Goal:** 8 tools inheriting from ai-toolkit BaseTool

| Tool | Description |
|------|-------------|
| `UDSSQLQueryTool` | Execute SQL against UDS, return results |
| `UDSSchemaInspectorTool` | List tables, describe columns |
| `DataAggregationTool` | Group/sum/stats on query results |
| `TrendAnalysisTool` | Time-series trend detection |
| `ComparisonReportTool` | Compare metrics across periods |
| `VisualizationTool` | Generate charts (Matplotlib/Plotly) |
| `ReportGeneratorTool` | Compile Markdown + PDF reports |
| `DataExportTool` | Export to CSV/Excel/JSON → OSS |

---

### Week 5–6: Task Planner + Data Analysis Agent

**Goal:** Agent that decomposes complex queries and executes multi-step analysis

- [ ] `TaskPlanner` — decompose natural language query into subtasks
- [ ] `DataAnalysisAgent` — ReActAgent subclass with UDS tools
- [ ] Integration with IC-RAG for schema documentation retrieval
- [ ] Conversation memory via Redis

---

### Week 7: Report Generation + API

**Goal:** Professional reports and REST API

- [ ] Markdown + PDF report generation (ReportLab)
- [ ] Report storage to Alibaba Cloud OSS
- [ ] FastAPI endpoints for the Data Analysis Agent
- [ ] Scheduled report generation

---

### Week 8: Testing + Documentation

- [ ] 80%+ test coverage
- [ ] Property-based tests (hypothesis)
- [ ] Integration tests against Alibaba Cloud
- [ ] API documentation
- [ ] UDS schema documentation

---

## File Structure

```
IC-RAG-Agent/
├── docker/
│   └── docker-compose.yml       ← ClickHouse + Redis + ChromaDB
├── scripts/uds/
│   ├── create_tables.sql        ← DDL for all 7 tables
│   ├── seed_data.py             ← Mock data generator
│   └── test_connection.py       ← Connection verification
├── config/
│   └── cloud.env.example        ← Connection config template
└── src/uds/
    ├── __init__.py
    ├── uds_client.py            ← ClickHouse client
    ├── task_planner.py          ← Query decomposition
    ├── uds_agent.py             ← Main agent
    ├── tools/
    │   ├── __init__.py
    │   ├── sql_query.py
    │   ├── schema_inspector.py
    │   ├── aggregation.py
    │   ├── trend_analysis.py
    │   ├── comparison_report.py
    │   ├── visualization.py
    │   ├── report_generator.py
    │   └── data_export.py
    └── tests/
        ├── __init__.py
        ├── test_uds_client.py
        ├── test_tools.py
        ├── test_task_planner.py
        └── test_properties.py
```

---

## Python Dependencies to Add

```txt
clickhouse-connect>=0.6.0    # ClickHouse Python client (recommended)
# OR
clickhouse-driver>=0.2.6     # Alternative driver

reportlab>=4.0.0             # PDF generation
plotly>=5.0.0                # Interactive charts
matplotlib>=3.7.0            # Static charts
openpyxl>=3.1.0              # Excel export
oss2>=2.18.0                 # Alibaba Cloud OSS SDK
```

---

## Connection Configuration

```env
# .env (never commit to git)
CLICKHOUSE_HOST=<your-ecs-ip>
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
CLICKHOUSE_DATABASE=uds
CLICKHOUSE_SECURE=false

REDIS_HOST=<your-ecs-ip>
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

CHROMA_HOST=<your-ecs-ip>
CHROMA_PORT=8000

OSS_ACCESS_KEY_ID=your_key
OSS_ACCESS_KEY_SECRET=your_secret
OSS_BUCKET=ic-rag-uds-reports
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
```

---

## Next Steps

1. **Immediate (Week 0):** Provision Alibaba Cloud services, run DDL scripts, verify connection
2. **After Part 1 completes:** Begin Week 1–2 UDS Client implementation
3. **Parallel opportunity:** DDL scripts and seed data can be written now by any team member

---

**Document Owner:** Kiro (AI Project Manager)  
**Last Updated:** 2026-03-03
