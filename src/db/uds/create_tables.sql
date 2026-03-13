-- UDS Database Schema
-- ClickHouse DDL — run on ApsaraDB for ClickHouse (Alibaba Cloud)
-- Region: cn-hangzhou or cn-shanghai
-- Database: uds

CREATE DATABASE IF NOT EXISTS uds;

-- ============================================================
-- Table 1: orders
-- One row per Amazon order
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.orders (
    order_id              String,
    purchase_date         DateTime,
    last_updated_date     DateTime,
    order_status          LowCardinality(String),      -- PENDING, SHIPPED, DELIVERED, CANCELED, RETURNED
    fulfillment_channel   LowCardinality(String),      -- AFN (FBA), MFN (FBM)
    marketplace_id        LowCardinality(String),      -- ATVPDKIKX0DER (US)
    buyer_email_hash      String,                      -- SHA256 hashed for privacy
    order_total           Decimal(12, 2),
    currency_code         LowCardinality(String),      -- USD
    number_of_items       UInt16,
    is_business_order     UInt8,                       -- 0/1
    is_prime              UInt8,                       -- 0/1
    ship_service_level    LowCardinality(String),      -- Standard, Expedited, Priority
    sales_channel         LowCardinality(String),      -- Amazon.com
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(purchase_date)
ORDER BY (purchase_date, order_id)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Table 2: order_items
-- One row per SKU per order
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.order_items (
    order_item_id         String,
    order_id              String,
    purchase_date         DateTime,
    asin                  String,
    seller_sku            String,
    title                 String,
    quantity_ordered      UInt16,
    quantity_shipped      UInt16,
    item_price            Decimal(12, 2),
    item_tax              Decimal(12, 2),
    shipping_price        Decimal(12, 2),
    shipping_tax          Decimal(12, 2),
    promotion_discount    Decimal(12, 2),
    cod_fee               Decimal(12, 2),
    condition_id          LowCardinality(String),      -- New, UsedGood, UsedAcceptable
    condition_note        String,
    created_at            DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(purchase_date)
ORDER BY (purchase_date, order_id, order_item_id)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Table 3: inventory
-- Daily inventory snapshots per SKU
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.inventory (
    snapshot_date             Date,
    seller_sku                String,
    asin                      String,
    fnsku                     String,
    condition                 LowCardinality(String),  -- NewItem, UsedGood, etc.
    warehouse_condition       LowCardinality(String),
    fulfillable_quantity      UInt32,
    inbound_working           UInt32,
    inbound_shipped           UInt32,
    inbound_receiving         UInt32,
    reserved_fc_transfers     UInt32,
    reserved_fc_processing    UInt32,
    reserved_customer_orders  UInt32,
    unfulfillable_quantity    UInt32,
    researching_quantity      UInt32,
    total_quantity            UInt32,
    days_of_supply            UInt16,
    created_at                DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (snapshot_date, seller_sku)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Table 4: products
-- Product catalog (slowly changing)
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.products (
    seller_sku            String,
    asin                  String,
    fnsku                 String,
    title                 String,
    brand                 String,
    category              LowCardinality(String),
    subcategory           String,
    product_type          LowCardinality(String),
    condition             LowCardinality(String),
    status                LowCardinality(String),      -- Active, Inactive, Suppressed
    price                 Decimal(12, 2),
    sale_price            Decimal(12, 2),
    currency              LowCardinality(String),
    -- Physical dimensions
    item_weight_kg        Float32,
    item_length_cm        Float32,
    item_width_cm         Float32,
    item_height_cm        Float32,
    -- FBA fee estimates
    fba_fulfillment_fee   Decimal(8, 2),
    referral_fee_rate     Float32,
    -- Metadata
    listing_date          Date,
    last_updated          DateTime,
    created_at            DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (seller_sku)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Table 5: shipments
-- Inbound FBA shipments
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.shipments (
    shipment_id               String,
    shipment_name             String,
    destination_fc            String,                  -- e.g. PHX7, LAX9
    shipment_status           LowCardinality(String),  -- WORKING, SHIPPED, IN_TRANSIT, RECEIVING, CLOSED, CANCELLED
    label_prep_type           LowCardinality(String),  -- SELLER_LABEL, AMAZON_LABEL, NO_LABEL
    are_cases_required        UInt8,
    created_date              DateTime,
    last_updated_date         DateTime,
    confirmed_need_by_date    Date,
    estimated_arrival_date    Date,
    total_units               UInt32,
    total_skus                UInt16,
    created_at                DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_date)
ORDER BY (created_date, shipment_id)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Table 6: shipment_items
-- Items within each inbound shipment
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.shipment_items (
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

-- ============================================================
-- Table 7: financials
-- Settlement-level P&L transactions
-- ============================================================
CREATE TABLE IF NOT EXISTS uds.financials (
    settlement_id             String,
    transaction_type          LowCardinality(String),  -- Order, Refund, FBAFee, ReferralFee, StorageFee, Adjustment
    order_id                  String,
    order_item_id             String,
    seller_sku                String,
    asin                      String,
    transaction_date          DateTime,
    settlement_start_date     DateTime,
    settlement_end_date       DateTime,
    -- Revenue components
    product_sales             Decimal(12, 2),
    product_sales_tax         Decimal(12, 2),
    shipping_credits          Decimal(12, 2),
    shipping_credits_tax      Decimal(12, 2),
    gift_wrap_credits         Decimal(12, 2),
    promotional_rebates       Decimal(12, 2),
    promotional_rebates_tax   Decimal(12, 2),
    -- Fee components
    marketplace_facilitator_tax Decimal(12, 2),
    selling_fees              Decimal(12, 2),          -- Referral fee
    fba_fees                  Decimal(12, 2),          -- Fulfillment fee
    other_transaction_fees    Decimal(12, 2),
    other                     Decimal(12, 2),
    -- Net
    total                     Decimal(12, 2),
    currency                  LowCardinality(String),
    created_at                DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(transaction_date)
ORDER BY (transaction_date, settlement_id, order_id)
SETTINGS index_granularity = 8192;

-- ============================================================
-- Useful views for common queries
-- ============================================================

-- Daily revenue summary
CREATE VIEW IF NOT EXISTS uds.v_daily_revenue AS
SELECT
    toDate(purchase_date) AS date,
    count() AS total_orders,
    sum(order_total) AS gross_revenue,
    countIf(order_status = 'CANCELED') AS canceled_orders,
    sumIf(order_total, order_status = 'CANCELED') AS canceled_revenue
FROM uds.orders
GROUP BY date;

-- SKU performance summary (last 90 days)
CREATE VIEW IF NOT EXISTS uds.v_sku_performance AS
SELECT
    oi.seller_sku,
    oi.asin,
    sum(oi.quantity_ordered) AS units_sold,
    sum(oi.item_price) AS gross_revenue,
    sum(f.fba_fees) AS total_fba_fees,
    sum(f.selling_fees) AS total_referral_fees,
    sum(f.total) AS net_revenue
FROM uds.order_items oi
LEFT JOIN uds.financials f ON oi.order_id = f.order_id AND oi.seller_sku = f.seller_sku
WHERE oi.purchase_date >= today() - 90
GROUP BY oi.seller_sku, oi.asin;

-- Current inventory status
CREATE VIEW IF NOT EXISTS uds.v_current_inventory AS
SELECT
    i.seller_sku,
    i.asin,
    i.fulfillable_quantity,
    i.inbound_shipped + i.inbound_receiving AS inbound_quantity,
    i.unfulfillable_quantity,
    i.days_of_supply,
    p.title,
    p.price
FROM uds.inventory i
LEFT JOIN uds.products p ON i.seller_sku = p.seller_sku
WHERE i.snapshot_date = (SELECT max(snapshot_date) FROM uds.inventory);
