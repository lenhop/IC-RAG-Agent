-- UDS Database Initialization Script
-- This script creates the UDS database and tables for the IC-RAG-Agent

-- Create UDS database
CREATE DATABASE IF NOT EXISTS uds;

-- Use UDS database
USE uds;

-- Create tables based on Amazon data schema
-- Note: These are simplified schemas for demonstration
-- In production, use the full schemas from IC-Data-Loader

-- Daily Inventory Ledger
CREATE TABLE IF NOT EXISTS amz_daily_inventory_ledger (
    date Date,
    fnsku String,
    asin String,
    product_name String,
    quantity Int32,
    fulfillment_channel String,
    disposition String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, fnsku, asin)
PARTITION BY toYYYYMM(date);

-- FBA Inventory All
CREATE TABLE IF NOT EXISTS amz_fba_inventory_all (
    date Date,
    fnsku String,
    asin String,
    product_name String,
    quantity Int32,
    fulfillment_center_id String,
    detailed_disposition String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, fnsku, asin)
PARTITION BY toYYYYMM(date);

-- Fee Data
CREATE TABLE IF NOT EXISTS amz_fee (
    date Date,
    fnsku String,
    asin String,
    product_name String,
    fee_type String,
    fee_amount Decimal(10,2),
    currency String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, fnsku, fee_type)
PARTITION BY toYYYYMM(date);

-- Listing Items
CREATE TABLE IF NOT EXISTS amz_listing_item (
    date Date,
    asin String,
    fnsku String,
    product_name String,
    product_category String,
    quantity Int32,
    price Decimal(10,2),
    currency String,
    fulfillment_channel String,
    status String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, asin, fnsku)
PARTITION BY toYYYYMM(date);

-- Monthly Search Terms
CREATE TABLE IF NOT EXISTS amz_monthly_search_term (
    date Date,
    search_term String,
    search_frequency_rank Int32,
    asin String,
    product_name String,
    clicks Int32,
    impressions Int32,
    ctr Decimal(5,4),
    country String
) ENGINE = MergeTree()
ORDER BY (date, search_term, search_frequency_rank)
PARTITION BY toYYYYMM(date);

-- Orders
CREATE TABLE IF NOT EXISTS amz_order (
    date Date,
    order_id String,
    asin String,
    fnsku String,
    product_name String,
    quantity Int32,
    item_price Decimal(10,2),
    item_tax Decimal(10,2),
    shipping_price Decimal(10,2),
    shipping_tax Decimal(10,2),
    currency String,
    fulfillment_channel String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, order_id, asin)
PARTITION BY toYYYYMM(date);

-- Products
CREATE TABLE IF NOT EXISTS amz_product (
    asin String,
    product_name String,
    product_category String,
    brand String,
    manufacturer String,
    bullet_point_1 String,
    bullet_point_2 String,
    bullet_point_3 String,
    bullet_point_4 String,
    bullet_point_5 String,
    product_description String,
    country String
) ENGINE = MergeTree()
ORDER BY asin;

-- Statements
CREATE TABLE IF NOT EXISTS amz_statement (
    date Date,
    transaction_type String,
    order_id String,
    marketplace String,
    amount_type String,
    amount_description String,
    amount Decimal(10,2),
    currency String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, transaction_type, order_id)
PARTITION BY toYYYYMM(date);

-- Transactions
CREATE TABLE IF NOT EXISTS amz_transaction (
    date Date,
    transaction_type String,
    order_id String,
    marketplace String,
    amount_type String,
    amount_description String,
    amount Decimal(10,2),
    currency String,
    country String
) ENGINE = MergeTree()
ORDER BY (date, transaction_type, order_id)
PARTITION BY toYYYYMM(date);

-- Create indexes for better performance
-- These are secondary indexes for common query patterns

-- Index for inventory queries by ASIN
CREATE INDEX IF NOT EXISTS idx_inventory_asin ON amz_daily_inventory_ledger (asin) TYPE minmax GRANULARITY 1;
CREATE INDEX IF NOT EXISTS idx_fba_asin ON amz_fba_inventory_all (asin) TYPE minmax GRANULARITY 1;

-- Index for order queries by date range
CREATE INDEX IF NOT EXISTS idx_orders_date ON amz_order (date) TYPE minmax GRANULARITY 1;

-- Index for fee queries by fee type
CREATE INDEX IF NOT EXISTS idx_fee_type ON amz_fee (fee_type) TYPE bloom_filter(0.01) GRANULARITY 1;

-- Index for search terms
CREATE INDEX IF NOT EXISTS idx_search_term ON amz_monthly_search_term (search_term) TYPE bloom_filter(0.01) GRANULARITY 1;

-- Index for product lookups
CREATE INDEX IF NOT EXISTS idx_product_asin ON amz_product (asin) TYPE bloom_filter(0.01) GRANULARITY 1;

-- Create a default user for the application
-- Note: In production, use proper authentication
CREATE USER IF NOT EXISTS uds_user IDENTIFIED WITH plaintext_password BY '';
GRANT ALL ON uds.* TO uds_user;

-- Auth: ic_agent database and user table (for login/register)
CREATE DATABASE IF NOT EXISTS ic_agent;

CREATE TABLE IF NOT EXISTS ic_agent.ic_rag_agent_user (
    user_id UUID,
    user_name String,
    email String DEFAULT '',
    password_hash String,
    role LowCardinality(String) DEFAULT 'general',
    status LowCardinality(String) DEFAULT 'active',
    created_time DateTime64(3),
    updated_time DateTime64(3),
    last_login_time Nullable(DateTime64(3)),
    last_login_ip Nullable(String),
    metadata String DEFAULT '{}'
) ENGINE = ReplacingMergeTree(updated_time)
ORDER BY (user_id);

-- Index for user_name and email lookups
CREATE INDEX IF NOT EXISTS idx_user_name ON ic_agent.ic_rag_agent_user (user_name) TYPE bloom_filter(0.01) GRANULARITY 1;
CREATE INDEX IF NOT EXISTS idx_user_email ON ic_agent.ic_rag_agent_user (email) TYPE bloom_filter(0.01) GRANULARITY 1;

GRANT ALL ON ic_agent.* TO uds_user;