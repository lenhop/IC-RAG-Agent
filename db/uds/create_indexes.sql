-- ClickHouse Index Creation Script
-- This file creates secondary indexes on key columns to improve query
-- performance. It includes minmax indexes for date columns and bloom
-- filter indexes for high-cardinality ID columns. Adjust table names,
-- columns and settings based on actual schema and usage patterns.

-- NOTE: run as a ClickHouse SQL script, e.g.
-- clickhouse-client --multiquery < db/uds/create_indexes.sql

-- amz_order table: queries often filter by start_date, amazon_order_id, sku
ALTER TABLE ic_agent.amz_order
    ADD INDEX idx_order_start_date (start_date) TYPE minmax GRANULARITY 8;

ALTER TABLE ic_agent.amz_order
    ADD INDEX idx_order_asin (asin) TYPE bloom_filter(0.01) GRANULARITY 1;

ALTER TABLE ic_agent.amz_order
    ADD INDEX idx_order_sku (sku) TYPE bloom_filter(0.01) GRANULARITY 1;

-- amz_transaction table: often filtered by transaction_date, amazon_order_id
ALTER TABLE ic_agent.amz_transaction
    ADD INDEX idx_trans_date (transaction_date) TYPE minmax GRANULARITY 8;

ALTER TABLE ic_agent.amz_transaction
    ADD INDEX idx_trans_order_id (amazon_order_id) TYPE bloom_filter(0.01) GRANULARITY 1;

-- amz_fba_inventory_all: queries filter by start_date, sku, asin
ALTER TABLE ic_agent.amz_fba_inventory_all
    ADD INDEX idx_inv_start_date (start_date) TYPE minmax GRANULARITY 8;

ALTER TABLE ic_agent.amz_fba_inventory_all
    ADD INDEX idx_inv_sku (sku) TYPE bloom_filter(0.01) GRANULARITY 1;

ALTER TABLE ic_agent.amz_fba_inventory_all
    ADD INDEX idx_inv_asin (asin) TYPE bloom_filter(0.01) GRANULARITY 1;

-- amz_fee: filter by start_date, asin, sku
ALTER TABLE ic_agent.amz_fee
    ADD INDEX idx_fee_start_date (start_date) TYPE minmax GRANULARITY 8;

ALTER TABLE ic_agent.amz_fee
    ADD INDEX idx_fee_asin (asin) TYPE bloom_filter(0.01) GRANULARITY 1;

-- amz_product: filter by ASIN
ALTER TABLE ic_agent.amz_product
    ADD INDEX idx_prod_asin (ASIN) TYPE bloom_filter(0.01) GRANULARITY 1;

-- add additional indexes for join optimization
-- example: join amz_order and amz_product on asin
ALTER TABLE ic_agent.amz_order
    ADD INDEX idx_order_asin_join (asin) TYPE bloom_filter(0.01) GRANULARITY 1;

-- Example minmax index on amz_order.purchase_date for analytical scans
ALTER TABLE ic_agent.amz_order
    ADD INDEX idx_order_purchase_date (purchase_date) TYPE minmax GRANULARITY 8;

-- You can extend indexes below as needed
-- END OF SCRIPT
