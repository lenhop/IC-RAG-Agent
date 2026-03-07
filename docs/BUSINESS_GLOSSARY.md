# UDS Business Glossary

## Amazon-Specific Terms

| Term | Definition |
|------|------------|
| **ASIN** | Amazon Standard Identification Number. Unique 10-character alphanumeric identifier for products in Amazon catalog. |
| **SKU** | Stock Keeping Unit. Merchant-defined unique identifier for a product. |
| **FNSKU** | Fulfillment Network Stock Keeping Unit. Barcode used by Amazon FBA to identify inventory. |
| **FBA** | Fulfillment by Amazon. Seller ships inventory to Amazon; Amazon handles storage, packing, shipping. |
| **FBM** | Fulfillment by Merchant. Seller handles storage and shipping. |
| **AFN** | Amazon Fulfillment Network. Same as FBA. |
| **MFN** | Merchant Fulfillment Network. Same as FBM. |
| **Settlement** | Periodic (typically 14-day) payment cycle when Amazon pays sellers for orders. |
| **Settlement ID** | Unique identifier for a settlement statement. |

## Business Metrics

| Metric | Definition | Typical SQL |
|--------|------------|------------|
| **GMV** | Gross Merchandise Value. Total sales before fees/refunds. | `SUM(item_price)` from amz_order |
| **AOV** | Average Order Value. | `AVG(total_amount)` or `SUM(item_price)/COUNT(DISTINCT amazon_order_id)` |
| **Conversion rate** | Orders / Sessions (sessions not in UDS). | N/A |
| **Inventory turnover** | How often inventory is sold and replaced. | Movements from amz_daily_inventory_ledger |
| **FBA storage fee** | Fee for storing inventory in Amazon warehouses. | From amz_fee |
| **Referral fee** | Percentage fee on each sale. | From amz_fee |

## Common Business Questions Mapped to Tables

| Question | Primary Table(s) | Key Columns |
|----------|------------------|-------------|
| What were total sales in October? | amz_order | start_date, item_price |
| How many orders per day? | amz_order | start_date, amazon_order_id |
| What are the top-selling SKUs? | amz_order | sku, item_price, quantity |
| What fees did we pay? | amz_fee | fee_type, amount |
| What is our current FBA inventory? | amz_fba_inventory_all | sku, fnsku, quantity |
| What inventory movements occurred? | amz_daily_inventory_ledger | fnsku, quantity, transaction_type |
| What is the product catalog? | amz_listing_item, amz_product | sku, ASIN, item_name |
| How do orders reconcile to settlements? | amz_transaction, amz_statement | settlement_id, amount |
| What search terms drove traffic? | amz_monthly_search_term | search_term, clicks, conversions |

## Table Quick Reference

| Table | Purpose |
|-------|---------|
| amz_order | Customer orders |
| amz_transaction | Financial transactions (payments, refunds) |
| amz_statement | Settlement statements |
| amz_fee | Amazon fees (FBA, referral, etc.) |
| amz_fba_inventory_all | FBA inventory snapshots |
| amz_daily_inventory_ledger | Daily inventory in/out movements |
| amz_listing_item | Product listing catalog (snapshots) |
| amz_product | Product master (ASIN-level) |
| amz_monthly_search_term | Search term performance |
