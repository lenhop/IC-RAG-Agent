"""
UDS Seed Data Generator
Generates realistic mock Amazon seller data for development/testing.

Usage:
    python tools/uds/seed_data.py --rows 100000 --host localhost

Requirements:
    pip install clickhouse-connect faker
"""

import argparse
import hashlib
import random
from datetime import date, datetime, timedelta

import clickhouse_connect

CATEGORIES = [
    "Electronics",
    "Home & Kitchen",
    "Sports & Outdoors",
    "Toys & Games",
    "Health & Personal Care",
    "Beauty",
    "Office Products",
    "Pet Supplies",
]

STATUSES = ["PENDING", "SHIPPED", "DELIVERED", "CANCELED", "RETURNED"]
STATUS_WEIGHTS = [0.05, 0.15, 0.70, 0.07, 0.03]

FC_CODES = ["PHX7", "LAX9", "JFK8", "ORD2", "DFW7", "ATL6", "SEA8", "BOS5"]


def random_date(start_days_ago: int = 730, end_days_ago: int = 0) -> datetime:
    """Generate random datetime in a configurable window."""
    delta = random.randint(end_days_ago, start_days_ago)
    return datetime.utcnow() - timedelta(
        days=delta,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def random_asin() -> str:
    """Generate random ASIN-like identifier."""
    return "B" + "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=9))


def random_sku(asin: str) -> str:
    """Generate random SKU from ASIN."""
    return f"SKU-{asin[-6:]}-{random.randint(1, 99):02d}"


def hash_email(email: str) -> str:
    """Hash email for privacy-safe synthetic data."""
    return hashlib.sha256(email.encode()).hexdigest()[:16]


def random_price(low: float = 9.99, high: float = 199.99) -> float:
    """Generate random price in configured range."""
    return round(random.uniform(low, high), 2)


def generate_products(n: int = 500) -> list[dict]:
    """Generate product catalog rows."""
    products = []
    for _ in range(n):
        asin = random_asin()
        sku = random_sku(asin)
        price = random_price()
        weight = round(random.uniform(0.1, 5.0), 2)
        fba_fee = round(weight * 1.5 + 2.5, 2)
        products.append(
            {
                "seller_sku": sku,
                "asin": asin,
                "fnsku": "X" + asin[1:],
                "title": f"Product {asin[-4:]} - {random.choice(CATEGORIES)}",
                "brand": f"Brand{random.randint(1, 50)}",
                "category": random.choice(CATEGORIES),
                "subcategory": f"Sub{random.randint(1, 10)}",
                "product_type": "STANDARD",
                "condition": "New",
                "status": random.choices(["Active", "Inactive", "Suppressed"], weights=[0.85, 0.10, 0.05])[0],
                "price": price,
                "sale_price": round(price * 0.9, 2) if random.random() < 0.2 else price,
                "currency": "USD",
                "item_weight_kg": weight,
                "item_length_cm": round(random.uniform(5, 50), 1),
                "item_width_cm": round(random.uniform(5, 40), 1),
                "item_height_cm": round(random.uniform(2, 30), 1),
                "fba_fulfillment_fee": fba_fee,
                "referral_fee_rate": random.choice([0.08, 0.12, 0.15, 0.17]),
                "listing_date": (datetime.utcnow() - timedelta(days=random.randint(30, 1000))).date(),
                "last_updated": datetime.utcnow() - timedelta(days=random.randint(0, 30)),
            }
        )
    return products


def generate_orders_and_items(products: list[dict], n_orders: int = 100000):
    """Generate order and order-item rows."""
    orders = []
    order_items = []

    for i in range(n_orders):
        order_id = f"114-{random.randint(1000000, 9999999)}-{random.randint(1000000, 9999999)}"
        purchase_date = random_date(730, 0)
        n_items = random.choices([1, 2, 3, 4], weights=[0.70, 0.20, 0.07, 0.03])[0]
        selected_products = random.sample(products, min(n_items, len(products)))

        order_total = 0.0
        for prod in selected_products:
            qty = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
            item_price = round(prod["price"] * qty, 2)
            order_total += item_price
            order_items.append(
                {
                    "order_item_id": f"OI{random.randint(10000000, 99999999)}",
                    "order_id": order_id,
                    "purchase_date": purchase_date,
                    "asin": prod["asin"],
                    "seller_sku": prod["seller_sku"],
                    "title": prod["title"],
                    "quantity_ordered": qty,
                    "quantity_shipped": qty,
                    "item_price": item_price,
                    "item_tax": round(item_price * 0.08, 2),
                    "shipping_price": 0.0,
                    "shipping_tax": 0.0,
                    "promotion_discount": 0.0,
                    "cod_fee": 0.0,
                    "condition_id": "New",
                    "condition_note": "",
                }
            )

        status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        orders.append(
            {
                "order_id": order_id,
                "purchase_date": purchase_date,
                "last_updated_date": purchase_date + timedelta(days=random.randint(1, 7)),
                "order_status": status,
                "fulfillment_channel": random.choices(["AFN", "MFN"], weights=[0.85, 0.15])[0],
                "marketplace_id": "ATVPDKIKX0DER",
                "buyer_email_hash": hash_email(f"buyer{i}@example.com"),
                "order_total": round(order_total, 2),
                "currency_code": "USD",
                "number_of_items": len(selected_products),
                "is_business_order": 0,
                "is_prime": random.choices([0, 1], weights=[0.3, 0.7])[0],
                "ship_service_level": random.choices(["Standard", "Expedited", "Priority"], weights=[0.6, 0.3, 0.1])[0],
                "sales_channel": "Amazon.com",
            }
        )

    return orders, order_items


def generate_inventory(products: list[dict], days: int = 90) -> list[dict]:
    """Generate inventory snapshot rows."""
    rows = []
    today = date.today()
    for day_offset in range(days):
        snapshot_date = today - timedelta(days=day_offset)
        for prod in products:
            fulfillable = random.randint(0, 500)
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "seller_sku": prod["seller_sku"],
                    "asin": prod["asin"],
                    "fnsku": prod["fnsku"],
                    "condition": "NewItem",
                    "warehouse_condition": "SELLABLE",
                    "fulfillable_quantity": fulfillable,
                    "inbound_working": random.randint(0, 50),
                    "inbound_shipped": random.randint(0, 100),
                    "inbound_receiving": random.randint(0, 30),
                    "reserved_fc_transfers": random.randint(0, 10),
                    "reserved_fc_processing": random.randint(0, 5),
                    "reserved_customer_orders": random.randint(0, 20),
                    "unfulfillable_quantity": random.randint(0, 5),
                    "researching_quantity": 0,
                    "total_quantity": fulfillable + random.randint(0, 50),
                    "days_of_supply": random.randint(0, 120),
                }
            )
    return rows


def generate_shipments(products: list[dict], n: int = 200):
    """Generate inbound shipment and shipment-item rows."""
    shipments = []
    shipment_items = []
    for i in range(n):
        shipment_id = f"FBA{random.randint(10000000, 99999999)}XYZ"
        created = random_date(365, 0)
        selected = random.sample(products, random.randint(1, 10))
        total_units = 0
        for prod in selected:
            qty = random.randint(10, 200)
            total_units += qty
            shipment_items.append(
                {
                    "shipment_id": shipment_id,
                    "seller_sku": prod["seller_sku"],
                    "fnsku": prod["fnsku"],
                    "asin": prod["asin"],
                    "condition": "NewItem",
                    "quantity_shipped": qty,
                    "quantity_received": qty if random.random() > 0.1 else int(qty * 0.95),
                    "quantity_in_case": random.choice([0, 6, 12, 24]),
                }
            )
        shipments.append(
            {
                "shipment_id": shipment_id,
                "shipment_name": f"Shipment {i+1:04d}",
                "destination_fc": random.choice(FC_CODES),
                "shipment_status": random.choices(
                    ["WORKING", "SHIPPED", "IN_TRANSIT", "RECEIVING", "CLOSED", "CANCELLED"],
                    weights=[0.05, 0.10, 0.15, 0.10, 0.55, 0.05],
                )[0],
                "label_prep_type": random.choice(["SELLER_LABEL", "AMAZON_LABEL"]),
                "are_cases_required": random.choice([0, 1]),
                "created_date": created,
                "last_updated_date": created + timedelta(days=random.randint(1, 30)),
                "confirmed_need_by_date": (created + timedelta(days=14)).date(),
                "estimated_arrival_date": (created + timedelta(days=21)).date(),
                "total_units": total_units,
                "total_skus": len(selected),
            }
        )
    return shipments, shipment_items


def generate_financials(orders: list[dict], order_items: list[dict]) -> list[dict]:
    """Generate financial rows from orders/order-items."""
    rows = []
    items_by_order = {}
    for item in order_items:
        items_by_order.setdefault(item["order_id"], []).append(item)

    settlement_id = f"SETTLE{random.randint(100000, 999999)}"
    for order in orders:
        if order["order_status"] == "CANCELED":
            continue
        for item in items_by_order.get(order["order_id"], []):
            product_sales = float(item["item_price"])
            fba_fee = round(product_sales * 0.12, 2)
            referral_fee = round(product_sales * 0.15, 2)
            rows.append(
                {
                    "settlement_id": settlement_id,
                    "transaction_type": "Order",
                    "order_id": order["order_id"],
                    "order_item_id": item["order_item_id"],
                    "seller_sku": item["seller_sku"],
                    "asin": item["asin"],
                    "transaction_date": order["purchase_date"],
                    "settlement_start_date": order["purchase_date"],
                    "settlement_end_date": order["purchase_date"] + timedelta(days=14),
                    "product_sales": product_sales,
                    "product_sales_tax": float(item["item_tax"]),
                    "shipping_credits": 0.0,
                    "shipping_credits_tax": 0.0,
                    "gift_wrap_credits": 0.0,
                    "promotional_rebates": 0.0,
                    "promotional_rebates_tax": 0.0,
                    "marketplace_facilitator_tax": 0.0,
                    "selling_fees": -referral_fee,
                    "fba_fees": -fba_fee,
                    "other_transaction_fees": 0.0,
                    "other": 0.0,
                    "total": round(product_sales - fba_fee - referral_fee, 2),
                    "currency": "USD",
                }
            )
    return rows


def insert_batch(client, table: str, rows: list[dict], batch_size: int = 10000):
    """Insert rows in batches for better throughput and lower memory pressure."""
    if not rows:
        return
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        client.insert(table, batch, column_names=list(batch[0].keys()))
        print(f"  Inserted {min(i + batch_size, len(rows))}/{len(rows)} rows into {table}")


def main():
    """CLI entrypoint for seed data generation."""
    parser = argparse.ArgumentParser(description="Generate UDS seed data")
    parser.add_argument("--host", default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=8443, help="ClickHouse port")
    parser.add_argument("--user", default="default", help="ClickHouse user")
    parser.add_argument("--password", default="", help="ClickHouse password")
    parser.add_argument("--database", default="uds", help="ClickHouse database")
    parser.add_argument("--secure", action="store_true", help="Use TLS")
    parser.add_argument("--products", type=int, default=500, help="Number of products")
    parser.add_argument("--orders", type=int, default=100000, help="Number of orders")
    parser.add_argument("--inventory-days", type=int, default=90, help="Days of inventory snapshots")
    parser.add_argument("--shipments", type=int, default=200, help="Number of shipments")
    args = parser.parse_args()

    print(f"Connecting to ClickHouse at {args.host}:{args.port}...")
    client = clickhouse_connect.get_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        database=args.database,
        secure=args.secure,
    )
    print("Connected.")

    print(f"\nGenerating {args.products} products...")
    products = generate_products(args.products)
    insert_batch(client, "uds.products", products)

    print(f"\nGenerating {args.orders} orders + items...")
    orders, order_items = generate_orders_and_items(products, args.orders)
    insert_batch(client, "uds.orders", orders)
    insert_batch(client, "uds.order_items", order_items)

    print(f"\nGenerating {args.inventory_days} days of inventory snapshots ({args.products * args.inventory_days} rows)...")
    inventory = generate_inventory(products, args.inventory_days)
    insert_batch(client, "uds.inventory", inventory)

    print(f"\nGenerating {args.shipments} shipments...")
    shipments, shipment_items = generate_shipments(products, args.shipments)
    insert_batch(client, "uds.shipments", shipments)
    insert_batch(client, "uds.shipment_items", shipment_items)

    print(f"\nGenerating financials from {len(orders)} orders...")
    financials = generate_financials(orders, order_items)
    insert_batch(client, "uds.financials", financials)

    print("\nSeed data generation complete.")
    print(f"  Products:       {len(products):,}")
    print(f"  Orders:         {len(orders):,}")
    print(f"  Order items:    {len(order_items):,}")
    print(f"  Inventory rows: {len(inventory):,}")
    print(f"  Shipments:      {len(shipments):,}")
    print(f"  Shipment items: {len(shipment_items):,}")
    print(f"  Financial rows: {len(financials):,}")


if __name__ == "__main__":
    main()
