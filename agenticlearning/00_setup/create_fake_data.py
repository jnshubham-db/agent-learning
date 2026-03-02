# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup: Create Fake Data for Customer Order Support Agent
# MAGIC
# MAGIC This notebook seeds the **sjdatabricks** catalog with three tables that all 14 tutorials share:
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `sjdatabricks.orders.order_details` | Customer orders with status tracking |
# MAGIC | `sjdatabricks.orders.returns` | Return requests linked to orders |
# MAGIC | `sjdatabricks.orders.products` | Product catalog with pricing and stock |
# MAGIC
# MAGIC **Run this notebook once before starting any tutorial.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create the Catalog and Schema
# MAGIC
# MAGIC Unity Catalog organizes data in a three-level namespace: **catalog → schema → table**.
# MAGIC We create `sjdatabricks` as our top-level catalog and `orders` as the schema that groups
# MAGIC all order-related tables together.

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS sjdatabricks")
spark.sql("CREATE SCHEMA IF NOT EXISTS sjdatabricks.orders")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Seed the `order_details` Table
# MAGIC
# MAGIC We generate 20 fake orders (IDs 1040–1059) with random products, quantities, and statuses.
# MAGIC Each order has a date within the last 90 days. In production, this table would be populated
# MAGIC by your order management system — here we use random data so the agent has something to query.

# COMMAND ----------

from datetime import date, timedelta
import random

random.seed(42)  # Reproducible fake data

orders = [
    (
        i,
        f"Customer_{i}",
        random.choice(["Laptop", "Phone", "Tablet", "Monitor"]),
        random.randint(1, 5),
        random.choice(["Shipped", "Processing", "Delivered"]),
        str(date.today() - timedelta(days=random.randint(1, 90))),
    )
    for i in range(1040, 1060)
]

orders_df = spark.createDataFrame(
    orders,
    ["order_id", "customer_name", "product", "quantity", "status", "order_date"],
)

orders_df.write.mode("overwrite").saveAsTable("sjdatabricks.orders.order_details")
display(spark.table("sjdatabricks.orders.order_details"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Seed the `returns` Table
# MAGIC
# MAGIC We generate 10 return records, each linked to a random order from the `order_details` table.
# MAGIC Returns have a reason (Defective, Wrong item, Changed mind) and a status (Pending, Approved, Rejected).

# COMMAND ----------

returns = [
    (
        r,
        random.randint(1040, 1059),
        random.choice(["Defective", "Wrong item", "Changed mind"]),
        random.choice(["Pending", "Approved", "Rejected"]),
        str(date.today() - timedelta(days=random.randint(1, 60))),
    )
    for r in range(5001, 5011)
]

returns_df = spark.createDataFrame(
    returns, ["return_id", "order_id", "reason", "status", "return_date"]
)

returns_df.write.mode("overwrite").saveAsTable("sjdatabricks.orders.returns")
display(spark.table("sjdatabricks.orders.returns"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Seed the `products` Table
# MAGIC
# MAGIC A small product catalog with 6 items across Electronics and Accessories categories.
# MAGIC The agent uses this table to answer questions like *"What products are in the Electronics category?"*

# COMMAND ----------

products = [
    (p, name, cat, round(random.uniform(50, 2000), 2), random.randint(0, 500))
    for p, (name, cat) in enumerate(
        [
            ("Laptop", "Electronics"),
            ("Phone", "Electronics"),
            ("Tablet", "Electronics"),
            ("Monitor", "Electronics"),
            ("Keyboard", "Accessories"),
            ("Mouse", "Accessories"),
        ],
        start=1,
    )
]

products_df = spark.createDataFrame(
    products, ["product_id", "name", "category", "price", "stock"]
)

products_df.write.mode("overwrite").saveAsTable("sjdatabricks.orders.products")
display(spark.table("sjdatabricks.orders.products"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Verify All Tables
# MAGIC
# MAGIC Quick sanity check — all three tables should now exist in the `sjdatabricks.orders` schema.

# COMMAND ----------

display(spark.sql("SHOW TABLES IN sjdatabricks.orders"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Your data is ready. For **Genie-based tutorials (Topics 7–14)**, you also need to create
# MAGIC Genie Spaces in the Databricks UI:
# MAGIC
# MAGIC 1. Go to **AI/BI Genie** in the left sidebar
# MAGIC 2. Create three spaces:
# MAGIC    - **Order Details Space** → add `sjdatabricks.orders.order_details`
# MAGIC    - **Returns Space** → add `sjdatabricks.orders.returns`
# MAGIC    - **Products Space** → add `sjdatabricks.orders.products`
# MAGIC 3. Note each space's ID — you'll paste them into the Genie tutorial notebooks
# MAGIC
# MAGIC Run the `create_genie_spaces.py` notebook to automate this via the API.
