# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Create Fake Data for Genie Spaces
# MAGIC
# MAGIC This notebook creates fake datasets in the `sjdatabricks` catalog for use with Genie Spaces.
# MAGIC
# MAGIC **Tables Created:**
# MAGIC - `sjdatabricks.genie_data.sales_orders` — Order transactions
# MAGIC - `sjdatabricks.genie_data.customers` — Customer profiles
# MAGIC - `sjdatabricks.genie_data.products` — Product catalog with inventory
# MAGIC - `sjdatabricks.genie_data.support_tickets` — Customer support tickets

# COMMAND ----------

# MAGIC %pip install faker dbldatagen
# MAGIC %restart_python

# COMMAND ----------

CATALOG = "sjdatabricks"
SCHEMA = "genie_data"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Products Table

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import *

product_spec = (
    dg.DataGenerator(spark, name="products", rowcount=50, partitions=1)
    .withColumn("product_id", StringType(), template=r"PROD-\k{4}")
    .withColumn(
        "product_name",
        StringType(),
        values=[
            "Wireless Headphones", "USB-C Hub", "Mechanical Keyboard",
            "4K Monitor", "Standing Desk", "Ergonomic Chair",
            "Laptop Stand", "Webcam HD", "Smart Mouse",
            "Portable Charger", "Noise Cancelling Earbuds", "Desk Lamp",
            "Monitor Arm", "Cable Management Kit", "Docking Station",
        ],
    )
    .withColumn(
        "category",
        StringType(),
        values=["Audio", "Accessories", "Peripherals", "Displays", "Furniture", "Power"],
    )
    .withColumn("price", FloatType(), minValue=19.99, maxValue=899.99, step=0.01)
    .withColumn("stock_quantity", IntegerType(), minValue=0, maxValue=500)
    .withColumn("supplier", StringType(), values=["TechCo", "GadgetWorld", "OfficePro", "ElectroMax"])
)

df_products = product_spec.build()
df_products.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.products")
display(df_products)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Customers Table

# COMMAND ----------

customer_spec = (
    dg.DataGenerator(spark, name="customers", rowcount=200, partitions=1)
    .withColumn("customer_id", StringType(), template=r"CUST-\k{6}")
    .withColumn(
        "first_name",
        StringType(),
        values=["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"],
    )
    .withColumn(
        "last_name",
        StringType(),
        values=["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Taylor"],
    )
    .withColumn("email", StringType(), template=r"\w.\w@example.com")
    .withColumn(
        "city",
        StringType(),
        values=["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Denver", "Boston", "Portland"],
    )
    .withColumn(
        "state",
        StringType(),
        values=["NY", "CA", "IL", "TX", "WA", "CO", "MA", "OR"],
    )
    .withColumn("signup_date", DateType(), begin="2023-01-01", end="2025-12-31")
    .withColumn(
        "tier",
        StringType(),
        values=["Bronze", "Silver", "Gold", "Platinum"],
        weights=[40, 30, 20, 10],
    )
)

df_customers = customer_spec.build()
df_customers.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.customers")
display(df_customers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sales Orders Table

# COMMAND ----------

from pyspark.sql import functions as F

product_ids = [row.product_id for row in df_products.select("product_id").collect()]
customer_ids = [row.customer_id for row in df_customers.select("customer_id").collect()]

order_spec = (
    dg.DataGenerator(spark, name="sales_orders", rowcount=5000, partitions=4)
    .withColumn("order_id", StringType(), template=r"ORD-\k{8}")
    .withColumn("customer_id", StringType(), values=customer_ids)
    .withColumn("product_id", StringType(), values=product_ids)
    .withColumn("quantity", IntegerType(), minValue=1, maxValue=10)
    .withColumn("unit_price", FloatType(), minValue=19.99, maxValue=899.99, step=0.01)
    .withColumn("order_date", DateType(), begin="2024-01-01", end="2026-02-27")
    .withColumn(
        "status",
        StringType(),
        values=["pending", "shipped", "delivered", "cancelled", "returned"],
        weights=[15, 25, 45, 10, 5],
    )
    .withColumn(
        "payment_method",
        StringType(),
        values=["credit_card", "debit_card", "paypal", "bank_transfer"],
    )
    .withColumn("discount_pct", FloatType(), minValue=0.0, maxValue=0.3, step=0.01)
)

df_orders = order_spec.build()
df_orders = df_orders.withColumn(
    "total_amount",
    F.round(F.col("quantity") * F.col("unit_price") * (1 - F.col("discount_pct")), 2),
)
df_orders.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.sales_orders")
display(df_orders)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Support Tickets Table

# COMMAND ----------

ticket_spec = (
    dg.DataGenerator(spark, name="support_tickets", rowcount=1000, partitions=2)
    .withColumn("ticket_id", StringType(), template=r"TKT-\k{6}")
    .withColumn("customer_id", StringType(), values=customer_ids)
    .withColumn("product_id", StringType(), values=product_ids)
    .withColumn(
        "issue_type",
        StringType(),
        values=["defective_product", "shipping_delay", "wrong_item", "refund_request", "general_inquiry"],
        weights=[20, 25, 15, 25, 15],
    )
    .withColumn(
        "priority",
        StringType(),
        values=["low", "medium", "high", "critical"],
        weights=[30, 40, 20, 10],
    )
    .withColumn(
        "status",
        StringType(),
        values=["open", "in_progress", "resolved", "closed"],
        weights=[20, 30, 35, 15],
    )
    .withColumn("created_date", DateType(), begin="2024-06-01", end="2026-02-27")
    .withColumn("resolution_days", IntegerType(), minValue=0, maxValue=14)
    .withColumn("satisfaction_score", IntegerType(), minValue=1, maxValue=5)
)

df_tickets = ticket_spec.build()
df_tickets.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.support_tickets")
display(df_tickets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify All Tables

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Table Comments for Genie Discoverability

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.products
    SET TBLPROPERTIES ('comment' = 'Product catalog with pricing, categories, stock levels, and supplier info.')
""")

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.customers
    SET TBLPROPERTIES ('comment' = 'Customer profiles with contact info, location, signup date, and loyalty tier.')
""")

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.sales_orders
    SET TBLPROPERTIES ('comment' = 'Sales order transactions with quantities, pricing, discounts, payment methods, and fulfillment status.')
""")

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.support_tickets
    SET TBLPROPERTIES ('comment' = 'Customer support tickets with issue types, priorities, resolution times, and satisfaction scores.')
""")

print("All table comments set successfully.")
