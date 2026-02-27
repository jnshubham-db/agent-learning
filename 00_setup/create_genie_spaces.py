# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Create Genie Spaces
# MAGIC
# MAGIC This notebook creates Genie Spaces on top of the fake data tables.
# MAGIC
# MAGIC **Genie Spaces Created:**
# MAGIC 1. **Sales Analytics** — Sales orders, revenue, and product performance
# MAGIC 2. **Customer Insights** — Customer profiles, support tickets, and satisfaction
# MAGIC
# MAGIC > **Prerequisites:** Run `create_fake_data` notebook first.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

CATALOG = "sjdatabricks"
SCHEMA = "genie_data"

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Sales Analytics Genie Space

# COMMAND ----------

sales_space = w.genie.create_space(
    title="Sales Analytics",
    description=(
        "Analyze sales orders, revenue trends, product performance, and order fulfillment. "
        "Ask questions like: 'What was total revenue last month?', "
        "'Which products have the most cancelled orders?', "
        "'Show me top 10 customers by total spend.'"
    ),
    table_identifiers=[
        f"{CATALOG}.{SCHEMA}.sales_orders",
        f"{CATALOG}.{SCHEMA}.products",
    ],
)

SALES_SPACE_ID = sales_space.space_id
print(f"Sales Analytics Genie Space created: {SALES_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Customer Insights Genie Space

# COMMAND ----------

customer_space = w.genie.create_space(
    title="Customer Insights",
    description=(
        "Explore customer demographics, support ticket trends, satisfaction scores, and loyalty tiers. "
        "Ask questions like: 'How many Platinum customers are in California?', "
        "'What is the average satisfaction score for refund requests?', "
        "'Show me open high-priority tickets.'"
    ),
    table_identifiers=[
        f"{CATALOG}.{SCHEMA}.customers",
        f"{CATALOG}.{SCHEMA}.support_tickets",
    ],
)

CUSTOMER_SPACE_ID = customer_space.space_id
print(f"Customer Insights Genie Space created: {CUSTOMER_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Space IDs for Use in Agent Notebooks
# MAGIC
# MAGIC Store these IDs — you'll reference them in topics 7-14.

# COMMAND ----------

print("=" * 60)
print("GENIE SPACE IDS — Copy these into your agent notebooks:")
print("=" * 60)
print(f"SALES_GENIE_SPACE_ID = \"{SALES_SPACE_ID}\"")
print(f"CUSTOMER_GENIE_SPACE_ID = \"{CUSTOMER_SPACE_ID}\"")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Spaces

# COMMAND ----------

spaces = w.genie.list_spaces()
for space in spaces:
    print(f"  - {space.title} (ID: {space.space_id})")
