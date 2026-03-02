# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup: Create Genie Spaces via API
# MAGIC
# MAGIC This notebook creates three **Genie Spaces** — one for each table in `sjdatabricks.orders`.
# MAGIC Genie Spaces let users ask natural-language questions that get translated into SQL automatically.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `create_fake_data.py` first so the tables exist.
# MAGIC - You need workspace admin or Genie permissions.
# MAGIC
# MAGIC **What is a Genie Space?**
# MAGIC A Genie Space is a Databricks AI/BI feature that wraps one or more tables with a natural language
# MAGIC interface. You point it at tables, and users can ask questions like "Show me all shipped orders"
# MAGIC — Genie translates that to SQL and returns the results. In our tutorials, we use Genie Spaces
# MAGIC as **tools** that our agents can call to answer data questions.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure the Databricks SDK Client
# MAGIC
# MAGIC The `WorkspaceClient` automatically picks up authentication from the notebook environment.
# MAGIC We also set the SQL warehouse ID — Genie needs a warehouse to execute queries.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Replace with your SQL warehouse ID (find in SQL Warehouses page)
SQL_WAREHOUSE_ID = "<YOUR_SQL_WAREHOUSE_ID>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the Genie Spaces
# MAGIC
# MAGIC We create three separate spaces so that each covers a focused domain:
# MAGIC - **Order Details** — for questions about order status, quantities, dates
# MAGIC - **Returns** — for questions about return reasons, approval status
# MAGIC - **Products** — for questions about product catalog, pricing, stock
# MAGIC
# MAGIC Separating spaces helps the agent route questions to the right domain and keeps SQL generation focused.

# COMMAND ----------

space_configs = [
    {
        "title": "Order Details - Customer Orders",
        "description": "Ask questions about customer orders: status, products ordered, quantities, and dates.",
        "table_identifiers": ["sjdatabricks.orders.order_details"],
    },
    {
        "title": "Returns - Order Returns",
        "description": "Ask questions about product returns: reasons, approval status, and return dates.",
        "table_identifiers": ["sjdatabricks.orders.returns"],
    },
    {
        "title": "Products - Product Catalog",
        "description": "Ask questions about the product catalog: names, categories, prices, and stock levels.",
        "table_identifiers": ["sjdatabricks.orders.products"],
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Call the Genie API to Create Each Space
# MAGIC
# MAGIC The Databricks SDK's Genie API creates the space and returns an ID.
# MAGIC Save these IDs — you'll need them in tutorials 7–14.

# COMMAND ----------

created_spaces = {}

for config in space_configs:
    space = w.genie.create_space(
        title=config["title"],
        description=config["description"],
        table_identifiers=config["table_identifiers"],
        warehouse_id=SQL_WAREHOUSE_ID,
    )
    created_spaces[config["title"]] = space.space_id
    print(f"Created: {config['title']} → Space ID: {space.space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Space IDs
# MAGIC
# MAGIC Copy these IDs into a widget or variable block in each Genie tutorial notebook.
# MAGIC The output below shows the mapping from space name to ID.

# COMMAND ----------

print("\n=== Genie Space IDs (copy these into your tutorial notebooks) ===\n")
for title, space_id in created_spaces.items():
    print(f"  {title}: {space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Copy the Space IDs printed above
# MAGIC 2. Open any Genie tutorial (Topics 7–14)
# MAGIC 3. Paste the relevant space ID into the configuration cell
# MAGIC 4. You can also visit each space in the Databricks UI under **AI/BI Genie** to test it manually
