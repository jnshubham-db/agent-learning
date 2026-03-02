# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 13: Multi-Genie Agent with DSPy — WHL Package
# MAGIC
# MAGIC This notebook deploys a **multi-agent supervisor** packaged as a Python wheel that routes
# MAGIC user questions to the correct **Databricks Genie space** using **DSPy ChainOfThought**.
# MAGIC
# MAGIC **Architecture Overview:**
# MAGIC - Three Genie spaces: **Orders**, **Returns**, **Products**
# MAGIC - A DSPy `ChainOfThought` router classifies each question into one of the three departments
# MAGIC - The selected Genie agent queries the corresponding space via `WorkspaceClient().genie`
# MAGIC - The entire agent is wrapped in an MLflow `ResponsesAgent` for tracing, evaluation, and serving
# MAGIC
# MAGIC **Why a WHL package?**
# MAGIC - Clean separation of agent code from deployment logic
# MAGIC - Versioned, reproducible builds with `pyproject.toml` + `hatchling`
# MAGIC - Easy to install on Databricks clusters or CI/CD pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and Install the Wheel
# MAGIC
# MAGIC We use `uv` to build the wheel from `pyproject.toml`, then install it into the current
# MAGIC Python environment. This ensures all imports (`from agent.genie import ...`) work correctly.

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import glob
import os
import subprocess
import sys

_pkg_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_pkg_root)

subprocess.check_call(["uv", "build", "--wheel"])

whl = sorted(glob.glob(os.path.join("dist", "*.whl")))[-1]
subprocess.check_call([sys.executable, "-m", "pip", "install", whl, "--force-reinstall"])
print(f"Built and installed: {whl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Genie Space IDs
# MAGIC
# MAGIC Each Genie space corresponds to a specific data domain. The agent reads these from
# MAGIC environment variables so that the same wheel works across environments (dev, staging, prod).
# MAGIC
# MAGIC Set the space IDs using Databricks widgets or hardcode them below for testing.

# COMMAND ----------

# Create widgets for space IDs (Databricks UI)
dbutils.widgets.text("ORDER_SPACE_ID", "<YOUR-ORDER-SPACE-ID>", "Orders Genie Space ID")
dbutils.widgets.text("RETURNS_SPACE_ID", "<YOUR-RETURNS-SPACE-ID>", "Returns Genie Space ID")
dbutils.widgets.text("PRODUCTS_SPACE_ID", "<YOUR-PRODUCTS-SPACE-ID>", "Products Genie Space ID")

ORDER_SPACE_ID = dbutils.widgets.get("ORDER_SPACE_ID")
RETURNS_SPACE_ID = dbutils.widgets.get("RETURNS_SPACE_ID")
PRODUCTS_SPACE_ID = dbutils.widgets.get("PRODUCTS_SPACE_ID")

# Set as environment variables so the agent module can read them
os.environ["ORDER_SPACE_ID"] = ORDER_SPACE_ID
os.environ["RETURNS_SPACE_ID"] = RETURNS_SPACE_ID
os.environ["PRODUCTS_SPACE_ID"] = PRODUCTS_SPACE_ID

print(f"ORDER_SPACE_ID:    {ORDER_SPACE_ID}")
print(f"RETURNS_SPACE_ID:  {RETURNS_SPACE_ID}")
print(f"PRODUCTS_SPACE_ID: {PRODUCTS_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log Model to MLflow
# MAGIC
# MAGIC We use the **models-from-code** approach: `mlflow.pyfunc.log_model` points to the
# MAGIC `agent.py` source file, which calls `set_model()` at import time. MLflow captures
# MAGIC the entire wheel as a dependency, so the model is fully self-contained.

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="multi_genie_dspy_whl"):
    logged_info = mlflow.pyfunc.log_model(
        python_model="src/agent/agent.py",
        artifact_path="model",
        name="agent",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri
    print(f"Logged agent. Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the Agent
# MAGIC
# MAGIC Load the logged model and test with questions from each department to verify routing.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Test Orders Question
# MAGIC
# MAGIC Questions about order status, revenue, shipping, and delivery should route to the Orders space.

# COMMAND ----------

orders_request = {
    "input": [{"role": "user", "content": "What is the total revenue from orders last month?"}],
}
orders_response = loaded_agent.predict(orders_request)
print("Orders response:")
for item in orders_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Test Returns Question
# MAGIC
# MAGIC Questions about refunds, exchanges, and return rates should route to the Returns space.

# COMMAND ----------

returns_request = {
    "input": [{"role": "user", "content": "How many returns were processed this week?"}],
}
returns_response = loaded_agent.predict(returns_request)
print("Returns response:")
for item in returns_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. Test Products Question
# MAGIC
# MAGIC Questions about inventory, pricing, and product catalog should route to the Products space.

# COMMAND ----------

products_request = {
    "input": [{"role": "user", "content": "What are the top 10 products by inventory count?"}],
}
products_response = loaded_agent.predict(products_request)
print("Products response:")
for item in products_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC Evaluate the agent with a dataset covering all three departments. The `Safety` scorer
# MAGIC checks for harmful content, and `RelevanceToQuery` checks that the answer addresses
# MAGIC the original question.

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the total revenue from orders last month?"}]},
    {"input": [{"role": "user", "content": "How many returns were processed this week?"}]},
    {"input": [{"role": "user", "content": "What are the top 10 products by inventory count?"}]},
    {"input": [{"role": "user", "content": "Show me the average order value by region"}]},
    {"input": [{"role": "user", "content": "What is the return rate for electronics?"}]},
    {"input": [{"role": "user", "content": "List all products with price above $500"}]},
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print("Evaluation complete. Check MLflow UI for detailed results.")
print(eval_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register to Unity Catalog
# MAGIC
# MAGIC Register the model to Unity Catalog for governance, lineage tracking, and deployment.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.multi_genie_dspy_agent"

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving Endpoint
# MAGIC
# MAGIC Create a serving endpoint. Environment variables (`ORDER_SPACE_ID`, `RETURNS_SPACE_ID`,
# MAGIC `PRODUCTS_SPACE_ID`) are set in the endpoint configuration so the agent can route
# MAGIC questions to the correct Genie Space at inference time.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

ENDPOINT_NAME = "multi-genie-dspy-whl-agent"

try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_NAME,
                    entity_version=str(model_info.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                    environment_vars={
                        "ORDER_SPACE_ID": ORDER_SPACE_ID,
                        "RETURNS_SPACE_ID": RETURNS_SPACE_ID,
                        "PRODUCTS_SPACE_ID": PRODUCTS_SPACE_ID,
                    },
                )
            ]
        ),
    )
    print(f"Endpoint '{ENDPOINT_NAME}' is ready!")
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint '{ENDPOINT_NAME}' already exists. Updating...")
        endpoint = w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_NAME,
                    entity_version=str(model_info.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                    environment_vars={
                        "ORDER_SPACE_ID": ORDER_SPACE_ID,
                        "RETURNS_SPACE_ID": RETURNS_SPACE_ID,
                        "PRODUCTS_SPACE_ID": PRODUCTS_SPACE_ID,
                    },
                )
            ],
        )
        print(f"Endpoint '{ENDPOINT_NAME}' updated and ready!")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Query the Deployed Endpoint — Non-Streaming
# MAGIC
# MAGIC Now that the endpoint is live, we can query it using the OpenAI-compatible Responses
# MAGIC API. The `openai` client points at the Databricks serving URL and authenticates with
# MAGIC a workspace token.

# COMMAND ----------

# MAGIC %pip install openai -q

# COMMAND ----------

from openai import OpenAI

client = OpenAI(
    base_url=f"{w.config.host}/serving-endpoints",
    api_key=w.tokens().token,
)

response = client.responses.create(
    model=ENDPOINT_NAME,
    input=[{"role": "user", "content": "What is the status of order 1042?"}],
)

for item in response.output:
    if hasattr(item, "content"):
        for content in item.content:
            if hasattr(content, "text"):
                print(content.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Query the Deployed Endpoint — Streaming
# MAGIC
# MAGIC Streaming lets you display tokens as they arrive, which improves perceived latency
# MAGIC for longer responses. The same `client.responses.create` call is used with `stream=True`.

# COMMAND ----------

stream = client.responses.create(
    model=ENDPOINT_NAME,
    input=[{"role": "user", "content": "Show me all returns for order 1045."}],
    stream=True,
)

print("Streaming response:")
for event in stream:
    if hasattr(event, "type") and event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()
