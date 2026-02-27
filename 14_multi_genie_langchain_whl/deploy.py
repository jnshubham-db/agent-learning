# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 14: Multi-Genie LangChain Agent (WHL Package)
# MAGIC
# MAGIC ## Use Case
# MAGIC A **supervisor agent** using LangChain that orchestrates two Genie spaces:
# MAGIC - **Sales Analytics** — Revenue, orders, product performance
# MAGIC - **Customer Insights** — Support tickets, customer profiles, satisfaction
# MAGIC
# MAGIC ## Architecture
# MAGIC - **LangChain** `create_tool_calling_agent` + `AgentExecutor` with two `@tool` functions:
# MAGIC   - `query_sales_data(question)` → Sales Analytics Genie
# MAGIC   - `query_customer_data(question)` → Customer Insights Genie
# MAGIC - **ChatDatabricks** LLM (`databricks-claude-sonnet-4-5`)
# MAGIC - **MLflow ResponsesAgent** with `to_chat_completions_input()` and `output_to_responses_items_stream()`
# MAGIC - **WHL package** for portable deployment
# MAGIC
# MAGIC **Prerequisites:** Run `create_genie_spaces` to create the Genie spaces, then set the Space IDs in the config section below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and Install Package
# MAGIC
# MAGIC Install `uv`, build the wheel using `hatchling` (via `pyproject.toml`), and install it.

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
# MAGIC Replace with your actual Space IDs from `create_genie_spaces` notebook.

# COMMAND ----------

SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set Genie Space IDs and Import Agent
# MAGIC
# MAGIC Override the placeholder IDs in the agent module before creating the agent instance.

# COMMAND ----------

import agent.agent as agent_module

# Override placeholder space IDs with actual values
agent_module.SALES_GENIE_SPACE_ID = SALES_GENIE_SPACE_ID
agent_module.CUSTOMER_GENIE_SPACE_ID = CUSTOMER_GENIE_SPACE_ID

from agent import MultiGenieLangChainAgent
import mlflow

mlflow.langchain.autolog()
agent = MultiGenieLangChainAgent()
mlflow.models.set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log Model to MLflow

# COMMAND ----------

mlflow.set_experiment("14_multi_genie_langchain_whl")

with mlflow.start_run():
    logged_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
        artifact_path="model",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri

print(f"Logged model. Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluation with Cross-Space Queries
# MAGIC
# MAGIC Evaluate with single-space (revenue, tickets) and multi-space (customers with tickets and order values) queries.

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

loaded_agent = mlflow.pyfunc.load_model(model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What was total revenue last month?"}]},
    {"input": [{"role": "user", "content": "How many open support tickets do we have?"}]},
    {"input": [{"role": "user", "content": "Which customers with open tickets have the highest order values?"}]},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print(f"Evaluation run ID: {results.run_id}")
if hasattr(results, "tables") and results.tables:
    print(results.tables.get("eval_results_table", "N/A"))
else:
    print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register to Unity Catalog

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.multi_genie_langchain"

mlflow.set_registry_uri("databricks-uc")
registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {registered.name} (version {registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deployment Example (Model Serving Endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the MLflow Deployments API to create a serving endpoint:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="multi-genie-langchain-agent",
# MAGIC     config={
# MAGIC         "served_entities": [
# MAGIC             {
# MAGIC                 "entity_name": UC_MODEL_NAME,
# MAGIC                 "entity_version": str(registered.version),
# MAGIC                 "workload_size": "Small",
# MAGIC                 "scale_to_zero_enabled": True,
# MAGIC             }
# MAGIC         ],
# MAGIC     },
# MAGIC )
# MAGIC print(f"Endpoint created: {endpoint.name}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Query Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local Test (Non-Streaming)

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(model_uri)
test_request = {"input": [{"role": "user", "content": "What was total revenue last month?"}]}
test_response = test_agent.predict(test_request)
print("Local test response:")
for item in test_response.output:
    if hasattr(item, "content"):
        for c in (item.content or []):
            if hasattr(c, "text") and c.text:
                print(c.text)
    elif isinstance(item, dict) and "content" in item:
        for c in item.get("content", []):
            if isinstance(c, dict) and "text" in c:
                print(c["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deployed Endpoint Query (Non-Streaming)
# MAGIC
# MAGIC Replace `endpoint_name` with your deployed endpoint:
# MAGIC
# MAGIC ```python
# MAGIC from databricks_openai import DatabricksOpenAI
# MAGIC
# MAGIC client = DatabricksOpenAI()
# MAGIC endpoint_name = "multi-genie-langchain-agent"
# MAGIC
# MAGIC response = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=[{"role": "user", "content": "What was total revenue last month?"}],
# MAGIC     max_output_tokens=512,
# MAGIC )
# MAGIC print("Response:", response)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Query
# MAGIC
# MAGIC ```python
# MAGIC stream = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=[{"role": "user", "content": "Which customers with open tickets have the highest order values?"}],
# MAGIC     max_output_tokens=512,
# MAGIC     stream=True,
# MAGIC )
# MAGIC for chunk in stream:
# MAGIC     if hasattr(chunk, "output"):
# MAGIC         for item in (chunk.output or []):
# MAGIC             if hasattr(item, "content"):
# MAGIC                 for c in (item.content or []):
# MAGIC                     if hasattr(c, "delta") and c.delta:
# MAGIC                         print(c.delta, end="", flush=True)
# MAGIC print()
# MAGIC ```
