# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy: Multi-Genie DSPy Agent (WHL)
# MAGIC
# MAGIC Supervisor agent using DSPy ReAct to orchestrate two Genie spaces:
# MAGIC - **Sales Analytics** — Revenue, orders, product performance
# MAGIC - **Customer Insights** — Support tickets, customer profiles, satisfaction
# MAGIC
# MAGIC This notebook: build whl, log model, evaluate with cross-space queries,
# MAGIC UC registration (`sjdatabricks.agents.multi_genie_dspy`), deploy, and query.

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
# MAGIC ## 3. Imports and Log Model

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Patch Space IDs into Agent Module (Optional)
# MAGIC
# MAGIC If you configured Space IDs above, patch the agent module before logging.

# COMMAND ----------

try:
    agent_module_path = os.path.join(_pkg_root, "src", "agent", "agent.py")
    if os.path.exists(agent_module_path):
        with open(agent_module_path, "r") as f:
            content = f.read()
        if SALES_GENIE_SPACE_ID != "<YOUR-SALES-GENIE-SPACE-ID>":
            content = content.replace('"<YOUR-SALES-GENIE-SPACE-ID>"', repr(SALES_GENIE_SPACE_ID))
        if CUSTOMER_GENIE_SPACE_ID != "<YOUR-CUSTOMER-GENIE-SPACE-ID>":
            content = content.replace('"<YOUR-CUSTOMER-GENIE-SPACE-ID>"', repr(CUSTOMER_GENIE_SPACE_ID))
        with open(agent_module_path, "w") as f:
            f.write(content)
        print("Patched agent.py with Space IDs.")
    else:
        print("agent.py not found; using package defaults.")
except Exception as e:
    print(f"Patch skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Log Model (Models-from-Code)
# MAGIC

# COMMAND ----------

mlflow.dspy.autolog()

with mlflow.start_run(run_name="multi_genie_dspy_agent"):
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
# MAGIC ## 4. Evaluation with Cross-Space Queries
# MAGIC
# MAGIC Evaluate with single-space (revenue, tickets) and multi-space (customers with tickets and order values) queries.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What was total revenue last month?"}]},
    {"input": [{"role": "user", "content": "How many open support tickets do we have?"}]},
    {"input": [{"role": "user", "content": "Which customers with open tickets have the highest order values?"}]},
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[
        Safety(),
        RelevanceToQuery(),
    ],
)

print("Evaluation complete. Check MLflow UI for detailed results.")
print(eval_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register to Unity Catalog
# MAGIC

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.multi_genie_dspy"

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    f"runs:/{run_id}/model",
    UC_MODEL_NAME,
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Local Testing (Before Deployment)
# MAGIC

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(model_uri)
test_request = {
    "input": [{"role": "user", "content": "What was total revenue last month?"}],
}
test_response = test_agent.predict(test_request)
print("Output items:")
for item in test_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving
# MAGIC
# MAGIC Create a serving endpoint via the Databricks UI or SDK:

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint_name = "multi-genie-dspy-agent"
# MAGIC
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name=endpoint_name,
# MAGIC     config={
# MAGIC         "served_entities": [
# MAGIC             {
# MAGIC                 "name": "multi-genie-agent",
# MAGIC                 "entity_name": UC_MODEL_NAME,
# MAGIC                 "entity_version": str(model_info.version),
# MAGIC                 "workload_size": "Small",
# MAGIC                 "scale_to_zero_enabled": True,
# MAGIC             }
# MAGIC         ],
# MAGIC         "traffic_config": {
# MAGIC             "routes": [
# MAGIC                 {"served_model_name": "multi-genie-agent", "traffic_percentage": 100}
# MAGIC             ]
# MAGIC         },
# MAGIC     },
# MAGIC )
# MAGIC print(f"Endpoint created: {endpoint.name}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Querying the Deployed Agent
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

w = WorkspaceClient()
client = DatabricksOpenAI(workspace_client=w)
endpoint_name = "multi-genie-dspy-agent"

input_msgs = [{"role": "user", "content": "What was total revenue last month?"}]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a. Non-Streaming Query
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC response = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=input_msgs,
# MAGIC     max_output_tokens=512,
# MAGIC )
# MAGIC
# MAGIC print("Non-streaming response:")
# MAGIC for item in getattr(response, "output", response):
# MAGIC     if hasattr(item, "content"):
# MAGIC         for c in (item.content or []):
# MAGIC             if hasattr(c, "text") and c.text:
# MAGIC                 print(c.text)
# MAGIC     elif hasattr(item, "text"):
# MAGIC         print(item.text)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b. Streaming Query
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC streaming_response = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=input_msgs,
# MAGIC     stream=True,
# MAGIC     max_output_tokens=512,
# MAGIC )
# MAGIC
# MAGIC print("Streaming response:")
# MAGIC for chunk in streaming_response:
# MAGIC     if hasattr(chunk, "output"):
# MAGIC         for item in (chunk.output or []):
# MAGIC             if hasattr(item, "content"):
# MAGIC                 for c in (item.content or []):
# MAGIC                     if hasattr(c, "delta") and c.delta:
# MAGIC                         print(c.delta, end="", flush=True)
# MAGIC             elif hasattr(item, "delta") and item.delta:
# MAGIC                 print(item.delta, end="", flush=True)
# MAGIC print()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Building Locally with uv
# MAGIC
# MAGIC To build a distributable wheel outside Databricks:
# MAGIC
# MAGIC ```bash
# MAGIC uv build --wheel
# MAGIC ```
# MAGIC
# MAGIC The resulting `.whl` will be in `dist/` and can be installed with:
# MAGIC
# MAGIC ```bash
# MAGIC uv pip install dist/multi_genie_dspy_agent-0.1.0-py3-none-any.whl --system
# MAGIC ```
