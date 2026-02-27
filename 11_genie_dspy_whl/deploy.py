# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy: Genie DSPy Agent (WHL)
# MAGIC
# MAGIC This notebook builds the whl package, logs the model to MLflow, evaluates it,
# MAGIC registers to Unity Catalog, and shows deployment/query examples.
# MAGIC
# MAGIC **Use Case:** An agent that queries a Sales Analytics Genie Space using DSPy CoT.

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
# MAGIC ## 2. Imports and Log Model

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Log Model with MLflow
# MAGIC
# MAGIC The agent module defines the model at import time via `set_model()`.
# MAGIC Log using the models-from-code path.

# COMMAND ----------

with mlflow.start_run(run_name="genie_dspy_agent"):
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
# MAGIC ## 3. Evaluation with LLM-as-a-Judge
# MAGIC
# MAGIC Evaluate the agent using `mlflow.genai.evaluate()` with Safety and RelevanceToQuery scorers.
# MAGIC **Note:** Requires a valid `SALES_GENIE_SPACE_ID` in `agent.py`.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

eval_dataset = [
    {
        "input": [
            {
                "role": "user",
                "content": "What were the top 5 products by revenue last quarter?",
            }
        ]
    },
    {
        "input": [
            {"role": "user", "content": "Show me sales by region for the current year"}
        ]
    },
    {
        "input": [
            {
                "role": "user",
                "content": "How does our Q3 revenue compare to Q2?",
            }
        ]
    },
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
# MAGIC ## 4. Register to Unity Catalog
# MAGIC

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri,
    "sjdatabricks.agents.genie_dspy_agent",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Local Testing (Before Deployment)
# MAGIC

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(model_uri)
test_request = {
    "input": [
        {"role": "user", "content": "What were the top 5 products by revenue?"}
    ],
}
test_response = test_agent.predict(test_request)
print("Output items:")
for item in test_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deployment to Model Serving
# MAGIC
# MAGIC Create a serving endpoint via the Databricks UI or SDK:

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="genie-dspy-sales-agent",
# MAGIC     config={
# MAGIC         "served_entities": [
# MAGIC             {
# MAGIC                 "entity_name": "sjdatabricks.agents.genie_dspy_agent",
# MAGIC                 "entity_version": str(model_info.version),
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
# MAGIC ## 7. Querying the Deployed Agent
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. Non-Streaming
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import DatabricksOpenAI
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC client = DatabricksOpenAI(workspace_client=w)
# MAGIC endpoint_name = "genie-dspy-sales-agent"
# MAGIC
# MAGIC input_msgs = [
# MAGIC     {"role": "user", "content": "What were the top 5 products by revenue last quarter?"}
# MAGIC ]
# MAGIC
# MAGIC response = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=input_msgs,
# MAGIC     max_output_tokens=512,
# MAGIC )
# MAGIC
# MAGIC print("Non-streaming response:")
# MAGIC for item in getattr(response, "output", response) or []:
# MAGIC     if hasattr(item, "content"):
# MAGIC         for c in (item.content or []):
# MAGIC             if hasattr(c, "text") and c.text:
# MAGIC                 print(c.text)
# MAGIC     elif hasattr(item, "text"):
# MAGIC         print(item.text)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Streaming
# MAGIC
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
# MAGIC         for item in chunk.output or []:
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
# MAGIC ## 8. Building Locally with uv
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
# MAGIC uv pip install dist/genie_dspy_agent-0.1.0-py3-none-any.whl --system
# MAGIC ```
