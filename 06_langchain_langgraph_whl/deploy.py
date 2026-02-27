# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 6: Customer Support Agent (LangChain + LangGraph) - WHL Package
# MAGIC
# MAGIC This notebook deploys the **langchain-langgraph-agent** whl package:
# MAGIC - Builds and installs the whl locally
# MAGIC - Logs the model to MLflow with ResponsesAgent
# MAGIC - Evaluates with Safety and RelevanceToQuery
# MAGIC - Registers to Unity Catalog
# MAGIC - Provides deployment and query examples

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
# MAGIC ## 3. Import Agent and Log Model

# COMMAND ----------

from agent import CustomerSupportAgent
import mlflow

mlflow.langchain.autolog()

agent = CustomerSupportAgent()
mlflow.models.set_model(agent)

# COMMAND ----------

mlflow.set_experiment("06_langchain_langgraph_whl")

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
# MAGIC ## 4. Evaluation with mlflow.genai.evaluate

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

loaded_agent = mlflow.pyfunc.load_model(model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}]},
    {"input": [{"role": "user", "content": "Tell me about product PROD-0001"}]},
    {"input": [{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}]},
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
# MAGIC ## 5. Register to Unity Catalog

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.langchain_langgraph_support_whl"

mlflow.set_registry_uri("databricks-uc")
registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {registered.name} (version {registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deployment Example (Model Serving Endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the MLflow Deployments API to create a serving endpoint:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="langchain-langgraph-support-whl",
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
# MAGIC ## 7. Query Examples

# MAGIC %md
# MAGIC ### Local Test (Non-Streaming)

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(model_uri)
test_request = {"input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}]}
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
# MAGIC endpoint_name = "langchain-langgraph-support-whl"
# MAGIC
# MAGIC response = client.responses.create(
# MAGIC     model=endpoint_name,
# MAGIC     input=[{"role": "user", "content": "What is the status of order ORD-12345678?"}],
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
# MAGIC     input=[{"role": "user", "content": "Tell me about product PROD-0001 and if it's in stock"}],
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
