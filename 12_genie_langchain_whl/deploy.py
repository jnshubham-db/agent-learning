# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 12: Simple Genie Agent with LangChain - WHL Package
# MAGIC
# MAGIC **USE CASE:** An agent that queries a Sales Analytics Genie Space using LangChain, packaged as a whl for deployment.
# MAGIC
# MAGIC This notebook:
# MAGIC - Builds and installs the whl locally
# MAGIC - Logs the model to MLflow with ResponsesAgent
# MAGIC - Evaluates with Safety and RelevanceToQuery
# MAGIC - Registers to Unity Catalog (`sjdatabricks.agents.genie_langchain_agent`)
# MAGIC - Deploys and provides query examples

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
# MAGIC ## 3. Configure Genie Space ID
# MAGIC
# MAGIC Replace with your Sales Genie Space ID from `create_genie_spaces`.

# COMMAND ----------

SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Import Agent, Create Instance, and Log Model

# COMMAND ----------

from agent import GenieLangChainAgent
import mlflow

mlflow.langchain.autolog()

agent = GenieLangChainAgent(space_id=SALES_GENIE_SPACE_ID)
mlflow.models.set_model(agent)

# COMMAND ----------

mlflow.set_experiment("12_genie_langchain_whl")

with mlflow.start_run(run_name="genie_langchain_agent"):
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
# MAGIC ## 5. Evaluation with mlflow.genai.evaluate

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

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

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: agent.predict(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print(f"Evaluation run ID: {results.run_id}")
if hasattr(results, "tables") and results.tables:
    print(results.tables.get("eval_results_table", "N/A"))
else:
    print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register to Unity Catalog

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.genie_langchain_agent"

mlflow.set_registry_uri("databricks-uc")
uc_registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {uc_registered.name} (version {uc_registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving Endpoint

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint_name = "genie-langchain-agent"

try:
    endpoint = client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": UC_MODEL_NAME,
                    "entity_version": uc_registered.version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
    )
    print(f"Endpoint created: {endpoint_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Endpoint {endpoint_name} already exists. Update if needed.")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Query Examples

# MAGIC ### Local Test (Non-Streaming)

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)
test_request = {
    "input": [
        {"role": "user", "content": "What were the top 5 products by revenue last quarter?"}
    ],
}
test_response = loaded_agent.predict(test_request)
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

# COMMAND ----------

from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()
input_msgs = [
    {"role": "user", "content": "What were the top 5 products by revenue last quarter?"}
]

response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    max_output_tokens=512,
)

print("Non-streaming response:")
for item in getattr(response, "output", []) or []:
    if hasattr(item, "content"):
        for c in (item.content or []):
            if hasattr(c, "text") and c.text:
                print(c.text)
    elif hasattr(item, "text"):
        print(item.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Query

# COMMAND ----------

streaming_response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    stream=True,
    max_output_tokens=512,
)

print("Streaming response:")
for chunk in streaming_response:
    if hasattr(chunk, "output"):
        for item in chunk.output or []:
            if hasattr(item, "content"):
                for c in (item.content or []):
                    if hasattr(c, "delta") and c.delta:
                        print(c.delta, end="", flush=True)
            elif hasattr(item, "delta") and item.delta:
                print(item.delta, end="", flush=True)
print()
