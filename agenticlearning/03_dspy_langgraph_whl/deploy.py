# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 3: Deploy DSPy + LangGraph Agent (WHL Package)
# MAGIC
# MAGIC This notebook deploys the **order_support_dspy** wheel package, which contains a
# MAGIC three-node LangGraph agent powered by DSPy Chain-of-Thought:
# MAGIC
# MAGIC ```
# MAGIC classify -> lookup -> respond
# MAGIC ```
# MAGIC
# MAGIC **What is a wheel package and why package agents this way?**
# MAGIC
# MAGIC A Python wheel (`.whl`) is a built distribution format that bundles your source code,
# MAGIC metadata, and dependency declarations into a single installable archive. Packaging an
# MAGIC agent as a wheel gives you:
# MAGIC
# MAGIC - **Reproducibility**: pinned dependencies and versioned releases
# MAGIC - **Modularity**: separate files for nodes, tools, signatures, and the agent class
# MAGIC - **Testability**: standard `pytest` / `unittest` workflows against the installed package
# MAGIC - **Deployment consistency**: the same `.whl` runs locally, in CI, and on Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and Install the Wheel
# MAGIC
# MAGIC We use `uv` to build the wheel from `pyproject.toml`, then install it into the
# MAGIC notebook environment. The `hatchling` build backend reads `src/agent/` as the
# MAGIC package root.

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
# MAGIC ## 2. Log the Agent with MLflow (Code-Based Logging)
# MAGIC
# MAGIC **How code-based logging works in MLflow:**
# MAGIC
# MAGIC Instead of pickling an in-memory object, code-based logging points MLflow at a
# MAGIC Python source file that defines the agent. When the model is loaded (locally or on
# MAGIC Model Serving), MLflow executes that file and calls `set_model()` to obtain the
# MAGIC agent instance. This approach:
# MAGIC
# MAGIC - Avoids serialisation issues with complex objects (Spark sessions, LLM clients)
# MAGIC - Lets you version the agent code alongside the model artifact
# MAGIC - Works naturally with wheel packages: the source file imports from the installed package

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/{user}/03_dspy_langgraph_whl")

# The agent.py file calls set_model(DSPyLangGraphAgent()) at import time,
# so we pass the path to that file for code-based logging.
agent_script = os.path.join(_pkg_root, "src", "agent", "agent.py")

with mlflow.start_run():
    logged_info = mlflow.pyfunc.log_model(
        python_model=agent_script,
        name="order_support_dspy_langgraph",
        artifact_path="model",
        pip_requirements=[
            "dspy",
            "langgraph",
            "langchain-core",
            "mlflow>=3",
            "databricks-agents",
            "pyspark",
        ],
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri

print(f"Logged model. Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Logged Model
# MAGIC
# MAGIC Load the model back from MLflow and run a prediction to verify end-to-end
# MAGIC correctness before registering to Unity Catalog.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

test_request = {
    "input": [{"role": "user", "content": "What is the status of order 1042?"}],
}

response = loaded_agent.predict(test_request)
print("Test response:")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional Test Cases

# COMMAND ----------

test_cases = [
    "Are there any returns for order 1045?",
    "Tell me about the Laptop product",
    "What is the price and stock of the Monitor?",
]

for question in test_cases:
    req = {"input": [{"role": "user", "content": question}]}
    resp = loaded_agent.predict(req)
    print(f"Q: {question}")
    print(f"A: {resp}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC We use `mlflow.genai.evaluate` with built-in scorers to assess the agent on
# MAGIC safety and relevance. The scorers use an LLM judge to rate each response.

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order 1042?"}]},
    {"input": [{"role": "user", "content": "Are there any returns for order 1050?"}]},
    {"input": [{"role": "user", "content": "Tell me about the Tablet product"}]},
    {"input": [{"role": "user", "content": "What products are in the Electronics category?"}]},
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
# MAGIC
# MAGIC Once we are satisfied with the evaluation results, we register the model to
# MAGIC Unity Catalog. This makes it available for deployment to a Model Serving endpoint.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.dspy_langgraph_support_whl"

mlflow.set_registry_uri("databricks-uc")
registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {registered.name} (version {registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deploy to Model Serving
# MAGIC
# MAGIC Databricks Model Serving loads the registered Unity Catalog model, installs its
# MAGIC pip dependencies (including our wheel), and executes the `agent.py` script to
# MAGIC instantiate `DSPyLangGraphAgent`. Incoming HTTP requests are routed to `predict()`
# MAGIC or `predict_stream()` depending on whether the client requests streaming.
# MAGIC
# MAGIC The code below uses the Databricks SDK to create (or update) a serving endpoint
# MAGIC and waits for it to become ready.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

ENDPOINT_NAME = "dspy-langgraph-support-whl"

try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_NAME,
                    entity_version=str(registered.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
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
                    entity_version=str(registered.version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ],
        )
        print(f"Endpoint '{ENDPOINT_NAME}' updated and ready!")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Query the Deployed Endpoint — Non-Streaming
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
# MAGIC ## 8. Query the Deployed Endpoint — Streaming
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
