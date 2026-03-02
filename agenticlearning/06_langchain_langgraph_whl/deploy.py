# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 6: Deploy LangChain + LangGraph Customer Support Agent (WHL Package)
# MAGIC
# MAGIC This notebook deploys the **Customer Support Agent** built with LangChain and LangGraph,
# MAGIC packaged as a Python wheel (`.whl`). The agent implements a **classify -> lookup -> respond**
# MAGIC graph that routes customer questions to the appropriate Spark table and generates
# MAGIC natural-language answers.
# MAGIC
# MAGIC **What you will learn:**
# MAGIC - What a wheel package is and why we package agents this way
# MAGIC - How code-based logging works in MLflow
# MAGIC - How LangChain + LangGraph work together inside a wheel
# MAGIC - How to deploy the agent to Databricks Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC ## Background: What is a Wheel Package?
# MAGIC
# MAGIC A Python **wheel** (`.whl`) is a built distribution format — a ZIP archive with a
# MAGIC standardized filename and structure that `pip` can install directly without running
# MAGIC `setup.py`. Packaging an agent as a wheel provides several advantages:
# MAGIC
# MAGIC 1. **Reproducibility**: Dependencies are pinned and the code is versioned as a single artifact
# MAGIC 2. **Separation of concerns**: Agent logic lives in `src/agent/` with clear module boundaries
# MAGIC    (`nodes.py` for graph nodes, `tools.py` for data access, `agent.py` for the MLflow wrapper)
# MAGIC 3. **Testability**: Each module can be unit-tested independently before packaging
# MAGIC 4. **Deployment simplicity**: Model Serving installs the wheel and imports the agent —
# MAGIC    no notebook code needs to run at serving time
# MAGIC
# MAGIC Our wheel structure:
# MAGIC ```
# MAGIC 06_langchain_langgraph_whl/
# MAGIC +-- pyproject.toml          # Build config: name, version, dependencies
# MAGIC +-- deploy.py               # This notebook: build, log, evaluate, register
# MAGIC +-- src/agent/
# MAGIC     +-- __init__.py         # Package exports
# MAGIC     +-- agent.py            # LangChainLangGraphAgent (ResponsesAgent subclass)
# MAGIC     +-- nodes.py            # classify_node, lookup_node, respond_node
# MAGIC     +-- tools.py            # Spark-based tool functions
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and Install the Wheel
# MAGIC
# MAGIC We use `uv` (a fast Python package installer) to build the wheel from `pyproject.toml`,
# MAGIC then install it into the current environment. The `hatchling` build backend reads the
# MAGIC `[tool.hatch.build.targets.wheel]` config to know that `src/agent/` is the package root.

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
# MAGIC ## 2. Log Model with Code-Based Logging
# MAGIC
# MAGIC ### How Code-Based Logging Works
# MAGIC
# MAGIC MLflow's **models-from-code** feature lets you log a model by pointing to a Python file
# MAGIC rather than pickling an object. When we call:
# MAGIC
# MAGIC ```python
# MAGIC mlflow.pyfunc.log_model(python_model="src/agent/agent.py", ...)
# MAGIC ```
# MAGIC
# MAGIC MLflow does the following:
# MAGIC 1. Reads `src/agent/agent.py` and stores it as an artifact
# MAGIC 2. At load time, it imports the file and looks for the model registered via `set_model()`
# MAGIC 3. The `LangChainLangGraphAgent` instance is reconstructed — including its compiled
# MAGIC    LangGraph — without needing pickle or cloudpickle
# MAGIC
# MAGIC This approach avoids serialization issues with complex objects (Spark sessions,
# MAGIC LangGraph compiled graphs, LangChain LLM clients) and keeps the model artifact
# MAGIC human-readable.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# COMMAND ----------

with mlflow.start_run():
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
# MAGIC ## 3. Test the Agent Locally
# MAGIC
# MAGIC Before evaluation and registration, verify the agent works by loading it from
# MAGIC the logged model URI and running a sample prediction. The `mlflow.models.predict`
# MAGIC function loads the model and invokes `predict()` in a subprocess, simulating
# MAGIC what Model Serving will do.

# COMMAND ----------

test_input = {
    "input": [{"role": "user", "content": "What is the status of order 1042?"}],
}

result = mlflow.models.predict(
    model_uri=model_uri,
    input_data=test_input,
)
print("Test prediction result:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC MLflow's `genai.evaluate` runs the agent over a dataset and scores each response
# MAGIC using LLM-based scorers:
# MAGIC
# MAGIC - **Safety**: Checks the response does not contain harmful, biased, or inappropriate content
# MAGIC - **RelevanceToQuery**: Checks the response is relevant to the user's question
# MAGIC
# MAGIC The evaluation loads the model, runs each input, and stores results (including
# MAGIC per-row scores) in the MLflow tracking server for review.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order 1042?"}]},
    {"input": [{"role": "user", "content": "Tell me about the Wireless Headphones product"}]},
    {"input": [{"role": "user", "content": "I want to return order 2001, it arrived damaged"}]},
    {"input": [{"role": "user", "content": "Do you have any laptop accessories in stock?"}]},
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
# MAGIC Registering the model to Unity Catalog makes it available for governed access,
# MAGIC versioning, and deployment. The model is stored at:
# MAGIC
# MAGIC ```
# MAGIC sjdatabricks.agents.langchain_langgraph_whl_support
# MAGIC ```
# MAGIC
# MAGIC Once registered, it can be deployed to a Model Serving endpoint or shared with
# MAGIC other teams in your Databricks workspace.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.langchain_langgraph_whl_support"

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    f"runs:/{run_id}/model",
    UC_MODEL_NAME,
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. How LangChain + LangGraph Work Together in the Wheel
# MAGIC
# MAGIC The agent combines two complementary frameworks inside the wheel package:
# MAGIC
# MAGIC **LangChain** provides:
# MAGIC - `ChatDatabricks`: A standardized LLM client that connects to Databricks model endpoints
# MAGIC - Message types (`HumanMessage`, `SystemMessage`): Structured conversation primitives
# MAGIC - Tool integration patterns for connecting LLMs to external data sources
# MAGIC
# MAGIC **LangGraph** provides:
# MAGIC - `StateGraph`: A directed graph where each node transforms a shared state dictionary
# MAGIC - Deterministic routing: `classify -> lookup -> respond` executes in a fixed sequence
# MAGIC - Streaming support: `graph.stream(state, stream_mode="updates")` yields per-node updates
# MAGIC
# MAGIC The three-node pipeline works as follows:
# MAGIC
# MAGIC 1. **classify_node** (`nodes.py`): Sends the question to ChatDatabricks with a system prompt
# MAGIC    that asks it to classify intent as `order`, `return`, or `product`
# MAGIC 2. **lookup_node** (`nodes.py`): Based on the category, calls the appropriate tool function
# MAGIC    from `tools.py` — each tool queries a Spark table in `sjdatabricks.orders`
# MAGIC 3. **respond_node** (`nodes.py`): Sends the question + retrieved data to ChatDatabricks,
# MAGIC    which generates a natural-language answer
# MAGIC
# MAGIC The `agent.py` file wraps this compiled graph in an MLflow `ResponsesAgent`, exposing
# MAGIC `predict()` and `predict_stream()` methods that Model Serving can call.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving
# MAGIC
# MAGIC When you deploy a wheel-packaged agent to Databricks Model Serving:
# MAGIC
# MAGIC 1. **Serving container starts**: A container is provisioned with the Python environment
# MAGIC 2. **Wheel is installed**: The `.whl` artifact is installed via `pip`, bringing in all
# MAGIC    dependencies declared in `pyproject.toml` (langchain, langgraph, pyspark, etc.)
# MAGIC 3. **Agent module loads**: MLflow imports `src/agent/agent.py`, which calls `set_model()`
# MAGIC    to register the `LangChainLangGraphAgent` instance
# MAGIC 4. **Graph compiles**: The `__init__` method calls `build_graph()`, which constructs and
# MAGIC    compiles the LangGraph `StateGraph`
# MAGIC 5. **Requests are served**: Each incoming request is routed to `predict()` or
# MAGIC    `predict_stream()`, which runs the classify -> lookup -> respond pipeline
# MAGIC
# MAGIC The code below uses the Databricks SDK to create (or update) a serving endpoint
# MAGIC and waits for it to become ready.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

ENDPOINT_NAME = "langchain-langgraph-support-whl"

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Building Locally with uv
# MAGIC
# MAGIC To build the wheel outside Databricks for local testing or CI/CD:
# MAGIC
# MAGIC ```bash
# MAGIC cd 06_langchain_langgraph_whl
# MAGIC uv build --wheel
# MAGIC uv pip install dist/order_support_langchain-0.1.0-py3-none-any.whl --system
# MAGIC ```
# MAGIC
# MAGIC The wheel can then be uploaded to a Databricks workspace or artifact registry.
