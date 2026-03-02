# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 11 — Deploy: Simple Genie Agent with DSPy (WHL)
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook deploys the **Genie DSPy Agent** packaged as a Python wheel.
# MAGIC
# MAGIC **Architecture:**
# MAGIC
# MAGIC | Component | Role |
# MAGIC |-----------|------|
# MAGIC | `genie.py` | DSPy `OrderQuery` signature + `ChainOfThought` router that decides if a question needs Genie data |
# MAGIC | `agent.py` | `GenieDSPyAgent(ResponsesAgent)` — routes via CoT, calls Genie Space or falls back to LLM |
# MAGIC | `pyproject.toml` | Hatchling build config for the `genie_dspy_agent` wheel |
# MAGIC | `deploy.py` | *This notebook* — build, log, evaluate, register |
# MAGIC
# MAGIC **Flow:** User question ➜ DSPy CoT router (`OrderQuery`) ➜ Genie Space (data) *or* DSPy CoT fallback (general) ➜ ResponsesAgent response
# MAGIC
# MAGIC The `GENIE_SPACE_ID` environment variable configures which Genie Space the agent queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and Install the Wheel
# MAGIC
# MAGIC We use `uv` + `hatchling` to build the wheel from `pyproject.toml`, then install it
# MAGIC into the current cluster environment.

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
# MAGIC ## 2. Install Additional Dependencies
# MAGIC
# MAGIC Ensure the agent's runtime dependencies are present.

# COMMAND ----------

import subprocess as _sp

_sp.check_call([
    "uv", "pip", "install",
    "dspy", "mlflow>=3", "databricks-agents", "databricks-sdk", "pyspark",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set GENIE_SPACE_ID
# MAGIC
# MAGIC Use a Databricks widget so the Genie Space ID can be configured at run time.
# MAGIC The agent reads this value from the `GENIE_SPACE_ID` environment variable.

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "", "Genie Space ID")
_genie_space_id = dbutils.widgets.get("genie_space_id")

if not _genie_space_id or _genie_space_id.startswith("<"):
    raise ValueError(
        "Please set the 'genie_space_id' widget to a valid Genie Space ID "
        "before running the remaining cells."
    )

os.environ["GENIE_SPACE_ID"] = _genie_space_id
print(f"GENIE_SPACE_ID set to: {_genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Model with MLflow (Code-Based Logging)
# MAGIC
# MAGIC The `agent.py` module calls `set_model(GenieDSPyAgent())` at import time.
# MAGIC We log using the **models-from-code** path so MLflow captures the full source.

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()
mlflow.set_experiment("/Users/{}/11_genie_dspy_whl".format(
    spark.sql("SELECT current_user()").first()[0]
))

with mlflow.start_run(run_name="genie_dspy_agent_whl") as run:
    logged_info = mlflow.pyfunc.log_model(
        python_model="src/agent/agent.py",
        artifact_path="model",
        name="agent",
    )
    run_id = run.info.run_id
    model_uri = logged_info.model_uri

print(f"Logged agent. Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test with `mlflow.models.predict()`
# MAGIC
# MAGIC Load the model back from MLflow and run a quick smoke test.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(model_uri)

test_request = {
    "input": [
        {"role": "user", "content": "What is the status of order 1042?"}
    ],
}
test_response = loaded_agent.predict(test_request)

print("Test output items:")
for item in test_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC Use `mlflow.genai.evaluate()` with **Safety** and **RelevanceToQuery** scorers
# MAGIC to assess the agent on representative questions.

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

eval_dataset = [
    {
        "input": [
            {"role": "user", "content": "What is the status of order 1042?"}
        ]
    },
    {
        "input": [
            {"role": "user", "content": "Show me all returns in the last 30 days"}
        ]
    },
    {
        "input": [
            {"role": "user", "content": "What products are in the Electronics category?"}
        ]
    },
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print("Evaluation complete. Check the MLflow UI for detailed results.")
print(eval_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register to Unity Catalog
# MAGIC
# MAGIC Register the logged model to the `sjdatabricks.agents` schema so it can be
# MAGIC deployed to a Model Serving endpoint.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.genie_dspy_agent"

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri,
    UC_MODEL_NAME,
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deploy to Model Serving
# MAGIC
# MAGIC Create a serving endpoint. The `GENIE_SPACE_ID` environment variable is configured
# MAGIC on the endpoint so the agent can reach the Genie Space at inference time. The code
# MAGIC below creates (or updates) the endpoint and waits for it to become ready.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

ENDPOINT_NAME = "genie-dspy-whl-agent"

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
                    environment_vars={"GENIE_SPACE_ID": _genie_space_id},
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
                    environment_vars={"GENIE_SPACE_ID": _genie_space_id},
                )
            ],
        )
        print(f"Endpoint '{ENDPOINT_NAME}' updated and ready!")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Query the Deployed Endpoint — Non-Streaming
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
# MAGIC ## 10. Query the Deployed Endpoint — Streaming
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
