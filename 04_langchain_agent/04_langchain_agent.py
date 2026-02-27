# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 4: Customer Support Agent with LangChain (without whl)
# MAGIC
# MAGIC ## Use Case
# MAGIC A **Customer Support Agent** that can:
# MAGIC - Look up **order status** by order ID
# MAGIC - Get **product information** by product ID
# MAGIC - Process **refund requests**
# MAGIC
# MAGIC ## Architecture
# MAGIC - **LangChain** `create_tool_calling_agent` with 3 `@tool`-decorated functions
# MAGIC - **ChatDatabricks** LLM (`databricks-claude-sonnet-4-5`)
# MAGIC - **MLflow ResponsesAgent** wrapper for logging, tracing, and model serving
# MAGIC - **Models-from-code** for framework-agnostic deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "langchain", "langchain-community", "langchain-databricks", "mlflow>=3.0", "databricks-sdk", "pydantic>=2",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports and Setup

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety
from mlflow.langchain import autolog as langchain_autolog

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Agent Definition (agent.py)
# MAGIC
# MAGIC The agent is defined in `agent.py` in the same directory. It contains:
# MAGIC - **Tools**: `get_order_status`, `get_product_info`, `process_refund`
# MAGIC - **LangChain agent**: `create_tool_calling_agent` + `AgentExecutor`
# MAGIC - **ResponsesAgent wrapper**: `CustomerSupportResponsesAgent` implementing `predict()` and `predict_stream()`
# MAGIC - `mlflow.langchain.autolog()` and `@mlflow.trace(span_type=SpanType.AGENT)` decorators

# COMMAND ----------

# Load the agent module (agent.py must be in the same folder or on path)
import importlib.util
import os

def _notebook_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

_agent_path = os.path.join(_notebook_dir(), "agent.py")
if not os.path.exists(_agent_path):
    raise FileNotFoundError(f"agent.py not found at {_agent_path}. Ensure it is in the same directory as this notebook.")

spec = importlib.util.spec_from_file_location("agent", _agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

agent = agent_module.agent
print("Agent loaded:", type(agent).__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Model with MLflow (Models-from-Code)

# COMMAND ----------

agent_dir = os.path.dirname(_agent_path)

with mlflow.start_run(run_name="customer_support_agent"):
    langchain_autolog()
    logged_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(agent_dir, "agent.py"),
        artifact_path="model",
        name="agent",
    )
    run_id = mlflow.active_run().info.run_id
    print(f"Logged model. Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate with MLflow GenAI (LLM-as-a-Judge)

# COMMAND ----------

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}], "inputs": {"messages": [{"role": "user", "content": "What is the status of order ORD-12345678?"}]}},
    {"input": [{"role": "user", "content": "Tell me about product PROD-0001"}], "inputs": {"messages": [{"role": "user", "content": "Tell me about product PROD-0001"}]}},
    {"input": [{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}], "inputs": {"messages": [{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}]}},
]


def predict_for_eval(inputs):
    """Wrapper to extract outputs for scorers (Safety, RelevanceToQuery expect inputs/outputs)."""
    response = agent.predict(inputs)
    texts = []
    for item in (response.output or []):
        if isinstance(item, dict) and item.get("content"):
            for c in item.get("content", []):
                if isinstance(c, dict) and c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
    return " ".join(texts) if texts else str(response)


eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: predict_for_eval(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print("Evaluation run ID:", eval_results.run_id)
eval_results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Serving Deployment (UC Registration)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Register to Unity Catalog

# COMMAND ----------

catalog = "main"  # Replace with your UC catalog
schema = "ml"     # Replace with your UC schema
model_name_uc = f"{catalog}.{schema}.customer_support_agent"

# Get the model URI from the logged run
model_uri = f"runs:/{run_id}/model"

mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=model_name_uc,
)
print(f"Registered model: {model_name_uc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Create/Update Model Serving Endpoint
# MAGIC
# MAGIC In Databricks, deploy the registered model via:
# MAGIC 1. **MLflow Model Serving** (UI): Serving → Create serving endpoint → Select the UC model
# MAGIC 2. Or use **Databricks SDK** to create/update the endpoint programmatically

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Query the Deployed Agent

# MAGIC Use the **Databricks OpenAI client** to query the agent via Chat Completions API (non-streaming and streaming).

# COMMAND ----------

# MAGIC %pip install databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI

# WorkspaceClient automatically picks up Databricks credentials
w = WorkspaceClient()

# Get the serving endpoint URL for your model
# If using foundation model or custom endpoint, set base_url and api_key accordingly
# For MLflow-served agents, use the endpoint URL from Model Serving
serving_endpoint_name = "customer-support-agent"  # Replace with your endpoint name

# Create OpenAI-compatible client for Databricks
# For Chat Completions API with Databricks, use the endpoint URL
try:
    oai_client = w.serving_endpoints.build_client(serving_endpoint_name)
except Exception as e:
    print(f"Could not build serving client: {e}")
    print("Ensure the endpoint exists and you have access.")
    oai_client = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Non-Streaming Query (Chat Completions)

# COMMAND ----------

def query_agent_non_streaming(messages: list[dict], client=None):
    """Query the agent using Chat Completions API (non-streaming)."""
    if client is None:
        # Fallback: load model locally for testing without deployment
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        request = {"input": messages}
        response = model.predict(request)
        return response
    # Use serving client when available
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4-5",  # or your served model
        messages=messages,
        max_tokens=512,
    )
    return response.choices[0].message.content


# Local test (no deployment)
test_messages = [{"role": "user", "content": "What is the status of order ORD-12345678?"}]
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
response = loaded_model.predict({"input": test_messages})
print("Response output items:", response.output[:1] if response.output else "N/A")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Streaming Query

# COMMAND ----------

def query_agent_streaming(messages: list[dict], client=None):
    """Query the agent with streaming response."""
    if client is None:
        loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        for event in loaded.predict_stream({"input": messages}):
            ev = event if hasattr(event, "model_dump") else (event.__dict__ if hasattr(event, "__dict__") else event)
            delta = getattr(event, "delta", None) or (ev.get("delta") if isinstance(ev, dict) else None)
            if delta:
                yield delta
            item = getattr(event, "item", None) or (ev.get("item") if isinstance(ev, dict) else None)
            if item and isinstance(item, dict):
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        yield c.get("text", "")
        return
    stream = client.chat.completions.create(
        model="databricks-claude-sonnet-4-5",
        messages=messages,
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# Local streaming test
print("Streaming response:")
for text in query_agent_streaming(test_messages):
    print(text, end="", flush=True)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Using DatabricksOpenAI (Responses API)

# MAGIC For agents deployed as **Responses API** endpoints:

# COMMAND ----------

# from databricks_openai import DatabricksOpenAI
#
# client = DatabricksOpenAI()
# response = client.responses.create(
#     model="databricks-customer-support-agent",
#     input=[{"role": "user", "content": "What is the status of order ORD-12345678?"}],
#     max_output_tokens=512,
# )
# # Non-streaming
# print(response)
#
# # Streaming
# for event in client.responses.create(
#     model="databricks-customer-support-agent",
#     input=[{"role": "user", "content": "Tell me about product PROD-0001"}],
#     max_output_tokens=512,
#     stream=True,
# ):
#     if hasattr(event, "delta") and event.delta:
#         print(event.delta, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC 1. **Agent** built with LangChain `create_tool_calling_agent` and 3 tools
# MAGIC 2. **ResponsesAgent** wrapper for MLflow compatibility
# MAGIC 3. **Models-from-code** logging with `mlflow.pyfunc.log_model(python_model="agent.py")`
# MAGIC 4. **Evaluation** with `mlflow.genai.evaluate` and Safety/RelevanceToQuery scorers
# MAGIC 5. **UC registration** for governance and deployment
# MAGIC 6. **Querying** via Chat Completions and Responses API (streaming and non-streaming)
