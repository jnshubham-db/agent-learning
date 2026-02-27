# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 9: Multi-Genie Agent with DSPy (without whl)
# MAGIC
# MAGIC This notebook implements a **supervisor agent** using **DSPy** that orchestrates two Databricks Genie agents:
# MAGIC - **Sales Analytics Genie** — Revenue, orders, product performance
# MAGIC - **Customer Insights Genie** — Support tickets, customer profiles, satisfaction
# MAGIC
# MAGIC **Architecture:**
# MAGIC - DSPy ReAct supervisor (CoT + tool calling) that routes queries to the appropriate Genie space(s)
# MAGIC - `query_sales_genie` and `query_customer_genie` tools using `WorkspaceClient().genie` API
# MAGIC - MLflow ResponsesAgent with streaming support
# MAGIC - Evaluation with cross-space queries, deployment, and querying
# MAGIC
# MAGIC **Prerequisites:** Run `create_genie_spaces` to create the Genie spaces, then paste the Space IDs below.

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
    "dspy", "mlflow>=3.0", "databricks-sdk", "pydantic>=2", "databricks-openai",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Genie Space IDs
# MAGIC
# MAGIC Replace with your actual Space IDs from `create_genie_spaces`, or leave as placeholders for local testing (tools will fail until configured).

# COMMAND ----------

SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create agent.py (Models-from-Code)
# MAGIC
# MAGIC Define the Genie query helper, tools, DSPy ReAct supervisor, and MLflow ResponsesAgent wrapper.

# COMMAND ----------

from pathlib import Path

_agent_code = '''\
"""
Multi-Genie DSPy Supervisor Agent.
Orchestrates Sales Analytics and Customer Insights Genie spaces via DSPy ReAct.
"""
import time
from typing import Generator

import dspy
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Genie space via WorkspaceClient().genie API."""
    w = WorkspaceClient()
    conversation = w.genie.start_conversation(space_id=space_id, content=question)
    result = w.genie.get_message(
        space_id=space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
    )
    while result.status in ("EXECUTING_QUERY", "SUBMITTED"):
        time.sleep(2)
        result = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )
    if result.attachments:
        for att in result.attachments:
            if att.text:
                return att.text.content
            if att.query:
                return f"Query: {att.query.query}\\nDescription: {att.query.description}"
    return "No results found."


def query_sales_genie(question: str) -> str:
    """Query the Sales Analytics Genie space."""
    return query_genie_space(SALES_GENIE_SPACE_ID, question)


def query_customer_genie(question: str) -> str:
    """Query the Customer Insights Genie space."""
    return query_genie_space(CUSTOMER_GENIE_SPACE_ID, question)


class MultiGenieDSPyAgent(ResponsesAgent):
    """Supervisor agent using DSPy ReAct to orchestrate Sales and Customer Genie spaces."""

    def __init__(self):
        super().__init__()
        lm = dspy.LM("databricks-claude-sonnet-4-5")
        dspy.configure(lm=lm)
        self.react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[query_sales_genie, query_customer_genie],
            max_iters=5,
        )

    def _run_agent(self, question: str) -> tuple[str, str]:
        result = self.react_agent(question=question)
        answer = getattr(result, "answer", "") or ""
        trajectory = getattr(result, "trajectory", []) or []
        reasoning = "\\n".join(str(t) for t in trajectory) if trajectory else ""
        return answer, reasoning

    def _messages_to_question(self, request: ResponsesAgentRequest) -> str:
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        if not input_list:
            return ""
        parts = []
        for msg in input_list:
            content = getattr(msg, "content", None) or (
                msg.get("content", "") if isinstance(msg, dict) else ""
            )
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif hasattr(c, "text"):
                        parts.append(c.text)
        return " ".join(parts).strip()

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        question = self._messages_to_question(request)
        answer, reasoning = self._run_agent(question)
        output_items = []
        if reasoning:
            output_items.append(self.create_reasoning_item(id="reason_1", reasoning_text=reasoning))
        output_items.append(self.create_text_output_item(text=answer, id="msg_1"))
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        question = self._messages_to_question(request)
        answer, reasoning = self._run_agent(question)
        item_id = "msg_1"
        if reasoning:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(id="reason_1", reasoning_text=reasoning),
            )
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=delta, item_id=item_id))
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=item_id),
        )


mlflow.dspy.autolog()
agent = MultiGenieDSPyAgent()
set_model(agent)
'''

Path("agent.py").write_text(_agent_code)
print("agent.py written successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Patch agent.py with Space IDs (if configured)
# MAGIC
# MAGIC If you set `SALES_GENIE_SPACE_ID` and `CUSTOMER_GENIE_SPACE_ID` above, patch agent.py so the logged model uses them.

# COMMAND ----------

# Patch agent.py with Space IDs if configured (replace placeholders)
try:
    with open("agent.py", "r") as f:
        c = f.read()
    if SALES_GENIE_SPACE_ID != "<YOUR-SALES-GENIE-SPACE-ID>":
        c = c.replace('"<YOUR-SALES-GENIE-SPACE-ID>"', repr(SALES_GENIE_SPACE_ID))
    if CUSTOMER_GENIE_SPACE_ID != "<YOUR-CUSTOMER-GENIE-SPACE-ID>":
        c = c.replace('"<YOUR-CUSTOMER-GENIE-SPACE-ID>"', repr(CUSTOMER_GENIE_SPACE_ID))
    with open("agent.py", "w") as f:
        f.write(c)
    print("agent.py patched with Space IDs.")
except FileNotFoundError:
    print("agent.py not found yet — run the previous cell first.")
except Exception as e:
    print(f"Patch skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Enable MLflow Autologging and Log the Model

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="multi_genie_dspy_agent"):
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        name="agent",
    )
    run_id = mlflow.active_run().info.run_id
    print(f"Logged agent. Run ID: {run_id}")
    print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluation with LLM-as-a-Judge
# MAGIC
# MAGIC Evaluate with cross-space queries: single-space (revenue, tickets) and multi-space (customers with tickets and order values).

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What was total revenue last month?"}]},
    {"input": [{"role": "user", "content": "How many open support tickets do we have?"}]},
    {"input": [{"role": "user", "content": "Which customers with open tickets have the highest order values?"}]},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[
        mlflow.genai.scorers.Safety(),
        mlflow.genai.scorers.RelevanceToQuery(),
    ],
)

print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Serving Deployment

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    f"runs:/{run_id}/agent",
    "sjdatabricks.agents.multi_genie_dspy",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy to Model Serving Endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="multi-genie-dspy-agent",
# MAGIC     config={
# MAGIC         "served_entities": [
# MAGIC             {
# MAGIC                 "name": "multi-genie-agent",
# MAGIC                 "entity_name": "sjdatabricks.agents.multi_genie_dspy",
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

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

w = WorkspaceClient()
client = DatabricksOpenAI(workspace_client=w)
endpoint_name = "multi-genie-dspy-agent"

input_msgs = [{"role": "user", "content": "What was total revenue last month?"}]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Streaming Call

# COMMAND ----------

response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    max_output_tokens=512,
)

print("Non-streaming response:")
for item in getattr(response, "output", response):
    if hasattr(item, "content"):
        for c in (item.content or []):
            if hasattr(c, "text") and c.text:
                print(c.text)
    elif hasattr(item, "text"):
        print(item.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Call

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
        for item in (chunk.output or []):
            if hasattr(item, "content"):
                for c in (item.content or []):
                    if hasattr(c, "delta") and c.delta:
                        print(c.delta, end="", flush=True)
            elif hasattr(item, "delta") and item.delta:
                print(item.delta, end="", flush=True)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Local Testing (Before Deployment)

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
test_request = {
    "input": [{"role": "user", "content": "What was total revenue last month?"}],
}
test_response = test_agent.predict(test_request)
print("Output items:")
for item in test_response.output:
    print(f"  - {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Set Model for MLflow

# COMMAND ----------

from mlflow.models import set_model

# Load agent from agent.py for set_model
import importlib.util
spec = importlib.util.spec_from_file_location("agent_module", "agent.py")
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
agent = agent_module.agent
mlflow.models.set_model(agent)
print("Model set. Agent ready for deployment.")
