# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 7: Simple Genie Agent with DSPy (without whl)
# MAGIC
# MAGIC ## Use Case
# MAGIC A **Sales Analytics Agent** that connects to a **Genie Space** for natural-language queries over sales data. The agent uses:
# MAGIC - **DSPy Chain-of-Thought (CoT)** to understand user questions
# MAGIC - **Databricks Genie API** to query the Sales Analytics space
# MAGIC - **MLflow ResponsesAgent** for logging, tracing, evaluation, and model serving
# MAGIC
# MAGIC ## Architecture
# MAGIC - **DSPy ReAct** with `query_genie_space` as the single tool
# MAGIC - **Databricks Genie** `WorkspaceClient().genie` for conversation API
# MAGIC - **MLflow ResponsesAgent** with `predict()` and `predict_stream()` (streaming via `create_text_delta()`, reasoning via `create_reasoning_item()`)
# MAGIC - **Databricks Claude Sonnet 4.5** via `dspy.LM("databricks-claude-sonnet-4-5")`

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
# MAGIC ## 2. Imports and Genie Query Function

# COMMAND ----------

import time
from typing import Generator

import dspy
import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Replace with your Sales Genie Space ID from Databricks Genie
SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Databricks Genie space with a natural language question.
    
    Args:
        space_id: The Genie space ID (e.g., Sales Analytics space)
        question: The user's natural language question
        
    Returns:
        The Genie response as text (from attachments or query description)
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    # Option 1: Use start_conversation with polling (as specified)
    conversation = w.genie.start_conversation(space_id=space_id, content=question)
    result = w.genie.get_message(
        space_id=space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
    )
    while getattr(result, "status", "") in ("EXECUTING_QUERY", "SUBMITTED"):
        time.sleep(2)
        result = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )
    # Extract result from attachments
    attachments = getattr(result, "attachments", None) or []
    for att in attachments:
        if getattr(att, "text", None) and att.text:
            return getattr(att.text, "content", str(att.text))
        if getattr(att, "query", None) and att.query:
            q = att.query
            return f"Query: {getattr(q, 'query', '')}\nDescription: {getattr(q, 'description', '')}"
    return "No results found."


# Create a version bound to the Sales Genie space for use as DSPy tool
def _query_sales_genie(question: str) -> str:
    """Query the Sales Analytics Genie space. Use for sales-related questions."""
    return query_genie_space(SALES_GENIE_SPACE_ID, question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure DSPy and Create Agent

# COMMAND ----------

_lm = dspy.LM("databricks-claude-sonnet-4-5")
dspy.configure(lm=_lm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap in MLflow ResponsesAgent

# MAGIC Subclass `ResponsesAgent`, implement `predict()` and `predict_stream()` with:
# MAGIC - CoT reasoning exposed via `create_reasoning_item()`
# MAGIC - Streaming via `create_text_delta()`

# COMMAND ----------

mlflow.dspy.autolog()


class GenieSalesAnalyticsAgent(ResponsesAgent):
    """DSPy-based agent wrapping a Genie Sales Analytics space in MLflow ResponsesAgent."""

    def __init__(self):
        super().__init__()
        lm = dspy.LM("databricks-claude-sonnet-4-5")
        dspy.configure(lm=lm)
        self.react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[_query_sales_genie],
            max_iters=5,
        )

    def _run_agent(self, question: str) -> tuple[str, str]:
        """Run DSPy ReAct and return (answer, trajectory/reasoning)."""
        result = self.react_agent(question=question)
        answer = getattr(result, "answer", "") or ""
        trajectory = getattr(result, "trajectory", []) or []
        reasoning = "\n".join(str(t) for t in trajectory) if trajectory else ""
        return answer, reasoning

    def _messages_to_question(self, request: ResponsesAgentRequest) -> str:
        """Extract user question from request input messages."""
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        if not input_list:
            return ""
        parts = []
        for msg in input_list:
            content = getattr(msg, "content", None) or msg.get("content", "")
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
            output_items.append(
                self.create_reasoning_item(id="reason_1", reasoning_text=reasoning)
            )
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
                item=self.create_reasoning_item(
                    id="reason_1", reasoning_text=reasoning
                ),
            )
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=item_id)
            )
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=item_id),
        )


agent = GenieSalesAnalyticsAgent()
set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log Model to MLflow

# MAGIC Enable DSPy autologging and log the agent using models-from-code.

# COMMAND ----------

mlflow.dspy.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="genie_dspy_sales_agent"):
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
        artifact_path="model",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_agent_info.model_uri
    print(f"Logged agent. Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluation with LLM-as-a-Judge

# MAGIC Evaluate the agent using `mlflow.genai.evaluate()` with Safety and RelevanceToQuery scorers.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

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

# Note: Evaluation requires a valid SALES_GENIE_SPACE_ID. Replace placeholder before running.
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

# MAGIC Register the model to Unity Catalog for deployment to Databricks Model Serving.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri,
    "sjdatabricks.agents.genie_dspy_sales",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy to Model Serving Endpoint

# MAGIC Use the Databricks SDK or MLflow deployments to create a serving endpoint.

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
# MAGIC                 "entity_name": "sjdatabricks.agents.genie_dspy_sales",
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
# MAGIC ## 8. Querying the Deployed Agent

# MAGIC Use the **DatabricksOpenAI** client to invoke the agent (Responses API) with non-streaming and streaming calls.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Streaming Call

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

w = WorkspaceClient()
client = DatabricksOpenAI(workspace_client=w)

# Replace with your deployed endpoint name
endpoint_name = "genie-dspy-sales-agent"

input_msgs = [
    {"role": "user", "content": "What were the top 5 products by revenue last quarter?"}
]

response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    max_output_tokens=512,
)

print("Non-streaming response:")
for item in getattr(response, "output", response) or []:
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
        for item in chunk.output or []:
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

# MAGIC Test the agent locally without deploying. **Requires valid `SALES_GENIE_SPACE_ID`**.

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
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
# MAGIC ## 10. Set Model for MLflow

# MAGIC Ensure MLflow knows the active agent for inference and serving.

# COMMAND ----------

mlflow.models.set_model(agent)
print("Model set for MLflow. Genie DSPy agent ready for deployment.")
