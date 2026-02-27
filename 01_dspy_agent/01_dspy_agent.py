# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 1: DSPy Customer Support Agent (without whl)
# MAGIC
# MAGIC This notebook implements a **Customer Support Agent** using **DSPy** with Chain-of-Thought (CoT) reasoning and tool-calling capabilities.
# MAGIC
# MAGIC **Use Case:** The agent can:
# MAGIC - Look up order status
# MAGIC - Check product information
# MAGIC - Handle refund requests
# MAGIC
# MAGIC **Key Components:**
# MAGIC - DSPy ReAct module (CoT + tool calling)
# MAGIC - MLflow ResponsesAgent interface for serving
# MAGIC - MLflow evaluation with LLM-as-a-judge
# MAGIC - Model serving deployment and querying

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies
# MAGIC
# MAGIC Install `dspy`, `mlflow`, `databricks-sdk`, and `pydantic` for the agent framework.

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
# MAGIC ## 2. Configure DSPy and Define Tools
# MAGIC
# MAGIC Configure DSPy to use Databricks Claude Sonnet and define the three support tools.

# COMMAND ----------

import dspy

lm = dspy.LM("databricks-claude-sonnet-4-5")
dspy.configure(lm=lm)

# COMMAND ----------

def get_order_status(order_id: str) -> str:
    """
    Get the status of an order by its ID.

    Args:
        order_id: The order ID (e.g., ORD-12345678)

    Returns:
        Status information for the order
    """
    # Simulated order data - in production, this would query a database/API
    order_db = {
        "ORD-12345678": {"status": "Shipped", "tracking": "1Z999AA10123456784", "eta": "Feb 28, 2025"},
        "ORD-87654321": {"status": "Delivered", "delivered_date": "Feb 25, 2025"},
        "ORD-11111111": {"status": "Processing", "estimated_ship": "Mar 1, 2025"},
    }
    if order_id in order_db:
        info = order_db[order_id]
        return f"Order {order_id}: {info.get('status', 'Unknown')}. " + " ".join(f"{k}: {v}" for k, v in info.items() if k != "status")
    return f"Order {order_id} not found. Please verify the order ID."


def get_product_info(product_id: str) -> str:
    """
    Get product information by product ID.

    Args:
        product_id: The product ID (e.g., PROD-0001)

    Returns:
        Product details including name, price, and availability
    """
    # Simulated product catalog
    product_db = {
        "PROD-0001": {"name": "Wireless Headphones", "price": "$99.99", "in_stock": True, "rating": 4.5},
        "PROD-0002": {"name": "USB-C Hub", "price": "$49.99", "in_stock": True, "rating": 4.2},
        "PROD-0003": {"name": "Mechanical Keyboard", "price": "$129.99", "in_stock": False, "rating": 4.8},
    }
    if product_id in product_db:
        p = product_db[product_id]
        return f"{p['name']} ({product_id}): {p['price']}, In stock: {p['in_stock']}, Rating: {p['rating']}/5"
    return f"Product {product_id} not found."


def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund request for an order.

    Args:
        order_id: The order ID to refund
        reason: The reason for the refund (e.g., defective, wrong item)

    Returns:
        Refund confirmation or status
    """
    # Simulated refund processing
    return f"Refund request submitted for order {order_id}. Reason: {reason}. Refund ID: REF-{order_id[-6:]}. Status: Pending approval. You will receive an email within 2-3 business days."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create agent.py (Models-from-Code)
# MAGIC
# MAGIC Write the full agent implementation to `agent.py` for MLflow models-from-code logging. The agent wraps DSPy ReAct in an MLflow ResponsesAgent interface.

# COMMAND ----------

from pathlib import Path

_agent_code = '''\
"""
DSPy Customer Support Agent - MLflow ResponsesAgent implementation.
Uses DSPy ReAct (Chain-of-Thought + tool calling) for order status, product info, and refunds.
"""
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


def get_order_status(order_id: str) -> str:
    """Get the status of an order by its ID."""
    order_db = {
        "ORD-12345678": {"status": "Shipped", "tracking": "1Z999AA10123456784", "eta": "Feb 28, 2025"},
        "ORD-87654321": {"status": "Delivered", "delivered_date": "Feb 25, 2025"},
        "ORD-11111111": {"status": "Processing", "estimated_ship": "Mar 1, 2025"},
    }
    if order_id in order_db:
        info = order_db[order_id]
        return f"Order {order_id}: {info.get(\'status\', \'Unknown\')}. " + " ".join(
            f"{k}: {v}" for k, v in info.items() if k != "status"
        )
    return f"Order {order_id} not found."


def get_product_info(product_id: str) -> str:
    """Get product information by product ID."""
    product_db = {
        "PROD-0001": {"name": "Wireless Headphones", "price": "$99.99", "in_stock": True, "rating": 4.5},
        "PROD-0002": {"name": "USB-C Hub", "price": "$49.99", "in_stock": True, "rating": 4.2},
        "PROD-0003": {"name": "Mechanical Keyboard", "price": "$129.99", "in_stock": False, "rating": 4.8},
    }
    if product_id in product_db:
        p = product_db[product_id]
        return f"{p[\'name\']} ({product_id}): {p[\'price\']}, In stock: {p[\'in_stock\']}, Rating: {p[\'rating\']}/5"
    return f"Product {product_id} not found."


def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order."""
    return (
        f"Refund request submitted for order {order_id}. "
        f"Reason: {reason}. Refund ID: REF-{order_id[-6:]}. Status: Pending approval."
    )


class DSPyCustomerSupportAgent(ResponsesAgent):
    """Customer Support Agent using DSPy ReAct (CoT + tools) wrapped in MLflow ResponsesAgent."""

    def __init__(self):
        super().__init__()
        lm = dspy.LM("databricks-claude-sonnet-4-5")
        dspy.configure(lm=lm)
        self.react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[get_order_status, get_product_info, process_refund],
            max_iters=5,
        )

    def _run_agent(self, question: str) -> tuple[str, str]:
        """Run DSPy ReAct and return (answer, trajectory/reasoning)."""
        result = self.react_agent(question=question)
        answer = getattr(result, "answer", "") or ""
        trajectory = getattr(result, "trajectory", []) or []
        reasoning = "\\n".join(str(t) for t in trajectory) if trajectory else ""
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
agent = DSPyCustomerSupportAgent()
set_model(agent)
'''

Path("agent.py").write_text(_agent_code)
print("agent.py written successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Enable MLflow Autologging and Log the Model
# MAGIC
# MAGIC Enable DSPy autologging for traces and log the agent using models-from-code.

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="dspy_customer_support_agent"):
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        name="agent",
    )
    run_id = mlflow.active_run().info.run_id
    print(f"Logged agent. Run ID: {run_id}")
    print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluation with LLM-as-a-Judge
# MAGIC
# MAGIC Evaluate the agent using MLflow GenAI evaluation with Safety and RelevanceToQuery scorers.

# COMMAND ----------

loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}]},
    {"input": [{"role": "user", "content": "Tell me about product PROD-0001"}]},
    {"input": [{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}]},
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
# MAGIC ## 6. Model Serving Deployment
# MAGIC
# MAGIC Register the model to Unity Catalog for deployment to Databricks Model Serving.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    f"runs:/{run_id}/agent",
    "sjdatabricks.agents.dspy_customer_support",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy to Model Serving Endpoint
# MAGIC
# MAGIC Use the Databricks SDK to create a serving endpoint (run in a separate cell or workflow after model is approved).

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC client = get_deploy_client("databricks")
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="dspy-customer-support-agent",
# MAGIC     config={
# MAGIC         "served_entities": [
# MAGIC             {
# MAGIC                 "name": "dspy-agent",
# MAGIC                 "entity_name": "sjdatabricks.agents.dspy_customer_support",
# MAGIC                 "entity_version": str(model_info.version),
# MAGIC                 "workload_size": "Small",
# MAGIC                 "scale_to_zero_enabled": True,
# MAGIC             }
# MAGIC         ],
# MAGIC         "traffic_config": {
# MAGIC             "routes": [
# MAGIC                 {"served_model_name": "dspy-agent", "traffic_percentage": 100}
# MAGIC             ]
# MAGIC         },
# MAGIC     },
# MAGIC )
# MAGIC print(f"Endpoint created: {endpoint.name}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Querying the Deployed Agent
# MAGIC
# MAGIC Use the Databricks OpenAI client to invoke the agent with both non-streaming and streaming calls.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Streaming Call
# MAGIC
# MAGIC Use `client.responses.create()` without `stream=True` for a single response.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

w = WorkspaceClient()
client = DatabricksOpenAI(workspace_client=w)

# Replace with your deployed endpoint name (e.g., "dspy-customer-support-agent" or "apps/<app-name>")
endpoint_name = "dspy-customer-support-agent"

input_msgs = [{"role": "user", "content": "What is the status of order ORD-12345678?"}]

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
# MAGIC
# MAGIC Use `stream=True` to receive incremental updates as they are generated.

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
# MAGIC ## 8. Local Testing (Before Deployment)
# MAGIC
# MAGIC Test the agent locally without deploying by loading and invoking it directly.

# COMMAND ----------

test_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
test_request = {
    "input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}],
}
test_response = test_agent.predict(test_request)
print("Output items:")
for item in test_response.output:
    print(f"  - {item}")
