# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 5: Customer Support Agent with LangChain and LangGraph
# MAGIC
# MAGIC This notebook implements a **Customer Support Agent** using LangChain and LangGraph with:
# MAGIC - **Router Node**: LLM classifies user intent (order, product, refund) and routes to specialized nodes
# MAGIC - **Order Handler**: Tools `get_order_status`, `get_tracking_info` for order inquiries
# MAGIC - **Product Handler**: Tools `get_product_info`, `check_inventory` for product questions
# MAGIC - **Refund Handler**: Tool `process_refund` for returns and escalations
# MAGIC
# MAGIC The agent uses `ChatDatabricks` (Claude Sonnet 4.5), is wrapped in MLflow `ResponsesAgent`, and supports evaluation, UC registration, and model serving queries.
# MAGIC
# MAGIC > **Note**: No WHL packaging; uses Chain-of-Thought (CoT) via LLM reasoning. All code is inline.

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "langchain", "langchain-community", "langchain-databricks", "langgraph", "mlflow>=3.0", "databricks-sdk", "databricks-openai", "pydantic>=2",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and Configuration

# COMMAND ----------

from typing import Generator, Literal

import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.chat_models.databricks import ChatDatabricks

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
UC_MODEL_NAME = "sjdatabricks.agents.langchain_langgraph_support"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Tools for Specialized Nodes
# MAGIC
# MAGIC Each handler node has its own set of tools. These simulate backend API calls.

# COMMAND ----------


@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of an order by order ID. Use for questions about order status, shipment, or delivery."""
    # Simulated order database
    orders = {
        "ORD-12345678": "Shipped - In transit, estimated delivery Feb 28",
        "ORD-87654321": "Delivered on Feb 20",
        "ORD-11111111": "Processing - Expected to ship within 2 days",
    }
    return orders.get(order_id.upper(), f"Order {order_id} not found. Please verify the order ID.")


@tool
def get_tracking_info(order_id: str) -> str:
    """Get tracking information for a shipped order. Use when the customer wants to track their package."""
    tracking = {
        "ORD-12345678": "Tracking: 1Z999AA10123456784 | Carrier: UPS | Last update: Departed Memphis, TN",
        "ORD-87654321": "Delivered. Final scan: Feb 20, 2:34 PM at front door.",
    }
    return tracking.get(order_id.upper(), f"No tracking found for {order_id}. Order may not be shipped yet.")


@tool
def get_product_info(product_id: str) -> str:
    """Get product details including name, description, and price. Use for product information questions."""
    products = {
        "PROD-0001": "Product: Wireless Headphones Pro | Price: $149.99 | Features: 30hr battery, noise cancellation, Bluetooth 5.3",
        "PROD-0002": "Product: USB-C Hub 7-in-1 | Price: $49.99 | Features: HDMI, USB 3.0, SD card reader",
    }
    return products.get(product_id.upper(), f"Product {product_id} not found in catalog.")


@tool
def check_inventory(product_id: str) -> str:
    """Check if a product is in stock and how many units are available."""
    inventory = {
        "PROD-0001": "In stock: 234 units",
        "PROD-0002": "Low stock: 12 units",
    }
    return inventory.get(product_id.upper(), f"Inventory status unknown for {product_id}.")


@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order. Use when customer wants to return, refund, or report defective items."""
    return (
        f"Refund request submitted for order {order_id} (reason: {reason}). "
        "Our team will review within 24-48 hours. You will receive an email with the outcome."
    )


# Order tools
ORDER_TOOLS = [get_order_status, get_tracking_info]
# Product tools
PRODUCT_TOOLS = [get_product_info, check_inventory]
# Refund tools
REFUND_TOOLS = [process_refund]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the LangGraph
# MAGIC
# MAGIC - **Router**: Uses LLM to classify intent → routes to `order`, `product`, or `refund`
# MAGIC - **Order Handler**: Binds order tools, invokes LLM with tool calling
# MAGIC - **Product Handler**: Binds product tools, invokes LLM with tool calling
# MAGIC - **Refund Handler**: Binds refund tool, invokes LLM with tool calling

# COMMAND ----------

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0)

# Router prompt
ROUTER_PROMPT = """You are a customer support intent classifier. Given the user message, respond with exactly one word:
- "order" if the user asks about order status, tracking, shipment, or delivery
- "product" if the user asks about product info, inventory, pricing, or specifications
- "refund" if the user wants to return, refund, or report defective items

Respond with only the single word. No punctuation."""


def router_node(state: MessagesState) -> dict:
    """Route to the appropriate handler based on user intent."""
    llm_router = llm.with_config({"tags": ["router"]})
    response = llm_router.invoke(
        [SystemMessage(content=ROUTER_PROMPT), state["messages"][-1]]
    )
    intent = response.content.strip().lower().replace(".", "")
    return {"messages": [response]}


def route_condition(state: MessagesState) -> Literal["order_handler", "product_handler", "refund_handler"]:
    """Conditional routing based on router output."""
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "") or ""
    text = str(content).strip().lower()
    if "order" in text:
        return "order_handler"
    if "product" in text:
        return "product_handler"
    if "refund" in text:
        return "refund_handler"
    return "order_handler"  # Default


# Specialized handler nodes with tool binding (LangChain tool calling)
order_llm = llm.bind_tools(ORDER_TOOLS)
product_llm = llm.bind_tools(PRODUCT_TOOLS)
refund_llm = llm.bind_tools(REFUND_TOOLS)

order_tool_node = ToolNode(ORDER_TOOLS)
product_tool_node = ToolNode(PRODUCT_TOOLS)
refund_tool_node = ToolNode(REFUND_TOOLS)


def order_handler_node(state: MessagesState) -> dict:
    """Handle order-related queries using tools."""
    response = order_llm.invoke(state["messages"])
    return {"messages": [response]}


def product_handler_node(state: MessagesState) -> dict:
    """Handle product-related queries using tools."""
    response = product_llm.invoke(state["messages"])
    return {"messages": [response]}


def refund_handler_node(state: MessagesState) -> dict:
    """Handle refund and return requests using tools."""
    response = refund_llm.invoke(state["messages"])
    return {"messages": [response]}


def _has_tool_calls(state: MessagesState) -> bool:
    """Check if the last message has tool calls."""
    if not state.get("messages"):
        return False
    last = state["messages"][-1]
    return bool(getattr(last, "tool_calls", None))


def order_tools_condition(state: MessagesState) -> Literal["order_tools", "__end__"]:
    """Route order handler: tools node or END."""
    return "order_tools" if _has_tool_calls(state) else "__end__"


def product_tools_condition(state: MessagesState) -> Literal["product_tools", "__end__"]:
    """Route product handler: tools node or END."""
    return "product_tools" if _has_tool_calls(state) else "__end__"


def refund_tools_condition(state: MessagesState) -> Literal["refund_tools", "__end__"]:
    """Route refund handler: tools node or END."""
    return "refund_tools" if _has_tool_calls(state) else "__end__"


# Build the graph
builder = StateGraph(MessagesState)

builder.add_node("router", router_node)
builder.add_node("order_handler", order_handler_node)
builder.add_node("product_handler", product_handler_node)
builder.add_node("refund_handler", refund_handler_node)
builder.add_node("order_tools", order_tool_node)
builder.add_node("product_tools", product_tool_node)
builder.add_node("refund_tools", refund_tool_node)

# Entry
builder.add_edge(START, "router")

# Router conditional edges -> specialized handlers
builder.add_conditional_edges("router", route_condition)

# Handler -> tools (if tool_calls) or END; tools -> back to handler
builder.add_conditional_edges("order_handler", order_tools_condition)
builder.add_edge("order_tools", "order_handler")

builder.add_conditional_edges("product_handler", product_tools_condition)
builder.add_edge("product_tools", "product_handler")

builder.add_conditional_edges("refund_handler", refund_tools_condition)
builder.add_edge("refund_tools", "refund_handler")

graph = builder.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap in MLflow ResponsesAgent
# MAGIC
# MAGIC Subclass `ResponsesAgent`, implement `predict` and `predict_stream` using `to_chat_completions_input` and `output_to_responses_items_stream`.

# COMMAND ----------

mlflow.langchain.autolog()


class LangGraphSupportAgent(ResponsesAgent):
    """Customer Support Agent wrapping a compiled LangGraph."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request) -> ResponsesAgentResponse:
        """Collect done items from predict_stream."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        custom = getattr(request, "custom_inputs", None) or (request.get("custom_inputs") if isinstance(request, dict) else None)
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream graph updates and yield response items."""
        inp = getattr(request, "input", None)
        if inp is None and isinstance(request, dict):
            inp = request.get("input", [])
        items = [i.model_dump() if hasattr(i, "model_dump") else i for i in inp]
        cc_msgs = to_chat_completions_input(items)

        for _, events in self.agent.stream(
            {"messages": cc_msgs}, stream_mode=["updates"]
        ):
            for node_data in events.values():
                if "messages" in node_data:
                    yield from output_to_responses_items_stream(node_data["messages"])


agent = LangGraphSupportAgent(graph)
set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Local Test

# COMMAND ----------

test_input = {
    "input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}],
}
result = agent.predict(test_input)
print("Test response (first text output):")
for item in result.output:
    if isinstance(item, dict):
        content = item.get("content", [])
        for c in content:
            if isinstance(c, dict) and "text" in c:
                print(c["text"])
                break
    elif hasattr(item, "content"):
        for c in (item.content or []):
            if hasattr(c, "text"):
                print(c.text)
                break

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Model to MLflow

# COMMAND ----------

# Use workspace-relative experiment path (adjust for your workspace)
mlflow.set_experiment("langchain_langgraph_agent")

with mlflow.start_run():
    logged_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
        artifact_path="model",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri
print(f"Logged model: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluation with mlflow.genai.evaluate
# MAGIC
# MAGIC Evaluate the agent on order status, product info, and refund intents using Safety and RelevanceToQuery scorers.

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order ORD-12345678?"}]},
    {"input": [{"role": "user", "content": "Tell me about product PROD-0001"}]},
    {"input": [{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}]},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda inputs: agent.predict(inputs),
    scorers=[Safety(), RelevanceToQuery()],
)

print(f"Evaluation run ID: {results.run_id}")
print(results.tables.get("eval_results_table", "N/A"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register to Unity Catalog and Deploy
# MAGIC
# MAGIC Register the model to `sjdatabricks.agents.langchain_langgraph_support` and create a serving endpoint.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
uc_registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {uc_registered.name} (version {uc_registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving Endpoint
# MAGIC
# MAGIC Deploy the registered model to a Databricks serving endpoint for real-time inference.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

endpoint_name = "langchain-langgraph-support-agent"

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
# MAGIC ## 9. Query Examples (Streaming and Non-Streaming)
# MAGIC
# MAGIC Use the **DatabricksOpenAI** client to query the deployed agent.

# COMMAND ----------

from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()
model_for_inference = endpoint_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Streaming Query

# COMMAND ----------

response = client.chat.completions.create(
    model=model_for_inference,
    messages=[{"role": "user", "content": "What is the status of order ORD-12345678?"}],
    max_tokens=512,
)
print("Non-streaming response:")
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Query

# COMMAND ----------

stream = client.chat.completions.create(
    model=model_for_inference,
    messages=[{"role": "user", "content": "Tell me about product PROD-0001 and if it's in stock"}],
    max_tokens=512,
    stream=True,
)

print("Streaming response:")
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Workspace OpenAI Client

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()
alt_response = openai_client.chat.completions.create(
    model=model_for_inference,
    messages=[{"role": "user", "content": "I want to return order ORD-87654321, it was defective"}],
    max_tokens=512,
)
print("Alternative client response:")
print(alt_response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Set Model for Models-From-Code
# MAGIC
# MAGIC Ensure MLflow knows the active agent for inference and serving.

# COMMAND ----------

mlflow.models.set_model(agent)
print("Model set for MLflow. Agent ready for deployment.")
