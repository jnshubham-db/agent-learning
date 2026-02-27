# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 2: Customer Support Agent with DSPy and LangGraph
# MAGIC
# MAGIC This notebook implements a **Customer Support Agent** using:
# MAGIC - **DSPy** for Chain-of-Thought (CoT) prompting
# MAGIC - **LangGraph** for multi-node routing (Order, Product, Refund handling)
# MAGIC - **MLflow ResponsesAgent** for model serving compatibility
# MAGIC
# MAGIC ## Architecture
# MAGIC - **Router Node**: Classifies user intent (order_query, product_query, refund_request)
# MAGIC - **Order Node**: Handles order status and tracking
# MAGIC - **Product Node**: Handles product info (pricing, availability, specs)
# MAGIC - **Refund Node**: Processes refund requests
# MAGIC
# MAGIC > No wheel packaging — runs directly from notebook code.

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "dspy", "langgraph", "langchain-core", "mlflow>=3.0", "databricks-sdk", "pydantic>=2",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure DSPy with Databricks Claude

# COMMAND ----------

import dspy

# Configure DSPy to use Databricks Claude Sonnet 4.5
lm = dspy.LM("databricks-claude-sonnet-4-5")
dspy.configure(lm=lm)

print("DSPy configured with databricks-claude-sonnet-4-5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define DSPy CoT Modules

# MAGIC Each node uses Chain-of-Thought for reasoning before producing output.

# COMMAND ----------

# Router: Classify user intent
class RouterSignature(dspy.Signature):
    """Classify the customer's intent from their message."""
    user_message: str = dspy.InputField(desc="The customer's message")
    intent: str = dspy.OutputField(
        desc="One of: order_query, product_query, refund_request"
    )

router_cot = dspy.ChainOfThought(RouterSignature)


# Order handler: Order status and tracking
class OrderHandlerSignature(dspy.Signature):
    """Handle order-related queries: status lookup, tracking, shipment info."""
    user_message: str = dspy.InputField(desc="The customer's order query")
    response: str = dspy.OutputField(
        desc="Helpful response about order status or tracking"
    )

order_cot = dspy.ChainOfThought(OrderHandlerSignature)


# Product handler: Product info
class ProductHandlerSignature(dspy.Signature):
    """Handle product queries: pricing, availability, specifications."""
    user_message: str = dspy.InputField(desc="The customer's product query")
    response: str = dspy.OutputField(
        desc="Helpful response about product pricing, availability, or specs"
    )

product_cot = dspy.ChainOfThought(ProductHandlerSignature)


# Refund handler
class RefundHandlerSignature(dspy.Signature):
    """Handle refund requests: eligibility, process, timeline."""
    user_message: str = dspy.InputField(desc="The customer's refund request")
    response: str = dspy.OutputField(
        desc="Helpful response about refund eligibility and process"
    )

refund_cot = dspy.ChainOfThought(RefundHandlerSignature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define LangGraph State and Build Graph

# COMMAND ----------

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    """Graph state with messages, intent, response, and reasoning trace."""
    messages: list
    intent: str
    response: str
    rationale: str


def router_node(state: AgentState) -> dict:
    """Route based on user intent using DSPy CoT."""
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    content = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
    if hasattr(last_msg, "content"):
        content = last_msg.content

    result = router_cot(user_message=content)
    intent = result.intent.strip().lower()
    # Normalize to valid intents
    if "order" in intent or intent == "order_query":
        intent = "order_query"
    elif "product" in intent or intent == "product_query":
        intent = "product_query"
    elif "refund" in intent or intent == "refund_request":
        intent = "refund_request"
    else:
        intent = "order_query"  # Default fallback

    return {"intent": intent, "rationale": getattr(result, "rationale", "") or ""}


def order_node(state: AgentState) -> dict:
    """Handle order queries with DSPy CoT."""
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    content = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
    if hasattr(last_msg, "content"):
        content = last_msg.content

    result = order_cot(user_message=content)
    rationale = getattr(result, "rationale", "") or ""
    return {"response": result.response, "rationale": rationale}


def product_node(state: AgentState) -> dict:
    """Handle product queries with DSPy CoT."""
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    content = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
    if hasattr(last_msg, "content"):
        content = last_msg.content

    result = product_cot(user_message=content)
    rationale = getattr(result, "rationale", "") or ""
    return {"response": result.response, "rationale": rationale}


def refund_node(state: AgentState) -> dict:
    """Handle refund requests with DSPy CoT."""
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    content = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
    if hasattr(last_msg, "content"):
        content = last_msg.content

    result = refund_cot(user_message=content)
    rationale = getattr(result, "rationale", "") or ""
    return {"response": result.response, "rationale": rationale}


def route_by_intent(state: AgentState) -> Literal["order_node", "product_node", "refund_node"]:
    """Conditional routing based on Router classification."""
    intent = state.get("intent", "order_query")
    if intent == "product_query":
        return "product_node"
    if intent == "refund_request":
        return "refund_node"
    return "order_node"


# Build the graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("router", router_node)
graph_builder.add_node("order_node", order_node)
graph_builder.add_node("product_node", product_node)
graph_builder.add_node("refund_node", refund_node)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges(
    "router",
    route_by_intent,
    {
        "order_node": "order_node",
        "product_node": "product_node",
        "refund_node": "refund_node",
    },
)
graph_builder.add_edge("order_node", END)
graph_builder.add_edge("product_node", END)
graph_builder.add_edge("refund_node", END)

graph = graph_builder.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap Graph in MLflow ResponsesAgent

# MAGIC The agent subclasses `ResponsesAgent`, uses `predict_stream` with LangGraph
# MAGIC `stream_mode=["updates"]`, and yields reasoning + text events.

# COMMAND ----------

from typing import Generator
from uuid import uuid4
import mlflow
from mlflow.entities.span import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)


class CustomerSupportResponsesAgent(ResponsesAgent):
    """Customer Support Agent wrapping DSPy+LangGraph in ResponsesAgent."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming: collect output items from predict_stream."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream reasoning (CoT trace) then answer text."""
        inp = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        msgs = to_chat_completions_input(
            [i.model_dump() if hasattr(i, "model_dump") else i for i in inp]
        )
        initial_state = {"messages": msgs, "intent": "", "response": "", "rationale": ""}

        final_state = {}
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for _node_name, node_update in event.items():
                final_state.update(node_update)

        rationale = final_state.get("rationale", "") or ""
        response = final_state.get("response", "") or "I'm sorry, I couldn't process your request."

        # 1. Yield reasoning item (CoT trace) first
        if rationale:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(id=str(uuid4()), reasoning_text=rationale),
            )

        # 2. Stream final answer text with create_text_delta events
        text_id = str(uuid4())
        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=chunk, item_id=text_id))

        # 3. Yield output_item.done for the text
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=response, id=text_id),
        )


# Instantiate agent
agent = CustomerSupportResponsesAgent(graph)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Enable MLflow Autologging and Tracing

# COMMAND ----------

mlflow.dspy.autolog()
mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Model to MLflow

# COMMAND ----------

with mlflow.start_run():
    logged_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri
    print(f"Model logged. Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluation with Safety and RelevanceToQuery Scorers

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

# Load the logged model for evaluation
loaded_agent = mlflow.pyfunc.load_model(logged_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "Where is my order #12345?"}]},
    {"input": [{"role": "user", "content": "How much does the Premium Widget cost?"}]},
    {"input": [{"role": "user", "content": "I want a refund for my last purchase"}]},
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri=model_uri,
    name="sjdatabricks.agents.dspy_langgraph_customer_support",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Serving Deployment

# MAGIC Deploy the logged model to Databricks Model Serving for real-time inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a. Create Model Serving Endpoint (via UI or SDK)
# MAGIC
# MAGIC 1. Go to **Workspace → Machine Learning → Model Serving**
# MAGIC 2. Click **Create serving endpoint**
# MAGIC 3. Select the registered model `dspy_langgraph_customer_support_agent`
# MAGIC 4. Configure compute and scaling
# MAGIC
# MAGIC Or use the SDK below:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Uncomment and customize to create endpoint programmatically:
# endpoint = w.serving_endpoints.create(
#     name="dspy-customer-support-agent",
#     config={
#         "served_models": [{
#             "model_name": "dspy_langgraph_customer_support_agent",
#             "model_version": "1",  # or latest
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True,
#         }],
#     },
# )
# print(f"Endpoint created: {endpoint.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Query Examples with Databricks Client

# MAGIC Use the OpenAI-compatible client to query the deployed model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9a. Test Locally (Before Deployment)

# COMMAND ----------

# Local test with the agent instance
test_requests = [
    "Where is my order #12345?",
    "How much does the Premium Widget cost and is it in stock?",
    "I want a refund for my defective product.",
]

for q in test_requests:
    req = ResponsesAgentRequest(input=[{"role": "user", "content": q}])
    resp = agent.predict(req)
    print(f"\nQ: {q}")
    for item in resp.output:
        if isinstance(item, dict):
            if item.get("type") == "reasoning":
                summary = item.get("summary", [{}])
                if summary:
                    txt = summary[0].get("text", "")
                    print(f"  Reasoning: {txt[:200]}{'...' if len(txt) > 200 else ''}")
            elif item.get("type") == "message":
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        print(f"  Answer: {c.get('text', '')}")
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9b. Query Deployed Endpoint with Databricks Client
# MAGIC
# MAGIC Once the model is deployed to Model Serving, use the Databricks OpenAI client.

# COMMAND ----------

# Example: Query the deployed endpoint (uncomment after deployment)
# from databricks.sdk import WorkspaceClient
#
# w = WorkspaceClient()
# client = w.serving_endpoints.get_open_ai_client()
#
# # Chat Completions API (OpenAI-compatible)
# response = client.chat.completions.create(
#     model="databricks-dspy_langgraph_customer_support_agent-1",
#     messages=[{"role": "user", "content": "Where is my order #12345?"}],
#     max_tokens=256,
# )
# print(response.choices[0].message.content)
#
# # Or use Responses API for full reasoning + text output:
# response = client.responses.create(
#     model="databricks-dspy_langgraph_customer_support_agent-1",
#     input=[{"role": "user", "content": "Where is my order #12345?"}],
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Set Model for Models-from-Code

# MAGIC This enables MLflow to resolve the model when loading from the registry.

# COMMAND ----------

mlflow.models.set_model(agent)

print("Model set for models-from-code pattern.")
