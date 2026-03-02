# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 4: Customer Support Agent with LangChain (No Whl)
# MAGIC
# MAGIC This notebook builds a **Customer Support Agent** using **LangChain** with tool-calling,
# MAGIC powered by **ChatDatabricks** and served through **MLflow ResponsesAgent**.
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC | Concept | Description |
# MAGIC |---------|-------------|
# MAGIC | **LangChain** | A framework for building LLM-powered applications with composable components |
# MAGIC | **ChatDatabricks** | LangChain's interface to Databricks Foundation Model APIs — lets you call any model endpoint as a chat model |
# MAGIC | **Tool-calling agents** | The LLM decides *which* tool to call and *what arguments* to pass, then uses the result to answer the user |
# MAGIC | **`create_react_agent`** | LangGraph's modern replacement for the older `AgentExecutor` — builds a ReAct loop as a compiled graph |
# MAGIC | **ResponsesAgent** | MLflow's agent interface for streaming-compatible model serving on Databricks |
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC User question
# MAGIC       |
# MAGIC       v
# MAGIC  create_react_agent (LangGraph ReAct loop)
# MAGIC       |
# MAGIC       +---> get_order_status(order_id)    --> Spark SQL on sjdatabricks.orders.order_details
# MAGIC       +---> search_returns(order_id)       --> Spark SQL on sjdatabricks.orders.returns
# MAGIC       +---> get_product_info(product_name)  --> Spark SQL on sjdatabricks.orders.products
# MAGIC       |
# MAGIC       v
# MAGIC  Final answer streamed via ResponsesAgent
# MAGIC ```
# MAGIC
# MAGIC **Data**: All tools query the `sjdatabricks` catalog tables created in `00_setup/create_fake_data.py`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies
# MAGIC
# MAGIC We need the following packages:
# MAGIC
# MAGIC | Package | Purpose |
# MAGIC |---------|---------|
# MAGIC | `mlflow>=3` | Model logging, tracing, evaluation, and the ResponsesAgent interface |
# MAGIC | `langchain` | Core LangChain framework for chains, prompts, and tool abstractions |
# MAGIC | `langchain-core` | Base classes shared across all LangChain integrations |
# MAGIC | `langchain-databricks` | The `ChatDatabricks` LLM wrapper that calls Databricks endpoints |
# MAGIC | `langgraph` | Graph-based orchestration — provides `create_react_agent` |
# MAGIC | `databricks-agents` | Helpers for deploying agents on Databricks Model Serving |
# MAGIC
# MAGIC > **Why `langgraph`?** The older `AgentExecutor` from LangChain is deprecated. The modern
# MAGIC > approach uses `create_react_agent` from `langgraph.prebuilt`, which builds a proper
# MAGIC > ReAct loop as a LangGraph `StateGraph` under the hood.

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "mlflow>=3", "langchain", "langchain-core", "langchain-databricks",
    "langgraph", "databricks-agents",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure the LLM with ChatDatabricks
# MAGIC
# MAGIC **ChatDatabricks** is LangChain's integration with Databricks Foundation Model APIs.
# MAGIC It wraps any Databricks model-serving endpoint (pay-per-token or provisioned throughput)
# MAGIC and exposes it as a standard LangChain `BaseChatModel`.
# MAGIC
# MAGIC Key points for beginners:
# MAGIC - You pass the **endpoint name** (not a model path) — Databricks routes the request.
# MAGIC - `temperature=0.0` makes the model deterministic, which is ideal for tool-calling agents
# MAGIC   because we want consistent, reliable function calls.
# MAGIC - Authentication is automatic inside a Databricks notebook (uses the notebook's credentials).

# COMMAND ----------

from langchain_databricks import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.0,
)

# Quick smoke test — make sure the endpoint responds
test_response = llm.invoke("Say hello in one sentence.")
print(f"LLM test: {test_response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Tools with the `@tool` Decorator
# MAGIC
# MAGIC ### What Are "Tools" in LangChain?
# MAGIC
# MAGIC A **tool** is a Python function that the LLM can decide to call. When you give an agent a
# MAGIC list of tools, the LLM receives their names, descriptions, and parameter schemas. During
# MAGIC a conversation the LLM can output a **tool call** — a structured JSON message saying
# MAGIC "call function X with arguments Y". The agent framework then:
# MAGIC
# MAGIC 1. Parses the tool call from the LLM output
# MAGIC 2. Executes the matching Python function
# MAGIC 3. Feeds the result back to the LLM
# MAGIC 4. The LLM uses the result to generate a final answer (or calls another tool)
# MAGIC
# MAGIC This loop is called **ReAct** (Reasoning + Acting).
# MAGIC
# MAGIC ### The `@tool` Decorator
# MAGIC
# MAGIC LangChain's `@tool` decorator converts a plain Python function into a `StructuredTool`
# MAGIC object. It automatically:
# MAGIC - Extracts the function name as the tool name
# MAGIC - Uses the docstring as the tool description (shown to the LLM)
# MAGIC - Infers the parameter schema from type hints
# MAGIC
# MAGIC ### Our Three Tools
# MAGIC
# MAGIC Each tool queries a different Spark SQL table in the `sjdatabricks.orders` schema:
# MAGIC
# MAGIC | Tool | Table | What It Does |
# MAGIC |------|-------|-------------|
# MAGIC | `get_order_status` | `order_details` | Looks up order status by order ID |
# MAGIC | `search_returns` | `returns` | Finds return records for an order |
# MAGIC | `get_product_info` | `products` | Gets product details by name |

# COMMAND ----------

from langchain_core.tools import tool


@tool
def get_order_status(order_id: int) -> str:
    """Look up the current status of a customer order by its order ID.

    Use this tool when the customer asks about an order status, shipment,
    or delivery. The order_id is an integer (e.g., 1040, 1045).

    Args:
        order_id: The numeric order ID to look up.

    Returns:
        A string with the order details or a not-found message.
    """
    rows = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
    ).collect()
    if not rows:
        return f"No order found with ID {order_id}. Please verify the order number."
    row = rows[0]
    return (
        f"Order {row['order_id']}: product={row['product']}, "
        f"quantity={row['quantity']}, status={row['status']}, "
        f"customer={row['customer_name']}, date={row['order_date']}"
    )


@tool
def search_returns(order_id: int) -> str:
    """Search for return or refund records associated with an order ID.

    Use this tool when the customer asks about a return, refund, or
    wants to know if a return has been filed for their order.

    Args:
        order_id: The numeric order ID to search returns for.

    Returns:
        A string listing all returns for that order, or a message if none exist.
    """
    rows = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
    ).collect()
    if not rows:
        return f"No returns found for order {order_id}."
    results = []
    for row in rows:
        results.append(
            f"Return {row['return_id']}: reason={row['reason']}, "
            f"status={row['status']}, date={row['return_date']}"
        )
    return "\n".join(results)


@tool
def get_product_info(product_name: str) -> str:
    """Get product details (price, stock, category) by product name.

    Use this tool when the customer asks about a product's price,
    availability, or category. The product_name is a string like
    'Laptop', 'Phone', 'Tablet', etc.

    Args:
        product_name: The name of the product to look up.

    Returns:
        A string with product details or a not-found message.
    """
    rows = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.products WHERE LOWER(name) = LOWER('{product_name}')"
    ).collect()
    if not rows:
        return f"No product found with name '{product_name}'. Available products: Laptop, Phone, Tablet, Monitor, Keyboard, Mouse."
    row = rows[0]
    return (
        f"Product: {row['name']} (ID={row['product_id']}), "
        f"category={row['category']}, price=${row['price']}, stock={row['stock']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the LLM "Sees" These Tools
# MAGIC
# MAGIC When we pass these tools to the agent, LangChain sends the LLM a description like:
# MAGIC
# MAGIC ```
# MAGIC Available tools:
# MAGIC - get_order_status(order_id: int): Look up the current status of a customer order...
# MAGIC - search_returns(order_id: int): Search for return or refund records...
# MAGIC - get_product_info(product_name: str): Get product details (price, stock, category)...
# MAGIC ```
# MAGIC
# MAGIC The LLM then decides which tool to call based on the user's question. For example:
# MAGIC - "What is the status of order 1045?" -> calls `get_order_status(order_id=1045)`
# MAGIC - "Has order 1042 been returned?" -> calls `search_returns(order_id=1042)`
# MAGIC - "How much does a Laptop cost?" -> calls `get_product_info(product_name="Laptop")`
# MAGIC
# MAGIC The LLM can also call **multiple tools** in sequence if the question requires it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create the Agent with `create_react_agent`
# MAGIC
# MAGIC ### What is `create_react_agent`?
# MAGIC
# MAGIC `create_react_agent` from `langgraph.prebuilt` is the **modern way** to build a tool-calling
# MAGIC agent in the LangChain ecosystem. It replaces the deprecated `AgentExecutor`.
# MAGIC
# MAGIC Under the hood, it builds a **LangGraph StateGraph** with this loop:
# MAGIC
# MAGIC ```
# MAGIC START -> call_model -> should_continue?
# MAGIC                           |
# MAGIC                   yes (tool call) -> call_tool -> call_model (loop back)
# MAGIC                   no (final answer) -> END
# MAGIC ```
# MAGIC
# MAGIC This is the **ReAct pattern**: the LLM *reasons* about what to do, *acts* by calling a tool,
# MAGIC *observes* the result, and repeats until it has enough information to answer.
# MAGIC
# MAGIC ### Parameters
# MAGIC
# MAGIC - `model`: The LLM (our `ChatDatabricks` instance)
# MAGIC - `tools`: List of tools the agent can call
# MAGIC - `prompt`: Optional system prompt to guide the agent's behavior

# COMMAND ----------

from langgraph.prebuilt import create_react_agent

# System prompt that guides the agent's behavior
system_prompt = (
    "You are a helpful customer support agent for an e-commerce company. "
    "You have access to tools that query order, return, and product databases. "
    "Always use the appropriate tool to look up real data before answering. "
    "Be concise, friendly, and accurate. If you cannot find the information, "
    "say so clearly and suggest what the customer can do next."
)

# Build the agent — this returns a compiled LangGraph
tools = [get_order_status, search_returns, get_product_info]
langgraph_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
)

print("Agent created successfully with 3 tools:")
for t in tools:
    print(f"  - {t.name}: {t.description[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Test of the Raw Agent
# MAGIC
# MAGIC Before wrapping in ResponsesAgent, let us verify the LangGraph agent works directly.
# MAGIC The agent's `invoke` method takes a dict with a `messages` key.

# COMMAND ----------

result = langgraph_agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of order 1045?"}]
})

# The result contains all messages including tool calls and responses
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:200] if msg.content else '(tool call)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wrap in MLflow ResponsesAgent
# MAGIC
# MAGIC ### Why Do We Need ResponsesAgent?
# MAGIC
# MAGIC Databricks **Model Serving** expects agents to implement a specific interface:
# MAGIC
# MAGIC - `predict(request)` for synchronous calls
# MAGIC - `predict_stream(request)` for streaming responses
# MAGIC
# MAGIC `ResponsesAgent` from `mlflow.pyfunc` provides this interface. It also gives us helper
# MAGIC methods like `create_text_output_item()` and `create_text_delta()` that produce correctly
# MAGIC formatted response objects.
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. A request comes in with `input` messages (user's conversation)
# MAGIC 2. We convert them to LangChain message format
# MAGIC 3. We invoke the LangGraph agent
# MAGIC 4. We extract the final AI message and stream it back
# MAGIC
# MAGIC The `predict_stream` method yields events that Model Serving sends to the client
# MAGIC as Server-Sent Events (SSE), enabling real-time token-by-token output.

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
)


class LangChainCustomerSupportAgent(ResponsesAgent):
    """Customer Support Agent using LangChain + LangGraph ReAct, wrapped in ResponsesAgent."""

    def __init__(self):
        super().__init__()
        # Re-create the LLM, tools, and agent inside the class so it is self-contained
        # when loaded from MLflow. In notebook testing we can also pass these in.
        self._build_agent()

    def _build_agent(self):
        """Build the LangChain agent with ChatDatabricks and tools."""
        from langchain_databricks import ChatDatabricks
        from langchain_core.tools import tool as tool_decorator
        from langgraph.prebuilt import create_react_agent

        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0.0,
        )

        # Define tools inside the class for self-contained logging
        @tool_decorator
        def get_order_status(order_id: int) -> str:
            """Look up the current status of a customer order by its order ID."""
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
            ).collect()
            if not rows:
                return f"No order found with ID {order_id}."
            row = rows[0]
            return (
                f"Order {row['order_id']}: product={row['product']}, "
                f"quantity={row['quantity']}, status={row['status']}, "
                f"customer={row['customer_name']}, date={row['order_date']}"
            )

        @tool_decorator
        def search_returns(order_id: int) -> str:
            """Search for return or refund records associated with an order ID."""
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
            ).collect()
            if not rows:
                return f"No returns found for order {order_id}."
            results = []
            for row in rows:
                results.append(
                    f"Return {row['return_id']}: reason={row['reason']}, "
                    f"status={row['status']}, date={row['return_date']}"
                )
            return "\n".join(results)

        @tool_decorator
        def get_product_info(product_name: str) -> str:
            """Get product details (price, stock, category) by product name."""
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.products WHERE LOWER(name) = LOWER('{product_name}')"
            ).collect()
            if not rows:
                return f"No product found with name '{product_name}'."
            row = rows[0]
            return (
                f"Product: {row['name']} (ID={row['product_id']}), "
                f"category={row['category']}, price=${row['price']}, stock={row['stock']}"
            )

        self.tools = [get_order_status, search_returns, get_product_info]

        system_prompt = (
            "You are a helpful customer support agent for an e-commerce company. "
            "You have access to tools that query order, return, and product databases. "
            "Always use the appropriate tool to look up real data before answering. "
            "Be concise, friendly, and accurate."
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt,
        )

    def _extract_question(self, request: ResponsesAgentRequest) -> list[dict]:
        """Convert ResponsesAgent input to LangChain message format."""
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        messages = []
        for msg in input_list:
            role = getattr(msg, "role", None) or (
                msg.get("role", "user") if isinstance(msg, dict) else "user"
            )
            content = getattr(msg, "content", None) or (
                msg.get("content", "") if isinstance(msg, dict) else ""
            )
            if isinstance(content, list):
                # Handle structured content (e.g., output_text items)
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text_parts.append(c.get("text", ""))
                    elif hasattr(c, "text"):
                        text_parts.append(c.text)
                content = " ".join(text_parts)
            messages.append({"role": role, "content": content})
        return messages

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous prediction: run the agent and return the full response."""
        messages = self._extract_question(request)
        result = self.agent.invoke({"messages": messages})

        # Extract the final AI message (last message in the conversation)
        final_messages = result.get("messages", [])
        answer = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                answer = msg.content
                break

        output_items = [
            self.create_text_output_item(text=answer, id=str(uuid4()))
        ]
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction: yield text deltas as the agent generates output."""
        messages = self._extract_question(request)

        # Use the agent's stream method to get incremental updates
        final_answer = ""
        for event in self.agent.stream({"messages": messages}, stream_mode="updates"):
            for node_name, node_update in event.items():
                # The last AI message from the agent node contains the answer
                if "messages" in node_update:
                    for msg in node_update["messages"]:
                        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                            final_answer = msg.content

        # Stream the final answer as text deltas
        text_id = str(uuid4())
        chunk_size = 20
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=chunk, item_id=text_id)
            )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=final_answer, id=text_id),
        )


# Instantiate the agent
agent = LangChainCustomerSupportAgent()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the ResponsesAgent Wrapper
# MAGIC
# MAGIC Let us break down what happens inside the class:
# MAGIC
# MAGIC 1. **`__init__`** — Rebuilds the LLM, tools, and LangGraph agent. This ensures the agent
# MAGIC    is fully self-contained when MLflow loads it from the registry.
# MAGIC
# MAGIC 2. **`_extract_question`** — Converts the Responses API input format (list of message objects)
# MAGIC    into the LangChain format (list of `{"role": ..., "content": ...}` dicts).
# MAGIC
# MAGIC 3. **`predict`** — Runs the agent synchronously with `agent.invoke()`, extracts the final
# MAGIC    AI message, and wraps it in a `ResponsesAgentResponse`.
# MAGIC
# MAGIC 4. **`predict_stream`** — Uses `agent.stream()` to process the agent graph incrementally,
# MAGIC    then yields the final answer as text deltas. Each delta is a small chunk of text that
# MAGIC    the client receives in real time.
# MAGIC
# MAGIC > **Why stream?** Streaming gives users immediate feedback. Instead of waiting 10 seconds
# MAGIC > for the full answer, they see text appearing word-by-word. This is especially important
# MAGIC > for customer support where responsiveness matters.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test the Agent Locally
# MAGIC
# MAGIC Before logging to MLflow, let us test the wrapped agent with a few questions.
# MAGIC This verifies that the ResponsesAgent interface works correctly.

# COMMAND ----------

from mlflow.types.responses import ResponsesAgentRequest

test_questions = [
    "What is the status of order 1045?",
    "Has there been a return filed for order 1042?",
    "How much does a Laptop cost and is it in stock?",
]

for question in test_questions:
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}]
    )
    response = agent.predict(request)
    print(f"\nQ: {question}")
    for item in response.output:
        if hasattr(item, "text"):
            print(f"A: {item.text}")
        elif isinstance(item, dict):
            for c in item.get("content", []):
                if isinstance(c, dict) and c.get("type") == "output_text":
                    print(f"A: {c['text']}")
    print("-" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Streaming
# MAGIC
# MAGIC Verify that `predict_stream` yields incremental text deltas correctly.
# MAGIC Each delta should be a small chunk of the answer.

# COMMAND ----------

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is the status of order 1050?"}]
)

print("Streaming response:")
for event in agent.predict_stream(request):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.output_item.done":
        print()  # newline after the full text
print("\nStreaming complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log the Agent with MLflow
# MAGIC
# MAGIC ### Code-Based Logging
# MAGIC
# MAGIC MLflow's **code-based logging** (also called "models-from-code") serializes the agent by
# MAGIC saving the Python object directly. When you call `mlflow.pyfunc.log_model(python_model=agent)`,
# MAGIC MLflow:
# MAGIC
# MAGIC 1. Pickles the agent object
# MAGIC 2. Records the Python environment (pip requirements)
# MAGIC 3. Stores everything as an MLflow artifact
# MAGIC
# MAGIC Later, `mlflow.pyfunc.load_model(uri)` reconstructs the agent from the pickle.
# MAGIC
# MAGIC > **Tip**: For production deployments you would use models-from-code with an `agent.py` file
# MAGIC > (see Topic 6). For notebook prototyping, passing the object directly is simpler.

# COMMAND ----------

import mlflow

mlflow.langchain.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="langchain_customer_support_agent") as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
    )
    run_id = run.info.run_id
    model_uri = logged_agent_info.model_uri
    print(f"Model logged successfully!")
    print(f"  Run ID:    {run_id}")
    print(f"  Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What Just Happened?
# MAGIC
# MAGIC MLflow created an **experiment run** containing:
# MAGIC
# MAGIC - **The serialized agent** — can be loaded with `mlflow.pyfunc.load_model()`
# MAGIC - **Pip requirements** — MLflow auto-detects langchain, langgraph, etc.
# MAGIC - **Traces** — if autologging is on, every `predict()` call is recorded
# MAGIC
# MAGIC You can view all of this in the MLflow Experiment UI (click "Experiments" in the left sidebar).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC ### What is LLM-as-a-Judge?
# MAGIC
# MAGIC Instead of manually reviewing every agent response, we use another LLM to **score** them.
# MAGIC MLflow provides built-in scorers:
# MAGIC
# MAGIC | Scorer | What It Measures |
# MAGIC |--------|-----------------|
# MAGIC | `Safety` | Is the response free of harmful, offensive, or dangerous content? |
# MAGIC | `RelevanceToQuery` | Does the response actually answer the user's question? |
# MAGIC
# MAGIC The evaluation runs each test question through the agent, collects the response, and then
# MAGIC asks a judge LLM to rate the output. Results are logged to MLflow for comparison.

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

# Load the logged model to ensure we evaluate the exact artifact that was saved
loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order 1045?"}]},
    {"input": [{"role": "user", "content": "Has order 1042 been returned?"}]},
    {"input": [{"role": "user", "content": "Tell me about the Tablet product — price and stock."}]},
    {"input": [{"role": "user", "content": "I need help with order 1055, is it shipped?"}]},
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: loaded_agent.predict(row),
    scorers=[
        Safety(),
        RelevanceToQuery(),
    ],
)

print("Evaluation complete. Check the MLflow Experiment UI for detailed scores.")
print(eval_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading Evaluation Results
# MAGIC
# MAGIC After evaluation, you will see a table in the MLflow UI with columns like:
# MAGIC
# MAGIC - **input**: The user's question
# MAGIC - **output**: The agent's response
# MAGIC - **safety/score**: 1 (safe) or 0 (unsafe)
# MAGIC - **relevance_to_query/score**: 1 (relevant) or 0 (not relevant)
# MAGIC
# MAGIC If any score is 0, click into the row to see the judge's reasoning — this helps you
# MAGIC understand what went wrong and improve the agent's system prompt or tools.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Register the Model to Unity Catalog
# MAGIC
# MAGIC Registering the model to **Unity Catalog** makes it available for:
# MAGIC
# MAGIC - **Version management** — track model versions over time
# MAGIC - **Access control** — govern who can deploy or query the model
# MAGIC - **Model Serving** — deploy to a real-time endpoint with one click
# MAGIC
# MAGIC The three-level name `catalog.schema.model_name` follows Unity Catalog conventions.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri=model_uri,
    name="sjdatabricks.agents.langchain_customer_support",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Deploy to Model Serving
# MAGIC
# MAGIC Now that the model is registered in Unity Catalog, we can deploy it to **Databricks Model Serving**.
# MAGIC This creates a real-time REST endpoint that can handle both synchronous and streaming requests.
# MAGIC
# MAGIC ### What Happens During Deployment
# MAGIC
# MAGIC 1. Databricks provisions a **serverless compute** container for the agent
# MAGIC 2. It installs the agent's pip dependencies (MLflow, LangChain, LangGraph, etc.)
# MAGIC 3. It loads the model artifact from Unity Catalog using `mlflow.pyfunc.load_model()`
# MAGIC 4. The endpoint starts accepting HTTP requests — each request calls `predict()` or `predict_stream()`
# MAGIC
# MAGIC We use the **Databricks SDK** (`databricks-sdk`) to create the endpoint programmatically.
# MAGIC The `create_and_wait` method blocks until the endpoint is fully provisioned and ready.
# MAGIC If the endpoint already exists (e.g., from a previous run), we fall back to `update_config_and_wait`
# MAGIC which updates the served model version without recreating the endpoint.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade -q
# MAGIC %restart_python

# COMMAND ----------

# Retrieve the latest model version from Unity Catalog so we deploy the most recent artifact.
# We also define the endpoint name and UC model name as constants for reuse in later cells.

import mlflow

mlflow.set_registry_uri("databricks-uc")

UC_MODEL_NAME = "sjdatabricks.agents.langchain_customer_support"
ENDPOINT_NAME = "langchain-customer-support-agent"

client = mlflow.tracking.MlflowClient()
latest_version = max(
    int(mv.version) for mv in client.search_model_versions(f"name='{UC_MODEL_NAME}'")
)
print(f"Latest version of '{UC_MODEL_NAME}': {latest_version}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_NAME,
                    entity_version=str(latest_version),
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
                    entity_version=str(latest_version),
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
# MAGIC ## 12. Query the Endpoint (Non-Streaming)
# MAGIC
# MAGIC With the endpoint deployed, we can query it using the **OpenAI SDK** pointed at the
# MAGIC Databricks serving URL. This works because Databricks Model Serving exposes an
# MAGIC OpenAI-compatible REST API.
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC - `base_url` is set to `{workspace_host}/serving-endpoints` — this tells the OpenAI client
# MAGIC   to send requests to Databricks instead of OpenAI's servers.
# MAGIC - `api_key` uses a Databricks workspace token for authentication.
# MAGIC - `model` is the endpoint name we created above.
# MAGIC - The `responses.create()` method sends a request and returns the full response at once.
# MAGIC
# MAGIC The response follows the OpenAI Responses API format: each output item may contain
# MAGIC `content` blocks, and each content block may have a `text` field with the actual answer.

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
# MAGIC ## 13. Query the Endpoint (Streaming)
# MAGIC
# MAGIC Streaming is essential for customer-facing applications where **perceived latency** matters.
# MAGIC Instead of waiting for the entire response to be generated, the client receives text
# MAGIC fragments (deltas) as they are produced.
# MAGIC
# MAGIC ### How Streaming Works
# MAGIC
# MAGIC 1. We pass `stream=True` to `responses.create()`.
# MAGIC 2. The method returns an **iterator** of Server-Sent Events (SSE).
# MAGIC 3. Each event has a `type` field — we look for `response.output_text.delta` events
# MAGIC    which contain a `delta` attribute holding the next chunk of text.
# MAGIC 4. We print each delta immediately with `flush=True` so it appears in real time.
# MAGIC
# MAGIC This mirrors what happens in a chat UI: the user sees the answer being "typed out"
# MAGIC token by token, which feels much more responsive than waiting several seconds for the
# MAGIC complete answer.

# COMMAND ----------

stream = client.responses.create(
    model=ENDPOINT_NAME,
    input=[{"role": "user", "content": "Show me all returns for order 1045."}],
    stream=True,
)
for event in stream:
    if hasattr(event, "type") and event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook you learned how to:
# MAGIC
# MAGIC 1. **Configure ChatDatabricks** as a LangChain LLM connected to a Databricks endpoint
# MAGIC 2. **Define tools** with the `@tool` decorator that query Spark SQL tables
# MAGIC 3. **Build a ReAct agent** using `create_react_agent` from LangGraph
# MAGIC 4. **Wrap the agent** in `ResponsesAgent` for MLflow-compatible streaming
# MAGIC 5. **Log, evaluate, and register** the agent with MLflow and Unity Catalog
# MAGIC 6. **Deploy to Model Serving** using the Databricks SDK with `create_and_wait`
# MAGIC 7. **Query the endpoint** using the OpenAI SDK in both non-streaming and streaming modes
# MAGIC
# MAGIC **Next**: Topic 5 shows how to use LangGraph's `StateGraph` directly for more control
# MAGIC over the agent's multi-step reasoning flow.
