# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Topic 1: Customer Order Support Agent with DSPy
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC In this notebook, we build a **Customer Order Support Agent** using **DSPy** and deploy it with
# MAGIC **MLflow 3** on Databricks. The agent can answer natural-language questions about customer orders,
# MAGIC returns, and products by querying live Spark tables.
# MAGIC
# MAGIC **What you will learn:**
# MAGIC
# MAGIC 1. How to configure DSPy with a Databricks-hosted LLM
# MAGIC 2. How to define Python tools that query Spark tables
# MAGIC 3. How to build a **ReAct** agent that reasons and calls tools automatically
# MAGIC 4. How to wrap the agent in a **ResponsesAgent** for MLflow serving compatibility
# MAGIC 5. How to enable **streaming** so frontends can display partial responses
# MAGIC 6. How to **log** the agent to MLflow using the code-based logging pattern
# MAGIC 7. How to **evaluate** the agent with LLM-as-a-judge
# MAGIC 8. How to **register** the agent to Unity Catalog for deployment
# MAGIC
# MAGIC **Architecture:**
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC      |
# MAGIC      v
# MAGIC  DSPy ReAct Agent  <--->  Tools (Spark SQL queries)
# MAGIC      |
# MAGIC      v
# MAGIC  ResponsesAgent (MLflow-compatible wrapper)
# MAGIC      |
# MAGIC      v
# MAGIC  Model Serving Endpoint
# MAGIC ```
# MAGIC
# MAGIC **Data tables** (in the `sjdatabricks` catalog):
# MAGIC - `sjdatabricks.orders.order_details` -- order id, customer name, product, quantity, status, order date
# MAGIC - `sjdatabricks.orders.returns` -- return id, order id, reason, status, return date
# MAGIC - `sjdatabricks.orders.products` -- product id, name, category, price, stock

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC We need three packages:
# MAGIC
# MAGIC - **`mlflow>=3`** -- The latest version of MLflow with first-class support for GenAI agents,
# MAGIC   the Responses API format, and code-based model logging.
# MAGIC - **`dspy`** -- Stanford's Declarative Self-improving Python framework. DSPy lets you build
# MAGIC   LLM pipelines as composable *modules* (like PyTorch nn.Module) rather than brittle prompt strings.
# MAGIC - **`databricks-agents`** -- Databricks-specific helpers for agent evaluation and deployment.
# MAGIC
# MAGIC After installing, we restart Python so the new packages are available in subsequent cells.

# COMMAND ----------

# MAGIC %pip install mlflow>=3 dspy databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure the LLM
# MAGIC
# MAGIC ### What is DSPy?
# MAGIC
# MAGIC **DSPy** (Declarative Self-improving Python) is a framework from Stanford NLP that treats LLM
# MAGIC calls as *modules* rather than raw prompt strings. Instead of hand-crafting prompts, you declare
# MAGIC **what** you want (a *Signature*) and DSPy figures out **how** to prompt the LLM. This makes your
# MAGIC agent code:
# MAGIC
# MAGIC - **Modular** -- swap LLMs without rewriting prompts
# MAGIC - **Composable** -- chain modules together like building blocks
# MAGIC - **Optimizable** -- DSPy can automatically tune prompts for better accuracy
# MAGIC
# MAGIC ### Configuring the LLM
# MAGIC
# MAGIC We point DSPy at the `databricks-meta-llama-3-3-70b-instruct` endpoint. On Databricks, this
# MAGIC is a **Foundation Model API** endpoint -- no API keys needed because authentication is handled
# MAGIC by the workspace. The `dspy.LM(...)` call creates a language model handle, and `dspy.configure(lm=...)`
# MAGIC sets it as the default for all DSPy modules in this notebook.

# COMMAND ----------

import dspy
import mlflow

# Point DSPy at the Databricks-hosted Llama 3.3 70B model.
# On Databricks, the "databricks-" prefix tells DSPy to use the Foundation Model API,
# which handles authentication automatically via your workspace credentials.
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

lm = dspy.LM(LLM_ENDPOINT)
dspy.configure(lm=lm)

# Enable MLflow tracing so every LLM call and tool invocation is recorded.
# This gives you full observability in the MLflow Tracking UI.
mlflow.dspy.autolog()

print(f"DSPy configured with LLM: {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Define Tools That Query Spark Tables
# MAGIC
# MAGIC ### What are "Tools" in an agent context?
# MAGIC
# MAGIC An **agent tool** is a plain Python function that the LLM can choose to call during its
# MAGIC reasoning process. The agent reads the tool's name, docstring, and type hints to understand
# MAGIC what the tool does and when to use it. This is how agents go beyond just generating text --
# MAGIC they can **take actions** like querying databases, calling APIs, or performing calculations.
# MAGIC
# MAGIC ### Our three tools
# MAGIC
# MAGIC Each tool runs a Spark SQL query against the `sjdatabricks.orders` schema:
# MAGIC
# MAGIC | Tool | Table | Purpose |
# MAGIC |------|-------|---------|
# MAGIC | `get_order_status` | `order_details` | Look up an order by ID |
# MAGIC | `get_returns` | `returns` | Find returns for a given order ID |
# MAGIC | `get_product_info` | `products` | Look up a product by name |
# MAGIC
# MAGIC **Why Spark SQL?** On Databricks, `spark.sql(...)` is the standard way to query Unity Catalog
# MAGIC tables. The results are returned as DataFrames, which we convert to a string for the LLM to read.
# MAGIC
# MAGIC **Important design choice:** Each tool has a clear docstring and typed parameters. The agent
# MAGIC uses these descriptions to decide *which* tool to call and *what arguments* to pass. Vague
# MAGIC docstrings lead to incorrect tool selection, so always be specific.

# COMMAND ----------

def get_order_status(order_id: int) -> str:
    """Look up the status and details of a customer order by its order ID.

    Use this tool when the user asks about a specific order, such as
    'What is the status of order 1042?' or 'Tell me about order 1050'.

    Args:
        order_id: The numeric order ID to look up (e.g., 1042).

    Returns:
        A string with the order details or a 'not found' message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
    )
    if df.count() == 0:
        return f"No order found with ID {order_id}."
    row = df.first()
    return (
        f"Order {row.order_id}: customer={row.customer_name}, product={row.product}, "
        f"quantity={row.quantity}, status={row.status}, order_date={row.order_date}"
    )


def get_returns(order_id: int) -> str:
    """Look up return requests associated with a specific order ID.

    Use this tool when the user asks about returns for an order, such as
    'Are there any returns for order 1045?' or 'Show me returns on order 1042'.

    Args:
        order_id: The numeric order ID to look up returns for.

    Returns:
        A string listing all returns for that order, or a message saying none were found.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
    )
    if df.count() == 0:
        return f"No returns found for order {order_id}."
    rows = df.collect()
    results = []
    for row in rows:
        results.append(
            f"Return {row.return_id}: reason={row.reason}, status={row.status}, "
            f"return_date={row.return_date}"
        )
    return f"Returns for order {order_id}:\n" + "\n".join(results)


def get_product_info(product_name: str) -> str:
    """Look up product details by product name.

    Use this tool when the user asks about a specific product, such as
    'What is the price of a Laptop?' or 'Is the Phone in stock?'.

    Args:
        product_name: The name of the product (e.g., 'Laptop', 'Phone', 'Tablet').

    Returns:
        A string with the product details or a 'not found' message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.products WHERE lower(name) = lower('{product_name}')"
    )
    if df.count() == 0:
        return f"No product found with name '{product_name}'."
    row = df.first()
    return (
        f"Product: {row.name}, category={row.category}, price=${row.price}, "
        f"stock={row.stock} units"
    )


# Quick test -- make sure the tools work before giving them to the agent
print(get_order_status(1042))
print(get_returns(1042))
print(get_product_info("Laptop"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build the DSPy ReAct Agent
# MAGIC
# MAGIC ### What is ReAct?
# MAGIC
# MAGIC **ReAct** stands for **Reasoning + Acting**. It is an agent architecture where the LLM alternates
# MAGIC between two phases in a loop:
# MAGIC
# MAGIC 1. **Reason (Think):** The LLM analyzes the question and decides what to do next.
# MAGIC 2. **Act (Tool Call):** The LLM calls a tool (e.g., query a database) and observes the result.
# MAGIC
# MAGIC This loop repeats until the agent has enough information to produce a final answer. The key
# MAGIC insight is that by *thinking out loud* before acting, the agent makes better decisions about
# MAGIC which tools to call and how to interpret the results.
# MAGIC
# MAGIC ```
# MAGIC Question: "What is the status of order 1042?"
# MAGIC   -> Think: "I need to look up order 1042. I'll use get_order_status."
# MAGIC   -> Act:   get_order_status(1042) -> "Order 1042: status=Shipped, ..."
# MAGIC   -> Think: "I have the info. The order is shipped."
# MAGIC   -> Answer: "Order 1042 has been shipped."
# MAGIC ```
# MAGIC
# MAGIC ### DSPy ReAct module
# MAGIC
# MAGIC In DSPy, `dspy.ReAct` is a built-in module that implements this pattern. You give it:
# MAGIC - A **signature** describing the input/output (question -> answer)
# MAGIC - A list of **tools** the agent can call
# MAGIC - A **max_iters** limit to prevent infinite loops
# MAGIC
# MAGIC DSPy handles all the prompt engineering internally -- you never write "You are a helpful
# MAGIC assistant..." prompts yourself.

# COMMAND ----------

# Define the agent's input/output signature.
# The docstring becomes the system instruction that guides the LLM's behavior.
class OrderSupport(dspy.Signature):
    """You are a helpful customer order support agent. Answer questions about
    orders, returns, and products by using the available tools. Always provide
    clear, concise answers based on the data returned by the tools. If a tool
    returns no results, tell the user politely."""

    question: str = dspy.InputField(desc="The customer's question about orders, returns, or products")
    answer: str = dspy.OutputField(desc="A helpful answer based on tool results")


# Build the ReAct agent with our three tools.
# max_iters=5 means the agent can do up to 5 think-act cycles before it must answer.
tools = [get_order_status, get_returns, get_product_info]
react_agent = dspy.ReAct(OrderSupport, tools=tools, max_iters=5)

print("ReAct agent created with tools:", [t.__name__ for t in tools])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test the Agent Locally
# MAGIC
# MAGIC Before wrapping the agent for deployment, we test it directly. This is a critical step --
# MAGIC always verify that your agent produces correct answers in a notebook before investing time
# MAGIC in packaging and serving.
# MAGIC
# MAGIC We test three scenarios, one for each tool:
# MAGIC 1. An order status question (exercises `get_order_status`)
# MAGIC 2. A returns question (exercises `get_returns`)
# MAGIC 3. A product question (exercises `get_product_info`)
# MAGIC
# MAGIC Notice that we just call `react_agent(question=...)` -- DSPy handles the entire ReAct loop
# MAGIC internally. The returned object has a `.answer` attribute matching our signature.

# COMMAND ----------

# Test 1: Order status
result1 = react_agent(question="What is the status of order 1042?")
print("Q: What is the status of order 1042?")
print(f"A: {result1.answer}\n")

# Test 2: Returns
result2 = react_agent(question="Are there any returns for order 1045?")
print("Q: Are there any returns for order 1045?")
print(f"A: {result2.answer}\n")

# Test 3: Product info
result3 = react_agent(question="What is the price of a Laptop?")
print("Q: What is the price of a Laptop?")
print(f"A: {result3.answer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Wrap in a ResponsesAgent for MLflow Serving
# MAGIC
# MAGIC ### What is ResponsesAgent?
# MAGIC
# MAGIC `ResponsesAgent` is a class from `mlflow.pyfunc` that provides a standard interface for
# MAGIC serving agents via Databricks Model Serving. It is the recommended way to deploy agents
# MAGIC with MLflow 3 because:
# MAGIC
# MAGIC 1. **Standard API format** -- It uses the OpenAI-compatible **Responses API** format, which
# MAGIC    means any frontend that speaks the Responses API can talk to your agent.
# MAGIC 2. **Streaming support** -- It defines both `predict()` (full response) and `predict_stream()`
# MAGIC    (server-sent events), which are essential for real-time chat UIs.
# MAGIC 3. **Deployment-ready** -- Databricks Model Serving knows how to load and serve a
# MAGIC    `ResponsesAgent` automatically. No custom serving code needed.
# MAGIC
# MAGIC ### Why not just use the DSPy agent directly?
# MAGIC
# MAGIC DSPy agents return DSPy-specific result objects (like `dspy.Prediction`). Model Serving
# MAGIC needs a standardized request/response format. `ResponsesAgent` bridges this gap -- it
# MAGIC translates between the Responses API format and your DSPy agent.
# MAGIC
# MAGIC ### Streaming: Why it matters for agents
# MAGIC
# MAGIC Agents can take 5-30 seconds to respond because they make multiple LLM calls and tool
# MAGIC invocations. Without streaming, the user stares at a blank screen the entire time.
# MAGIC With streaming, partial results appear immediately:
# MAGIC
# MAGIC - The user sees "Thinking..." as the agent reasons
# MAGIC - Tool call results appear as they happen
# MAGIC - The final answer streams token by token
# MAGIC
# MAGIC We implement streaming using `dspy.streamify()`, which wraps a DSPy module to yield
# MAGIC intermediate results as they are produced.

# COMMAND ----------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
import json


class DSPyOrderAgent(ResponsesAgent):
    """A customer order support agent built with DSPy ReAct, wrapped for MLflow serving.

    This class translates between the Responses API format (used by Model Serving)
    and the DSPy ReAct agent (which does the actual reasoning and tool calling).
    """

    def __init__(self):
        """Initialize the DSPy agent, LLM, and tools.

        This runs once when the model is loaded for serving. We configure
        the LLM and build the ReAct agent here so it is ready for requests.
        """
        import dspy

        lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
        dspy.configure(lm=lm)

        self.tools = [get_order_status, get_returns, get_product_info]
        self.agent = dspy.ReAct(OrderSupport, tools=self.tools, max_iters=5)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle a single request and return the full response.

        This method:
        1. Extracts the user's question from the Responses API request
        2. Runs the DSPy ReAct agent
        3. Packages the answer into a Responses API response

        Args:
            request: A ResponsesAgentRequest with the user's messages.

        Returns:
            A ResponsesAgentResponse containing the agent's answer.
        """
        # Extract the last user message from the request
        question = self._extract_question(request)

        # Run the ReAct agent
        result = self.agent(question=question)

        # Build and return the Responses API response
        return ResponsesAgentResponse.from_text(result.answer)

    def predict_stream(self, request: ResponsesAgentRequest):
        """Handle a request and yield streaming events.

        This method uses dspy.streamify() to get intermediate results from
        the ReAct agent as they are produced, then yields them as
        ResponsesAgentStreamEvent objects that Model Serving can send as
        server-sent events (SSE) to the client.

        Args:
            request: A ResponsesAgentRequest with the user's messages.

        Yields:
            ResponsesAgentStreamEvent objects for each chunk of the response.
        """
        import dspy

        question = self._extract_question(request)

        # streamify wraps the ReAct module to yield partial results
        streaming_agent = dspy.streamify(self.agent)
        stream = streaming_agent(question=question)

        accumulated_text = ""
        for chunk in stream:
            # dspy.streamify yields different types: dspy.Prediction for the final
            # result, and intermediate strings/objects during reasoning.
            if isinstance(chunk, dspy.Prediction):
                # Final prediction -- yield the complete answer
                final_text = chunk.answer
                # Yield any remaining text that was not streamed yet
                remaining = final_text[len(accumulated_text):]
                if remaining:
                    yield ResponsesAgentStreamEvent.from_text(remaining)
            elif isinstance(chunk, str):
                accumulated_text += chunk
                yield ResponsesAgentStreamEvent.from_text(chunk)

    def _extract_question(self, request: ResponsesAgentRequest) -> str:
        """Extract the user's question from the Responses API request.

        The request contains a list of messages. We take the last message with
        role='user' as the question to answer.
        """
        # The input field contains the conversation messages
        messages = request.input
        # Find the last user message
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "user":
                # The content may be a string or a list of content parts
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content parts
                    texts = [
                        part.text for part in msg.content
                        if hasattr(part, "text")
                    ]
                    return " ".join(texts)
        return "No question provided."


# Instantiate the agent for local testing
agent_wrapper = DSPyOrderAgent()
print("DSPyOrderAgent instantiated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test the Wrapped Agent Locally
# MAGIC
# MAGIC Now we test the `DSPyOrderAgent` by sending it a `ResponsesAgentRequest` -- the exact same
# MAGIC format that Model Serving will use in production. This is important because it verifies
# MAGIC the full translation pipeline: Responses API request -> DSPy ReAct -> Responses API response.
# MAGIC
# MAGIC If this works correctly, the agent is ready for deployment.

# COMMAND ----------

# Build a test request in Responses API format
test_request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "What is the status of order 1042?"}
    ]
)

# Test predict (non-streaming)
response = agent_wrapper.predict(test_request)
print("Non-streaming response:")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Test Streaming
# MAGIC
# MAGIC We test the `predict_stream()` method to verify that streaming works. In production,
# MAGIC Model Serving calls this method and sends each yielded event to the client as a
# MAGIC server-sent event (SSE). The client can then display partial results in real time.
# MAGIC
# MAGIC Here we simply collect and print the stream events to verify they are produced correctly.

# COMMAND ----------

# Test predict_stream (streaming)
test_request_stream = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "What is the price of a Phone?"}
    ]
)

print("Streaming response:")
for event in agent_wrapper.predict_stream(test_request_stream):
    print(event, end="", flush=True)
print()  # Final newline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Log the Agent with MLflow (Code-Based Logging)
# MAGIC
# MAGIC ### What is MLflow model logging?
# MAGIC
# MAGIC When you **log** a model with MLflow, you save a snapshot of the model (or the code that
# MAGIC defines it) along with its dependencies, parameters, and metadata. This snapshot is stored
# MAGIC as an **MLflow artifact** and can be:
# MAGIC
# MAGIC - **Loaded** later for testing or inference
# MAGIC - **Registered** to Unity Catalog for governance
# MAGIC - **Deployed** to Model Serving with one click
# MAGIC
# MAGIC ### The code-based logging pattern
# MAGIC
# MAGIC There are two ways to log a model with MLflow:
# MAGIC
# MAGIC 1. **Object-based** -- serialize the model object directly (e.g., pickle). This works for
# MAGIC    simple models but breaks for agents that depend on Spark, database connections, or
# MAGIC    runtime configuration.
# MAGIC
# MAGIC 2. **Code-based** -- save the Python source code that defines the agent, and MLflow
# MAGIC    re-executes it at load time. This is the recommended pattern for agents because:
# MAGIC    - The agent can re-establish connections (to Spark, LLMs, etc.) at serving time
# MAGIC    - You avoid serialization issues with complex objects
# MAGIC    - The code is human-readable in the MLflow artifact viewer
# MAGIC
# MAGIC With code-based logging, you:
# MAGIC 1. Write the agent code to a `.py` file
# MAGIC 2. Call `mlflow.pyfunc.log_model(python_model="path/to/agent.py", ...)`
# MAGIC 3. MLflow saves the file and re-runs it when loading the model
# MAGIC
# MAGIC The file must define a class that inherits from `ResponsesAgent` (or `ChatModel`) and
# MAGIC must set a module-level variable `AGENT` to an instance of that class, OR the file must
# MAGIC define exactly one `ResponsesAgent` subclass that MLflow will instantiate automatically.

# COMMAND ----------

import os

# Write the agent code to a file. This file must be self-contained --
# it must import everything it needs and define all tools and the agent class.
agent_code = '''
import dspy
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

mlflow.dspy.autolog()

# ---- LLM Configuration ----
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# ---- Tools ----
def get_order_status(order_id: int) -> str:
    """Look up the status and details of a customer order by its order ID.

    Use this tool when the user asks about a specific order, such as
    'What is the status of order 1042?' or 'Tell me about order 1050'.

    Args:
        order_id: The numeric order ID to look up (e.g., 1042).

    Returns:
        A string with the order details or a 'not found' message.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
    )
    if df.count() == 0:
        return f"No order found with ID {order_id}."
    row = df.first()
    return (
        f"Order {row.order_id}: customer={row.customer_name}, product={row.product}, "
        f"quantity={row.quantity}, status={row.status}, order_date={row.order_date}"
    )


def get_returns(order_id: int) -> str:
    """Look up return requests associated with a specific order ID.

    Use this tool when the user asks about returns for an order, such as
    'Are there any returns for order 1045?' or 'Show me returns on order 1042'.

    Args:
        order_id: The numeric order ID to look up returns for.

    Returns:
        A string listing all returns for that order, or a message saying none were found.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
    )
    if df.count() == 0:
        return f"No returns found for order {order_id}."
    rows = df.collect()
    results = []
    for row in rows:
        results.append(
            f"Return {row.return_id}: reason={row.reason}, status={row.status}, "
            f"return_date={row.return_date}"
        )
    return f"Returns for order {order_id}:\\n" + "\\n".join(results)


def get_product_info(product_name: str) -> str:
    """Look up product details by product name.

    Use this tool when the user asks about a specific product, such as
    'What is the price of a Laptop?' or 'Is the Phone in stock?'.

    Args:
        product_name: The name of the product (e.g., 'Laptop', 'Phone', 'Tablet').

    Returns:
        A string with the product details or a 'not found' message.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.products WHERE lower(name) = lower(\\'{product_name}\\')"
    )
    if df.count() == 0:
        return f"No product found with name \\'{product_name}\\'."
    row = df.first()
    return (
        f"Product: {row.name}, category={row.category}, price=${row.price}, "
        f"stock={row.stock} units"
    )


# ---- DSPy Signature ----
class OrderSupport(dspy.Signature):
    """You are a helpful customer order support agent. Answer questions about
    orders, returns, and products by using the available tools. Always provide
    clear, concise answers based on the data returned by the tools. If a tool
    returns no results, tell the user politely."""

    question: str = dspy.InputField(desc="The customer\\'s question about orders, returns, or products")
    answer: str = dspy.OutputField(desc="A helpful answer based on tool results")


# ---- Agent Class ----
class DSPyOrderAgent(ResponsesAgent):
    """Customer order support agent built with DSPy ReAct."""

    def __init__(self):
        import dspy as _dspy
        lm = _dspy.LM(LLM_ENDPOINT)
        _dspy.configure(lm=lm)
        self.tools = [get_order_status, get_returns, get_product_info]
        self.agent = _dspy.ReAct(OrderSupport, tools=self.tools, max_iters=5)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        question = self._extract_question(request)
        result = self.agent(question=question)
        return ResponsesAgentResponse.from_text(result.answer)

    def predict_stream(self, request: ResponsesAgentRequest):
        import dspy as _dspy
        question = self._extract_question(request)
        streaming_agent = _dspy.streamify(self.agent)
        stream = streaming_agent(question=question)
        accumulated_text = ""
        for chunk in stream:
            if isinstance(chunk, _dspy.Prediction):
                final_text = chunk.answer
                remaining = final_text[len(accumulated_text):]
                if remaining:
                    yield ResponsesAgentStreamEvent.from_text(remaining)
            elif isinstance(chunk, str):
                accumulated_text += chunk
                yield ResponsesAgentStreamEvent.from_text(chunk)

    def _extract_question(self, request: ResponsesAgentRequest) -> str:
        messages = request.input
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "user":
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    texts = [
                        part.text for part in msg.content
                        if hasattr(part, "text")
                    ]
                    return " ".join(texts)
        return "No question provided."


# MLflow looks for a set_agent() call or a single ResponsesAgent subclass.
# We use set_agent() to be explicit.
AGENT = DSPyOrderAgent()
mlflow.pyfunc.set_agent(AGENT)
'''

# Write the agent code to a file in the current working directory
agent_code_path = os.path.join(os.getcwd(), "dspy_order_agent.py")
with open(agent_code_path, "w") as f:
    f.write(agent_code)

print(f"Agent code written to: {agent_code_path}")

# COMMAND ----------

# Log the model using code-based logging
with mlflow.start_run(run_name="dspy_order_agent") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="dspy_order_agent",
        python_model=agent_code_path,
        pip_requirements=[
            "mlflow>=3",
            "dspy",
            "databricks-agents",
        ],
    )

    print(f"Model logged to: {model_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What just happened?
# MAGIC
# MAGIC We used the **code-based logging pattern** to save our agent to MLflow:
# MAGIC
# MAGIC 1. **Wrote the agent code** to `dspy_order_agent.py` -- a self-contained file that defines
# MAGIC    all tools, the DSPy signature, and the `DSPyOrderAgent` class.
# MAGIC
# MAGIC 2. **Called `mlflow.pyfunc.log_model()`** with:
# MAGIC    - `python_model=agent_code_path` -- tells MLflow to save this Python file as the model
# MAGIC    - `pip_requirements` -- the packages needed to run the agent in a serving environment
# MAGIC    - `artifact_path` -- where to store the model within the MLflow run
# MAGIC
# MAGIC 3. **MLflow saved** the code, dependencies, and metadata as an artifact. When this model is
# MAGIC    loaded (for serving or testing), MLflow will:
# MAGIC    - Install the pip requirements
# MAGIC    - Execute the Python file
# MAGIC    - Find the `AGENT` variable (or the `set_agent()` call)
# MAGIC    - Use it to handle incoming requests
# MAGIC
# MAGIC This pattern is powerful because the agent code re-creates its own Spark session, LLM
# MAGIC connections, and tools at serving time -- no pickling of complex objects needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Evaluate the Agent with LLM-as-a-Judge
# MAGIC
# MAGIC ### What is LLM-as-a-judge evaluation?
# MAGIC
# MAGIC Traditional software testing uses exact assertions (`assert result == expected`). But agent
# MAGIC outputs are natural language -- there are many correct ways to say "Order 1042 has been shipped."
# MAGIC Exact string matching would fail even for perfect answers.
# MAGIC
# MAGIC **LLM-as-a-judge** solves this by using a *separate* LLM to evaluate the agent's answers.
# MAGIC The judge LLM receives:
# MAGIC - The original question
# MAGIC - The expected answer (ground truth)
# MAGIC - The agent's actual answer
# MAGIC
# MAGIC It then scores the answer on criteria like:
# MAGIC - **Correctness** -- Does the answer contain the right information?
# MAGIC - **Relevance** -- Does it answer the question that was asked?
# MAGIC - **Groundedness** -- Is it based on retrieved data (not hallucinated)?
# MAGIC
# MAGIC ### Using mlflow.evaluate()
# MAGIC
# MAGIC MLflow provides built-in LLM-as-a-judge metrics through `mlflow.evaluate()`. We pass:
# MAGIC - An evaluation dataset with questions and expected answers
# MAGIC - The logged model URI
# MAGIC - The metric names to compute
# MAGIC
# MAGIC The results are logged to the MLflow run and visible in the MLflow UI.

# COMMAND ----------

import pandas as pd

# Define an evaluation dataset with questions and expected answers.
# The expected answers do not need to be exact -- the LLM judge evaluates
# semantic similarity, not string equality.
eval_data = pd.DataFrame(
    {
        "request": [
            '{"input": [{"role": "user", "content": "What is the status of order 1042?"}]}',
            '{"input": [{"role": "user", "content": "Are there any returns for order 1045?"}]}',
            '{"input": [{"role": "user", "content": "What is the price of a Laptop?"}]}',
        ],
        "expected_response": [
            "The status of order 1042 should include information about the order such as status, product, and customer.",
            "The response should indicate whether returns exist for order 1045 and provide details if they do.",
            "The response should include the price of a Laptop from the products table.",
        ],
    }
)

# Run evaluation using the logged model
eval_results = mlflow.evaluate(
    data=eval_data,
    model=model_info.model_uri,
    model_type="databricks-agent",
)

# Display the evaluation results
print("Evaluation metrics:")
print(eval_results.metrics)

# Show per-row results
display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the Evaluation Results
# MAGIC
# MAGIC The evaluation produces several metrics:
# MAGIC
# MAGIC - **`response/llm_judged/relevance/rating`** -- How relevant is the answer to the question?
# MAGIC   (1-5 scale, higher is better)
# MAGIC - **`response/llm_judged/groundedness/rating`** -- Is the answer grounded in retrieved data?
# MAGIC   (1-5 scale)
# MAGIC - **`response/llm_judged/safety/rating`** -- Is the answer safe and appropriate? (1-5 scale)
# MAGIC
# MAGIC These metrics are computed by a judge LLM and logged to the MLflow run. You can view them
# MAGIC in the MLflow Tracking UI alongside your model artifacts.
# MAGIC
# MAGIC **Best practices for evaluation:**
# MAGIC - Use at least 10-20 diverse test questions for meaningful results
# MAGIC - Include edge cases (e.g., "What is order 9999?" for a nonexistent order)
# MAGIC - Review the per-row results to understand where your agent struggles
# MAGIC - Re-run evaluation after making changes to measure improvement

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Register the Model to Unity Catalog
# MAGIC
# MAGIC ### What is Unity Catalog model registration?
# MAGIC
# MAGIC Unity Catalog (UC) is Databricks' governance layer for all data and AI assets. When you
# MAGIC **register** a model to UC, you:
# MAGIC
# MAGIC 1. **Version** the model -- each registration creates a new version, so you can roll back
# MAGIC 2. **Govern** access -- UC permissions control who can view, use, or deploy the model
# MAGIC 3. **Deploy** easily -- Model Serving can load directly from a UC model path
# MAGIC 4. **Audit** usage -- UC tracks who registered, modified, or deployed each version
# MAGIC
# MAGIC The registration path follows the three-level namespace: `catalog.schema.model_name`.
# MAGIC After registration, you can deploy the model to a serving endpoint from the Databricks UI
# MAGIC or programmatically.

# COMMAND ----------

import mlflow

# Set the registry to Unity Catalog (not the legacy workspace registry)
mlflow.set_registry_uri("databricks-uc")

# Define the model name in Unity Catalog
UC_MODEL_NAME = "sjdatabricks.orders.dspy_order_agent"

# Register the model from the logged run
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=UC_MODEL_NAME,
)

print(f"Model registered: {UC_MODEL_NAME}")
print(f"Version: {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Test the Registered Model (Optional)
# MAGIC
# MAGIC As a final validation, we load the model back from MLflow and send it a test request.
# MAGIC This simulates exactly what Model Serving will do: load the model artifact, instantiate
# MAGIC the agent, and call `predict()`.
# MAGIC
# MAGIC If this cell works, your agent is fully ready for deployment to a serving endpoint.

# COMMAND ----------

# Load the model back from the logged run
loaded_agent = mlflow.pyfunc.load_model(model_info.model_uri)

# Send a test request
test_input = {
    "input": [
        {"role": "user", "content": "Tell me about order 1050 and whether it has any returns."}
    ]
}
result = loaded_agent.predict(test_input)
print("Loaded model response:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving
# MAGIC
# MAGIC Now that the model is registered in Unity Catalog, we can deploy it to a **Model Serving endpoint**.
# MAGIC This creates a real-time REST API that any application can call to interact with the agent.
# MAGIC
# MAGIC **What happens during deployment:**
# MAGIC 1. Databricks provisions a container with the specified workload size
# MAGIC 2. It installs the model's pip dependencies
# MAGIC 3. It loads the model artifact and calls `set_model()` to instantiate the agent
# MAGIC 4. The endpoint starts accepting HTTP requests routed to `predict()` or `predict_stream()`
# MAGIC
# MAGIC **Note:** The first deployment takes ~15 minutes to provision. Subsequent updates are faster.
# MAGIC Scale-to-zero is enabled so you won't be charged when the endpoint is idle.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

ENDPOINT_NAME = "dspy-order-support-agent"
UC_MODEL_NAME = "sjdatabricks.orders.dspy_order_agent"

latest_version = registered_model.version

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
# MAGIC ## Endpoint Provisioning
# MAGIC
# MAGIC The `create_and_wait` call above blocks until the endpoint is fully provisioned.
# MAGIC During this time, Databricks is:
# MAGIC - Spinning up compute resources for the serving container
# MAGIC - Installing all pip dependencies from the logged model
# MAGIC - Loading the model artifact and calling `set_model()` to instantiate the agent
# MAGIC - Running health checks to verify the endpoint is ready to serve traffic
# MAGIC
# MAGIC If the cell above completed successfully, the endpoint is live and ready for queries.
# MAGIC You can also monitor the endpoint status in the Databricks UI under **Serving**.

# COMMAND ----------

# Check endpoint status
status = w.serving_endpoints.get(name=ENDPOINT_NAME)
print(f"Endpoint: {status.name}")
print(f"State: {status.state.ready}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Deployed Endpoint — Non-Streaming
# MAGIC
# MAGIC Once the endpoint is live, you can call it from **any Python environment** — not just Databricks.
# MAGIC The Responses API follows the OpenAI-compatible format, so you can use the `openai` SDK or
# MAGIC `databricks-sdk` to query it.
# MAGIC
# MAGIC **Non-streaming** returns the complete response in a single HTTP reply. This is simpler
# MAGIC to implement but the caller must wait for the full response before showing anything to the user.

# COMMAND ----------

# MAGIC %pip install openai -q

# COMMAND ----------

import os
from openai import OpenAI

# The OpenAI SDK works with Databricks Model Serving endpoints
client = OpenAI(
    base_url=f"{w.config.host}/serving-endpoints",
    api_key=w.tokens().token,
)

# Non-streaming call
response = client.responses.create(
    model=ENDPOINT_NAME,
    input=[{"role": "user", "content": "What is the status of order 1042?"}],
)

# Extract the text from the response
for item in response.output:
    if hasattr(item, "content"):
        for content in item.content:
            if hasattr(content, "text"):
                print(content.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Deployed Endpoint — Streaming
# MAGIC
# MAGIC **Streaming** sends response chunks as they are generated, so your frontend can show
# MAGIC partial answers in real time. This is the standard pattern for chat-style UIs.
# MAGIC
# MAGIC The streaming response sends Server-Sent Events (SSE), and the OpenAI SDK handles
# MAGIC the parsing automatically. Each chunk may contain text deltas that you concatenate
# MAGIC to build the full response.

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
print()  # newline at end

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we built a complete **Customer Order Support Agent** using DSPy and MLflow:
# MAGIC
# MAGIC | Step | What we did | Why it matters |
# MAGIC |------|------------|----------------|
# MAGIC | 1 | Installed dependencies | MLflow 3 + DSPy + databricks-agents |
# MAGIC | 2 | Configured DSPy LLM | Points to Databricks Foundation Model API |
# MAGIC | 3 | Defined tools | Spark SQL queries for orders, returns, products |
# MAGIC | 4 | Built ReAct agent | Reasoning + Acting loop for tool use |
# MAGIC | 5 | Tested locally | Verified correctness before packaging |
# MAGIC | 6 | Wrapped in ResponsesAgent | Standard API format for Model Serving |
# MAGIC | 7-8 | Tested wrapper + streaming | Verified end-to-end compatibility |
# MAGIC | 9 | Logged with MLflow | Code-based logging for reproducibility |
# MAGIC | 10 | Evaluated with LLM-as-a-judge | Automated quality scoring |
# MAGIC | 11 | Registered to Unity Catalog | Governance + deployment readiness |
# MAGIC | 12 | Tested loaded model | Final validation before deployment |
# MAGIC | 13 | Deployed to Model Serving | Real-time REST API endpoint |
# MAGIC | 14 | Queried non-streaming | Full response in one HTTP call |
# MAGIC | 15 | Queried streaming | Real-time token-by-token output |
# MAGIC
# MAGIC The agent is now **live** and can be called from any application using the OpenAI-compatible API.
