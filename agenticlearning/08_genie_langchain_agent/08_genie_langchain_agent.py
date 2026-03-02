# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Topic 8: Simple Genie Agent with LangChain
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC This tutorial builds the **same agent** as Topic 7 — a single Genie Space with an LLM router —
# MAGIC but uses **LangChain** instead of DSPy for the routing layer. By comparing both approaches,
# MAGIC you will understand the trade-offs and can pick the framework that fits your team.
# MAGIC
# MAGIC ### DSPy vs. LangChain for Routing — Quick Comparison
# MAGIC
# MAGIC | Aspect | DSPy (Topic 7) | LangChain (This Topic) |
# MAGIC |--------|---------------|----------------------|
# MAGIC | Style | Declarative signatures | Imperative chains / prompt templates |
# MAGIC | Routing | `dspy.ChainOfThought` auto-generates CoT | You write the prompt and parse the output |
# MAGIC | Flexibility | Great for structured classification | Great for complex multi-step chains |
# MAGIC | Learning curve | Smaller API surface | Larger ecosystem, more to learn |
# MAGIC
# MAGIC ### Architecture (Same as Topic 7)
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC      |
# MAGIC      v
# MAGIC [LangChain Router] -- needs_data? --> YES --> [GenieAgent] --> SQL answer
# MAGIC      |                                                          |
# MAGIC      +-- needs_data? --> NO --> Fallback text response          |
# MAGIC                                                                 v
# MAGIC                                                          Final Response
# MAGIC ```
# MAGIC
# MAGIC The only difference from Topic 7 is that the classification box uses LangChain's
# MAGIC `ChatDatabricks` model with a prompt template instead of a DSPy signature.

# COMMAND ----------

# MAGIC %pip install mlflow>=3 langchain langchain-databricks databricks-agents -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC | Package | Purpose |
# MAGIC |---------|---------|
# MAGIC | `mlflow>=3` | ResponsesAgent interface, model logging, evaluation |
# MAGIC | `langchain` | Core LangChain framework for chains and prompts |
# MAGIC | `langchain-databricks` | `ChatDatabricks` — LangChain wrapper for Databricks Foundation Models |
# MAGIC | `databricks-agents` | `GenieAgent` for calling Genie Spaces |
# MAGIC
# MAGIC `langchain-databricks` is a separate package (not bundled with `langchain`) that provides
# MAGIC the `ChatDatabricks` class. This class connects to Databricks Model Serving endpoints and
# MAGIC handles authentication automatically within a Databricks notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure Widgets and the LLM
# MAGIC
# MAGIC We create a widget for the Genie Space ID (same as Topic 7) and initialize the LangChain
# MAGIC `ChatDatabricks` model.
# MAGIC
# MAGIC **ChatDatabricks** is LangChain's interface to Databricks-hosted LLMs. You specify the
# MAGIC `endpoint` (the Model Serving endpoint name) and optionally `temperature` and `max_tokens`.
# MAGIC
# MAGIC - `temperature=0` makes the model deterministic — given the same input, it returns the same
# MAGIC   output every time. This is ideal for classification tasks where you want consistent routing.
# MAGIC - `max_tokens=10` limits the response length since we only need a short "yes" or "no" answer.
# MAGIC   This saves tokens (cost) and prevents the model from generating unnecessary text.

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "", "Genie Space ID")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")

if not GENIE_SPACE_ID:
    raise ValueError(
        "Please set the 'genie_space_id' widget at the top of this notebook. "
        "You can find the Space ID in the Databricks UI under AI/BI Genie."
    )

print(f"Using Genie Space ID: {GENIE_SPACE_ID}")

# COMMAND ----------

from langchain_databricks import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0,
    max_tokens=10,
)

print("LangChain ChatDatabricks model initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create the GenieAgent
# MAGIC
# MAGIC Same as Topic 7 — we create a single `GenieAgent` pointing at the configured Genie Space.
# MAGIC The GenieAgent is framework-agnostic; it does not care whether DSPy or LangChain does the
# MAGIC routing. It simply receives a question and returns data.

# COMMAND ----------

from databricks_agents.genie import GenieAgent

genie = GenieAgent(genie_space_id=GENIE_SPACE_ID)

print(f"GenieAgent created for space: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: LangChain Routing — Classify the Question
# MAGIC
# MAGIC In LangChain, routing is done by **invoking the LLM with a prompt** and parsing the output.
# MAGIC We write a prompt that asks the model to answer "yes" or "no" — does this question require
# MAGIC looking up data from a database?
# MAGIC
# MAGIC **How this differs from DSPy:**
# MAGIC - In DSPy, you define a typed signature (`needs_data: bool`) and DSPy handles the prompt.
# MAGIC - In LangChain, you write the prompt yourself and parse the string output.
# MAGIC
# MAGIC Both approaches work. LangChain gives you more control over the exact prompt wording, while
# MAGIC DSPy abstracts it away. For simple yes/no classification, the difference is minimal.
# MAGIC
# MAGIC **Parsing strategy**: We convert the LLM output to lowercase and check if it starts with "yes".
# MAGIC This is simple but robust enough for a classification task. For more complex parsing, LangChain
# MAGIC offers `OutputParser` classes, but they are overkill here.

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage

ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier. The user will ask a question. "
    "Determine if the question requires looking up data from an orders/returns/products database. "
    "Answer ONLY 'yes' or 'no'. Nothing else."
)

def classify_question(question: str) -> bool:
    """Return True if the question needs a database lookup."""
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    answer = response.content.strip().lower()
    return answer.startswith("yes")

# Quick test
print(f"'How many shipped orders?' -> needs_data={classify_question('How many shipped orders?')}")
print(f"'What is the weather?' -> needs_data={classify_question('What is the weather?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Build the Handle Function
# MAGIC
# MAGIC This is identical in structure to Topic 7's `handle_question()`. The only difference is that
# MAGIC we call `classify_question()` (LangChain-based) instead of the DSPy router.
# MAGIC
# MAGIC The pattern is always the same:
# MAGIC 1. Classify the question (router layer)
# MAGIC 2. If data needed, call Genie
# MAGIC 3. Otherwise, return a fallback
# MAGIC
# MAGIC This separation of concerns (routing vs. execution) is a fundamental agent design pattern.
# MAGIC It makes each component independently testable and replaceable.

# COMMAND ----------

def handle_question(question: str) -> str:
    """Route a question to Genie or return a fallback."""
    needs_data = classify_question(question)

    if needs_data:
        try:
            response = genie.predict(
                {"messages": [{"role": "user", "content": question}]}
            )
            if response and response.get("messages"):
                return response["messages"][-1].get("content", "No response from Genie.")
            return "No response from Genie."
        except Exception as e:
            return f"Error querying data: {str(e)}"
    else:
        return (
            "I'm a data assistant for order information. "
            "I can help you look up orders, returns, and product details. "
            "Could you rephrase your question to be about our order data?"
        )

# Test
print(handle_question("How many orders are in Processing status?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Wrap in a ResponsesAgent
# MAGIC
# MAGIC The `ResponsesAgent` interface is the same regardless of whether you use DSPy or LangChain
# MAGIC internally. This is one of the strengths of the MLflow agent framework — it provides a
# MAGIC **consistent deployment interface** while letting you use any LLM framework underneath.
# MAGIC
# MAGIC **Key implementation details:**
# MAGIC
# MAGIC - `__init__` re-creates the LLM, GenieAgent, and routing function. This is important because
# MAGIC   when MLflow loads the model from storage, it calls `__init__` fresh — any state from the
# MAGIC   notebook session is not available.
# MAGIC - `predict()` extracts the last user message, routes it, and returns a `Response` object.
# MAGIC - `predict_stream()` wraps `predict()` in streaming events. For a more sophisticated
# MAGIC   implementation, you could stream tokens from `ChatDatabricks` directly.
# MAGIC - `@mlflow.trace()` decorators enable **MLflow Tracing** — each call generates a trace
# MAGIC   that shows the routing decision, Genie call, and response. You can view these traces
# MAGIC   in the MLflow UI for debugging.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    Response,
    ResponseMessage,
    TextContent,
    ResponsesAgentStreamEvent,
)

class GenieLangChainAgent(ResponsesAgent):
    """
    A LangChain-powered agent that routes data questions to a Genie Space
    and handles general questions with a fallback response.
    """

    def __init__(self):
        from langchain_databricks import ChatDatabricks
        from langchain_core.messages import HumanMessage, SystemMessage
        from databricks_agents.genie import GenieAgent

        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0,
            max_tokens=10,
        )
        self.genie = GenieAgent(genie_space_id=GENIE_SPACE_ID)
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
        self.router_prompt = (
            "You are a routing classifier. The user will ask a question. "
            "Determine if the question requires looking up data from an "
            "orders/returns/products database. Answer ONLY 'yes' or 'no'."
        )

    def _classify(self, question: str) -> bool:
        messages = [
            self._SystemMessage(content=self.router_prompt),
            self._HumanMessage(content=question),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip().lower().startswith("yes")

    @mlflow.trace(span_type="AGENT")
    def predict(self, request):
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        if not last_user_message:
            answer = "I didn't receive a question. Please ask about orders, returns, or products."
        else:
            answer = self._route(last_user_message)

        return Response(
            id=f"resp-{hash(last_user_message) % 10**8}",
            choices=[],
            output=[
                ResponseMessage(
                    role="assistant",
                    content=[TextContent(text=answer)],
                )
            ],
        )

    def predict_stream(self, request):
        full_response = self.predict(request)
        yield ResponsesAgentStreamEvent(
            type="response.output_text.delta",
            delta=full_response.output[0].content[0].text,
        )
        yield ResponsesAgentStreamEvent(
            type="response.completed",
            response=full_response,
        )

    @mlflow.trace(span_type="CHAIN")
    def _route(self, question: str) -> str:
        needs_data = self._classify(question)

        if needs_data:
            try:
                response = self.genie.predict(
                    {"messages": [{"role": "user", "content": question}]}
                )
                if response and response.get("messages"):
                    return response["messages"][-1].get("content", "No response from Genie.")
                return "No response from Genie."
            except Exception as e:
                return f"Error querying data: {str(e)}"
        else:
            return (
                "I'm a data assistant for order information. "
                "I can help with orders, returns, and product details. "
                "Please ask a data-related question."
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test the Agent Locally
# MAGIC
# MAGIC We run the same two test cases as Topic 7:
# MAGIC 1. A **data question** that should be routed to the Genie Space.
# MAGIC 2. A **general question** that should trigger the fallback response.
# MAGIC
# MAGIC If both work correctly, the LangChain routing layer is functioning properly.

# COMMAND ----------

agent = GenieLangChainAgent()

# Data question
data_request = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}
response = agent.predict(data_request)
print("=== Data Question ===")
print(response.output[0].content[0].text)

print()

# General question
general_request = {
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}
response = agent.predict(general_request)
print("=== General Question ===")
print(response.output[0].content[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Log the Agent with MLflow
# MAGIC
# MAGIC Code-based logging works the same way regardless of the framework inside the agent. MLflow
# MAGIC does not need to know that you used LangChain — it just records the code and dependencies.
# MAGIC
# MAGIC Notice the `pip_requirements` now includes `langchain` and `langchain-databricks` instead
# MAGIC of `dspy`. This ensures the correct packages are installed when the model is loaded for serving.

# COMMAND ----------

mlflow.set_experiment("/Users/{}/08_genie_langchain_agent".format(
    spark.sql("SELECT current_user()").first()[0]
))

input_example = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}

with mlflow.start_run(run_name="genie_langchain_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=GenieLangChainAgent(),
        pip_requirements=[
            "mlflow>=3",
            "langchain",
            "langchain-databricks",
            "databricks-agents",
        ],
        input_example=input_example,
    )

print(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC We use the same evaluation dataset as Topic 7. This makes it easy to compare the two
# MAGIC approaches — if both agents receive the same questions, you can directly compare their
# MAGIC evaluation scores in the MLflow UI.
# MAGIC
# MAGIC **Tip:** In a real project, maintain a single evaluation dataset in a shared location
# MAGIC (e.g., a Unity Catalog table or a Delta table). This ensures consistent benchmarking
# MAGIC across different agent implementations and versions.

# COMMAND ----------

import pandas as pd

eval_data = pd.DataFrame(
    {
        "messages": [
            [{"role": "user", "content": "How many orders are in Processing status?"}],
            [{"role": "user", "content": "What is your favorite color?"}],
            [{"role": "user", "content": "Show me all orders for Customer_1042"}],
        ],
        "expected_response": [
            "The agent should return a count of processing orders from the database.",
            "The agent should politely redirect to data-related questions.",
            "The agent should return order details for customer 1042.",
        ],
    }
)

eval_results = mlflow.evaluate(
    data=eval_data,
    model=model_info.model_uri,
    model_type="databricks-agent",
)

display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Register to Unity Catalog
# MAGIC
# MAGIC We register under a different name than Topic 7 so both versions can coexist. This lets
# MAGIC you deploy both agents side by side and A/B test them with real traffic.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.orders.genie_langchain_agent"

mlflow.set_registry_uri("databricks-uc")

registered = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=UC_MODEL_NAME,
)

print(f"Registered model: {registered.name}, version: {registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving
# MAGIC
# MAGIC Now that the model is registered in Unity Catalog, we deploy it to a **Model Serving endpoint**.
# MAGIC This creates a REST API that any application can call to interact with the agent.
# MAGIC
# MAGIC **What happens during deployment:**
# MAGIC 1. Databricks provisions a container with the specified workload size
# MAGIC 2. It installs the model's pip dependencies
# MAGIC 3. It loads the model artifact and calls `set_model()` to instantiate the agent
# MAGIC 4. The endpoint starts accepting HTTP requests routed to `predict()` or `predict_stream()`
# MAGIC
# MAGIC **Note:** First deployment takes ~15 minutes. Scale-to-zero means no charge when idle.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

ENDPOINT_NAME = "genie-langchain-order-agent"

try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_NAME,
                    entity_version=str(registered.version),
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
                    entity_version=str(registered.version),
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
# MAGIC ## Query the Deployed Endpoint — Non-Streaming
# MAGIC
# MAGIC Once live, you can call the endpoint from **any Python environment** using the OpenAI SDK.
# MAGIC Non-streaming returns the complete response in one HTTP reply.

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
# MAGIC ## Query the Deployed Endpoint — Streaming
# MAGIC
# MAGIC Streaming sends response chunks as they are generated for real-time chat UIs.

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
# MAGIC ## Summary
# MAGIC
# MAGIC You built the same Genie routing agent as Topic 7, but with LangChain handling the
# MAGIC classification step. The key takeaway is that **the framework choice (DSPy vs. LangChain)
# MAGIC only affects the routing layer** — everything else (GenieAgent, ResponsesAgent, MLflow
# MAGIC logging, evaluation, registration, deployment) stays exactly the same.
# MAGIC
# MAGIC | Component | Topic 7 (DSPy) | Topic 8 (LangChain) |
# MAGIC |-----------|---------------|---------------------|
# MAGIC | Router | `dspy.ChainOfThought(OrderQuery)` | `ChatDatabricks.invoke()` with prompt |
# MAGIC | Genie | `GenieAgent` | `GenieAgent` (same) |
# MAGIC | Agent wrapper | `ResponsesAgent` | `ResponsesAgent` (same) |
# MAGIC | Logging | `mlflow.pyfunc.log_model` | `mlflow.pyfunc.log_model` (same) |
# MAGIC | Deployment | Model Serving endpoint | Model Serving endpoint (same) |
# MAGIC | Querying | OpenAI SDK (streaming + non-streaming) | OpenAI SDK (same) |
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **Topic 9**: Multi-Genie routing with DSPy (3 Genie Spaces)
# MAGIC - **Topic 10**: Multi-Genie routing with LangChain (3 Genie Spaces)
