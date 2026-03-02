# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Topic 7: Simple Genie Agent with DSPy
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC In this tutorial, you will build an **AI agent** that combines two powerful ideas:
# MAGIC
# MAGIC 1. **Genie Spaces** — Databricks AI/BI feature that lets you ask natural-language questions
# MAGIC    against your data tables, and Genie translates them to SQL automatically.
# MAGIC 2. **DSPy routing** — a lightweight LLM framework that decides whether a user's question
# MAGIC    should be sent to the Genie Space or handled with a simple text response.
# MAGIC
# MAGIC ### What is a Genie Space?
# MAGIC
# MAGIC A Genie Space is a natural-language interface over one or more Unity Catalog tables. You
# MAGIC configure it by pointing it at tables (e.g., `sjdatabricks.orders.order_details`), and then
# MAGIC anyone can ask questions like *"How many orders shipped last week?"* — Genie writes the SQL,
# MAGIC runs it on a SQL warehouse, and returns the answer. Think of it as a **data analyst in a box**.
# MAGIC
# MAGIC ### What is a Genie Agent?
# MAGIC
# MAGIC A **Genie Agent** is a Python wrapper (`databricks_agents.genie.GenieAgent`) that lets your
# MAGIC code call a Genie Space programmatically. Instead of using the Databricks UI, your agent sends
# MAGIC a question string and gets back structured results. This means you can embed Genie as a **tool**
# MAGIC inside a larger agent pipeline.
# MAGIC
# MAGIC ### Why Combine DSPy Routing with Genie?
# MAGIC
# MAGIC Not every user question requires a database lookup. If someone asks *"What is your return policy?"*,
# MAGIC there is no SQL to run — it is a general knowledge question. A routing layer inspects the question
# MAGIC first and decides:
# MAGIC
# MAGIC - **Data question** (e.g., "Show shipped orders") --> send to Genie Space
# MAGIC - **General question** (e.g., "What are your hours?") --> reply with a fallback message
# MAGIC
# MAGIC DSPy makes this routing clean and declarative. You define a **signature** (input/output fields),
# MAGIC and DSPy uses Chain-of-Thought prompting to classify the question. No prompt engineering needed.
# MAGIC
# MAGIC ### Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC      |
# MAGIC      v
# MAGIC [DSPy Router] -- needs_data? --> YES --> [GenieAgent] --> SQL answer
# MAGIC      |                                                      |
# MAGIC      +-- needs_data? --> NO --> Fallback text response      |
# MAGIC                                                             v
# MAGIC                                                      Final Response
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install mlflow>=3 dspy databricks-agents -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC We install three packages:
# MAGIC
# MAGIC | Package | Purpose |
# MAGIC |---------|---------|
# MAGIC | `mlflow>=3` | MLflow 3 introduces the **ResponsesAgent** interface and code-based model logging |
# MAGIC | `dspy` | Declarative framework for LLM pipelines — we use it for classification / routing |
# MAGIC | `databricks-agents` | Provides `GenieAgent` for calling Genie Spaces and evaluation helpers |
# MAGIC
# MAGIC The `dbutils.library.restartPython()` call restarts the Python process so that the newly
# MAGIC installed packages are available in subsequent cells. This is standard practice in Databricks
# MAGIC notebooks when installing packages at runtime.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure the Genie Space ID
# MAGIC
# MAGIC Every Genie Space has a unique ID (a string like `"01efabc123..."`). You get this ID when you
# MAGIC create the space in the Databricks UI or via the API (see `00_setup/create_genie_spaces.py`).
# MAGIC
# MAGIC We use **Databricks widgets** to make this configurable. Widgets create a text input at the top
# MAGIC of the notebook so you (or anyone running this notebook) can paste in their own Space ID without
# MAGIC editing code. This is a best practice for parameterized notebooks.

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "", "Genie Space ID")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")

if not GENIE_SPACE_ID:
    raise ValueError(
        "Please set the 'genie_space_id' widget at the top of this notebook. "
        "You can find the Space ID in the Databricks UI under AI/BI Genie, "
        "or run 00_setup/create_genie_spaces.py to create spaces via the API."
    )

print(f"Using Genie Space ID: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure the DSPy Language Model
# MAGIC
# MAGIC DSPy needs an LLM backend for its classification modules. We use `databricks-meta-llama-3-3-70b-instruct`,
# MAGIC which is a **Foundation Model** hosted on Databricks Model Serving. This means:
# MAGIC
# MAGIC - No API keys needed — authentication is handled by the notebook environment.
# MAGIC - Low latency — the model runs inside your Databricks workspace.
# MAGIC - Cost-effective — Llama 3.3 70B is powerful enough for classification but cheaper than GPT-4 class models.
# MAGIC
# MAGIC The `dspy.configure(lm=...)` call sets this as the **default LM** for all DSPy modules in this notebook.
# MAGIC Every time a DSPy module needs to call an LLM (e.g., our router), it will use this model automatically.

# COMMAND ----------

import dspy

lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
dspy.configure(lm=lm)

print("DSPy configured with Llama 3.3 70B Instruct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create the GenieAgent
# MAGIC
# MAGIC `GenieAgent` is a thin wrapper around the Genie Space API. When you call it with a question,
# MAGIC it sends the question to the Genie Space, waits for the SQL to be generated and executed,
# MAGIC and returns the result as structured text.
# MAGIC
# MAGIC **Key points:**
# MAGIC - One `GenieAgent` instance maps to exactly one Genie Space.
# MAGIC - The Space must already exist and have tables configured (run `00_setup/` first).
# MAGIC - The agent handles SQL generation, execution, and result formatting — you just send a question.
# MAGIC
# MAGIC In this single-Genie tutorial, we have one agent for one space. In Topics 9-10, we will
# MAGIC create multiple GenieAgent instances for multi-space routing.

# COMMAND ----------

from databricks_agents.genie import GenieAgent

genie = GenieAgent(genie_space_id=GENIE_SPACE_ID)

print(f"GenieAgent created for space: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Define the DSPy Routing Signature
# MAGIC
# MAGIC A **DSPy Signature** defines the input and output fields for an LLM call. Think of it as a
# MAGIC typed function signature for a language model. Here we define:
# MAGIC
# MAGIC - **Input**: `question` (str) — the user's question
# MAGIC - **Output**: `needs_data` (bool) — True if the question needs a database lookup, False otherwise
# MAGIC
# MAGIC We wrap this in `dspy.ChainOfThought`, which tells DSPy to:
# MAGIC 1. Generate a reasoning trace (step-by-step thinking)
# MAGIC 2. Then produce the final boolean answer
# MAGIC
# MAGIC Chain-of-Thought (CoT) improves classification accuracy because the model "thinks aloud" before
# MAGIC deciding. For example, it might reason: *"The user asks about shipped orders — this requires
# MAGIC querying a database, so needs_data = True"*.
# MAGIC
# MAGIC **Why not just write a prompt?** DSPy signatures are more maintainable than raw prompts. If you
# MAGIC later want to add fields (e.g., `confidence_score`), you just add them to the signature — no
# MAGIC prompt rewriting needed. DSPy also supports automatic prompt optimization, which we do not cover
# MAGIC here but is a major advantage for production systems.

# COMMAND ----------

class OrderQuery(dspy.Signature):
    """Classify whether a customer question requires looking up order data from the database."""

    question: str = dspy.InputField(desc="The customer's question")
    needs_data: bool = dspy.OutputField(
        desc="True if the question requires querying order/product/return data from the database, "
        "False if it can be answered with general knowledge"
    )

router = dspy.ChainOfThought(OrderQuery)

# Quick test
test_result = router(question="How many orders shipped last week?")
print(f"Question: 'How many orders shipped last week?'")
print(f"Needs data: {test_result.needs_data}")
print(f"Reasoning: {test_result.reasoning}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Build the Routing Function
# MAGIC
# MAGIC Now we combine the DSPy router and the GenieAgent into a single function. The logic is simple:
# MAGIC
# MAGIC 1. Run the question through the DSPy `router` to get `needs_data`.
# MAGIC 2. If `needs_data` is True, send the question to the GenieAgent and return its response.
# MAGIC 3. If `needs_data` is False, return a polite fallback message.
# MAGIC
# MAGIC This pattern is called **tool routing** — the LLM decides which tool to use, and the framework
# MAGIC executes it. In more complex agents (Topics 9-10), the router picks from multiple tools.
# MAGIC
# MAGIC **Error handling**: We wrap the Genie call in a try/except because network issues, Genie
# MAGIC timeouts, or malformed questions can cause failures. A production agent should always handle
# MAGIC errors gracefully rather than crashing.

# COMMAND ----------

def handle_question(question: str) -> str:
    """Route a question to the Genie Space or return a fallback."""
    # Step 1: Classify
    classification = router(question=question)

    if classification.needs_data:
        # Step 2a: Send to Genie
        try:
            response = genie.predict(
                {"messages": [{"role": "user", "content": question}]}
            )
            # Extract the text content from the Genie response
            if response and response.get("messages"):
                return response["messages"][-1].get("content", "No response from Genie.")
            return "No response from Genie."
        except Exception as e:
            return f"I tried to look that up but encountered an error: {str(e)}"
    else:
        # Step 2b: Fallback
        return (
            "I'm a data assistant for order information. "
            "I can help you look up orders, returns, and product details. "
            "Could you rephrase your question to be about our order data?"
        )

# Quick test
print(handle_question("How many orders are in Processing status?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Wrap in a ResponsesAgent
# MAGIC
# MAGIC MLflow 3 introduces the **ResponsesAgent** interface — a standard way to package agents for
# MAGIC deployment on Databricks. Every ResponsesAgent must implement two methods:
# MAGIC
# MAGIC | Method | Purpose |
# MAGIC |--------|---------|
# MAGIC | `predict(request)` | Synchronous — takes a request, returns a complete `Response` |
# MAGIC | `predict_stream(request)` | Streaming — yields `ResponsesAgentStreamEvent` objects one at a time |
# MAGIC
# MAGIC The `request` object follows the OpenAI Responses API format. It contains a `messages` list
# MAGIC where each message has a `role` ("user" or "assistant") and `content` (the text). We extract
# MAGIC the last user message to get the current question.
# MAGIC
# MAGIC **Why ResponsesAgent?** This interface is what Databricks Model Serving expects. When you
# MAGIC deploy your agent as an endpoint, Serving calls `predict()` or `predict_stream()` automatically.
# MAGIC It also integrates with MLflow tracing, evaluation, and the Review App.
# MAGIC
# MAGIC **Streaming**: The `predict_stream` method yields events incrementally. For simplicity,
# MAGIC we generate the full response first and yield it as a single event. In production, you
# MAGIC could yield token-by-token for a better user experience.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    Response,
    ResponseMessage,
    TextContent,
    ResponsesAgentStreamEvent,
)

class GenieOrderAgent(ResponsesAgent):
    """
    A DSPy-powered agent that routes data questions to a Genie Space
    and handles general questions with a fallback response.
    """

    def __init__(self):
        import dspy
        from databricks_agents.genie import GenieAgent

        # Re-initialize inside __init__ so the agent works when loaded from MLflow
        self.lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
        dspy.configure(lm=self.lm)

        self.genie = GenieAgent(genie_space_id=GENIE_SPACE_ID)

        class _OrderQuery(dspy.Signature):
            """Classify whether a question requires database lookup."""
            question: str = dspy.InputField()
            needs_data: bool = dspy.OutputField(
                desc="True if the question requires querying order data"
            )

        self.router = dspy.ChainOfThought(_OrderQuery)

    @mlflow.trace(span_type="AGENT")
    def predict(self, request):
        """Handle a single request synchronously."""
        # Extract the last user message
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        if not last_user_message:
            answer = "I didn't receive a question. Please ask me about orders, returns, or products."
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
        """Handle a request with streaming output."""
        # Generate the full response, then yield as a single stream event
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
        """Internal routing logic."""
        classification = self.router(question=question)

        if classification.needs_data:
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
# MAGIC ## Step 8: Test the Agent Locally
# MAGIC
# MAGIC Before deploying, always test locally. We create an instance of `GenieOrderAgent` and call
# MAGIC `predict()` with a sample request that mimics the Responses API format.
# MAGIC
# MAGIC **What to look for:**
# MAGIC - Data questions should return SQL-derived answers from Genie.
# MAGIC - General questions should return the fallback message.
# MAGIC - Errors should be caught gracefully, not crash the agent.
# MAGIC
# MAGIC This local testing loop is much faster than deploying and testing via an endpoint.

# COMMAND ----------

agent = GenieOrderAgent()

# Test with a data question
data_request = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}
response = agent.predict(data_request)
print("=== Data Question ===")
print(response.output[0].content[0].text)

print()

# Test with a general question
general_request = {
    "messages": [{"role": "user", "content": "What is your return policy?"}]
}
response = agent.predict(general_request)
print("=== General Question ===")
print(response.output[0].content[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Log the Agent with MLflow
# MAGIC
# MAGIC **Code-based logging** is the recommended way to log agents in MLflow 3. Instead of serializing
# MAGIC the Python object (pickle), MLflow records the **source code file** that defines the agent class.
# MAGIC When the model is loaded later, MLflow imports the module and instantiates the class fresh.
# MAGIC
# MAGIC Benefits of code-based logging:
# MAGIC - **Reproducibility**: The exact code is stored, not a fragile pickle.
# MAGIC - **Transparency**: You can inspect what the agent does by reading its code.
# MAGIC - **Dependency tracking**: `pip_requirements` ensures the right packages are installed.
# MAGIC
# MAGIC The `log_model` call creates a new **MLflow Run** and stores:
# MAGIC - The agent code (this notebook or a separate `.py` file)
# MAGIC - The pip requirements
# MAGIC - An input example (for documentation and testing)

# COMMAND ----------

mlflow.set_experiment("/Users/{}/07_genie_dspy_agent".format(
    spark.sql("SELECT current_user()").first()[0]
))

input_example = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}

with mlflow.start_run(run_name="genie_dspy_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=GenieOrderAgent(),
        pip_requirements=[
            "mlflow>=3",
            "dspy",
            "databricks-agents",
        ],
        input_example=input_example,
    )

print(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC `mlflow.evaluate()` with `databricks-agents` uses an **LLM judge** to score your agent's
# MAGIC responses on dimensions like:
# MAGIC
# MAGIC - **Relevance**: Does the answer address the question?
# MAGIC - **Groundedness**: Is the answer supported by the retrieved data?
# MAGIC - **Safety**: Does the answer avoid harmful content?
# MAGIC
# MAGIC We pass a small evaluation dataset — a list of questions with optional expected answers.
# MAGIC The judge LLM (hosted by Databricks) reads each question-answer pair and assigns scores.
# MAGIC
# MAGIC **Why evaluate?** Without evaluation, you have no way to know if a code change improved or
# MAGIC degraded your agent. Evaluation lets you track quality over time, compare different routing
# MAGIC strategies, and catch regressions before deploying to production.

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
# MAGIC ## Step 11: Register to Unity Catalog
# MAGIC
# MAGIC Registering the model in **Unity Catalog** makes it available for:
# MAGIC
# MAGIC - **Deployment**: Create a Model Serving endpoint from the registered model.
# MAGIC - **Versioning**: Each registration creates a new version, so you can roll back if needed.
# MAGIC - **Access control**: Unity Catalog governs who can deploy or invoke the model.
# MAGIC - **Discovery**: Other team members can find and reuse your agent.
# MAGIC
# MAGIC The three-level name (`catalog.schema.model_name`) follows the same Unity Catalog namespace
# MAGIC as tables. After registration, you can deploy the agent to a serving endpoint with a single
# MAGIC click in the Databricks UI or via the API.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.orders.genie_dspy_agent"

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

ENDPOINT_NAME = "genie-dspy-order-agent"

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
# MAGIC In this tutorial you built a complete agent pipeline:
# MAGIC
# MAGIC 1. **Genie Space** — natural-language SQL interface over your order data
# MAGIC 2. **DSPy Router** — Chain-of-Thought classifier that decides if a question needs data
# MAGIC 3. **ResponsesAgent** — MLflow 3 standard interface for serving agents
# MAGIC 4. **Evaluation** — LLM-as-a-judge to measure response quality
# MAGIC 5. **Registration** — Unity Catalog for versioning and deployment
# MAGIC 6. **Deployment** — Model Serving endpoint with REST API access
# MAGIC 7. **Querying** — Non-streaming and streaming calls via OpenAI SDK
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **Topic 8**: Same pattern but using LangChain instead of DSPy for routing
# MAGIC - **Topic 9**: Multi-Genie routing with DSPy (3 Genie Spaces)
# MAGIC - **Topic 11**: Package this agent as a wheel for production deployment
