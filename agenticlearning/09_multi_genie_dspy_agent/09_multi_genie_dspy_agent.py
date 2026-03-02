# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Topic 9: Multi-Genie Agent with DSPy
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC In Topics 7-8, you built an agent that routes questions to a **single** Genie Space. But in
# MAGIC real-world applications, your data is often spread across multiple domains — orders, returns,
# MAGIC products, inventory, customer profiles, and so on. Each domain may have its own Genie Space.
# MAGIC
# MAGIC This tutorial teaches you to build a **multi-Genie agent** that:
# MAGIC 1. Accepts a user question
# MAGIC 2. Classifies it into one of three departments (orders, returns, products)
# MAGIC 3. Routes it to the correct Genie Space
# MAGIC 4. Returns the data-driven answer
# MAGIC
# MAGIC ### What is Multi-Agent Routing?
# MAGIC
# MAGIC Multi-agent routing is a pattern where a **router agent** sits in front of multiple **specialist
# MAGIC agents** (in our case, Genie Spaces). The router's job is to understand the user's intent and
# MAGIC forward the question to the right specialist.
# MAGIC
# MAGIC ```
# MAGIC User Question: "What products are in the Electronics category?"
# MAGIC      |
# MAGIC      v
# MAGIC [DSPy Router] --> department = "products"
# MAGIC      |
# MAGIC      +---> orders_genie   (skipped)
# MAGIC      +---> returns_genie  (skipped)
# MAGIC      +---> products_genie (selected!) --> SQL answer about Electronics products
# MAGIC ```
# MAGIC
# MAGIC ### Why Multiple Genie Spaces Instead of One Big One?
# MAGIC
# MAGIC You might wonder: why not put all tables into one Genie Space? There are several reasons:
# MAGIC
# MAGIC 1. **Focused SQL generation**: When a Genie Space has fewer tables, the SQL generator is more
# MAGIC    accurate. It does not get confused by irrelevant columns from other domains.
# MAGIC 2. **Access control**: Different teams may own different data. Separate spaces let you apply
# MAGIC    different permissions.
# MAGIC 3. **Performance**: Smaller schemas mean faster query planning.
# MAGIC 4. **Maintainability**: When the products team adds a column, it does not affect the orders space.
# MAGIC
# MAGIC ### DSPy for Multi-Class Classification
# MAGIC
# MAGIC In Topic 7, we used DSPy for binary classification (needs_data: True/False). Here we extend it
# MAGIC to **multi-class classification** — the model outputs one of three string labels. DSPy handles
# MAGIC this naturally: you define the output field as a `str` with a description listing the valid
# MAGIC categories, and Chain-of-Thought reasoning picks the best one.

# COMMAND ----------

# MAGIC %pip install mlflow>=3 dspy databricks-agents -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC Same packages as Topic 7. The multi-Genie pattern does not require any additional libraries —
# MAGIC it is purely a design pattern on top of the same building blocks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure Three Genie Space IDs
# MAGIC
# MAGIC We create three widgets — one for each Genie Space. Each space covers a different domain:
# MAGIC
# MAGIC | Widget | Domain | Table |
# MAGIC |--------|--------|-------|
# MAGIC | `orders_space_id` | Customer orders | `sjdatabricks.orders.order_details` |
# MAGIC | `returns_space_id` | Return requests | `sjdatabricks.orders.returns` |
# MAGIC | `products_space_id` | Product catalog | `sjdatabricks.orders.products` |
# MAGIC
# MAGIC You get these IDs from the `00_setup/create_genie_spaces.py` notebook or from the Databricks
# MAGIC UI under AI/BI Genie. Each space must be created and configured before running this notebook.

# COMMAND ----------

dbutils.widgets.text("orders_space_id", "", "Orders Genie Space ID")
dbutils.widgets.text("returns_space_id", "", "Returns Genie Space ID")
dbutils.widgets.text("products_space_id", "", "Products Genie Space ID")

ORDERS_SPACE_ID = dbutils.widgets.get("orders_space_id")
RETURNS_SPACE_ID = dbutils.widgets.get("returns_space_id")
PRODUCTS_SPACE_ID = dbutils.widgets.get("products_space_id")

for name, sid in [("Orders", ORDERS_SPACE_ID), ("Returns", RETURNS_SPACE_ID), ("Products", PRODUCTS_SPACE_ID)]:
    if not sid:
        raise ValueError(
            f"Please set the '{name.lower()}_space_id' widget. "
            "Run 00_setup/create_genie_spaces.py to create Genie Spaces."
        )
    print(f"{name} Space ID: {sid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure DSPy Language Model
# MAGIC
# MAGIC Same as Topic 7 — we use Llama 3.3 70B Instruct as the classification model. For multi-class
# MAGIC classification, this model is more than capable. The Chain-of-Thought reasoning helps it
# MAGIC distinguish between similar categories (e.g., "return status" goes to returns, not orders).

# COMMAND ----------

import dspy

lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
dspy.configure(lm=lm)

print("DSPy configured with Llama 3.3 70B Instruct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Three GenieAgent Instances
# MAGIC
# MAGIC We instantiate one `GenieAgent` per Genie Space. Each agent is independent — it only knows
# MAGIC about its own space's tables. The routing logic (Step 5) decides which one to call.
# MAGIC
# MAGIC We also create a **genie_map** dictionary that maps department names to GenieAgent instances.
# MAGIC This makes the routing code clean: `genie_map[department].predict(...)`.

# COMMAND ----------

from databricks_agents.genie import GenieAgent

orders_genie = GenieAgent(genie_space_id=ORDERS_SPACE_ID)
returns_genie = GenieAgent(genie_space_id=RETURNS_SPACE_ID)
products_genie = GenieAgent(genie_space_id=PRODUCTS_SPACE_ID)

genie_map = {
    "orders": orders_genie,
    "returns": returns_genie,
    "products": products_genie,
}

print("Created 3 GenieAgent instances:")
for dept, agent in genie_map.items():
    print(f"  {dept}: {agent}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Define the DSPy Multi-Class Routing Signature
# MAGIC
# MAGIC This is the key difference from Topic 7. Instead of a boolean `needs_data` field, we define
# MAGIC a string `department` field that must be one of three values: "orders", "returns", or "products".
# MAGIC
# MAGIC **How DSPy handles multi-class classification:**
# MAGIC
# MAGIC 1. The `desc` parameter tells the model what values are valid.
# MAGIC 2. `ChainOfThought` generates reasoning first: *"The user asks about product prices, which
# MAGIC    relates to the product catalog, so department = products"*.
# MAGIC 3. The model outputs the department label as a string.
# MAGIC
# MAGIC **Edge cases**: What if the question does not fit any department (e.g., "What is the weather?")?
# MAGIC We handle this in the routing function by treating unknown departments as a fallback. In a
# MAGIC production system, you might add a fourth category like "general" or "unknown" to the signature.
# MAGIC
# MAGIC **Why Chain-of-Thought matters for multi-class**: With binary classification, the model just
# MAGIC picks yes/no. With three categories, the reasoning trace helps the model distinguish between
# MAGIC similar domains. For example, "return status for order 1042" mentions both returns and orders —
# MAGIC the CoT reasoning helps the model realize it is primarily about returns.

# COMMAND ----------

class RouteQuestion(dspy.Signature):
    """Classify a customer question into the appropriate department for routing."""

    question: str = dspy.InputField(desc="The customer's question")
    department: str = dspy.OutputField(
        desc="The department to route to. Must be one of: 'orders' (for questions about "
        "order status, quantities, shipping, order dates), 'returns' (for questions about "
        "return requests, refund status, return reasons), or 'products' (for questions about "
        "product catalog, pricing, stock levels, categories)"
    )

router = dspy.ChainOfThought(RouteQuestion)

# Test the router with three different question types
test_questions = [
    "How many orders shipped last week?",
    "Show me all pending returns",
    "What products are in the Electronics category?",
]

for q in test_questions:
    result = router(question=q)
    print(f"Q: {q}")
    print(f"  Department: {result.department}")
    print(f"  Reasoning: {result.reasoning[:100]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Build the Multi-Genie Handle Function
# MAGIC
# MAGIC The handle function now has three possible destinations instead of two (data vs. fallback).
# MAGIC The flow is:
# MAGIC
# MAGIC 1. **Classify** the question into a department using the DSPy router.
# MAGIC 2. **Normalize** the department string (lowercase, strip whitespace) to handle minor variations.
# MAGIC 3. **Look up** the department in the `genie_map` dictionary.
# MAGIC 4. If found, **call** the corresponding GenieAgent.
# MAGIC 5. If not found (unknown department), **return a fallback** listing the available departments.
# MAGIC
# MAGIC **Why normalize?** LLMs can be inconsistent with capitalization and whitespace. The model might
# MAGIC output "Orders" or " orders " instead of "orders". Normalizing prevents routing failures due
# MAGIC to minor formatting differences.

# COMMAND ----------

def handle_question(question: str) -> str:
    """Route a question to the appropriate Genie Space."""
    # Step 1: Classify
    classification = router(question=question)
    department = classification.department.strip().lower()

    # Step 2: Look up the Genie agent
    selected_genie = genie_map.get(department)

    if selected_genie is None:
        return (
            f"I could not determine the right department for your question. "
            f"I can help with: orders, returns, and products. "
            f"(Router suggested: '{department}')"
        )

    # Step 3: Call the Genie Space
    try:
        response = selected_genie.predict(
            {"messages": [{"role": "user", "content": question}]}
        )
        if response and response.get("messages"):
            return response["messages"][-1].get("content", "No response from Genie.")
        return "No response from Genie."
    except Exception as e:
        return f"Error querying the {department} database: {str(e)}"

# Test
print(handle_question("What products are in the Electronics category?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Wrap in a ResponsesAgent
# MAGIC
# MAGIC The `ResponsesAgent` structure is the same as Topics 7-8. The only internal difference is
# MAGIC that `_route()` now classifies into three departments instead of doing binary classification.
# MAGIC
# MAGIC **Design note**: We pass all three space IDs into the constructor. In a production deployment,
# MAGIC these would come from environment variables or a configuration service rather than notebook
# MAGIC widgets. The ResponsesAgent pattern supports this because `__init__` runs at model load time.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    Response,
    ResponseMessage,
    TextContent,
    ResponsesAgentStreamEvent,
)

class MultiGenieOrderAgent(ResponsesAgent):
    """
    A DSPy-powered agent that routes questions to one of three Genie Spaces
    based on the question's department (orders, returns, or products).
    """

    def __init__(self):
        import dspy
        from databricks_agents.genie import GenieAgent

        # Configure DSPy
        self.lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
        dspy.configure(lm=self.lm)

        # Create three Genie agents
        self.genie_map = {
            "orders": GenieAgent(genie_space_id=ORDERS_SPACE_ID),
            "returns": GenieAgent(genie_space_id=RETURNS_SPACE_ID),
            "products": GenieAgent(genie_space_id=PRODUCTS_SPACE_ID),
        }

        # Define the routing signature
        class _RouteQuestion(dspy.Signature):
            """Classify a customer question into orders, returns, or products."""
            question: str = dspy.InputField()
            department: str = dspy.OutputField(
                desc="Must be one of: 'orders', 'returns', 'products'"
            )

        self.router = dspy.ChainOfThought(_RouteQuestion)

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
        """Classify the question and route to the appropriate Genie Space."""
        classification = self.router(question=question)
        department = classification.department.strip().lower()

        selected_genie = self.genie_map.get(department)

        if selected_genie is None:
            return (
                f"I could not determine the right department for your question. "
                f"I can help with: orders, returns, and products. "
                f"(Router classified as: '{department}')"
            )

        try:
            response = selected_genie.predict(
                {"messages": [{"role": "user", "content": question}]}
            )
            if response and response.get("messages"):
                return response["messages"][-1].get("content", "No response from Genie.")
            return "No response from Genie."
        except Exception as e:
            return f"Error querying the {department} database: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Test with Three Different Question Types
# MAGIC
# MAGIC We test one question per department to verify that routing works correctly. Each question
# MAGIC should be sent to its respective Genie Space:
# MAGIC
# MAGIC - **Orders question** -> `orders_genie` -> queries `order_details` table
# MAGIC - **Returns question** -> `returns_genie` -> queries `returns` table
# MAGIC - **Products question** -> `products_genie` -> queries `products` table
# MAGIC
# MAGIC **Debugging tip**: If a question gets routed to the wrong department, check the MLflow trace
# MAGIC to see the router's reasoning. You might need to adjust the DSPy signature description to
# MAGIC give the model clearer guidance about what each department covers.

# COMMAND ----------

agent = MultiGenieOrderAgent()

test_cases = [
    ("Orders", "How many orders are in Shipped status?"),
    ("Returns", "Show me all pending return requests"),
    ("Products", "What is the price of the Laptop?"),
]

for dept, question in test_cases:
    print(f"=== {dept} Question ===")
    print(f"Q: {question}")
    response = agent.predict(
        {"messages": [{"role": "user", "content": question}]}
    )
    print(f"A: {response.output[0].content[0].text}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Log the Agent with MLflow
# MAGIC
# MAGIC Same logging pattern as before. The model artifact captures the multi-Genie routing logic,
# MAGIC all three GenieAgent configurations, and the DSPy router.

# COMMAND ----------

mlflow.set_experiment("/Users/{}/09_multi_genie_dspy_agent".format(
    spark.sql("SELECT current_user()").first()[0]
))

input_example = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}

with mlflow.start_run(run_name="multi_genie_dspy_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=MultiGenieOrderAgent(),
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
# MAGIC Our evaluation dataset now covers all three departments. This is important because we need
# MAGIC to verify that routing works for each category, not just one. A common mistake is testing
# MAGIC only the "happy path" — make sure to include edge cases like ambiguous questions.
# MAGIC
# MAGIC **Evaluation best practice**: Include at least 2-3 questions per department, plus a few
# MAGIC edge cases (ambiguous questions, off-topic questions). For production, aim for 50+ examples.

# COMMAND ----------

import pandas as pd

eval_data = pd.DataFrame(
    {
        "messages": [
            [{"role": "user", "content": "How many orders are in Processing status?"}],
            [{"role": "user", "content": "Show me all approved returns"}],
            [{"role": "user", "content": "What products are in the Accessories category?"}],
            [{"role": "user", "content": "What is the return reason for order 1045?"}],
            [{"role": "user", "content": "How much stock do we have for Monitors?"}],
        ],
        "expected_response": [
            "Should return a count of processing orders from the orders database.",
            "Should return approved return records from the returns database.",
            "Should return Keyboard and Mouse from the products database.",
            "Should query the returns database for return reason linked to order 1045.",
            "Should query the products database for Monitor stock levels.",
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
# MAGIC We register the multi-Genie agent as a separate model from the single-Genie agents in
# MAGIC Topics 7-8. This allows independent versioning and deployment.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.orders.multi_genie_dspy_agent"

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

ENDPOINT_NAME = "multi-genie-dspy-agent"

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
# MAGIC You extended the single-Genie pattern from Topic 7 into a **multi-Genie routing agent**:
# MAGIC
# MAGIC | Concept | Single Genie (Topic 7) | Multi-Genie (This Topic) |
# MAGIC |---------|----------------------|------------------------|
# MAGIC | Genie Spaces | 1 | 3 (orders, returns, products) |
# MAGIC | Classification | Binary (needs_data: bool) | Multi-class (department: str) |
# MAGIC | Routing | Data vs. fallback | Orders vs. returns vs. products |
# MAGIC | DSPy Signature | `OrderQuery` | `RouteQuestion` |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. **Separate Genie Spaces by domain** for better SQL accuracy and access control.
# MAGIC 2. **Use multi-class classification** (not nested if/else) for clean routing logic.
# MAGIC 3. **Normalize LLM output** (lowercase, strip) to handle formatting inconsistencies.
# MAGIC 4. **Test all routes** — evaluate with questions from every department.
# MAGIC 5. **Deploy to Model Serving** — create a REST API endpoint for production access.
# MAGIC 6. **Query via OpenAI SDK** — use non-streaming and streaming calls to interact with the agent.
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **Topic 10**: Same multi-Genie pattern but with LangChain routing
# MAGIC - **Topic 13**: Package this agent as a wheel for production deployment
