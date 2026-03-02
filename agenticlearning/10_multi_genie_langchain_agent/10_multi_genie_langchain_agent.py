# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Topic 10: Multi-Genie Agent with LangChain
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC This tutorial builds the same **multi-Genie routing agent** as Topic 9, but uses **LangChain**
# MAGIC instead of DSPy for the classification step. You will route user questions to one of three
# MAGIC Genie Spaces (orders, returns, products) using `ChatDatabricks`.
# MAGIC
# MAGIC ### Comparing Approaches Across All Four Genie Tutorials
# MAGIC
# MAGIC | Tutorial | Framework | Genie Spaces | Classification Type |
# MAGIC |----------|-----------|-------------|-------------------|
# MAGIC | Topic 7 | DSPy | 1 | Binary (needs_data: bool) |
# MAGIC | Topic 8 | LangChain | 1 | Binary (yes/no string) |
# MAGIC | Topic 9 | DSPy | 3 | Multi-class (department: str) |
# MAGIC | **Topic 10** | **LangChain** | **3** | **Multi-class (department string)** |
# MAGIC
# MAGIC ### LangChain for Multi-Class Classification
# MAGIC
# MAGIC In Topic 8, we asked the LLM a yes/no question. Now we need it to pick one of three categories.
# MAGIC The approach is the same — send a prompt and parse the output — but the prompt is more detailed
# MAGIC and the parsing logic handles three categories instead of two.
# MAGIC
# MAGIC **Prompt engineering matters**: For multi-class classification with LangChain, the quality of
# MAGIC your system prompt directly affects routing accuracy. We include:
# MAGIC - Clear category definitions with examples
# MAGIC - Explicit instruction to output ONLY the category name
# MAGIC - A fallback instruction for ambiguous questions
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC User Question: "Show me all pending return requests"
# MAGIC      |
# MAGIC      v
# MAGIC [LangChain Classifier] --> department = "returns"
# MAGIC      |
# MAGIC      +---> orders_genie   (skipped)
# MAGIC      +---> returns_genie  (selected!) --> SQL answer about pending returns
# MAGIC      +---> products_genie (skipped)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install mlflow>=3 langchain langchain-databricks databricks-agents -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC Same packages as Topic 8. The multi-Genie pattern is a design choice, not a library change.
# MAGIC All the complexity lives in the routing prompt and the dictionary of Genie agents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure Three Genie Space IDs and the LLM
# MAGIC
# MAGIC We set up three widgets for the space IDs and initialize `ChatDatabricks`. Notice that we
# MAGIC increase `max_tokens` from 10 (Topic 8) to 20. This gives the model more room to output the
# MAGIC department name, especially if it adds minor formatting. We still use `temperature=0` for
# MAGIC deterministic classification.
# MAGIC
# MAGIC **Why determinism matters for routing**: If the same question gets routed to different
# MAGIC departments on different calls, the agent becomes unpredictable. Setting `temperature=0`
# MAGIC ensures the same question always goes to the same Genie Space.

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

from langchain_databricks import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0,
    max_tokens=20,
)

print("LangChain ChatDatabricks model initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Three GenieAgent Instances
# MAGIC
# MAGIC Identical to Topic 9 — one `GenieAgent` per domain, stored in a `genie_map` dictionary.
# MAGIC The framework used for routing (DSPy vs. LangChain) does not affect how Genie agents are
# MAGIC created or called.

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
for dept in genie_map:
    print(f"  - {dept}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: LangChain Multi-Class Classification
# MAGIC
# MAGIC This is the core of the LangChain routing approach. We write a **system prompt** that:
# MAGIC
# MAGIC 1. Defines the three departments with clear descriptions
# MAGIC 2. Gives examples of questions for each department
# MAGIC 3. Instructs the model to output ONLY the department name
# MAGIC
# MAGIC **Prompt design tips for multi-class classification:**
# MAGIC
# MAGIC - **Be specific about boundaries**: "Return reasons" goes to returns, not orders — make this
# MAGIC   explicit in the prompt so the model does not guess.
# MAGIC - **Include edge cases**: Questions like "return status for order 1042" touch both domains.
# MAGIC   The prompt should clarify which department owns this (returns, because the primary action
# MAGIC   is about a return).
# MAGIC - **Keep the output format strict**: "Answer ONLY with one word" prevents the model from
# MAGIC   generating explanations that are harder to parse.
# MAGIC
# MAGIC **Parsing**: We extract the first word from the response and match it against the known
# MAGIC department names. This is more robust than exact string matching because the model might
# MAGIC output "orders." (with a period) or "Orders" (capitalized).

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage

ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a customer support system.
Given a user question, classify it into exactly one department.

DEPARTMENTS:
- orders: Questions about customer orders, order status, shipping, order dates, quantities ordered
- returns: Questions about return requests, refund status, return reasons, return dates, return approvals
- products: Questions about the product catalog, product names, categories, pricing, stock levels

RULES:
- Output ONLY the department name (orders, returns, or products). Nothing else.
- If the question mentions returns or refunds, classify as "returns" even if it references an order ID.
- If the question is about product details (price, stock, category), classify as "products".
- If the question is about order status, shipping, or when something was ordered, classify as "orders".
- If the question does not fit any department, output "unknown".

Examples:
- "How many orders shipped last week?" -> orders
- "What is the return reason for order 1045?" -> returns
- "What products are in Electronics?" -> products
- "Show me all pending returns" -> returns
- "What is the price of a Laptop?" -> products"""

def classify_question(question: str) -> str:
    """Classify a question into a department: orders, returns, products, or unknown."""
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    department = response.content.strip().lower().split()[0] if response.content.strip() else "unknown"

    # Clean up any trailing punctuation
    department = department.rstrip(".,!?;:")

    return department if department in ("orders", "returns", "products") else "unknown"

# Test with three question types
test_questions = [
    "How many orders are in Shipped status?",
    "Show me all pending return requests",
    "What products are in the Electronics category?",
]

for q in test_questions:
    dept = classify_question(q)
    print(f"Q: {q}")
    print(f"  Department: {dept}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Build the Multi-Genie Handle Function
# MAGIC
# MAGIC The handle function follows the same pattern as Topic 9:
# MAGIC
# MAGIC 1. **Classify** using the LangChain-based `classify_question()`.
# MAGIC 2. **Look up** the GenieAgent from `genie_map`.
# MAGIC 3. **Call Genie** or return a fallback for unknown departments.
# MAGIC
# MAGIC The function is identical in structure to Topic 9's version. The only difference is that
# MAGIC `classify_question()` uses LangChain internally instead of DSPy. This demonstrates how
# MAGIC the **routing interface** (input: question, output: department string) is framework-agnostic.

# COMMAND ----------

def handle_question(question: str) -> str:
    """Route a question to the appropriate Genie Space."""
    department = classify_question(question)

    selected_genie = genie_map.get(department)

    if selected_genie is None:
        return (
            f"I could not determine the right department for your question. "
            f"I can help with: orders, returns, and products. "
            f"(Classifier suggested: '{department}')"
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

# Test
print(handle_question("What products are in the Electronics category?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Wrap in a ResponsesAgent
# MAGIC
# MAGIC The `ResponsesAgent` wrapper follows the exact same pattern as all previous tutorials. This
# MAGIC consistency is intentional — MLflow's agent interface is designed to be a **stable contract**
# MAGIC between your agent logic and the deployment infrastructure.
# MAGIC
# MAGIC **What changes between tutorials:**
# MAGIC - The internal `__init__` creates different LLM clients (DSPy vs. LangChain)
# MAGIC - The `_route()` method calls different classification functions
# MAGIC - The `_classify()` method uses different frameworks
# MAGIC
# MAGIC **What stays the same:**
# MAGIC - `predict()` signature and return type
# MAGIC - `predict_stream()` signature and event types
# MAGIC - Message extraction logic
# MAGIC - Error handling patterns
# MAGIC - MLflow tracing decorators
# MAGIC
# MAGIC This means you can swap the routing framework without changing your deployment pipeline,
# MAGIC evaluation setup, or serving configuration.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    Response,
    ResponseMessage,
    TextContent,
    ResponsesAgentStreamEvent,
)

class MultiGenieLangChainAgent(ResponsesAgent):
    """
    A LangChain-powered agent that routes questions to one of three Genie Spaces
    based on the question's department (orders, returns, or products).
    """

    def __init__(self):
        from langchain_databricks import ChatDatabricks
        from langchain_core.messages import HumanMessage, SystemMessage
        from databricks_agents.genie import GenieAgent

        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0,
            max_tokens=20,
        )
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage

        # Create three Genie agents
        self.genie_map = {
            "orders": GenieAgent(genie_space_id=ORDERS_SPACE_ID),
            "returns": GenieAgent(genie_space_id=RETURNS_SPACE_ID),
            "products": GenieAgent(genie_space_id=PRODUCTS_SPACE_ID),
        }

        self.router_prompt = """You are a routing classifier for a customer support system.
Given a user question, classify it into exactly one department.

DEPARTMENTS:
- orders: Questions about customer orders, order status, shipping, order dates, quantities
- returns: Questions about return requests, refund status, return reasons, return dates
- products: Questions about the product catalog, names, categories, pricing, stock levels

RULES:
- Output ONLY the department name (orders, returns, or products). Nothing else.
- If the question mentions returns or refunds, classify as "returns".
- If the question is about product details, classify as "products".
- If the question is about order status or shipping, classify as "orders".
- If unsure, output "unknown"."""

    def _classify(self, question: str) -> str:
        """Classify a question into a department."""
        messages = [
            self._SystemMessage(content=self.router_prompt),
            self._HumanMessage(content=question),
        ]
        response = self.llm.invoke(messages)
        department = response.content.strip().lower().split()[0] if response.content.strip() else "unknown"
        department = department.rstrip(".,!?;:")
        return department if department in ("orders", "returns", "products") else "unknown"

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
        """Classify and route to the appropriate Genie Space."""
        department = self._classify(question)

        selected_genie = self.genie_map.get(department)

        if selected_genie is None:
            return (
                f"I could not determine the right department for your question. "
                f"I can help with: orders, returns, and products. "
                f"(Classifier output: '{department}')"
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
# MAGIC ## Step 7: Test with Three Question Types
# MAGIC
# MAGIC We test one question per department, same as Topic 9. Comparing the outputs between
# MAGIC Topics 9 and 10 should yield identical or very similar results — both agents route to
# MAGIC the same Genie Spaces and get the same SQL-generated answers. The only difference is
# MAGIC the classification mechanism.

# COMMAND ----------

agent = MultiGenieLangChainAgent()

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
# MAGIC ## Step 8: Log the Agent with MLflow
# MAGIC
# MAGIC We log with LangChain-specific pip requirements. Notice how the logging call is identical
# MAGIC in structure across all four Genie tutorials — only the class name and requirements change.

# COMMAND ----------

mlflow.set_experiment("/Users/{}/10_multi_genie_langchain_agent".format(
    spark.sql("SELECT current_user()").first()[0]
))

input_example = {
    "messages": [{"role": "user", "content": "How many orders are in Shipped status?"}]
}

with mlflow.start_run(run_name="multi_genie_langchain_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=MultiGenieLangChainAgent(),
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
# MAGIC Same evaluation dataset as Topic 9, covering all three departments. By using the same dataset
# MAGIC across Topics 9 and 10, you can directly compare DSPy vs. LangChain routing accuracy in the
# MAGIC MLflow UI. Look at the evaluation metrics side by side to see which framework produces better
# MAGIC routing decisions for your specific data.
# MAGIC
# MAGIC **Comparison tip**: In the MLflow Experiments page, select both runs (Topic 9 and Topic 10)
# MAGIC and click "Compare". This shows metrics side by side and highlights any differences in
# MAGIC relevance, groundedness, or safety scores.

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
# MAGIC ## Step 10: Register to Unity Catalog
# MAGIC
# MAGIC Final step — register the multi-Genie LangChain agent for deployment.

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.orders.multi_genie_langchain_agent"

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

ENDPOINT_NAME = "multi-genie-langchain-agent"

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
# MAGIC You have now completed all four Genie agent tutorials. Here is the full comparison:
# MAGIC
# MAGIC | Tutorial | Framework | Spaces | Classification | Key Strength |
# MAGIC |----------|-----------|--------|---------------|-------------|
# MAGIC | Topic 7 | DSPy | 1 | Binary | Minimal code, auto-generated prompts |
# MAGIC | Topic 8 | LangChain | 1 | Binary | Full control over prompt wording |
# MAGIC | Topic 9 | DSPy | 3 | Multi-class | Clean typed signatures for routing |
# MAGIC | Topic 10 | LangChain | 3 | Multi-class | Detailed prompt engineering for accuracy |
# MAGIC
# MAGIC ### When to Use Which Approach
# MAGIC
# MAGIC - **DSPy** is best when you want **simplicity and automatic prompt optimization**. Define
# MAGIC   your inputs/outputs as typed fields and let DSPy handle the rest. Great for teams that
# MAGIC   prefer declarative code and want to experiment with prompt optimization later.
# MAGIC
# MAGIC - **LangChain** is best when you need **fine-grained control over prompts and chains**.
# MAGIC   Write exactly the prompt you want, add few-shot examples, chain multiple steps together.
# MAGIC   Great for teams already using the LangChain ecosystem.
# MAGIC
# MAGIC - **Single Genie** (Topics 7-8) is sufficient when all your data lives in one domain or
# MAGIC   one Genie Space covers all relevant tables.
# MAGIC
# MAGIC - **Multi-Genie** (Topics 9-10) is needed when your data spans multiple domains with
# MAGIC   different access controls, different update frequencies, or different table schemas.
# MAGIC
# MAGIC ### What Stayed the Same Across All Four Tutorials
# MAGIC
# MAGIC - `GenieAgent` creation and invocation
# MAGIC - `ResponsesAgent` interface (`predict` / `predict_stream`)
# MAGIC - MLflow model logging (`log_model`)
# MAGIC - LLM-as-a-judge evaluation (`mlflow.evaluate`)
# MAGIC - Unity Catalog registration (`register_model`)
# MAGIC - Model Serving deployment and OpenAI SDK querying
# MAGIC
# MAGIC This consistency is by design. The MLflow agent ecosystem provides stable building blocks
# MAGIC so you can focus on the part that matters most — the routing and reasoning logic.
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **Topics 11-14**: Package these agents as Python wheels for production deployment
# MAGIC - **Review the MLflow UI**: Compare evaluation results across all four tutorials
# MAGIC - **Experiment**: Try adding a fourth Genie Space or a different classification model
