# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 2: Agent with DSPy + LangGraph (Classify -> Lookup -> Respond)
# MAGIC
# MAGIC This notebook builds a **Customer Order Support Agent** that combines two powerful frameworks:
# MAGIC
# MAGIC - **LangGraph** for orchestrating a multi-step pipeline as a directed graph
# MAGIC - **DSPy** with **ChainOfThought** for structured, reasoning-aware LLM calls at each step
# MAGIC
# MAGIC ## What is LangGraph?
# MAGIC
# MAGIC LangGraph is a library for building stateful, multi-step AI applications as **graphs**. Instead of writing
# MAGIC a single monolithic prompt, you break your logic into discrete **nodes** (functions) connected by **edges**
# MAGIC (transitions). A shared **state** dictionary flows through the graph, accumulating results at each step.
# MAGIC This makes complex agent workflows easier to reason about, test, and debug.
# MAGIC
# MAGIC ## Why combine DSPy with LangGraph?
# MAGIC
# MAGIC - **DSPy** excels at structured LLM interactions: you define typed *Signatures* (input/output schemas) and
# MAGIC   use modules like `ChainOfThought` that automatically prompt the LLM to show its reasoning before answering.
# MAGIC - **LangGraph** excels at *orchestration*: deciding which step runs next, passing data between steps, and
# MAGIC   managing the overall workflow.
# MAGIC - Together, each LangGraph node becomes a well-defined DSPy call with explicit inputs, outputs, and reasoning.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC [START] --> classify --> lookup --> respond --> [END]
# MAGIC ```
# MAGIC
# MAGIC | Node | What it does | DSPy module |
# MAGIC |------|-------------|-------------|
# MAGIC | **classify** | Determines if the question is about an order, a return, or a product | `ChainOfThought(ClassifyQuestion)` |
# MAGIC | **lookup** | Queries the appropriate Spark table in `sjdatabricks.orders` | Pure Python (Spark SQL) |
# MAGIC | **respond** | Formulates a natural-language answer from the raw data | `ChainOfThought(AnswerFromData)` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies
# MAGIC
# MAGIC We install:
# MAGIC - **mlflow>=3** -- for model logging, tracing, and evaluation
# MAGIC - **dspy** -- for structured LLM signatures and ChainOfThought reasoning
# MAGIC - **langgraph** -- for the StateGraph orchestration framework
# MAGIC - **langchain-core** -- required by LangGraph for message types
# MAGIC - **databricks-agents** -- helpers for Databricks agent deployment

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "mlflow>=3.0", "dspy", "langgraph", "langchain-core",
    "databricks-agents", "databricks-sdk", "pydantic>=2", "databricks-openai",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure DSPy LLM
# MAGIC
# MAGIC DSPy needs a language model to power its modules. We point it at the Databricks-hosted
# MAGIC **Meta-Llama 3.3 70B Instruct** endpoint. The `dspy.configure(lm=lm)` call sets this as the
# MAGIC global default so every `ChainOfThought` call automatically uses it.

# COMMAND ----------

import dspy

lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
dspy.configure(lm=lm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define DSPy Signatures
# MAGIC
# MAGIC A **DSPy Signature** is a typed contract describing what goes into and comes out of an LLM call.
# MAGIC Think of it like a function signature for a language model: it tells DSPy the *shape* of the
# MAGIC conversation so it can generate the right prompt automatically.
# MAGIC
# MAGIC ### ClassifyQuestion
# MAGIC
# MAGIC - **Input:** a natural-language `question` (string)
# MAGIC - **Output:** a `category` (string) that must be one of `"order"`, `"return"`, or `"product"`
# MAGIC - **Output:** `reasoning` -- the step-by-step thought process (provided by ChainOfThought)
# MAGIC
# MAGIC ### AnswerFromData
# MAGIC
# MAGIC - **Inputs:** the original `question`, the `category`, and the raw `data` fetched from Spark
# MAGIC - **Output:** a friendly `answer` string
# MAGIC
# MAGIC ### What is ChainOfThought?
# MAGIC
# MAGIC `dspy.ChainOfThought` wraps a Signature and adds an automatic "reasoning" step. Instead of
# MAGIC asking the LLM to jump straight to an answer, it first asks the LLM to *think step by step*.
# MAGIC This is the core idea behind Chain-of-Thought prompting (Wei et al., 2022) and it significantly
# MAGIC improves accuracy on tasks that require multi-step reasoning. DSPy handles the prompt engineering
# MAGIC for you -- you just define the Signature and DSPy adds the reasoning scaffolding.

# COMMAND ----------

class ClassifyQuestion(dspy.Signature):
    """Classify a customer support question into exactly one category.

    Rules:
    - "order" for questions about order status, shipping, delivery, order details
    - "return" for questions about returns, refunds, exchanges
    - "product" for questions about product info, pricing, availability, stock
    """
    question: str = dspy.InputField(desc="The customer's natural-language question")
    category: str = dspy.OutputField(desc="One of: order, return, product")


class AnswerFromData(dspy.Signature):
    """Given raw data from the database, compose a helpful customer support answer.

    Be concise, friendly, and include relevant details from the data.
    If the data is empty or says 'No records found', say so politely.
    """
    question: str = dspy.InputField(desc="The original customer question")
    category: str = dspy.InputField(desc="The category: order, return, or product")
    data: str = dspy.InputField(desc="Raw data rows fetched from the database")
    answer: str = dspy.OutputField(desc="A helpful natural-language answer for the customer")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick test: Classification
# MAGIC
# MAGIC Let us verify that the `ClassifyQuestion` signature correctly categorizes a sample question.
# MAGIC The `ChainOfThought` module will print its reasoning before the final answer.

# COMMAND ----------

classify_cot = dspy.ChainOfThought(ClassifyQuestion)
test_result = classify_cot(question="What is the status of order 1042?")
print(f"Category : {test_result.category}")
print(f"Reasoning: {test_result.reasoning}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define the Graph State
# MAGIC
# MAGIC LangGraph passes a **state** dictionary through every node. We use Python's `TypedDict` to
# MAGIC give it a clear schema. Each node reads from and writes to this shared state.
# MAGIC
# MAGIC | Field | Type | Set by |
# MAGIC |-------|------|--------|
# MAGIC | `question` | `str` | The user (initial input) |
# MAGIC | `category` | `str` | The `classify` node |
# MAGIC | `data` | `str` | The `lookup` node |
# MAGIC | `answer` | `str` | The `respond` node |
# MAGIC | `messages` | `list` | Accumulated for tracing / Responses API |
# MAGIC
# MAGIC The `messages` field keeps a log of what happened at each step, which is useful for debugging
# MAGIC and for constructing the final Responses API output.

# COMMAND ----------

from typing import TypedDict

class AgentState(TypedDict):
    question: str
    category: str
    data: str
    answer: str
    messages: list

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define Node Functions
# MAGIC
# MAGIC Each node is a plain Python function that takes the current state and returns updates to it.
# MAGIC LangGraph merges the returned dictionary into the existing state before passing it to the next node.
# MAGIC
# MAGIC ### Node 1: `classify_node`
# MAGIC Uses `ChainOfThought(ClassifyQuestion)` to determine the question category. The LLM reasons
# MAGIC about the question, then outputs one of `"order"`, `"return"`, or `"product"`. We normalize the
# MAGIC output to lowercase and validate it against the known categories.
# MAGIC
# MAGIC ### Node 2: `lookup_node`
# MAGIC Based on the category, this node runs a Spark SQL query against the appropriate table in
# MAGIC `sjdatabricks.orders`. It uses `spark.sql(...)` to fetch rows and converts them to a string
# MAGIC representation. This is a pure data-fetching step with no LLM call.
# MAGIC
# MAGIC ### Node 3: `respond_node`
# MAGIC Uses `ChainOfThought(AnswerFromData)` to turn the raw data into a friendly, human-readable
# MAGIC answer. It receives the original question, the category, and the data so the LLM has full
# MAGIC context for composing its response.

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Compile DSPy ChainOfThought modules
classify_module = dspy.ChainOfThought(ClassifyQuestion)
respond_module = dspy.ChainOfThought(AnswerFromData)

# Table mapping: category -> fully qualified Spark table name
TABLE_MAP = {
    "order": "sjdatabricks.orders.order_details",
    "return": "sjdatabricks.orders.returns",
    "product": "sjdatabricks.orders.products",
}


def classify_node(state: AgentState) -> dict:
    """Classify the customer question into order / return / product."""
    result = classify_module(question=state["question"])
    raw_category = result.category.strip().lower()
    # Normalize: ensure it is one of the valid categories
    valid = {"order", "return", "product"}
    category = raw_category if raw_category in valid else "order"
    messages = state.get("messages", []) + [
        {"step": "classify", "category": category, "reasoning": result.reasoning}
    ]
    return {"category": category, "messages": messages}


def lookup_node(state: AgentState) -> dict:
    """Query the Spark table that matches the classified category."""
    category = state["category"]
    table = TABLE_MAP.get(category, TABLE_MAP["order"])
    try:
        df = spark.sql(f"SELECT * FROM {table} LIMIT 20")
        rows = df.toPandas().to_string(index=False)
        if not rows.strip():
            rows = "No records found."
    except Exception as e:
        rows = f"Error querying {table}: {e}"
    messages = state.get("messages", []) + [
        {"step": "lookup", "table": table, "row_count": rows.count("\n")}
    ]
    return {"data": rows, "messages": messages}


def respond_node(state: AgentState) -> dict:
    """Compose a natural-language answer from the fetched data."""
    result = respond_module(
        question=state["question"],
        category=state["category"],
        data=state["data"],
    )
    messages = state.get("messages", []) + [
        {"step": "respond", "reasoning": result.reasoning}
    ]
    return {"answer": result.answer, "messages": messages}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Build the StateGraph
# MAGIC
# MAGIC ### What is a StateGraph?
# MAGIC
# MAGIC A `StateGraph` is LangGraph's core abstraction. You:
# MAGIC 1. **Create** it with a state schema (our `AgentState` TypedDict)
# MAGIC 2. **Add nodes** -- each node is a name + function pair
# MAGIC 3. **Add edges** -- define the order in which nodes execute
# MAGIC 4. **Set entry and finish points** -- tell LangGraph where to start and where to end
# MAGIC 5. **Compile** -- produces a runnable graph object
# MAGIC
# MAGIC ### How state flows between nodes
# MAGIC
# MAGIC When the graph runs:
# MAGIC 1. The initial state `{"question": "...", "category": "", "data": "", "answer": "", "messages": []}` enters the `classify` node.
# MAGIC 2. `classify` returns `{"category": "order", "messages": [...]}` -- LangGraph **merges** this into the state.
# MAGIC 3. The updated state (now with `category` filled in) flows to `lookup`.
# MAGIC 4. `lookup` returns `{"data": "...", "messages": [...]}` -- merged again.
# MAGIC 5. The state (now with `data` filled in) flows to `respond`.
# MAGIC 6. `respond` returns `{"answer": "...", "messages": [...]}` -- the final state has all fields populated.
# MAGIC
# MAGIC This is a simple **linear pipeline**: classify -> lookup -> respond. LangGraph also supports
# MAGIC conditional edges and cycles for more complex workflows, but a linear chain is perfect here.

# COMMAND ----------

from langgraph.graph import StateGraph

# Build the graph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("lookup", lookup_node)
graph_builder.add_node("respond", respond_node)

# Add edges: classify -> lookup -> respond
graph_builder.add_edge("classify", "lookup")
graph_builder.add_edge("lookup", "respond")

# Set entry and finish points
graph_builder.set_entry_point("classify")
graph_builder.set_finish_point("respond")

# Compile into a runnable
compiled_graph = graph_builder.compile()
print("Graph compiled successfully.")
print(f"Nodes: {list(compiled_graph.nodes.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick test: Run the graph locally
# MAGIC
# MAGIC Let us invoke the compiled graph with a sample question to verify the full pipeline works.

# COMMAND ----------

test_state = {
    "question": "What is the status of order 1042?",
    "category": "",
    "data": "",
    "answer": "",
    "messages": [],
}
result = compiled_graph.invoke(test_state)
print(f"Category: {result['category']}")
print(f"Answer  : {result['answer']}")
print(f"Steps   : {[m['step'] for m in result['messages']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Wrap in ResponsesAgent
# MAGIC
# MAGIC ### Why do we need a ResponsesAgent wrapper?
# MAGIC
# MAGIC Databricks Model Serving uses the **Responses API** format for input and output. The Responses API
# MAGIC expects requests like:
# MAGIC
# MAGIC ```json
# MAGIC {"input": [{"role": "user", "content": "What is the status of order 1042?"}]}
# MAGIC ```
# MAGIC
# MAGIC And returns responses with structured `output` items (text, reasoning, etc.).
# MAGIC
# MAGIC Our LangGraph operates on a plain `AgentState` dictionary, so we need a thin adapter layer that:
# MAGIC 1. **Converts** incoming Responses API messages into a plain `question` string
# MAGIC 2. **Runs** the LangGraph pipeline
# MAGIC 3. **Converts** the graph output back into Responses API format
# MAGIC
# MAGIC `mlflow.pyfunc.ResponsesAgent` is the base class that provides this interface. We override two methods:
# MAGIC - `predict()` -- for synchronous (non-streaming) calls
# MAGIC - `predict_stream()` -- for streaming calls that yield events as each node completes
# MAGIC
# MAGIC ### How streaming works
# MAGIC
# MAGIC For streaming, we use LangGraph's `.stream()` method which yields state updates after each node
# MAGIC completes. We convert each update into a `ResponsesAgentStreamEvent` so the caller gets incremental
# MAGIC progress (e.g., "classification done", "data fetched", "answer generated").

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write `agent.py` (Models-from-Code)
# MAGIC
# MAGIC MLflow's **models-from-code** approach logs a Python file as the model artifact. When the model
# MAGIC is loaded later (for serving or evaluation), MLflow executes this file and looks for the object
# MAGIC set via `set_model()`. This keeps the model self-contained and reproducible.

# COMMAND ----------

from pathlib import Path

_agent_code = '''\
"""
Topic 2: DSPy + LangGraph Customer Support Agent.

Architecture: classify -> lookup -> respond
- classify: DSPy ChainOfThought determines question category (order/return/product)
- lookup: Queries the appropriate Spark table in sjdatabricks.orders
- respond: DSPy ChainOfThought composes a natural-language answer from the data

Wrapped in MLflow ResponsesAgent for Databricks Model Serving compatibility.
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
from langgraph.graph import StateGraph
from pyspark.sql import SparkSession
from typing import TypedDict


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    category: str
    data: str
    answer: str
    messages: list


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------
class ClassifyQuestion(dspy.Signature):
    """Classify a customer support question into exactly one category.
    Rules:
    - "order" for questions about order status, shipping, delivery, order details
    - "return" for questions about returns, refunds, exchanges
    - "product" for questions about product info, pricing, availability, stock
    """
    question: str = dspy.InputField(desc="The customer question")
    category: str = dspy.OutputField(desc="One of: order, return, product")


class AnswerFromData(dspy.Signature):
    """Given raw data from the database, compose a helpful customer support answer.
    Be concise, friendly, and include relevant details from the data.
    If the data is empty or says No records found, say so politely.
    """
    question: str = dspy.InputField(desc="The original customer question")
    category: str = dspy.InputField(desc="The category: order, return, or product")
    data: str = dspy.InputField(desc="Raw data rows fetched from the database")
    answer: str = dspy.OutputField(desc="A helpful natural-language answer")


# ---------------------------------------------------------------------------
# Table mapping
# ---------------------------------------------------------------------------
TABLE_MAP = {
    "order": "sjdatabricks.orders.order_details",
    "return": "sjdatabricks.orders.returns",
    "product": "sjdatabricks.orders.products",
}


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def _make_nodes():
    """Create node functions with their own DSPy module instances."""
    lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
    dspy.configure(lm=lm)

    classify_mod = dspy.ChainOfThought(ClassifyQuestion)
    respond_mod = dspy.ChainOfThought(AnswerFromData)
    spark = SparkSession.builder.getOrCreate()

    def classify_node(state: AgentState) -> dict:
        result = classify_mod(question=state["question"])
        raw = result.category.strip().lower()
        category = raw if raw in {"order", "return", "product"} else "order"
        msgs = state.get("messages", []) + [
            {"step": "classify", "category": category, "reasoning": result.reasoning}
        ]
        return {"category": category, "messages": msgs}

    def lookup_node(state: AgentState) -> dict:
        table = TABLE_MAP.get(state["category"], TABLE_MAP["order"])
        try:
            df = spark.sql(f"SELECT * FROM {table} LIMIT 20")
            rows = df.toPandas().to_string(index=False)
            if not rows.strip():
                rows = "No records found."
        except Exception as e:
            rows = f"Error querying {table}: {e}"
        msgs = state.get("messages", []) + [
            {"step": "lookup", "table": table}
        ]
        return {"data": rows, "messages": msgs}

    def respond_node(state: AgentState) -> dict:
        result = respond_mod(
            question=state["question"],
            category=state["category"],
            data=state["data"],
        )
        msgs = state.get("messages", []) + [
            {"step": "respond", "reasoning": result.reasoning}
        ]
        return {"answer": result.answer, "messages": msgs}

    return classify_node, lookup_node, respond_node


def _build_graph():
    """Build and compile the LangGraph StateGraph."""
    classify_fn, lookup_fn, respond_fn = _make_nodes()
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_fn)
    builder.add_node("lookup", lookup_fn)
    builder.add_node("respond", respond_fn)
    builder.add_edge("classify", "lookup")
    builder.add_edge("lookup", "respond")
    builder.set_entry_point("classify")
    builder.set_finish_point("respond")
    return builder.compile()


# ---------------------------------------------------------------------------
# ResponsesAgent wrapper
# ---------------------------------------------------------------------------
class DSPyLangGraphAgent(ResponsesAgent):
    """Wraps the classify->lookup->respond LangGraph in an MLflow ResponsesAgent."""

    def __init__(self):
        super().__init__()
        self.graph = _build_graph()

    def _extract_question(self, request: ResponsesAgentRequest) -> str:
        """Pull the user question text out of a Responses API request."""
        input_list = getattr(request, "input", None)
        if input_list is None and isinstance(request, dict):
            input_list = request.get("input", [])
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

    def _initial_state(self, question: str) -> AgentState:
        return {
            "question": question,
            "category": "",
            "data": "",
            "answer": "",
            "messages": [],
        }

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        question = self._extract_question(request)
        final_state = self.graph.invoke(self._initial_state(question))

        output_items = []
        # Include reasoning from classify and respond steps
        reasoning_parts = []
        for msg in final_state.get("messages", []):
            if "reasoning" in msg:
                reasoning_parts.append(f"[{msg['step']}] {msg['reasoning']}")
        if reasoning_parts:
            output_items.append(
                self.create_reasoning_item(
                    id="reason_1",
                    reasoning_text="\\n\\n".join(reasoning_parts),
                )
            )
        output_items.append(
            self.create_text_output_item(text=final_state["answer"], id="msg_1")
        )
        return ResponsesAgentResponse(output=output_items)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        question = self._extract_question(request)
        state = self._initial_state(question)

        # Stream node-by-node using LangGraph .stream()
        final_answer = ""
        reasoning_parts = []

        for update in self.graph.stream(state):
            # update is a dict {node_name: {state_updates}}
            for node_name, node_output in update.items():
                # Collect reasoning from messages
                for msg in node_output.get("messages", []):
                    if "reasoning" in msg:
                        reasoning_parts.append(f"[{msg['step']}] {msg['reasoning']}")
                if "answer" in node_output and node_output["answer"]:
                    final_answer = node_output["answer"]

        # Yield reasoning first
        if reasoning_parts:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(
                    id="reason_1",
                    reasoning_text="\\n\\n".join(reasoning_parts),
                ),
            )

        # Stream the answer in small chunks
        item_id = "msg_1"
        chunk_size = 5
        for i in range(0, len(final_answer), chunk_size):
            delta = final_answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=item_id)
            )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=final_answer, id=item_id),
        )


mlflow.dspy.autolog()
agent = DSPyLangGraphAgent()
set_model(agent)
'''

Path("agent.py").write_text(_agent_code)
print("agent.py written successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the `agent.py` file
# MAGIC
# MAGIC The file above is a self-contained Python module that MLflow will execute when loading the model.
# MAGIC Here is what each section does:
# MAGIC
# MAGIC 1. **State schema (`AgentState`)** -- the TypedDict that flows through the graph.
# MAGIC 2. **DSPy Signatures** -- `ClassifyQuestion` and `AnswerFromData` define the LLM contracts.
# MAGIC 3. **`_make_nodes()`** -- factory function that initializes DSPy, creates ChainOfThought modules,
# MAGIC    and returns three node functions (`classify_node`, `lookup_node`, `respond_node`).
# MAGIC 4. **`_build_graph()`** -- constructs the StateGraph, wires nodes with edges, and compiles.
# MAGIC 5. **`DSPyLangGraphAgent`** -- the ResponsesAgent subclass that bridges Responses API <-> LangGraph:
# MAGIC    - `_extract_question()` pulls the user's text from the Responses API input format
# MAGIC    - `predict()` runs the full graph synchronously and returns a ResponsesAgentResponse
# MAGIC    - `predict_stream()` streams events as nodes complete, yielding reasoning and answer chunks
# MAGIC 6. **`set_model(agent)`** -- tells MLflow which object to use when the model is loaded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test the Agent Locally (Non-Streaming)
# MAGIC
# MAGIC Before logging to MLflow, we test the agent by loading it from the file and calling `predict()`.

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()

# Load the agent from the code file (same as MLflow will do during serving)
loaded_agent = mlflow.pyfunc.load_model("agent.py")

# Test with a sample request in Responses API format
test_request = {
    "input": [{"role": "user", "content": "What is the status of order 1042?"}],
}
response = loaded_agent.predict(test_request)
print("=== Non-Streaming Response ===")
for item in response.output:
    print(item)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Test Streaming
# MAGIC
# MAGIC Streaming lets a frontend display incremental progress. Our `predict_stream()` yields:
# MAGIC 1. A reasoning item (once all nodes have run) showing the classify and respond reasoning
# MAGIC 2. Text delta events that stream the final answer character by character
# MAGIC 3. A final `output_item.done` event with the complete answer

# COMMAND ----------

test_request_stream = {
    "input": [{"role": "user", "content": "Show me all available products"}],
}

print("=== Streaming Response ===")
for event in loaded_agent.predict_stream(test_request_stream):
    if hasattr(event, "type") and event.type == "response.output_item.done":
        item = event.item
        if hasattr(item, "summary") and item.summary:
            print(f"\n[Reasoning] {item.summary[0].text[:200]}...")
        elif hasattr(item, "content"):
            for c in item.content:
                if hasattr(c, "text"):
                    print(f"\n[Final Answer] {c.text}")
    elif hasattr(event, "delta"):
        print(event.delta, end="", flush=True)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Log Model with MLflow (Code-Based Logging)
# MAGIC
# MAGIC We use MLflow's **models-from-code** approach: instead of serializing the Python objects, we
# MAGIC log the `agent.py` file itself. When the model is loaded later, MLflow executes the file and
# MAGIC uses the object registered via `set_model()`.
# MAGIC
# MAGIC This approach is ideal for agents because:
# MAGIC - The code is fully transparent and auditable
# MAGIC - Dependencies (DSPy, LangGraph) are captured in the pip requirements
# MAGIC - The model can be loaded in any environment that has the right packages

# COMMAND ----------

with mlflow.start_run(run_name="dspy_langgraph_classify_lookup_respond") as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        name="agent",
    )
    run_id = run.info.run_id
    print(f"Logged agent. Run ID: {run_id}")
    print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC MLflow GenAI evaluation uses an LLM to automatically score agent responses on dimensions like
# MAGIC **safety** (is the response harmful?) and **relevance** (does it actually answer the question?).
# MAGIC
# MAGIC We create a small evaluation dataset with representative questions from each category (order,
# MAGIC return, product) and run the evaluation. The scores help us decide if the agent is ready for
# MAGIC production deployment.

# COMMAND ----------

eval_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order 1042?"}]},
    {"input": [{"role": "user", "content": "Show me recent returns"}]},
    {"input": [{"role": "user", "content": "What products do you have in Electronics?"}]},
    {"input": [{"role": "user", "content": "I want to return order 1045, it was defective"}]},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: eval_agent.predict(row),
    scorers=[
        mlflow.genai.scorers.Safety(),
        mlflow.genai.scorers.RelevanceToQuery(),
    ],
)

print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Register Model to Unity Catalog
# MAGIC
# MAGIC Registering the model in Unity Catalog makes it available for deployment to Databricks Model
# MAGIC Serving. Once registered, you can create a serving endpoint from the Databricks UI or SDK.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    f"runs:/{run_id}/agent",
    "sjdatabricks.agents.dspy_langgraph_order_support",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

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

ENDPOINT_NAME = "dspy-langgraph-order-support"
UC_MODEL_NAME = "sjdatabricks.agents.dspy_langgraph_order_support"

latest_version = model_info.version

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
# MAGIC In this notebook, we built a complete **DSPy + LangGraph Order Support Agent** and deployed it end-to-end:
# MAGIC
# MAGIC | Step | What we did | Why it matters |
# MAGIC |------|------------|----------------|
# MAGIC | 1 | Installed dependencies | MLflow 3 + DSPy + LangGraph |
# MAGIC | 2 | Configured DSPy LLM | Points to Databricks Foundation Model API |
# MAGIC | 3 | Defined tools | Spark SQL queries for orders, returns, products |
# MAGIC | 4 | Built LangGraph workflow | Classifier node routes to specialized handlers |
# MAGIC | 5 | Tested locally | Verified correctness before packaging |
# MAGIC | 6 | Wrapped in ResponsesAgent | Standard API format for Model Serving |
# MAGIC | 7-8 | Tested wrapper + streaming | Verified end-to-end compatibility |
# MAGIC | 9 | Logged with MLflow | Code-based logging for reproducibility |
# MAGIC | 10 | Evaluated with LLM-as-a-judge | Automated quality scoring |
# MAGIC | 11 | Registered to Unity Catalog | Governance + deployment readiness |
# MAGIC | 12 | Deployed to Model Serving | Real-time REST API endpoint |
# MAGIC | 13 | Queried non-streaming | Full response in one HTTP call |
# MAGIC | 14 | Queried streaming | Real-time token-by-token output |
# MAGIC
# MAGIC The agent is now **live** and can be called from any application using the OpenAI-compatible API.
