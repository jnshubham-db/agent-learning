# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 5: Customer Support Agent with LangChain + LangGraph (Multi-Node CoT)
# MAGIC
# MAGIC This notebook builds a **Customer Support Agent** using **LangGraph** to orchestrate
# MAGIC a multi-node pipeline where each node is powered by **LangChain ChatDatabricks**.
# MAGIC
# MAGIC ## What You Will Learn
# MAGIC
# MAGIC | Concept | Description |
# MAGIC |---------|-------------|
# MAGIC | **LangGraph** | A library for building stateful, multi-step agent workflows as directed graphs |
# MAGIC | **StateGraph** | LangGraph's core abstraction — nodes are functions, edges define flow, state carries data between nodes |
# MAGIC | **Chain-of-Thought (CoT)** | Each node uses the LLM to reason step-by-step before producing output |
# MAGIC | **Multi-node routing** | A classifier node routes to different lookup nodes based on the question category |
# MAGIC | **ChatDatabricks** | LangChain's wrapper for Databricks model endpoints |
# MAGIC
# MAGIC ## Why LangGraph Instead of a Simple ReAct Agent?
# MAGIC
# MAGIC In Topic 4, we used `create_react_agent` which gives the LLM full control over tool selection.
# MAGIC That works well, but sometimes you want **explicit control** over the workflow:
# MAGIC
# MAGIC - **Predictable routing** — you decide which code path handles which question type
# MAGIC - **Separation of concerns** — each node has a single responsibility
# MAGIC - **Easier debugging** — you can inspect state between nodes
# MAGIC - **Custom logic** — nodes can contain arbitrary Python, not just LLM calls
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC User question
# MAGIC       |
# MAGIC       v
# MAGIC  [classify] -- ChatDatabricks classifies into: order / return / product
# MAGIC       |
# MAGIC       +--- "order"   --> [lookup] queries sjdatabricks.orders.order_details
# MAGIC       +--- "return"  --> [lookup] queries sjdatabricks.orders.returns
# MAGIC       +--- "product" --> [lookup] queries sjdatabricks.orders.products
# MAGIC       |
# MAGIC       v
# MAGIC  [respond] -- ChatDatabricks formulates a friendly answer from the data
# MAGIC       |
# MAGIC       v
# MAGIC  Final answer streamed via ResponsesAgent
# MAGIC ```
# MAGIC
# MAGIC **Three nodes, one graph, clear data flow.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies
# MAGIC
# MAGIC | Package | Purpose |
# MAGIC |---------|---------|
# MAGIC | `mlflow>=3` | Agent logging, tracing, evaluation, ResponsesAgent |
# MAGIC | `langchain` | Core framework for LLM application building |
# MAGIC | `langchain-core` | Base message types and prompt templates |
# MAGIC | `langchain-databricks` | `ChatDatabricks` — connects LangChain to Databricks endpoints |
# MAGIC | `langgraph` | `StateGraph` for building the multi-node pipeline |
# MAGIC | `databricks-agents` | Databricks agent deployment helpers |

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
# MAGIC ## 2. Configure ChatDatabricks
# MAGIC
# MAGIC We create a single `ChatDatabricks` instance that all three nodes will share. This avoids
# MAGIC creating multiple connections to the same endpoint.
# MAGIC
# MAGIC **`ChatDatabricks`** wraps a Databricks Foundation Model API endpoint and exposes it
# MAGIC as a LangChain `BaseChatModel`. You can call it with:
# MAGIC - `llm.invoke([messages])` — returns a single `AIMessage`
# MAGIC - `llm.stream([messages])` — yields message chunks for streaming
# MAGIC
# MAGIC We use `temperature=0.0` because we want deterministic classification and consistent answers.

# COMMAND ----------

from langchain_databricks import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.0,
)

# Quick smoke test
test = llm.invoke("Say hello in one word.")
print(f"LLM ready: {test.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define the Graph State
# MAGIC
# MAGIC ### What is "State" in LangGraph?
# MAGIC
# MAGIC LangGraph passes a **state dictionary** between nodes. Each node reads from the state,
# MAGIC does some work, and returns updates to merge back into the state. Think of it as a
# MAGIC shared blackboard that each node writes to.
# MAGIC
# MAGIC We define the state as a `TypedDict` so we get type checking and clear documentation
# MAGIC of what data flows through the graph.
# MAGIC
# MAGIC ### Our State Fields
# MAGIC
# MAGIC | Field | Type | Purpose |
# MAGIC |-------|------|---------|
# MAGIC | `question` | `str` | The user's original question |
# MAGIC | `category` | `str` | Classification result: `"order"`, `"return"`, or `"product"` |
# MAGIC | `data` | `str` | Raw data retrieved from the Spark table |
# MAGIC | `answer` | `str` | The final human-friendly answer |
# MAGIC | `messages` | `list` | The original input messages (for ResponsesAgent compatibility) |

# COMMAND ----------

from typing import TypedDict


class AgentState(TypedDict):
    """State that flows through the classify -> lookup -> respond pipeline."""
    question: str
    category: str
    data: str
    answer: str
    messages: list

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Node Functions
# MAGIC
# MAGIC Each node is a plain Python function that takes the current state and returns a dict
# MAGIC of state updates. LangGraph merges these updates into the state automatically.
# MAGIC
# MAGIC ### Node 1: Classify
# MAGIC
# MAGIC The **classify** node asks the LLM to categorize the user's question into one of three
# MAGIC categories: `order`, `return`, or `product`. This determines which database table to query.
# MAGIC
# MAGIC We use a structured prompt that forces the LLM to output exactly one word. This is a
# MAGIC simple but effective approach — for production you might use structured output parsing.

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage


def classify_node(state: AgentState) -> dict:
    """Classify the user's question into: order, return, or product.

    This node uses ChatDatabricks with a strict system prompt that constrains
    the output to a single category word. The LLM analyzes the question and
    determines which domain it belongs to.
    """
    question = state["question"]

    classification_prompt = [
        SystemMessage(content=(
            "You are a question classifier for a customer support system. "
            "Classify the following question into exactly one category. "
            "Reply with ONLY one word — no explanation, no punctuation:\n\n"
            "- 'order' — if the question is about order status, shipping, delivery, or tracking\n"
            "- 'return' — if the question is about returns, refunds, or exchanges\n"
            "- 'product' — if the question is about product details, pricing, stock, or availability\n\n"
            "If unsure, default to 'order'."
        )),
        HumanMessage(content=question),
    ]

    response = llm.invoke(classification_prompt)
    raw_category = response.content.strip().lower()

    # Normalize — ensure we get a valid category
    if "return" in raw_category or "refund" in raw_category:
        category = "return"
    elif "product" in raw_category or "price" in raw_category or "stock" in raw_category:
        category = "product"
    else:
        category = "order"

    return {"category": category}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Node 2: Lookup
# MAGIC
# MAGIC The **lookup** node queries the appropriate Spark SQL table based on the classification.
# MAGIC It does NOT use the LLM — it runs pure SQL against the `sjdatabricks.orders` schema.
# MAGIC
# MAGIC This is one of the advantages of the multi-node approach: the lookup node is deterministic
# MAGIC and fast. No LLM tokens are spent on data retrieval.
# MAGIC
# MAGIC The node checks the `category` field in the state and queries the right table:
# MAGIC
# MAGIC | Category | Table | Query Strategy |
# MAGIC |----------|-------|---------------|
# MAGIC | `order` | `order_details` | Tries to extract an order ID; falls back to showing recent orders |
# MAGIC | `return` | `returns` | Tries to extract an order ID; falls back to showing recent returns |
# MAGIC | `product` | `products` | Tries to extract a product name; falls back to listing all products |

# COMMAND ----------

import re


def lookup_node(state: AgentState) -> dict:
    """Query the appropriate Spark table based on the classified category.

    This node performs deterministic SQL lookups — no LLM calls. It extracts
    relevant identifiers from the question using simple regex patterns and
    queries the corresponding table.
    """
    question = state["question"]
    category = state["category"]

    if category == "order":
        # Try to extract an order ID (integer) from the question
        match = re.search(r"\b(\d{4,})\b", question)
        if match:
            order_id = match.group(1)
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
            ).collect()
            if rows:
                row = rows[0]
                data = (
                    f"Order {row['order_id']}: product={row['product']}, "
                    f"quantity={row['quantity']}, status={row['status']}, "
                    f"customer={row['customer_name']}, date={row['order_date']}"
                )
            else:
                data = f"No order found with ID {order_id}."
        else:
            # No specific order ID — show recent orders
            rows = spark.sql(
                "SELECT * FROM sjdatabricks.orders.order_details ORDER BY order_date DESC LIMIT 5"
            ).collect()
            data = "Recent orders:\n" + "\n".join(
                f"  Order {r['order_id']}: {r['product']} ({r['status']})" for r in rows
            )

    elif category == "return":
        # Try to extract an order ID for return lookup
        match = re.search(r"\b(\d{4,})\b", question)
        if match:
            order_id = match.group(1)
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
            ).collect()
            if rows:
                data = "\n".join(
                    f"Return {r['return_id']}: order={r['order_id']}, reason={r['reason']}, "
                    f"status={r['status']}, date={r['return_date']}"
                    for r in rows
                )
            else:
                data = f"No returns found for order {order_id}."
        else:
            # No specific ID — show recent returns
            rows = spark.sql(
                "SELECT * FROM sjdatabricks.orders.returns ORDER BY return_date DESC LIMIT 5"
            ).collect()
            data = "Recent returns:\n" + "\n".join(
                f"  Return {r['return_id']}: order={r['order_id']}, reason={r['reason']} ({r['status']})"
                for r in rows
            )

    elif category == "product":
        # Try to extract a product name from the question
        known_products = ["laptop", "phone", "tablet", "monitor", "keyboard", "mouse"]
        found_product = None
        for prod in known_products:
            if prod in question.lower():
                found_product = prod
                break

        if found_product:
            rows = spark.sql(
                f"SELECT * FROM sjdatabricks.orders.products WHERE LOWER(name) = '{found_product}'"
            ).collect()
            if rows:
                row = rows[0]
                data = (
                    f"Product: {row['name']} (ID={row['product_id']}), "
                    f"category={row['category']}, price=${row['price']}, stock={row['stock']}"
                )
            else:
                data = f"No product found matching '{found_product}'."
        else:
            # No specific product — list all
            rows = spark.sql(
                "SELECT * FROM sjdatabricks.orders.products ORDER BY name"
            ).collect()
            data = "Available products:\n" + "\n".join(
                f"  {r['name']}: ${r['price']} (stock: {r['stock']})" for r in rows
            )
    else:
        data = "Unable to determine the appropriate data source."

    return {"data": data}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Node 3: Respond
# MAGIC
# MAGIC The **respond** node takes the raw data from the lookup and uses ChatDatabricks to
# MAGIC formulate a friendly, human-readable answer. This is where the LLM adds value — turning
# MAGIC raw database output into a conversational response.
# MAGIC
# MAGIC The system prompt instructs the LLM to:
# MAGIC - Be concise and helpful
# MAGIC - Only use the provided data (no hallucination)
# MAGIC - Acknowledge when data is missing

# COMMAND ----------


def respond_node(state: AgentState) -> dict:
    """Formulate a friendly customer support answer from the retrieved data.

    This node sends the original question and the retrieved data to ChatDatabricks,
    which generates a natural language response suitable for a customer.
    """
    question = state["question"]
    data = state["data"]

    response_prompt = [
        SystemMessage(content=(
            "You are a friendly customer support agent. The user asked a question and "
            "we retrieved the following data from our database. Use ONLY this data to "
            "answer the question. Be concise, helpful, and accurate. If the data shows "
            "the information was not found, acknowledge that and suggest next steps.\n\n"
            f"Retrieved data:\n{data}"
        )),
        HumanMessage(content=question),
    ]

    response = llm.invoke(response_prompt)
    return {"answer": response.content}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build the StateGraph
# MAGIC
# MAGIC ### How StateGraph Works
# MAGIC
# MAGIC A `StateGraph` is defined by:
# MAGIC
# MAGIC 1. **State schema** — the `TypedDict` that defines what data flows through the graph
# MAGIC 2. **Nodes** — Python functions that process the state
# MAGIC 3. **Edges** — connections between nodes (can be unconditional or conditional)
# MAGIC 4. **Entry point** — where the graph starts (`START`)
# MAGIC 5. **Exit point** — where the graph ends (`END`)
# MAGIC
# MAGIC ### Our Graph Structure
# MAGIC
# MAGIC ```
# MAGIC START --> classify --> lookup --> respond --> END
# MAGIC ```
# MAGIC
# MAGIC This is a simple **linear pipeline** — each step feeds into the next. The "routing" happens
# MAGIC inside the `lookup` node, which checks the `category` field to decide which table to query.
# MAGIC
# MAGIC > **Why not conditional edges?** We *could* have three separate lookup nodes connected by
# MAGIC > conditional edges from `classify`. But since the routing logic is simple (just checking a
# MAGIC > string), putting it inside one `lookup` function is cleaner. Use conditional edges when
# MAGIC > you have fundamentally different processing pipelines.

# COMMAND ----------

from langgraph.graph import StateGraph, START, END

# Create the graph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("lookup", lookup_node)
graph_builder.add_node("respond", respond_node)

# Add edges: START -> classify -> lookup -> respond -> END
graph_builder.add_edge(START, "classify")
graph_builder.add_edge("classify", "lookup")
graph_builder.add_edge("lookup", "respond")
graph_builder.add_edge("respond", END)

# Compile the graph — this validates the structure and makes it callable
graph = graph_builder.compile()

print("Graph compiled successfully!")
print("Nodes: classify -> lookup -> respond")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Raw Graph
# MAGIC
# MAGIC Before wrapping in ResponsesAgent, let us test the graph directly. The `invoke` method
# MAGIC takes the initial state and returns the final state after all nodes have run.

# COMMAND ----------

# Test the graph with a sample question
test_state = {
    "question": "What is the status of order 1045?",
    "category": "",
    "data": "",
    "answer": "",
    "messages": [],
}

result = graph.invoke(test_state)
print(f"Category: {result['category']}")
print(f"Data:     {result['data']}")
print(f"Answer:   {result['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect State Flow
# MAGIC
# MAGIC You can use `graph.stream()` with `stream_mode="updates"` to see what each node
# MAGIC contributes to the state. This is invaluable for debugging.

# COMMAND ----------

test_state = {
    "question": "How much does a Laptop cost?",
    "category": "",
    "data": "",
    "answer": "",
    "messages": [],
}

print("Tracing state through each node:\n")
for event in graph.stream(test_state, stream_mode="updates"):
    for node_name, node_output in event.items():
        print(f"[{node_name}] returned: {node_output}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wrap in MLflow ResponsesAgent
# MAGIC
# MAGIC Just like in Topic 4, we need to wrap our graph in `ResponsesAgent` so it can be
# MAGIC deployed to Databricks Model Serving. The key differences from Topic 4:
# MAGIC
# MAGIC - Instead of calling a ReAct agent, we invoke our custom StateGraph
# MAGIC - We extract the `answer` field from the final state (not from LangChain messages)
# MAGIC - The graph itself handles all the routing and data retrieval

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


class LangGraphCustomerSupportAgent(ResponsesAgent):
    """Customer Support Agent using a LangGraph StateGraph (classify -> lookup -> respond)."""

    def __init__(self):
        super().__init__()
        self._build_graph()

    def _build_graph(self):
        """Build the LLM, node functions, and StateGraph."""
        from langchain_databricks import ChatDatabricks
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import StateGraph, START, END
        import re

        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0.0,
        )

        _llm = self.llm  # local reference for closures

        def classify_node(state: dict) -> dict:
            question = state["question"]
            messages = [
                SystemMessage(content=(
                    "Classify the question into exactly one category. "
                    "Reply with ONLY one word:\n"
                    "- 'order' — order status, shipping, delivery\n"
                    "- 'return' — returns, refunds, exchanges\n"
                    "- 'product' — product details, pricing, stock\n"
                    "Default to 'order' if unsure."
                )),
                HumanMessage(content=question),
            ]
            response = _llm.invoke(messages)
            raw = response.content.strip().lower()
            if "return" in raw or "refund" in raw:
                category = "return"
            elif "product" in raw or "price" in raw or "stock" in raw:
                category = "product"
            else:
                category = "order"
            return {"category": category}

        def lookup_node(state: dict) -> dict:
            question = state["question"]
            category = state["category"]

            if category == "order":
                match = re.search(r"\b(\d{4,})\b", question)
                if match:
                    order_id = match.group(1)
                    rows = spark.sql(
                        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
                    ).collect()
                    if rows:
                        r = rows[0]
                        data = (
                            f"Order {r['order_id']}: product={r['product']}, "
                            f"quantity={r['quantity']}, status={r['status']}, "
                            f"customer={r['customer_name']}, date={r['order_date']}"
                        )
                    else:
                        data = f"No order found with ID {order_id}."
                else:
                    rows = spark.sql(
                        "SELECT * FROM sjdatabricks.orders.order_details ORDER BY order_date DESC LIMIT 5"
                    ).collect()
                    data = "Recent orders:\n" + "\n".join(
                        f"  Order {r['order_id']}: {r['product']} ({r['status']})" for r in rows
                    )

            elif category == "return":
                match = re.search(r"\b(\d{4,})\b", question)
                if match:
                    order_id = match.group(1)
                    rows = spark.sql(
                        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
                    ).collect()
                    if rows:
                        data = "\n".join(
                            f"Return {r['return_id']}: order={r['order_id']}, reason={r['reason']}, "
                            f"status={r['status']}, date={r['return_date']}"
                            for r in rows
                        )
                    else:
                        data = f"No returns found for order {order_id}."
                else:
                    rows = spark.sql(
                        "SELECT * FROM sjdatabricks.orders.returns ORDER BY return_date DESC LIMIT 5"
                    ).collect()
                    data = "Recent returns:\n" + "\n".join(
                        f"  Return {r['return_id']}: order={r['order_id']}, reason={r['reason']} ({r['status']})"
                        for r in rows
                    )

            elif category == "product":
                known_products = ["laptop", "phone", "tablet", "monitor", "keyboard", "mouse"]
                found = None
                for prod in known_products:
                    if prod in question.lower():
                        found = prod
                        break
                if found:
                    rows = spark.sql(
                        f"SELECT * FROM sjdatabricks.orders.products WHERE LOWER(name) = '{found}'"
                    ).collect()
                    if rows:
                        r = rows[0]
                        data = (
                            f"Product: {r['name']} (ID={r['product_id']}), "
                            f"category={r['category']}, price=${r['price']}, stock={r['stock']}"
                        )
                    else:
                        data = f"No product found matching '{found}'."
                else:
                    rows = spark.sql(
                        "SELECT * FROM sjdatabricks.orders.products ORDER BY name"
                    ).collect()
                    data = "Available products:\n" + "\n".join(
                        f"  {r['name']}: ${r['price']} (stock: {r['stock']})" for r in rows
                    )
            else:
                data = "Unable to determine the appropriate data source."

            return {"data": data}

        def respond_node(state: dict) -> dict:
            question = state["question"]
            data = state["data"]
            messages = [
                SystemMessage(content=(
                    "You are a friendly customer support agent. Use ONLY the retrieved data "
                    "to answer the question. Be concise and helpful.\n\n"
                    f"Retrieved data:\n{data}"
                )),
                HumanMessage(content=question),
            ]
            response = _llm.invoke(messages)
            return {"answer": response.content}

        builder = StateGraph(AgentState)
        builder.add_node("classify", classify_node)
        builder.add_node("lookup", lookup_node)
        builder.add_node("respond", respond_node)
        builder.add_edge(START, "classify")
        builder.add_edge("classify", "lookup")
        builder.add_edge("lookup", "respond")
        builder.add_edge("respond", END)
        self.graph = builder.compile()

    def _extract_question(self, request: ResponsesAgentRequest) -> str:
        """Extract the user's question text from the request."""
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
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
        """Synchronous prediction: run the full graph and return the answer."""
        question = self._extract_question(request)

        initial_state = {
            "question": question,
            "category": "",
            "data": "",
            "answer": "",
            "messages": [],
        }

        final_state = self.graph.invoke(initial_state)
        answer = final_state.get("answer", "I could not process your request.")

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
        """Streaming prediction: run the graph, then stream the answer as text deltas."""
        question = self._extract_question(request)

        initial_state = {
            "question": question,
            "category": "",
            "data": "",
            "answer": "",
            "messages": [],
        }

        # Run the graph and collect intermediate state for reasoning
        reasoning_parts = []
        final_state = {}
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                if node_name == "classify":
                    reasoning_parts.append(f"Classified as: {node_output.get('category', '?')}")
                elif node_name == "lookup":
                    reasoning_parts.append(f"Retrieved data: {node_output.get('data', '')[:200]}")

        # Yield reasoning item (shows the intermediate classify + lookup steps)
        reasoning_text = "\n".join(reasoning_parts)
        if reasoning_text:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(
                    id=str(uuid4()),
                    reasoning_text=reasoning_text,
                ),
            )

        # Stream the final answer as text deltas
        answer = final_state.get("answer", "I could not process your request.")
        text_id = str(uuid4())
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=chunk, item_id=text_id)
            )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=text_id),
        )


# Instantiate the agent
agent = LangGraphCustomerSupportAgent()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the Streaming Flow
# MAGIC
# MAGIC When `predict_stream` is called:
# MAGIC
# MAGIC 1. **Graph execution** — The full `classify -> lookup -> respond` pipeline runs. As each
# MAGIC    node completes, we capture its output for the reasoning trace.
# MAGIC
# MAGIC 2. **Reasoning event** — We yield a `reasoning_item` that shows the intermediate steps
# MAGIC    (what category was detected, what data was retrieved). This appears as a collapsible
# MAGIC    "thinking" section in UIs that support it.
# MAGIC
# MAGIC 3. **Text deltas** — The final answer is broken into 20-character chunks and yielded as
# MAGIC    `response.output_text.delta` events. The client renders these in real time.
# MAGIC
# MAGIC 4. **Done event** — A final `response.output_item.done` event with the complete text.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Locally
# MAGIC
# MAGIC Let us verify the wrapped agent with a variety of question types.

# COMMAND ----------

from mlflow.types.responses import ResponsesAgentRequest

test_questions = [
    "What is the status of order 1045?",
    "Has there been a return for order 1042?",
    "How much does a Laptop cost?",
    "What products do you have available?",
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
# MAGIC ## 8. Test Streaming
# MAGIC
# MAGIC Verify that streaming works — you should see the reasoning trace followed by
# MAGIC the answer appearing incrementally.

# COMMAND ----------

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is the status of order 1050?"}]
)

print("Streaming response:\n")
for event in agent.predict_stream(request):
    if event.type == "response.output_item.done" and hasattr(event.item, "summary"):
        # This is the reasoning item
        for s in event.item.summary:
            if hasattr(s, "text"):
                print(f"[Reasoning] {s.text}")
        print()
    elif event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.output_item.done":
        print()
print("\nStreaming complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Log the Agent with MLflow
# MAGIC
# MAGIC We enable LangChain autologging for traces and log the agent object.

# COMMAND ----------

import mlflow

mlflow.langchain.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="langgraph_classify_lookup_respond_agent") as run:
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
# MAGIC ## 10. Evaluate with LLM-as-a-Judge
# MAGIC
# MAGIC We evaluate the agent on a set of test questions using MLflow's built-in scorers.
# MAGIC The judge LLM rates each response for safety and relevance.

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery

loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

eval_dataset = [
    {"input": [{"role": "user", "content": "What is the status of order 1045?"}]},
    {"input": [{"role": "user", "content": "Has order 1042 been returned?"}]},
    {"input": [{"role": "user", "content": "How much does a Tablet cost and is it in stock?"}]},
    {"input": [{"role": "user", "content": "Show me all available products."}]},
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
# MAGIC ## 11. Register to Unity Catalog
# MAGIC
# MAGIC Register the model for version management and deployment.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.register_model(
    model_uri=model_uri,
    name="sjdatabricks.agents.langgraph_customer_support",
)
print(f"Registered model: {model_info.name} v{model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Deploy to Model Serving
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

UC_MODEL_NAME = "sjdatabricks.agents.langgraph_customer_support"
ENDPOINT_NAME = "langgraph-customer-support-agent"

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
# MAGIC ## 13. Query the Endpoint (Non-Streaming)
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
# MAGIC ## 14. Query the Endpoint (Streaming)
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
# MAGIC 1. **Define a StateGraph** with a `TypedDict` state schema
# MAGIC 2. **Build three nodes** — `classify` (LLM), `lookup` (Spark SQL), `respond` (LLM)
# MAGIC 3. **Connect nodes with edges** to create a linear classify -> lookup -> respond pipeline
# MAGIC 4. **Wrap in ResponsesAgent** with both `predict` and `predict_stream` support
# MAGIC 5. **Log, evaluate, and register** the agent with MLflow
# MAGIC 6. **Deploy to Model Serving** using the Databricks SDK with `create_and_wait`
# MAGIC 7. **Query the endpoint** using the OpenAI SDK in both non-streaming and streaming modes
# MAGIC
# MAGIC ### Comparing Topic 4 vs Topic 5
# MAGIC
# MAGIC | Aspect | Topic 4 (ReAct Agent) | Topic 5 (StateGraph) |
# MAGIC |--------|----------------------|---------------------|
# MAGIC | **Control** | LLM decides which tool to call | You define the exact routing logic |
# MAGIC | **Flexibility** | LLM can call tools in any order | Fixed pipeline: classify -> lookup -> respond |
# MAGIC | **LLM calls** | Multiple (one per ReAct step) | Exactly 2 (classify + respond) |
# MAGIC | **Debugging** | Harder (LLM controls flow) | Easier (inspect state between nodes) |
# MAGIC | **Best for** | Open-ended questions | Well-defined question categories |
# MAGIC
# MAGIC **Next**: Topic 6 packages this LangChain + LangGraph agent as a Python wheel (`.whl`)
# MAGIC for production deployment.
