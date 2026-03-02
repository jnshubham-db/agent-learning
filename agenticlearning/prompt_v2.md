# Agent Development Learning Path — Databricks + MLflow

## Goal

Learn the end-to-end lifecycle of agent development with **DSPy**, **LangChain**, **LangGraph**, and **Genie** on Databricks.
Each topic is a self-contained Databricks notebook (or a folder with multiple code files when a `.whl` is required).

---

## Consistent Example Use Case: **Customer Order Support Agent**

All topics below use the same fictional **"Customer Order Support"** scenario so the patterns stay comparable:

- **Catalog:** `sjdatabricks`
- **Tables:**
  - `sjdatabricks.orders.order_details` — order id, customer name, product, quantity, status, order date
  - `sjdatabricks.orders.returns` — return id, order id, reason, status, return date
  - `sjdatabricks.orders.products` — product id, name, category, price, stock
- **Agent task:** A user asks natural-language questions like *"What is the status of order 1042?"* or *"Show me all returns in the last 30 days"* and the agent answers by querying the data or reasoning over it.

For **Genie** topics, create fake data in the `sjdatabricks` catalog and build Genie Spaces on top of it (a script is provided to seed the data and create the spaces).

---

## Cross-Cutting Requirements (apply to every topic)

1. **MLflow integration** — latest version (`mlflow>=3`). Log the agent, parameters, metrics, and artifacts.
2. **Model Serving** — every agent must be deployable to Databricks Model Serving.
3. **Streaming output** — DSPy CoT does not natively expose LLM reasoning as a stream. Each topic must demonstrate how to surface streaming output from Model Serving so frontend apps can consume it (e.g., via `mlflow.pyfunc.ChatModel` with streaming support).
4. **Evaluation** — include an evaluation cell with an LLM-as-a-judge setup to score the agent's answers.
5. **Functional style** — use simple functional programming concepts (pure functions, composition) to keep notebooks readable.

---

## Topics

### 1. Agent with DSPy — notebook (no whl)

A single Databricks notebook. The agent uses a DSPy signature + Chain-of-Thought to answer order questions.

**Example:**
```
import dspy, mlflow

lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
dspy.configure(lm=lm)

class OrderStatus(dspy.Signature):
    """Answer questions about customer orders."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

agent = dspy.ChainOfThought(OrderStatus)
result = agent(question="What is the status of order 1042?")

with mlflow.start_run():
    mlflow.dspy.log_model(agent, "order_status_agent")
```

---

### 2. Agent with DSPy + LangGraph (2–3 nodes, CoT) — notebook (no whl)

A single notebook. LangGraph orchestrates 2–3 nodes: **classify** the question, **lookup** order data, **respond** with CoT reasoning.

**Example:**
```
from langgraph.graph import StateGraph
import dspy, mlflow

class ClassifyQuestion(dspy.Signature):
    """Classify if the question is about orders, returns, or products."""
    question: str = dspy.InputField()
    category: str = dspy.OutputField()

class AnswerQuestion(dspy.Signature):
    """Answer the customer question given the category."""
    question: str = dspy.InputField()
    category: str = dspy.InputField()
    answer: str = dspy.OutputField()

classify = dspy.ChainOfThought(ClassifyQuestion)
answer = dspy.ChainOfThought(AnswerQuestion)

def classify_node(state):
    result = classify(question=state["question"])
    return {**state, "category": result.category}

def answer_node(state):
    result = answer(question=state["question"], category=state["category"])
    return {**state, "answer": result.answer}

graph = StateGraph(dict)
graph.add_node("classify", classify_node)
graph.add_node("answer", answer_node)
graph.add_edge("classify", "answer")
graph.set_entry_point("classify")
graph.set_finish_point("answer")
agent = graph.compile()

with mlflow.start_run():
    mlflow.langchain.log_model(agent, "order_support_langgraph_dspy")
```

---

### 3. Agent with DSPy + LangGraph (2–3 nodes, CoT) — whl package

Same logic as Topic 2 but split into a proper Python package:

```
order_support_dspy/
├── order_support_dspy/
│   ├── __init__.py
│   ├── signatures.py      # ClassifyQuestion, AnswerQuestion
│   ├── nodes.py            # classify_node, answer_node
│   └── graph.py            # build_graph() -> compiled LangGraph
├── setup.py
└── register_model.py       # mlflow.langchain.log_model(...)
```

**Example (`signatures.py`):**
```python
import dspy

class ClassifyQuestion(dspy.Signature):
    """Classify if the question is about orders, returns, or products."""
    question: str = dspy.InputField()
    category: str = dspy.OutputField()

class AnswerQuestion(dspy.Signature):
    """Answer the customer question given the category."""
    question: str = dspy.InputField()
    category: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

### 4. Agent with LangChain — notebook (no whl)

A single notebook. Uses LangChain's `ChatDatabricks` + tools to answer order questions.

**Example:**
```
from langchain_databricks import ChatDatabricks
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import mlflow

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

@tool
def get_order_status(order_id: int) -> str:
    """Look up the status of a customer order by its ID."""
    orders = {1042: "Shipped", 1043: "Processing", 1044: "Delivered"}
    return orders.get(order_id, "Order not found")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [get_order_status], prompt)
executor = AgentExecutor(agent=agent, tools=[get_order_status])

result = executor.invoke({"input": "What is the status of order 1042?"})

with mlflow.start_run():
    mlflow.langchain.log_model(executor, "order_support_langchain")
```

---

### 5. Agent with LangChain + LangGraph (2–3 nodes, CoT) — notebook (no whl)

A single notebook. LangGraph orchestrates **classify → lookup → respond** nodes, each powered by LangChain.

**Example:**
```
from langgraph.graph import StateGraph
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
import mlflow

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

def classify_node(state):
    resp = llm.invoke([HumanMessage(
        content=f"Classify this question as 'order', 'return', or 'product': {state['question']}"
    )])
    return {**state, "category": resp.content.strip()}

def answer_node(state):
    resp = llm.invoke([HumanMessage(
        content=f"You are a customer support agent. Category: {state['category']}. "
                f"Answer: {state['question']}"
    )])
    return {**state, "answer": resp.content}

graph = StateGraph(dict)
graph.add_node("classify", classify_node)
graph.add_node("answer", answer_node)
graph.add_edge("classify", "answer")
graph.set_entry_point("classify")
graph.set_finish_point("answer")
agent = graph.compile()

result = agent.invoke({"question": "What is the status of order 1042?"})

with mlflow.start_run():
    mlflow.langchain.log_model(agent, "order_support_langgraph_langchain")
```

---

### 6. Agent with LangChain + LangGraph (2–3 nodes, CoT) — whl package

Same logic as Topic 5, packaged as a `.whl`:

```
order_support_langchain/
├── order_support_langchain/
│   ├── __init__.py
│   ├── nodes.py            # classify_node, answer_node
│   └── graph.py            # build_graph() -> compiled LangGraph
├── setup.py
└── register_model.py       # mlflow.langchain.log_model(...)
```

**Example (`nodes.py`):**
```python
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

def classify_node(state):
    resp = llm.invoke([HumanMessage(
        content=f"Classify this question as 'order', 'return', or 'product': {state['question']}"
    )])
    return {**state, "category": resp.content.strip()}

def answer_node(state):
    resp = llm.invoke([HumanMessage(
        content=f"You are a customer support agent. Category: {state['category']}. "
                f"Answer: {state['question']}"
    )])
    return {**state, "answer": resp.content}
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

### 7. Simple Genie Agent with DSPy — notebook (no whl)

A single notebook. The DSPy agent delegates data questions to a **single Genie Space** (built on the `sjdatabricks.orders` tables).

**Example:**
```
import dspy, mlflow
from databricks_agents.genie import GenieAgent

genie = GenieAgent(genie_space_id="<ORDER_GENIE_SPACE_ID>")

class OrderQuery(dspy.Signature):
    """Decide whether to query the Genie Space for order data."""
    question: str = dspy.InputField()
    needs_data: bool = dspy.OutputField()

router = dspy.ChainOfThought(OrderQuery)

def handle(question: str) -> str:
    decision = router(question=question)
    if decision.needs_data:
        return genie.invoke({"messages": [{"role": "user", "content": question}]})
    return "I can only help with order-related questions."

print(handle("What is the status of order 1042?"))

with mlflow.start_run():
    mlflow.pyfunc.log_model(...)  # wrap handle as ChatModel
```

---

### 8. Simple Genie Agent with LangChain — notebook (no whl)

Same as Topic 7 but the routing is done with LangChain instead of DSPy.

**Example:**
```
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
from databricks_agents.genie import GenieAgent
import mlflow

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
genie = GenieAgent(genie_space_id="<ORDER_GENIE_SPACE_ID>")

def handle(question: str) -> str:
    resp = llm.invoke([HumanMessage(
        content=f"Does this question need order data? Reply 'yes' or 'no': {question}"
    )])
    if "yes" in resp.content.lower():
        return genie.invoke({"messages": [{"role": "user", "content": question}]})
    return "I can only help with order-related questions."

print(handle("What is the status of order 1042?"))

with mlflow.start_run():
    mlflow.pyfunc.log_model(...)  # wrap handle as ChatModel
```

---

### 9. Multi-Genie Agent with DSPy (main agent + multiple Genie agents) — notebook (no whl)

A main DSPy agent routes to **multiple Genie Spaces**: Orders, Returns, and Products. Each space covers one table group.

**Example:**
```
import dspy, mlflow
from databricks_agents.genie import GenieAgent

orders_genie = GenieAgent(genie_space_id="<ORDER_SPACE_ID>")
returns_genie = GenieAgent(genie_space_id="<RETURNS_SPACE_ID>")
products_genie = GenieAgent(genie_space_id="<PRODUCTS_SPACE_ID>")

class RouteQuestion(dspy.Signature):
    """Route the question to the correct department: orders, returns, or products."""
    question: str = dspy.InputField()
    department: str = dspy.OutputField()

router = dspy.ChainOfThought(RouteQuestion)
genie_map = {"orders": orders_genie, "returns": returns_genie, "products": products_genie}

def handle(question: str) -> str:
    route = router(question=question)
    genie = genie_map.get(route.department)
    if genie:
        return genie.invoke({"messages": [{"role": "user", "content": question}]})
    return "Sorry, I could not route your question."

print(handle("What is the status of order 1042?"))
print(handle("Show me all returns in the last 30 days"))
print(handle("What products are in the Electronics category?"))

with mlflow.start_run():
    mlflow.pyfunc.log_model(...)  # wrap handle as ChatModel
```

---

### 10. Multi-Genie Agent with LangChain (main agent + multiple Genie agents) — notebook (no whl)

Same routing as Topic 9 but the main agent uses LangChain.

**Example:**
```
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
from databricks_agents.genie import GenieAgent
import mlflow

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

orders_genie = GenieAgent(genie_space_id="<ORDER_SPACE_ID>")
returns_genie = GenieAgent(genie_space_id="<RETURNS_SPACE_ID>")
products_genie = GenieAgent(genie_space_id="<PRODUCTS_SPACE_ID>")

genie_map = {"orders": orders_genie, "returns": returns_genie, "products": products_genie}

def handle(question: str) -> str:
    resp = llm.invoke([HumanMessage(
        content=f"Classify this question into one of: orders, returns, products. "
                f"Reply with only the category.\nQuestion: {question}"
    )])
    department = resp.content.strip().lower()
    genie = genie_map.get(department)
    if genie:
        return genie.invoke({"messages": [{"role": "user", "content": question}]})
    return "Sorry, I could not route your question."

print(handle("What is the status of order 1042?"))
print(handle("Show me all returns in the last 30 days"))

with mlflow.start_run():
    mlflow.pyfunc.log_model(...)  # wrap handle as ChatModel
```

---

### 11. Simple Genie Agent with DSPy — whl package

Same as Topic 7 packaged as a `.whl`:

```
genie_dspy/
├── genie_dspy/
│   ├── __init__.py
│   ├── router.py           # OrderQuery signature + router logic
│   └── agent.py            # handle() function using GenieAgent
├── setup.py
└── register_model.py       # mlflow.pyfunc.log_model(...)
```

**Example (`router.py`):**
```python
import dspy

class OrderQuery(dspy.Signature):
    """Decide whether to query the Genie Space for order data."""
    question: str = dspy.InputField()
    needs_data: bool = dspy.OutputField()

router = dspy.ChainOfThought(OrderQuery)
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

### 12. Simple Genie Agent with LangChain — whl package

Same as Topic 8 packaged as a `.whl`:

```
genie_langchain/
├── genie_langchain/
│   ├── __init__.py
│   ├── router.py           # LLM-based routing logic
│   └── agent.py            # handle() function using GenieAgent
├── setup.py
└── register_model.py       # mlflow.pyfunc.log_model(...)
```

**Example (`router.py`):**
```python
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

def needs_order_data(question: str) -> bool:
    resp = llm.invoke([HumanMessage(
        content=f"Does this question need order data? Reply 'yes' or 'no': {question}"
    )])
    return "yes" in resp.content.lower()
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

### 13. Multi-Genie Agent with DSPy (main agent + multiple Genie agents) — whl package

Same as Topic 9 packaged as a `.whl`:

```
multi_genie_dspy/
├── multi_genie_dspy/
│   ├── __init__.py
│   ├── router.py           # RouteQuestion signature + CoT router
│   ├── genies.py           # GenieAgent instances for orders, returns, products
│   └── agent.py            # handle() dispatches to the right Genie
├── setup.py
└── register_model.py       # mlflow.pyfunc.log_model(...)
```

**Example (`router.py`):**
```python
import dspy

class RouteQuestion(dspy.Signature):
    """Route the question to the correct department: orders, returns, or products."""
    question: str = dspy.InputField()
    department: str = dspy.OutputField()

router = dspy.ChainOfThought(RouteQuestion)
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

### 14. Multi-Genie Agent with LangChain (main agent + multiple Genie agents) — whl package

Same as Topic 10 packaged as a `.whl`:

```
multi_genie_langchain/
├── multi_genie_langchain/
│   ├── __init__.py
│   ├── router.py           # LLM-based classification into orders/returns/products
│   ├── genies.py           # GenieAgent instances
│   └── agent.py            # handle() dispatches to the right Genie
├── setup.py
└── register_model.py       # mlflow.pyfunc.log_model(...)
```

**Example (`router.py`):**
```python
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

def classify_question(question: str) -> str:
    resp = llm.invoke([HumanMessage(
        content=f"Classify this question into one of: orders, returns, products. "
                f"Reply with only the category.\nQuestion: {question}"
    )])
    return resp.content.strip().lower()
```

Build the `.whl`, install, then log with MLflow and deploy to Model Serving.

---

## Fake Data & Genie Space Setup Script

Before starting the Genie topics (7–14), run this script to seed `sjdatabricks` and create Genie Spaces:

```python
from pyspark.sql import SparkSession
from datetime import date, timedelta
import random

spark = SparkSession.builder.getOrCreate()

spark.sql("CREATE CATALOG IF NOT EXISTS sjdatabricks")
spark.sql("CREATE SCHEMA IF NOT EXISTS sjdatabricks.orders")

orders = [
    (i, f"Customer_{i}", random.choice(["Laptop", "Phone", "Tablet", "Monitor"]),
     random.randint(1, 5), random.choice(["Shipped", "Processing", "Delivered"]),
     str(date.today() - timedelta(days=random.randint(1, 90))))
    for i in range(1040, 1060)
]
spark.createDataFrame(orders, ["order_id", "customer_name", "product", "quantity", "status", "order_date"]) \
     .write.mode("overwrite").saveAsTable("sjdatabricks.orders.order_details")

returns = [
    (r, random.randint(1040, 1059), random.choice(["Defective", "Wrong item", "Changed mind"]),
     random.choice(["Pending", "Approved", "Rejected"]),
     str(date.today() - timedelta(days=random.randint(1, 60))))
    for r in range(5001, 5011)
]
spark.createDataFrame(returns, ["return_id", "order_id", "reason", "status", "return_date"]) \
     .write.mode("overwrite").saveAsTable("sjdatabricks.orders.returns")

products = [
    (p, name, cat, round(random.uniform(50, 2000), 2), random.randint(0, 500))
    for p, (name, cat) in enumerate([
        ("Laptop", "Electronics"), ("Phone", "Electronics"),
        ("Tablet", "Electronics"), ("Monitor", "Electronics"),
        ("Keyboard", "Accessories"), ("Mouse", "Accessories")
    ], start=1)
]
spark.createDataFrame(products, ["product_id", "name", "category", "price", "stock"]) \
     .write.mode("overwrite").saveAsTable("sjdatabricks.orders.products")
```

After running this, manually create three Genie Spaces in the Databricks UI (or via API), one each for `order_details`, `returns`, and `products`.

---

## Key Resources

| Resource | Link |
|----------|------|
| ML docs | https://docs.databricks.com/aws/en/machine-learning/ |
| Agent dev workflow | https://docs.databricks.com/aws/en/generative-ai/guide/agents-dev-workflow |
| Author an agent | https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent |
| Chat app | https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app |
| Query an agent | https://docs.databricks.com/aws/en/generative-ai/agent-framework/query-agent |
| MLflow 3 GenAI | https://docs.databricks.com/aws/en/mlflow3/genai/ |
| DSPy on Databricks | https://docs.databricks.com/aws/en/generative-ai/dspy/ |
| Multi-agent Genie | https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie |

All sub-pages can be discovered from the sitemap.xml linked under the ML docs root.
