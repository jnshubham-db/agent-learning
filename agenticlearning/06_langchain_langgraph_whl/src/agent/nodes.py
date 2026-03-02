"""
LangGraph node functions for the classify -> lookup -> respond pipeline.

Uses ChatDatabricks (Meta Llama 3.3 70B Instruct) for classification and
response generation, and Spark-based tool functions for data lookup.
"""

from typing import Literal, TypedDict

from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .tools import get_order_details, get_product_info, search_returns

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Graph state flowing through classify -> lookup -> respond.

    Fields:
        question: The original user question.
        category: Classified intent — one of 'order', 'return', 'product'.
        data:     Raw data retrieved from Spark tables.
        answer:   Final natural-language response for the user.
        messages: Message history for MLflow compatibility.
    """

    question: str
    category: str
    data: str
    answer: str
    messages: list


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------


def classify_node(state: AgentState) -> dict:
    """
    Classify the user's question as 'order', 'return', or 'product'.

    Sends the question to ChatDatabricks with a system prompt instructing it
    to respond with exactly one word. The response is normalized to a valid
    category.
    """
    question = state["question"]
    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a customer support intent classifier. "
                "Given the user message, respond with exactly one word:\n"
                "- 'order' if the user asks about order status, tracking, shipment, or delivery\n"
                "- 'return' if the user asks about returns, refunds, or exchanges\n"
                "- 'product' if the user asks about product info, inventory, pricing, or specifications\n\n"
                "Respond with only the single word. No punctuation."
            )
        ),
        HumanMessage(content=question),
    ])
    category = response.content.strip().lower().replace(".", "")

    # Normalize to valid categories
    if "return" in category or "refund" in category:
        category = "return"
    elif "product" in category:
        category = "product"
    else:
        category = "order"

    return {"category": category}


def lookup_node(state: AgentState) -> dict:
    """
    Query the appropriate Spark table based on the classified category.

    Routes to:
    - get_order_details() for 'order' questions
    - search_returns() for 'return' questions
    - get_product_info() for 'product' questions

    Extracts relevant identifiers from the question text to pass as arguments.
    """
    category = state["category"]
    question = state["question"]

    if category == "order":
        order_id = _extract_number(question)
        data = get_order_details(order_id) if order_id else "Please provide an order ID."
    elif category == "return":
        order_id = _extract_number(question)
        data = search_returns(order_id) if order_id else "Please provide an order ID for return lookup."
    elif category == "product":
        product_name = _extract_product_name(question)
        data = get_product_info(product_name) if product_name else "Please provide a product name."
    else:
        data = "Unable to determine the appropriate data source."

    return {"data": data}


def respond_node(state: AgentState) -> dict:
    """
    Generate a natural-language answer using ChatDatabricks.

    Takes the original question, classified category, and retrieved data,
    then produces a helpful customer-support response.
    """
    question = state["question"]
    category = state["category"]
    data = state["data"]

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a helpful customer support agent. "
                "Use the data provided to answer the customer's question clearly and concisely. "
                "If the data indicates nothing was found, let the customer know politely "
                "and suggest next steps."
            )
        ),
        HumanMessage(
            content=(
                f"Customer question: {question}\n"
                f"Category: {category}\n"
                f"Retrieved data:\n{data}"
            )
        ),
    ])

    return {"answer": response.content}


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def route_after_classify(state: AgentState) -> Literal["lookup_node"]:
    """
    Route after classification. All categories proceed to lookup_node.
    This function exists to allow future conditional routing if needed.
    """
    return "lookup_node"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Build and compile the classify -> lookup -> respond StateGraph.

    Returns:
        A compiled LangGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("lookup_node", lookup_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "lookup_node")
    builder.add_edge("lookup_node", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _extract_number(text: str) -> int | None:
    """Extract the first integer from a string, or None if not found."""
    import re

    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _extract_product_name(text: str) -> str:
    """
    Extract a likely product name from the question.

    Uses simple heuristics: looks for quoted strings first, then words
    after common keywords like 'product', 'about', 'info on'.
    """
    import re

    # Try quoted strings first
    quoted = re.search(r'["\']([^"\']+)["\']', text)
    if quoted:
        return quoted.group(1)

    # Try extracting after keywords
    for keyword in ["product", "about", "info on", "details on", "tell me about"]:
        idx = text.lower().find(keyword)
        if idx >= 0:
            after = text[idx + len(keyword) :].strip().strip("?.,!")
            if after:
                return after

    # Fallback: return the question minus common stop words
    return text
