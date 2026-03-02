"""
LangGraph node functions and DSPy signatures for the Customer Order Support Agent.

Pipeline: classify -> lookup -> respond
- classify_node: uses DSPy CoT to classify the question into orders, returns, or products
- lookup_node: queries Spark tables via tool functions based on classification
- respond_node: uses DSPy CoT to formulate a natural-language answer from the data
"""

import re
from typing import TypedDict, List, Any

import dspy
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .tools import get_order_details, search_returns, get_product_info

# ---------------------------------------------------------------------------
# DSPy language model configuration
# ---------------------------------------------------------------------------
LM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
lm = dspy.LM(LM_ENDPOINT)
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class ClassifyQuestion(dspy.Signature):
    """Classify a customer support question into one of three categories:
    'orders', 'returns', or 'products'."""

    question: str = dspy.InputField(desc="The customer's question")
    category: str = dspy.OutputField(
        desc="Exactly one of: orders, returns, products"
    )


class AnswerFromData(dspy.Signature):
    """Given a customer question and relevant data retrieved from the database,
    produce a helpful, concise answer."""

    question: str = dspy.InputField(desc="The original customer question")
    data: str = dspy.InputField(desc="Data retrieved from the database")
    answer: str = dspy.OutputField(desc="A helpful answer for the customer")


# ---------------------------------------------------------------------------
# Pre-built DSPy modules
# ---------------------------------------------------------------------------
classify_cot = dspy.ChainOfThought(ClassifyQuestion)
answer_cot = dspy.ChainOfThought(AnswerFromData)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph nodes."""

    question: str
    category: str
    data: str
    answer: str
    messages: List[BaseMessage]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def classify_node(state: AgentState) -> dict:
    """Use DSPy Chain-of-Thought to classify the question category.

    Reads `state["question"]` and writes `state["category"]`.
    """
    result = classify_cot(question=state["question"])
    category = result.category.strip().lower()
    # Normalise to one of the three expected values
    if "order" in category:
        category = "orders"
    elif "return" in category or "refund" in category:
        category = "returns"
    elif "product" in category:
        category = "products"
    else:
        category = "orders"  # safe default
    return {"category": category}


def lookup_node(state: AgentState) -> dict:
    """Query Spark tables based on the classified category.

    Reads `state["category"]` and `state["question"]`, writes `state["data"]`.
    Attempts to extract a numeric order_id or product name from the question.
    """
    question = state["question"]
    category = state["category"]

    if category == "orders":
        order_id = _extract_order_id(question)
        if order_id is not None:
            data = get_order_details(order_id)
        else:
            data = "Could not determine order ID from the question."

    elif category == "returns":
        order_id = _extract_order_id(question)
        if order_id is not None:
            data = search_returns(order_id)
        else:
            data = "Could not determine order ID for return lookup."

    elif category == "products":
        product_name = _extract_product_name(question)
        if product_name:
            data = get_product_info(product_name)
        else:
            data = "Could not determine product name from the question."

    else:
        data = "Unknown category; unable to look up data."

    return {"data": data}


def respond_node(state: AgentState) -> dict:
    """Use DSPy Chain-of-Thought to formulate a customer-friendly answer.

    Reads `state["question"]` and `state["data"]`, writes `state["answer"]`
    and appends an AIMessage to `state["messages"]`.
    """
    result = answer_cot(question=state["question"], data=state["data"])
    answer_text = result.answer

    messages = list(state.get("messages") or [])
    messages.append(AIMessage(content=answer_text))

    return {"answer": answer_text, "messages": messages}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_order_id(text: str) -> int | None:
    """Try to pull a numeric order ID from a question string."""
    match = re.search(r"\b(\d{4,})\b", text)
    return int(match.group(1)) if match else None


def _extract_product_name(text: str) -> str | None:
    """Try to pull a product name from a question string.

    Looks for patterns like 'product <Name>' or known product keywords.
    """
    known_products = ["laptop", "phone", "tablet", "monitor", "keyboard", "mouse"]
    lower = text.lower()
    for product in known_products:
        if product in lower:
            return product.capitalize()
    # Fallback: look for quoted strings
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None
