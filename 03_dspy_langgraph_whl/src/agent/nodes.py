"""
LangGraph nodes powered by DSPy Chain-of-Thought modules.
Router classifies intent; Order, Product, and Refund nodes handle specialized queries using tools.
"""

from typing import Literal, TypedDict

import dspy
from langgraph.graph import END, START, StateGraph

from .tools import get_order_status, get_product_info, process_refund


# Configure DSPy with Databricks Claude
_lm = dspy.LM("databricks-claude-sonnet-4-5")
dspy.configure(lm=_lm)


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Graph state with messages, intent, response, and CoT rationale."""

    messages: list
    intent: str
    response: str
    rationale: str


# ---------------------------------------------------------------------------
# DSPy CoT Signatures and Modules
# ---------------------------------------------------------------------------


class RouterSignature(dspy.Signature):
    """Classify the customer's intent from their message."""

    user_message: str = dspy.InputField(desc="The customer's message")
    intent: str = dspy.OutputField(
        desc="One of: order_query, product_query, refund_request"
    )


router_cot = dspy.ChainOfThought(RouterSignature)


# Order node uses ReAct (CoT + tools) for order queries
order_react = dspy.ReAct(
    signature="user_message -> response",
    tools=[get_order_status],
    max_iters=3,
)


# Product node uses ReAct (CoT + tools) for product queries
product_react = dspy.ReAct(
    signature="user_message -> response",
    tools=[get_product_info],
    max_iters=3,
)


# Refund node uses ReAct (CoT + tools) for refund requests
refund_react = dspy.ReAct(
    signature="user_message -> response",
    tools=[process_refund],
    max_iters=3,
)


# ---------------------------------------------------------------------------
# Helper to extract last user message from state
# ---------------------------------------------------------------------------


def _get_user_content(state: AgentState) -> str:
    """Extract the last user message content from state."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    last_msg = messages[-1]
    content = (
        getattr(last_msg, "content", None)
        or (last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg))
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(c.get("text", ""))
            elif hasattr(c, "text"):
                parts.append(getattr(c, "text", ""))
        return " ".join(parts).strip()
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> dict:
    """
    Router node: DSPy CoT module that classifies intent.
    Returns updated state with intent and rationale.
    """
    content = _get_user_content(state)
    result = router_cot(user_message=content)
    intent = getattr(result, "intent", "").strip().lower()

    # Normalize to valid intents
    if "order" in intent or intent == "order_query":
        intent = "order_query"
    elif "product" in intent or intent == "product_query":
        intent = "product_query"
    elif "refund" in intent or intent == "refund_request":
        intent = "refund_request"
    else:
        intent = "order_query"

    rationale = getattr(result, "rationale", "") or getattr(result, "reasoning", "") or ""
    return {"intent": intent, "rationale": rationale}


def order_node(state: AgentState) -> dict:
    """
    Order node: DSPy ReAct (CoT + tools) that handles order queries using get_order_status.
    """
    content = _get_user_content(state)
    result = order_react(user_message=content)
    response = getattr(result, "response", getattr(result, "answer", "")) or ""
    trajectory = getattr(result, "trajectory", []) or []
    rationale = "\n".join(str(t) for t in trajectory) if trajectory else ""
    return {"response": response, "rationale": rationale}


def product_node(state: AgentState) -> dict:
    """
    Product node: DSPy ReAct (CoT + tools) that handles product queries using get_product_info.
    """
    content = _get_user_content(state)
    result = product_react(user_message=content)
    response = getattr(result, "response", getattr(result, "answer", "")) or ""
    trajectory = getattr(result, "trajectory", []) or []
    rationale = "\n".join(str(t) for t in trajectory) if trajectory else ""
    return {"response": response, "rationale": rationale}


def refund_node(state: AgentState) -> dict:
    """
    Refund node: DSPy ReAct (CoT + tools) that handles refund requests using process_refund.
    """
    content = _get_user_content(state)
    result = refund_react(user_message=content)
    response = getattr(result, "response", getattr(result, "answer", "")) or ""
    trajectory = getattr(result, "trajectory", []) or []
    rationale = "\n".join(str(t) for t in trajectory) if trajectory else ""
    return {"response": response, "rationale": rationale}


def route_by_intent(state: AgentState) -> Literal["order_node", "product_node", "refund_node"]:
    """Conditional routing based on Router classification."""
    intent = state.get("intent", "order_query")
    if intent == "product_query":
        return "product_node"
    if intent == "refund_request":
        return "refund_node"
    return "order_node"


# ---------------------------------------------------------------------------
# Graph Builder and Compilation
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the LangGraph with conditional edges from router."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router", router_node)
    graph_builder.add_node("order_node", order_node)
    graph_builder.add_node("product_node", product_node)
    graph_builder.add_node("refund_node", refund_node)

    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "order_node": "order_node",
            "product_node": "product_node",
            "refund_node": "refund_node",
        },
    )
    graph_builder.add_edge("order_node", END)
    graph_builder.add_edge("product_node", END)
    graph_builder.add_edge("refund_node", END)

    return graph_builder.compile()
