"""
LangGraph nodes for the Customer Support Agent.
Uses ChatDatabricks, MessagesState, and ToolNode from langgraph.prebuilt.
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_community.chat_models.databricks import ChatDatabricks
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import ToolNode

from .tools import (
    get_order_status,
    get_tracking_info,
    get_product_info,
    check_inventory,
    process_refund,
)

# Tool groupings for specialized handlers
ORDER_TOOLS = [get_order_status, get_tracking_info]
PRODUCT_TOOLS = [get_product_info, check_inventory]
REFUND_TOOLS = [process_refund]
ALL_TOOLS = ORDER_TOOLS + PRODUCT_TOOLS + REFUND_TOOLS

LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0)

# Specialized LLMs with tool binding
order_llm = llm.bind_tools(ORDER_TOOLS)
product_llm = llm.bind_tools(PRODUCT_TOOLS)
refund_llm = llm.bind_tools(REFUND_TOOLS)

# Single tool node for all tool execution
tool_node = ToolNode(ALL_TOOLS)


class AgentState(MessagesState, total=False):
    """Extended state with current_handler for routing from tool_node back to handler."""

    current_handler: str


ROUTER_PROMPT = """You are a customer support intent classifier. Given the user message, respond with exactly one word:
- "order" if the user asks about order status, tracking, shipment, or delivery
- "product" if the user asks about product info, inventory, pricing, or specifications
- "refund" if the user wants to return, refund, or report defective items

Respond with only the single word. No punctuation."""


def router_node(state: AgentState) -> dict:
    """Route to the appropriate handler based on user intent (CoT-style reasoning via LLM)."""
    llm_router = llm.with_config({"tags": ["router"]})
    response = llm_router.invoke(
        [SystemMessage(content=ROUTER_PROMPT), state["messages"][-1]]
    )
    return {"messages": [response]}


def route_condition(
    state: AgentState,
) -> Literal["order_handler", "product_handler", "refund_handler"]:
    """Conditional routing based on router output."""
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "") or ""
    text = str(content).strip().lower()
    if "order" in text:
        return "order_handler"
    if "product" in text:
        return "product_handler"
    if "refund" in text:
        return "refund_handler"
    return "order_handler"  # Default


def order_handler_node(state: AgentState) -> dict:
    """Handle order-related queries using order tools."""
    response = order_llm.invoke(state["messages"])
    return {"messages": [response], "current_handler": "order_handler"}


def product_handler_node(state: AgentState) -> dict:
    """Handle product-related queries using product tools."""
    response = product_llm.invoke(state["messages"])
    return {"messages": [response], "current_handler": "product_handler"}


def refund_handler_node(state: AgentState) -> dict:
    """Handle refund and return requests using refund tool."""
    response = refund_llm.invoke(state["messages"])
    return {"messages": [response], "current_handler": "refund_handler"}


def _has_tool_calls(state: AgentState) -> bool:
    """Check if the last message has tool calls."""
    if not state.get("messages"):
        return False
    last = state["messages"][-1]
    return bool(getattr(last, "tool_calls", None))


def tools_condition(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Route handler: tool node or END."""
    return "tool_node" if _has_tool_calls(state) else "__end__"


def tool_to_handler_condition(state: AgentState) -> str:
    """Route from tool_node back to the handler that invoked it."""
    return state.get("current_handler", "order_handler")
