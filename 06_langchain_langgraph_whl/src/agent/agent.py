"""
Customer Support Agent - LangGraph with LangChain, wrapped in MLflow ResponsesAgent.
Uses ChatDatabricks, CoT-style routing, specialized handler nodes, and a shared tool_node.
"""

from typing import Generator

import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    router_node,
    route_condition,
    order_handler_node,
    product_handler_node,
    refund_handler_node,
    tool_node,
    tools_condition,
    tool_to_handler_condition,
    AgentState,
)


def build_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph with conditional edges."""
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("order_handler", order_handler_node)
    builder.add_node("product_handler", product_handler_node)
    builder.add_node("refund_handler", refund_handler_node)
    builder.add_node("tool_node", tool_node)

    # Entry
    builder.add_edge(START, "router")

    # Router conditional edges -> specialized handlers
    builder.add_conditional_edges("router", route_condition)

    # Handler -> tool_node (if tool_calls) or END
    builder.add_conditional_edges("order_handler", tools_condition)
    builder.add_conditional_edges("product_handler", tools_condition)
    builder.add_conditional_edges("refund_handler", tools_condition)

    # tool_node -> back to the handler that invoked it
    builder.add_conditional_edges("tool_node", tool_to_handler_condition)

    return builder.compile()


# Compiled graph instance
graph = build_graph()


class CustomerSupportAgent(ResponsesAgent):
    """Customer Support Agent wrapping a compiled LangGraph with order, product, and refund handlers."""

    def __init__(self, agent=None):
        super().__init__()
        self.agent = agent or graph

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Collect done items from predict_stream."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream graph updates and yield response items."""
        inp = getattr(request, "input", None)
        if inp is None and isinstance(request, dict):
            inp = request.get("input", [])
        items = [i.model_dump() if hasattr(i, "model_dump") else i for i in (inp or [])]
        cc_msgs = to_chat_completions_input(items)

        for _, events in self.agent.stream(
            {"messages": cc_msgs}, stream_mode=["updates"]
        ):
            for node_data in events.values():
                if "messages" in node_data:
                    yield from output_to_responses_items_stream(node_data["messages"])


mlflow.langchain.autolog()
agent = CustomerSupportAgent()
set_model(agent)
