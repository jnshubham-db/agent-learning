"""
Customer Order Support Agent - DSPy + LangGraph, wrapped in MLflow ResponsesAgent.

Builds a three-node LangGraph (classify -> lookup -> respond) where:
- classify uses DSPy CoT to determine the question category
- lookup queries Spark tables via tool functions
- respond uses DSPy CoT to formulate a natural-language answer

The agent is exposed as an MLflow ResponsesAgent for seamless deployment
to Databricks Model Serving.
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
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    classify_node,
    lookup_node,
    respond_node,
    AgentState,
)


def build_graph() -> CompiledStateGraph:
    """Build and compile the classify -> lookup -> respond LangGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("lookup", lookup_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "lookup")
    builder.add_edge("lookup", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


class DSPyLangGraphAgent(ResponsesAgent):
    """Customer Order Support Agent using DSPy + LangGraph, served via MLflow ResponsesAgent."""

    def __init__(self):
        super().__init__()
        self.graph = build_graph()

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Run the full graph and return a complete ResponsesAgentResponse.

        Extracts the user message from the request, invokes the graph,
        and wraps the final answer in the Responses API format.
        """
        user_message = self._extract_user_message(request)

        result = self.graph.invoke({
            "question": user_message,
            "messages": [],
        })

        answer = result.get("answer", "I'm sorry, I couldn't process your request.")

        output_message = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer}],
            "status": "completed",
        }

        return ResponsesAgentResponse(output=[output_message])

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream graph execution and yield ResponsesAgentStreamEvent objects.

        Each node completion yields a stream event so callers can observe
        incremental progress (classify -> lookup -> respond).
        """
        user_message = self._extract_user_message(request)

        # Stream node updates from the graph
        final_answer = None
        for event in self.graph.stream(
            {"question": user_message, "messages": []},
            stream_mode="updates",
        ):
            # Each event is a dict {node_name: state_update}
            for node_name, state_update in event.items():
                if node_name == "respond" and "answer" in state_update:
                    final_answer = state_update["answer"]

        # Yield the completed message as a stream event
        if final_answer:
            output_message = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_answer}],
                "status": "completed",
            }
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=output_message,
            )

    @staticmethod
    def _extract_user_message(request: ResponsesAgentRequest) -> str:
        """Pull the latest user message text from a ResponsesAgentRequest."""
        inp = getattr(request, "input", None)
        if inp is None and isinstance(request, dict):
            inp = request.get("input", [])

        # Walk input items in reverse to find the last user message
        for item in reversed(inp or []):
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if isinstance(item, dict):
                role = item.get("role", "")
                if role == "user":
                    content = item.get("content", "")
                    if isinstance(content, str):
                        return content
                    # content may be a list of content parts
                    if isinstance(content, list):
                        texts = [
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        ]
                        return " ".join(texts)
            elif isinstance(item, str):
                return item

        return ""


# Register the agent with MLflow
set_model(DSPyLangGraphAgent())
