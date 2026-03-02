"""
Customer Support Agent: LangChain + LangGraph wrapped in MLflow ResponsesAgent.
Packages the classify -> lookup -> respond graph into a deployable agent.
"""

from typing import Generator
from uuid import uuid4

import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from .nodes import build_graph


def _extract_question(request: ResponsesAgentRequest) -> str:
    """
    Extract the user's question text from a ResponsesAgentRequest.

    Handles both object-style and dict-style request formats, converting
    through to_chat_completions_input to normalize the message structure.
    """
    input_list = getattr(request, "input", None) or (
        request.get("input", []) if isinstance(request, dict) else []
    )
    msgs = to_chat_completions_input(
        [i.model_dump() if hasattr(i, "model_dump") else i for i in input_list]
    )
    # Find the last user message
    for msg in reversed(msgs):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "user":
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif isinstance(c, dict) and "text" in c:
                        parts.append(c["text"])
                    elif hasattr(c, "text"):
                        parts.append(getattr(c, "text", ""))
                return " ".join(parts).strip()
            return str(content)
    return ""


class LangChainLangGraphAgent(ResponsesAgent):
    """
    Customer Support Agent wrapping a LangChain + LangGraph pipeline
    in MLflow ResponsesAgent.

    The graph follows a three-node pattern:
      classify -> lookup -> respond

    - classify_node: Uses ChatDatabricks to determine intent (order/return/product)
    - lookup_node:   Queries the appropriate Spark table via tool functions
    - respond_node:  Uses ChatDatabricks to formulate a natural-language answer
    """

    def __init__(self):
        super().__init__()
        self.graph = build_graph()

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Non-streaming prediction.

        Runs the full graph and collects all output items from the stream,
        returning them as a ResponsesAgentResponse.
        """
        output_items = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done" and event.item is not None
        ]
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Streaming prediction.

        Runs the classify -> lookup -> respond graph, then yields:
        1. Text delta events that stream the answer in chunks
        2. A response.output_item.done event with the complete text output
        """
        question = _extract_question(request)

        # Build initial state for the graph
        initial_state = {
            "question": question,
            "category": "",
            "data": "",
            "answer": "",
            "messages": [],
        }

        # Run graph and collect final state
        final_state = dict(initial_state)
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for _node_name, node_update in event.items():
                final_state.update(node_update)

        answer = final_state.get("answer", "") or "I'm sorry, I couldn't process your request."

        # Stream the answer in chunks via text_delta events
        text_id = str(uuid4())
        chunk_size = 40
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=chunk, item_id=text_id)
            )

        # Yield the final complete text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=text_id),
        )


# ---------------------------------------------------------------------------
# Module-level setup for models-from-code
# ---------------------------------------------------------------------------

mlflow.langchain.autolog()

_agent = LangChainLangGraphAgent()
set_model(_agent)
