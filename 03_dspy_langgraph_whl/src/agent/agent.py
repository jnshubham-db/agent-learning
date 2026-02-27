"""
Customer Support Agent: LangGraph + DSPy CoT wrapped in MLflow ResponsesAgent.
Uses specialized nodes for order handling, product info, and refund processing.
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


def _messages_to_state(request: ResponsesAgentRequest) -> dict:
    """Convert request input to graph initial state."""
    input_list = getattr(request, "input", None) or (
        request.get("input", []) if isinstance(request, dict) else []
    )
    msgs = to_chat_completions_input(
        [i.model_dump() if hasattr(i, "model_dump") else i for i in input_list]
    )
    return {
        "messages": msgs,
        "intent": "",
        "response": "",
        "rationale": "",
    }


class CustomerSupportAgent(ResponsesAgent):
    """
    Customer Support Agent wrapping DSPy CoT + LangGraph in MLflow ResponsesAgent.
    Router classifies intent; Order, Product, and Refund nodes handle specialized queries.
    """

    def __init__(self):
        super().__init__()
        self.graph = build_graph()

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction: run graph and return output items."""
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
        Streaming prediction: run graph, then yield reasoning + text events.
        DSPy CoT doesn't natively stream, so we run the full module then:
        1. Yield create_reasoning_item() with CoT rationale
        2. Yield create_text_delta() events for answer text
        3. Yield response.output_item.done with create_text_output_item()
        """
        initial_state = _messages_to_state(request)

        # Run graph with stream_mode="updates"
        final_state = {}
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for _node_name, node_update in event.items():
                final_state.update(node_update)

        rationale = final_state.get("rationale", "") or ""
        response = (
            final_state.get("response", "") or "I'm sorry, I couldn't process your request."
        )

        # 1. Yield reasoning item (CoT rationale) first
        if rationale:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(id=str(uuid4()), reasoning_text=rationale),
            )

        # 2. Stream final answer text with create_text_delta events
        text_id = str(uuid4())
        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=chunk, item_id=text_id))

        # 3. Yield output_item.done for the text
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=response, id=text_id),
        )


# Module-level setup for models-from-code
mlflow.dspy.autolog()
mlflow.langchain.autolog()

_agent = CustomerSupportAgent()
set_model(_agent)
