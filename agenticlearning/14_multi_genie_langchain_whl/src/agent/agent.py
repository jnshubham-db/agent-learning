"""
Multi-Genie LangChain Agent — supervisor agent that routes questions to the correct
Genie space (orders, returns, or products) using a ChatDatabricks LLM classifier.

Wrapped in MLflow ResponsesAgent for logging, tracing, evaluation, and model serving.
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
)

from .genie import classify_question, genie_map


class MultiGenieLangChainAgent(ResponsesAgent):
    """LangChain-based supervisor agent that routes to Orders, Returns, or Products Genie spaces."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__()

    def _messages_to_question(self, request: ResponsesAgentRequest) -> str:
        """Extract the user question from the request input messages."""
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        if not input_list:
            return ""
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
        """Non-streaming prediction: classify the question, route to the correct Genie, return response."""
        question = self._messages_to_question(request)
        department = classify_question(question)
        agent = genie_map[department]
        answer = f"[Routed to {department}] {agent.query(question)}"

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
        """Stream the response: classify, route, emit text deltas, then output_item.done."""
        question = self._messages_to_question(request)
        department = classify_question(question)
        agent = genie_map[department]
        answer = f"[Routed to {department}] {agent.query(question)}"
        item_id = str(uuid4())

        # Stream answer text with create_text_delta events
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=item_id)
            )

        # Yield final text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=item_id),
        )


mlflow.langchain.autolog()
agent = MultiGenieLangChainAgent()
set_model(agent)
