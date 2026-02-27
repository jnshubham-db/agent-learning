"""
Genie DSPy Agent - Sales Analytics Genie Space agent with DSPy ReAct (CoT).

Uses DSPy ReAct with a query_sales_data tool that calls the Databricks Genie API.
Wrapped in MLflow ResponsesAgent for logging, tracing, evaluation, and model serving.
"""

from typing import Generator
from uuid import uuid4

import dspy
import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from .genie import query_genie_space

# Replace with your Sales Analytics Genie Space ID from Databricks Genie
SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"


def query_sales_data(question: str) -> str:
    """Query the Sales Analytics Genie space for sales-related questions.

    Use this tool to answer questions about sales data, revenue, products,
    regions, and other analytics available in the Sales Genie Space.

    Args:
        question: Natural language question about sales data.

    Returns:
        The Genie response with query results or descriptions.
    """
    return query_genie_space(SALES_GENIE_SPACE_ID, question)


class GenieDSPyAgent(ResponsesAgent):
    """DSPy ReAct agent that queries a Sales Analytics Genie Space via CoT."""

    def __init__(self, space_id: str | None = None):
        """Initialize the agent.

        Args:
            space_id: Optional Genie space ID. If None, uses SALES_GENIE_SPACE_ID.
        """
        super().__init__()
        self._space_id = space_id or SALES_GENIE_SPACE_ID
        lm = dspy.LM("databricks-claude-sonnet-4-5")
        dspy.configure(lm=lm)

        def query_sales_data(question: str) -> str:
            return query_genie_space(self._space_id, question)

        self.react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[query_sales_data],
            max_iters=5,
        )

    def _run_agent(self, question: str) -> tuple[str, str]:
        """Run DSPy ReAct and return (answer, reasoning/trajectory)."""
        result = self.react_agent(question=question)
        answer = getattr(result, "answer", "") or ""
        trajectory = getattr(result, "trajectory", []) or []
        reasoning = "\n".join(str(t) for t in trajectory) if trajectory else ""
        return answer, reasoning

    def _messages_to_question(self, request: ResponsesAgentRequest) -> str:
        """Extract user question from request input messages."""
        input_list = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        if not input_list:
            return ""
        parts = []
        for msg in input_list:
            content = getattr(msg, "content", None) or msg.get("content", "")
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
        """Non-streaming prediction: run agent and return output items."""
        question = self._messages_to_question(request)
        answer, reasoning = self._run_agent(question)
        output_items = []
        if reasoning:
            output_items.append(
                self.create_reasoning_item(id=str(uuid4()), reasoning_text=reasoning)
            )
        output_items.append(
            self.create_text_output_item(text=answer, id=str(uuid4()))
        )
        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream reasoning (CoT) first, then text deltas, then output_item.done."""
        question = self._messages_to_question(request)
        answer, reasoning = self._run_agent(question)
        text_id = str(uuid4())
        reason_id = str(uuid4())

        # 1. Yield reasoning item (CoT) first
        if reasoning:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(id=reason_id, reasoning_text=reasoning),
            )

        # 2. Stream answer text with create_text_delta events
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=delta, item_id=text_id))

        # 3. Yield final text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=text_id),
        )


mlflow.dspy.autolog()
agent = GenieDSPyAgent()
set_model(agent)
