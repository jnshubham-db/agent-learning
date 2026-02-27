"""
Multi-Genie DSPy Supervisor Agent.
Orchestrates Sales Analytics and Customer Insights Genie spaces via DSPy ReAct.
Subclass of mlflow.pyfunc.ResponsesAgent with predict/predict_stream, tool-call output,
CoT reasoning, and streaming support.
"""

import json
from typing import Any, Generator, List, Tuple
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

# Placeholder IDs — replace with actual Genie Space IDs from create_genie_spaces
SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"


def _create_tool_wrapper(
    space_id: str,
    tool_name: str,
    recorder: List[Tuple[str, str, str, str]],
) -> Any:
    """Create a tool that records invocations for ResponsesAgent output."""

    def _tool(question: str) -> str:
        call_id = str(uuid4())
        arguments = json.dumps({"question": question})
        try:
            output = query_genie_space(space_id, question)
        except Exception as e:
            output = str(e)
        recorder.append((call_id, tool_name, arguments, output))
        return output

    _tool.__name__ = tool_name
    _tool.__doc__ = (
        f"Query the {tool_name.replace('_', ' ').title()} Genie space. "
        f"Pass the user's question directly."
    )
    return _tool


def query_sales_data(question: str) -> str:
    """Query the Sales Analytics Genie space for revenue, orders, product performance."""
    return query_genie_space(SALES_GENIE_SPACE_ID, question)


def query_customer_data(question: str) -> str:
    """Query the Customer Insights Genie space for support tickets, customer profiles."""
    return query_genie_space(CUSTOMER_GENIE_SPACE_ID, question)


class MultiGenieDSPyAgent(ResponsesAgent):
    """
    Supervisor agent using DSPy ReAct to orchestrate Sales Analytics and Customer Insights
    Genie spaces. Subclass of mlflow.pyfunc.ResponsesAgent.
    """

    def __init__(
        self,
        sales_space_id: str | None = None,
        customer_space_id: str | None = None,
    ):
        super().__init__()
        global SALES_GENIE_SPACE_ID, CUSTOMER_GENIE_SPACE_ID
        if sales_space_id and sales_space_id != "<YOUR-SALES-GENIE-SPACE-ID>":
            SALES_GENIE_SPACE_ID = sales_space_id
        if customer_space_id and customer_space_id != "<YOUR-CUSTOMER-GENIE-SPACE-ID>":
            CUSTOMER_GENIE_SPACE_ID = customer_space_id

        lm = dspy.LM("databricks-claude-sonnet-4-5")
        dspy.configure(lm=lm)

        # Tool call recorder for emitting create_function_call_item / create_function_call_output_item
        self._tool_calls: List[Tuple[str, str, str, str]] = []

        _recorded_query_sales = _create_tool_wrapper(
            SALES_GENIE_SPACE_ID, "query_sales_data", self._tool_calls
        )
        _recorded_query_customer = _create_tool_wrapper(
            CUSTOMER_GENIE_SPACE_ID, "query_customer_data", self._tool_calls
        )

        self.react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[_recorded_query_sales, _recorded_query_customer],
            max_iters=5,
        )

    def _run_agent(self, question: str) -> Tuple[str, str, List[Tuple[str, str, str, str]]]:
        """
        Run DSPy ReAct and return (answer, trajectory/reasoning, tool_calls).
        """
        self._tool_calls.clear()
        result = self.react_agent(question=question)
        answer = getattr(result, "answer", "") or ""
        trajectory = getattr(result, "trajectory", []) or []
        reasoning = "\n".join(str(t) for t in trajectory) if trajectory else ""
        tool_calls = list(self._tool_calls)
        return answer, reasoning, tool_calls

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
        question = self._messages_to_question(request)
        answer, reasoning, tool_calls = self._run_agent(question)

        output_items = []

        # 1. CoT reasoning
        if reasoning:
            output_items.append(self.create_reasoning_item(id="reason_1", reasoning_text=reasoning))

        # 2. Function call items and outputs
        for idx, (call_id, name, arguments, output) in enumerate(tool_calls):
            item_id = f"fc_{idx}"
            output_items.append(
                self.create_function_call_item(
                    id=item_id,
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
            output_items.append(
                self.create_function_call_output_item(call_id=call_id, output=output)
            )

        # 3. Final text response
        output_items.append(self.create_text_output_item(text=answer, id="msg_1"))

        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        question = self._messages_to_question(request)
        answer, reasoning, tool_calls = self._run_agent(question)
        item_id = "msg_1"

        # 1. CoT reasoning
        if reasoning:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(id="reason_1", reasoning_text=reasoning),
            )

        # 2. Function call items and outputs
        for idx, (call_id, name, arguments, output) in enumerate(tool_calls):
            fc_item_id = f"fc_{idx}"
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_function_call_item(
                    id=fc_item_id,
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                ),
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_function_call_output_item(call_id=call_id, output=output),
            )

        # 3. Stream text delta for final answer
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=item_id)
            )

        # 4. Final text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=item_id),
        )


# Module-level setup for MLflow
mlflow.dspy.autolog()
_agent = MultiGenieDSPyAgent()
set_model(_agent)
