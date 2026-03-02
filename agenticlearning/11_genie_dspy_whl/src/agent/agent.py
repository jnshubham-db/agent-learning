"""
Genie DSPy Agent — ResponsesAgent that routes questions through DSPy CoT,
delegates data questions to a Databricks Genie Space, and falls back to
the LLM for general conversation.

Architecture:
  1. DSPy ChainOfThought (OrderQuery) decides if the question needs data.
  2. If yes  -> query the Genie Space and return the result.
  3. If no   -> generate a fallback answer with DSPy CoT.
  4. Wrapped in MLflow ResponsesAgent with predict() and predict_stream().
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

from .genie import GENIE_SPACE_ID, query_genie_space, route_question


# ---------------------------------------------------------------------------
# Fallback signature for non-data questions
# ---------------------------------------------------------------------------


class GeneralAnswer(dspy.Signature):
    """Answer a general customer question that does not require data lookup."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# ResponsesAgent implementation
# ---------------------------------------------------------------------------


class GenieDSPyAgent(ResponsesAgent):
    """DSPy-based agent that routes to a Genie Space for data questions.

    - Uses ``dspy.LM("databricks-meta-llama-3-3-70b-instruct")``
    - Routes via the ``OrderQuery`` ChainOfThought signature
    - Delegates data questions to the Genie Space identified by ``GENIE_SPACE_ID``
    - Falls back to a ``GeneralAnswer`` CoT for non-data questions
    """

    def __init__(self):
        super().__init__()
        # Configure DSPy LM
        self._lm = dspy.LM("databricks-meta-llama-3-3-70b-instruct")
        dspy.configure(lm=self._lm)

        # Genie space ID (from environment variable via genie module)
        self._space_id = GENIE_SPACE_ID

        # Fallback CoT for general questions
        self._general_cot = dspy.ChainOfThought(GeneralAnswer)

    # ----- helpers --------------------------------------------------------

    def _extract_question(self, request: ResponsesAgentRequest) -> str:
        """Extract the user's question text from the request input messages."""
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

    def _run(self, question: str) -> tuple[str, str]:
        """Route, query Genie or fallback, return (answer, reasoning)."""
        needs_data, reasoning = route_question(question)

        if needs_data:
            answer = query_genie_space(self._space_id, question)
        else:
            result = self._general_cot(question=question)
            answer = getattr(result, "answer", "") or ""
            fallback_rationale = getattr(result, "rationale", "") or ""
            if fallback_rationale:
                reasoning = f"{reasoning}\n{fallback_rationale}".strip()

        return answer, reasoning

    # ----- predict (non-streaming) ----------------------------------------

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction: route, query, and return Response."""
        question = self._extract_question(request)
        answer, reasoning = self._run(question)

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

    # ----- predict_stream (streaming) -------------------------------------

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction: yield reasoning, text deltas, and final item."""
        question = self._extract_question(request)
        answer, reasoning = self._run(question)

        text_id = str(uuid4())

        # 1. Yield reasoning item (CoT rationale)
        if reasoning:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_reasoning_item(
                    id=str(uuid4()), reasoning_text=reasoning
                ),
            )

        # 2. Stream answer text via create_text_delta
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=text_id)
            )

        # 3. Yield final text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=text_id),
        )


# ---------------------------------------------------------------------------
# Module-level: register agent with MLflow models-from-code
# ---------------------------------------------------------------------------

mlflow.dspy.autolog()

set_model(GenieDSPyAgent())
