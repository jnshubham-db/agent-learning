"""
Genie LangChain Agent — ResponsesAgent that routes questions through an
LLM-based classifier, delegates data questions to a Databricks Genie Space,
and falls back to the LLM for general conversation.

Architecture:
  1. needs_order_data() (LangChain ChatDatabricks) decides if the question needs data.
  2. If yes  -> query the Genie Space and return the result.
  3. If no   -> generate a conversational answer with the LLM.
  4. Wrapped in MLflow ResponsesAgent with predict() and predict_stream().
"""

from typing import Generator
from uuid import uuid4

import mlflow
from langchain_core.messages import HumanMessage
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from .genie import GENIE_SPACE_ID, llm, needs_order_data, query_genie_space


# ---------------------------------------------------------------------------
# ResponsesAgent implementation
# ---------------------------------------------------------------------------


class GenieLangChainAgent(ResponsesAgent):
    """LangChain-based agent that routes to a Genie Space for data questions.

    - Uses ``ChatDatabricks("databricks-meta-llama-3-3-70b-instruct")``
    - Routes via ``needs_order_data()`` LLM classifier
    - Delegates data questions to the Genie Space identified by ``GENIE_SPACE_ID``
    - Falls back to the LLM for general questions
    """

    def __init__(self):
        super().__init__()
        self._space_id = GENIE_SPACE_ID
        self._llm = llm

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

    def _run(self, question: str) -> str:
        """Route, query Genie or fallback, return the answer string."""
        if needs_order_data(question):
            return query_genie_space(self._space_id, question)
        else:
            resp = self._llm.invoke([
                HumanMessage(
                    content=(
                        "You are a helpful customer support assistant. "
                        "Answer the following question concisely.\n\n"
                        f"Question: {question}"
                    )
                )
            ])
            return resp.content

    # ----- predict (non-streaming) ----------------------------------------

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction: route, query, and return Response."""
        question = self._extract_question(request)
        answer = self._run(question)

        output_items = [
            self.create_text_output_item(text=answer, id=str(uuid4()))
        ]

        custom = getattr(request, "custom_inputs", None) or (
            request.get("custom_inputs") if isinstance(request, dict) else None
        )
        return ResponsesAgentResponse(output=output_items, custom_outputs=custom)

    # ----- predict_stream (streaming) -------------------------------------

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction: yield text deltas and final output item."""
        question = self._extract_question(request)
        answer = self._run(question)

        text_id = str(uuid4())

        # Stream answer text via create_text_delta
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            delta = answer[i : i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=delta, item_id=text_id)
            )

        # Yield final text output item
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=text_id),
        )


# ---------------------------------------------------------------------------
# Module-level: register agent with MLflow models-from-code
# ---------------------------------------------------------------------------

mlflow.langchain.autolog()

set_model(GenieLangChainAgent())
