"""
Genie LangChain Agent - Sales Analytics agent using LangChain with ChatDatabricks and Genie tool.
"""

import os
from typing import Generator
from uuid import uuid4

import mlflow
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
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

from .genie import query_genie_space


def _make_query_sales_data_tool(space_id: str):
    """Create the query_sales_data tool bound to the given Genie space_id."""

    @tool
    def query_sales_data(question: str) -> str:
        """Query the Sales Analytics Genie space for sales data. Use for questions about revenue, products, regions, quarters, sales trends, and analytics. Pass the user's question directly."""
        return query_genie_space(space_id, question)

    return query_sales_data


SYSTEM_PROMPT = """You are a Sales Analytics assistant. You have access to a Genie space with sales data.
Use the query_sales_data tool to answer questions about sales, revenue, products, regions, and trends.
Formulate clear, natural language questions for the tool based on what the user asks.
Be concise and data-driven in your responses."""


def _chat_to_langchain_messages(cc_input: list) -> list[BaseMessage]:
    """Convert chat completion format to LangChain messages."""
    messages = []
    for m in cc_input:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


mlflow.langchain.autolog()


class GenieLangChainAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapping the LangChain Genie sales agent."""

    def __init__(self, space_id: str | None = None):
        super().__init__()
        self._space_id = space_id or os.environ.get("GENIE_SPACE_ID", "<YOUR-SALES-GENIE-SPACE-ID>")
        tool_fn = _make_query_sales_data_tool(self._space_id)
        tools = [tool_fn]
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
        agent_runnable = create_tool_calling_agent(llm, tools, prompt)
        self.agent = AgentExecutor(
            agent=agent_runnable,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
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
        inp = getattr(request, "input", None)
        if inp is None and isinstance(request, dict):
            inp = request.get("input", [])
        items = [i.model_dump() if hasattr(i, "model_dump") else i for i in inp]
        cc_msgs = to_chat_completions_input(items)
        lc_messages = _chat_to_langchain_messages(cc_msgs)

        input_content = ""
        chat_history = []
        if lc_messages:
            last_msg = lc_messages[-1]
            input_content = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )
            chat_history = lc_messages[:-1] if len(lc_messages) > 1 else []

        inv_input = {"input": input_content, "chat_history": chat_history}

        try:
            def _message_stream():
                try:
                    stream_kw = (
                        {"stream_mode": "messages"}
                        if hasattr(self.agent, "stream")
                        and "messages"
                        in str(
                            getattr(
                                self.agent, "stream", lambda **kw: None
                            ).__code__.co_varnames
                        )
                        else {}
                    )
                    for chunk in self.agent.stream(inv_input, **stream_kw):
                        if isinstance(chunk, BaseMessage):
                            yield chunk
                        elif isinstance(chunk, (list, tuple)):
                            for msg in chunk:
                                if isinstance(msg, BaseMessage):
                                    yield msg
                except (TypeError, AttributeError):
                    pass
                result = self.agent.invoke(inv_input)
                if "output" in result:
                    yield AIMessage(content=result["output"])
                elif "messages" in result:
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and msg.content:
                            yield msg
                            break

            yield from output_to_responses_items_stream(_message_stream())
        except Exception as e:
            result = self.agent.invoke(inv_input)
            output_text = result.get("output", str(e))
            item_id = str(uuid4())
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": item_id,
                    "content": [
                        {
                            "type": "output_text",
                            "text": output_text,
                            "annotations": [],
                        }
                    ],
                    "role": "assistant",
                    "type": "message",
                },
            )
