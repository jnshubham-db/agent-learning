"""
Multi-Genie LangChain Agent - Supervisor agent orchestrating Sales Analytics and Customer Insights Genie spaces.

Uses LangChain create_tool_calling_agent + AgentExecutor with two Genie tools,
wrapped in MLflow ResponsesAgent for logging, tracing, and model serving.
"""

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
    to_chat_completions_input,
    output_to_responses_items_stream,
)

from .genie import query_genie_space

# Placeholder Genie space IDs - replace with actual IDs from create_genie_spaces
SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"


# --- Tools ---


@tool
def query_sales_data(question: str) -> str:
    """Query the Sales Analytics Genie space for sales data.

    Use for: revenue, orders, product performance, sales trends, regional sales,
    quarterly analytics, top customers by spend, cancelled orders.

    Args:
        question: The user's natural language question about sales data.

    Returns:
        Sales analytics response from the Genie space.
    """
    return query_genie_space(SALES_GENIE_SPACE_ID, question)


@tool
def query_customer_data(question: str) -> str:
    """Query the Customer Insights Genie space for customer and support data.

    Use for: support tickets, customer profiles, satisfaction scores,
    loyalty tiers, open tickets, demographics, refund requests.

    Args:
        question: The user's natural language question about customers or support.

    Returns:
        Customer insights response from the Genie space.
    """
    return query_genie_space(CUSTOMER_GENIE_SPACE_ID, question)


# --- Agent Setup ---

SYSTEM_PROMPT = """You are a business analytics assistant with access to two Genie spaces:

1. **Sales Analytics** (query_sales_data): Use for revenue, orders, product performance,
   sales trends, regional/quarterly analytics, top customers by spend.

2. **Customer Insights** (query_customer_data): Use for support tickets, customer profiles,
   satisfaction scores, loyalty tiers, open tickets, demographics.

For questions spanning both domains (e.g., "customers with open tickets who have highest order values"),
call BOTH tools and synthesize a coherent answer. Formulate clear natural language questions
for each tool. Be concise and data-driven."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
tools = [query_sales_data, query_customer_data]
agent_runnable = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)


# --- ResponsesAgent Wrapper ---


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


class MultiGenieLangChainAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapping the LangChain multi-Genie supervisor agent.

    Orchestrates two Genie spaces (Sales Analytics + Customer Insights) via
    query_sales_data and query_customer_data tools.
    """

    def __init__(self):
        super().__init__()
        self.agent = agent_executor

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction: collect output items from predict_stream."""
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
        """Streaming prediction using to_chat_completions_input and output_to_responses_items_stream."""
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


# --- Model for MLflow Models-from-Code ---

mlflow.langchain.autolog()
agent = MultiGenieLangChainAgent()
set_model(agent)
