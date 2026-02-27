"""
Customer Support Agent - LangChain tool-calling agent wrapped in MLflow ResponsesAgent.
Uses: order status, product info, refund processing.
"""

from typing import Generator
from uuid import uuid4

import mlflow
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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

# --- Tools ---


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of an order by order ID.
    
    Args:
        order_id: The order ID (e.g., ORD-12345678)
        
    Returns:
        Order status information.
    """
    # Simulated order status lookup
    orders = {
        "ORD-12345678": "Order ORD-12345678 is Shipped. Delivered on 2024-02-15.",
        "ORD-87654321": "Order ORD-87654321 is Processing. Expected delivery: 2024-02-28.",
    }
    return orders.get(order_id, f"Order {order_id} not found. Please verify the order ID.")


@tool
def get_product_info(product_id: str) -> str:
    """Get detailed information about a product by product ID.
    
    Args:
        product_id: The product ID (e.g., PROD-0001)
        
    Returns:
        Product details including name, price, and availability.
    """
    # Simulated product catalog
    products = {
        "PROD-0001": "Product PROD-0001: Wireless Headphones. Price: $79.99. In stock. 4.5★ rating.",
        "PROD-0002": "Product PROD-0002: Smart Watch. Price: $199.99. Low stock. 4.7★ rating.",
    }
    return products.get(product_id, f"Product {product_id} not found. Please verify the product ID.")


@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order.
    
    Args:
        order_id: The order ID to refund
        reason: Reason for the refund request
        
    Returns:
        Refund processing confirmation.
    """
    return (
        f"Refund request submitted for order {order_id}.\n"
        f"Reason: {reason}\n"
        f"Status: Pending approval. You will receive an email within 24-48 hours.\n"
        f"Refund ID: REF-{order_id}-001"
    )


# --- Agent Setup ---

SYSTEM_PROMPT = """You are a helpful customer support agent. You can:
- Look up order status by order ID
- Get product information by product ID  
- Process refund requests

Be professional, concise, and empathetic. Always confirm order/product IDs with the user when relevant."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
tools = [get_order_status, get_product_info, process_refund]
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
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    
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


class CustomerSupportResponsesAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapping the LangChain customer support agent."""

    def __init__(self):
        super().__init__()
        self.agent = agent_executor

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
        inp = getattr(request, "input", None) or (
            request.get("input", []) if isinstance(request, dict) else []
        )
        cc_msgs = to_chat_completions_input(
            [i.model_dump() if hasattr(i, "model_dump") else i for i in inp]
        )
        lc_messages = _chat_to_langchain_messages(cc_msgs)
        
        input_content = ""
        chat_history = []
        if lc_messages:
            last_msg = lc_messages[-1]
            input_content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            chat_history = lc_messages[:-1] if len(lc_messages) > 1 else []
        
        inv_input = {"input": input_content, "chat_history": chat_history}
        
        try:
            # Use agent.astream_events() for true token-level streaming when available
            # Fallback: collect messages from stream or invoke
            def _message_stream():
                try:
                    # AgentExecutor.stream(stream_mode="messages") yields messages as produced
                    stream_kw = {"stream_mode": "messages"} if hasattr(
                        self.agent, "stream"
                    ) and "messages" in str(getattr(self.agent, "stream", lambda **kw: None).__code__.co_varnames):
                        {}
                    else:
                        stream_kw = {}
                    for chunk in self.agent.stream(inv_input, **stream_kw):
                        if isinstance(chunk, BaseMessage):
                            yield chunk
                        elif isinstance(chunk, (list, tuple)):
                            for msg in chunk:
                                if isinstance(msg, BaseMessage):
                                    yield msg
                except (TypeError, AttributeError):
                    pass
                # Fallback: invoke and yield final AIMessage
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
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                    "role": "assistant",
                    "type": "message",
                },
            )


# --- Model for MLflow Models-from-Code ---

mlflow.langchain.autolog()
agent = CustomerSupportResponsesAgent()
set_model(agent)
