"""
Multi-Genie routing with LangChain — routes questions to the correct Genie space
(orders, returns, or products) using a ChatDatabricks LLM classifier.
"""

import os
import time

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

from databricks.sdk import WorkspaceClient


# ---------------------------------------------------------------------------
# Genie Space helper
# ---------------------------------------------------------------------------

class GenieAgent:
    """Wraps a Databricks Genie space for querying via the WorkspaceClient API."""

    def __init__(self, space_id: str, name: str):
        self.space_id = space_id
        self.name = name

    def query(self, question: str) -> str:
        """Query this Genie space with a natural language question.

        Starts a conversation, polls until the message is ready, and extracts
        the response from text or query attachments.

        Args:
            question: The user's natural language question.

        Returns:
            The Genie response as text, or "No results found." if empty.
        """
        w = WorkspaceClient()
        conversation = w.genie.start_conversation(
            space_id=self.space_id, content=question
        )
        result = w.genie.get_message(
            space_id=self.space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )
        while getattr(result, "status", "") in ("EXECUTING_QUERY", "SUBMITTED"):
            time.sleep(2)
            result = w.genie.get_message(
                space_id=self.space_id,
                conversation_id=conversation.conversation_id,
                message_id=conversation.message_id,
            )
        attachments = getattr(result, "attachments", None) or []
        for att in attachments:
            if getattr(att, "text", None) and att.text:
                return getattr(att.text, "content", str(att.text))
            if getattr(att, "query", None) and att.query:
                q = att.query
                return (
                    f"Query: {getattr(q, 'query', '')}\n"
                    f"Description: {getattr(q, 'description', '')}"
                )
        return "No results found."


# ---------------------------------------------------------------------------
# Genie agent instances (space IDs from environment variables)
# ---------------------------------------------------------------------------

orders_agent = GenieAgent(
    space_id=os.environ.get("ORDER_SPACE_ID", "<YOUR-ORDER-SPACE-ID>"),
    name="orders",
)
returns_agent = GenieAgent(
    space_id=os.environ.get("RETURNS_SPACE_ID", "<YOUR-RETURNS-SPACE-ID>"),
    name="returns",
)
products_agent = GenieAgent(
    space_id=os.environ.get("PRODUCTS_SPACE_ID", "<YOUR-PRODUCTS-SPACE-ID>"),
    name="products",
)

genie_map: dict[str, GenieAgent] = {
    "orders": orders_agent,
    "returns": returns_agent,
    "products": products_agent,
}


# ---------------------------------------------------------------------------
# LangChain-based question classifier
# ---------------------------------------------------------------------------

_llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

CLASSIFY_SYSTEM_PROMPT = """You are a routing classifier. Given a user question, classify it into exactly one department.

Respond with ONLY one word — the department name. No explanation, no punctuation.

Departments:
- orders: questions about order status, order history, shipping, delivery, revenue, sales
- returns: questions about returns, refunds, exchanges, return policies, return rates
- products: questions about product catalog, inventory, pricing, product details, stock levels"""


def classify_question(question: str) -> str:
    """Classify a question into one of: orders, returns, products.

    Uses ChatDatabricks LLM with a system prompt to perform single-word classification.

    Args:
        question: The user's natural language question.

    Returns:
        One of "orders", "returns", or "products".
    """
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = _llm.invoke(messages)
    department = response.content.strip().lower()

    # Fallback: if the LLM returns something unexpected, default to orders
    if department not in genie_map:
        department = "orders"

    return department
