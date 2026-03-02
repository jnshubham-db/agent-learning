"""
Multi-Genie routing with DSPy — routes questions to the correct Genie space
(orders, returns, or products) using a DSPy ChainOfThought classifier.
"""

import os
import time

import dspy
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


# ---------------------------------------------------------------------------
# DSPy routing signature and module
# ---------------------------------------------------------------------------

class RouteQuestion(dspy.Signature):
    """Classify a user question into the correct department.

    Choose exactly one of: orders, returns, products.
    - orders: questions about order status, order history, shipping, delivery, revenue
    - returns: questions about returns, refunds, exchanges, return policies
    - products: questions about product catalog, inventory, pricing, product details
    """

    question: str = dspy.InputField(desc="The user's natural language question")
    department: str = dspy.OutputField(
        desc="One of: orders, returns, products"
    )


router = dspy.ChainOfThought(RouteQuestion)

genie_map: dict[str, GenieAgent] = {
    "orders": orders_agent,
    "returns": returns_agent,
    "products": products_agent,
}


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------

def route_and_query(question: str) -> str:
    """Route a question to the appropriate Genie space and return the answer.

    Uses DSPy ChainOfThought to classify the question into one of three
    departments (orders, returns, products), then queries the matching
    Genie space.

    Args:
        question: The user's natural language question.

    Returns:
        The Genie response from the selected department's space.
    """
    result = router(question=question)
    department = result.department.strip().lower()

    # Fallback: if the router returns something unexpected, default to orders
    if department not in genie_map:
        department = "orders"

    agent = genie_map[department]
    return f"[Routed to {department}] {agent.query(question)}"
