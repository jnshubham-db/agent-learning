"""
Genie Space helper and DSPy-based routing for the Simple Genie DSPy Agent.

- GenieAgent initialised with space_id from GENIE_SPACE_ID environment variable
- OrderQuery DSPy signature for question -> needs_data routing
- ChainOfThought router that decides whether a question requires Genie data
- route_question() returns (needs_data: bool, reasoning: str)
"""

import os
import time

import dspy
from databricks.sdk import WorkspaceClient


# ---------------------------------------------------------------------------
# Genie Space query helper
# ---------------------------------------------------------------------------

GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID", "<YOUR-GENIE-SPACE-ID>")


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Databricks Genie space with a natural language question.

    Starts a conversation, polls until the message is ready, and extracts
    the response from text or query attachments.

    Args:
        space_id: The Genie space ID.
        question: The user's natural language question.

    Returns:
        The Genie response as text (from attachments or query description).
        Returns "No results found." if no usable attachment is present.
    """
    w = WorkspaceClient()
    conversation = w.genie.start_conversation(space_id=space_id, content=question)
    result = w.genie.get_message(
        space_id=space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
    )
    while getattr(result, "status", "") in ("EXECUTING_QUERY", "SUBMITTED"):
        time.sleep(2)
        result = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )
    attachments = getattr(result, "attachments", None) or []
    for att in attachments:
        if getattr(att, "text", None) and att.text:
            return getattr(att.text, "content", str(att.text))
        if getattr(att, "query", None) and att.query:
            q = att.query
            return f"Query: {getattr(q, 'query', '')}\nDescription: {getattr(q, 'description', '')}"
    return "No results found."


# ---------------------------------------------------------------------------
# DSPy signature and ChainOfThought router
# ---------------------------------------------------------------------------


class OrderQuery(dspy.Signature):
    """Determine whether a customer question requires order/sales data from the Genie space.

    Analyse the question and decide if it needs data lookup (e.g. order status,
    revenue figures, product inventory) or can be answered with general knowledge.
    """

    question: str = dspy.InputField(desc="The user's natural-language question")
    needs_data: bool = dspy.OutputField(
        desc="True if the question requires data from the Genie space, False otherwise"
    )


# ChainOfThought router produces reasoning before the boolean decision
_router = dspy.ChainOfThought(OrderQuery)


def route_question(question: str) -> tuple[bool, str]:
    """Route a question through the DSPy CoT router.

    Args:
        question: The user's natural-language question.

    Returns:
        A tuple of (needs_data, reasoning).
        - needs_data: True when the question should be sent to Genie.
        - reasoning: The chain-of-thought rationale from the router.
    """
    result = _router(question=question)
    needs_data = bool(result.needs_data)
    reasoning = getattr(result, "rationale", "") or getattr(result, "reasoning", "") or ""
    return needs_data, reasoning
