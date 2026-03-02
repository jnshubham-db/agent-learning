"""
Genie Space helper and LangChain-based routing for the Simple Genie LangChain Agent.

- ChatDatabricks LLM configuration
- GenieAgent with space_id from the GENIE_SPACE_ID environment variable
- needs_order_data() — LLM-based function that decides if a question needs Genie data
"""

import os
import time

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

llm = ChatDatabricks(endpoint=LLM_ENDPOINT)


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
    from databricks.sdk import WorkspaceClient

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
# LLM-based routing
# ---------------------------------------------------------------------------


def needs_order_data(question: str) -> bool:
    """Determine whether a question requires order/sales data from the Genie Space.

    Uses the ChatDatabricks LLM to classify the question. Returns True if the
    question needs data lookup, False for general conversation.

    Args:
        question: The user's natural-language question.

    Returns:
        True if the question needs Genie data, False otherwise.
    """
    prompt = (
        "Does the following question require looking up order, sales, product, "
        "or return data from a database? Reply with exactly 'yes' or 'no'.\n\n"
        f"Question: {question}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return "yes" in resp.content.lower()
