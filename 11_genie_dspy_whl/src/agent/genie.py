"""
Genie Space helper for querying Databricks Genie Spaces.
"""

import time

from databricks.sdk import WorkspaceClient


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Databricks Genie space with a natural language question.

    Starts a conversation, polls until the message is ready, and extracts
    the response from text or query attachments.

    Args:
        space_id: The Genie space ID (e.g., Sales Analytics space).
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
    while result.status in ("EXECUTING_QUERY", "SUBMITTED"):
        time.sleep(2)
        result = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )
    if result.attachments:
        for att in result.attachments:
            if att.text:
                return att.text.content
            if att.query:
                return f"Query: {att.query.query}\nDescription: {att.query.description}"
    return "No results found."
