"""
Genie Space query helper - queries a Databricks Genie space via WorkspaceClient with polling.
"""

import time


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Databricks Genie space with a natural language question.

    Uses WorkspaceClient().genie to start a conversation and polls until the
    message status is no longer EXECUTING_QUERY or SUBMITTED.

    Args:
        space_id: The Genie space ID (e.g., Sales Analytics or Customer Insights space)
        question: The user's natural language question

    Returns:
        The Genie response as text, from text attachments or query description.
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
    # Poll until query execution completes
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
