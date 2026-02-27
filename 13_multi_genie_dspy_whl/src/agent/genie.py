"""
Genie Space helper for querying Databricks Genie spaces via WorkspaceClient API.
Uses polling to wait for query completion.
"""

import time
from typing import Optional


def query_genie_space(space_id: str, question: str) -> str:
    """
    Query a Databricks Genie space with a natural language question.

    Uses WorkspaceClient().genie API with polling until the query completes.
    Same implementation as Topics 11.

    Args:
        space_id: The Genie space ID (e.g., Sales Analytics or Customer Insights)
        question: The user's natural language question

    Returns:
        The Genie response as text (from attachments or query description),
        or "No results found." if no attachments.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    conversation = w.genie.start_conversation(space_id=space_id, content=question)
    result = w.genie.get_message(
        space_id=space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
    )

    # Poll until query completes
    while getattr(result, "status", "") in ("EXECUTING_QUERY", "SUBMITTED"):
        time.sleep(2)
        result = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            message_id=conversation.message_id,
        )

    # Extract result from attachments
    attachments = getattr(result, "attachments", None) or []
    for att in attachments:
        if getattr(att, "text", None) and att.text:
            return getattr(att.text, "content", str(att.text))
        if getattr(att, "query", None) and att.query:
            q = att.query
            return f"Query: {getattr(q, 'query', '')}\nDescription: {getattr(q, 'description', '')}"

    return "No results found."
