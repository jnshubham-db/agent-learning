# Databricks notebook source
# MAGIC %md
# MAGIC # Topic 10: Multi-Genie Agent with LangChain (without whl)
# MAGIC
# MAGIC This notebook implements a **supervisor agent** using **LangChain** that orchestrates two Databricks Genie agents:
# MAGIC - **Sales Analytics Genie** — Revenue, orders, product performance
# MAGIC - **Customer Insights Genie** — Support tickets, customer profiles, satisfaction
# MAGIC
# MAGIC **Architecture:**
# MAGIC - LangChain `create_tool_calling_agent` + `AgentExecutor` with two Genie tools
# MAGIC - `query_sales_genie` and `query_customer_genie` via `WorkspaceClient().genie` API
# MAGIC - System prompt guides routing to the appropriate Genie space(s)
# MAGIC - MLflow ResponsesAgent with `to_chat_completions_input()` and `output_to_responses_items_stream()`
# MAGIC - Evaluation, deployment, and querying
# MAGIC
# MAGIC **Prerequisites:** Run `create_genie_spaces` to create the Genie spaces, then paste the Space IDs below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install uv
# MAGIC %restart_python

# COMMAND ----------

import subprocess
subprocess.check_call([
    "uv", "pip", "install",
    "langchain", "langchain-community", "langchain-databricks", "mlflow>=3.0", "databricks-sdk", "pydantic>=2", "databricks-openai",
    "--system",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Genie Space IDs
# MAGIC
# MAGIC Replace with your actual Space IDs from `create_genie_spaces`.

# COMMAND ----------

SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
CUSTOMER_GENIE_SPACE_ID = "<YOUR-CUSTOMER-GENIE-SPACE-ID>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Genie Query Helper and Tools

# COMMAND ----------

import time
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


def query_genie_space(space_id: str, question: str) -> str:
    """Query a Databricks Genie space with a natural language question."""
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


@tool
def query_sales_genie(question: str) -> str:
    """Query the Sales Analytics Genie space. Use for: revenue, orders, product performance, sales trends, regional sales, quarterly analytics. Pass the user's question directly."""
    return query_genie_space(SALES_GENIE_SPACE_ID, question)


@tool
def query_customer_genie(question: str) -> str:
    """Query the Customer Insights Genie space. Use for: support tickets, customer profiles, satisfaction scores, loyalty tiers, open tickets. Pass the user's question directly."""
    return query_genie_space(CUSTOMER_GENIE_SPACE_ID, question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build LangChain Agent
# MAGIC
# MAGIC System prompt tells the agent which Genie space to use for each type of query. For cross-space questions (e.g., "customers with open tickets and highest order values"), use both tools and combine results.

# COMMAND ----------

SYSTEM_PROMPT = """You are a business analytics assistant with access to two Genie spaces:

1. **Sales Analytics** (query_sales_genie): Use for revenue, orders, product performance, sales trends, regional/quarterly analytics.
2. **Customer Insights** (query_customer_genie): Use for support tickets, customer profiles, satisfaction scores, loyalty tiers.

For questions spanning both domains (e.g., "customers with open tickets who have highest order values"), call BOTH tools and synthesize a coherent answer.
Formulate clear natural language questions for each tool. Be concise and data-driven."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
tools = [query_sales_genie, query_customer_genie]
agent_runnable = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wrap in MLflow ResponsesAgent
# MAGIC
# MAGIC Implement `predict()` and `predict_stream()` using `to_chat_completions_input()` and `output_to_responses_items_stream()`.

# COMMAND ----------

def _chat_to_langchain_messages(cc_input: list) -> list[BaseMessage]:
    """Convert chat completion format to LangChain messages."""
    from langchain_core.messages import SystemMessage

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


class MultiGenieLangChainAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapping the LangChain multi-Genie supervisor agent."""

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


agent = MultiGenieLangChainAgent()
set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Local Test

# COMMAND ----------

test_input = {
    "input": [{"role": "user", "content": "What was total revenue last month?"}],
}
result = agent.predict(test_input)
print("Test response (first text output):")
for item in result.output:
    if isinstance(item, dict):
        content = item.get("content", [])
        for c in content:
            if isinstance(c, dict) and "text" in c:
                print(c["text"])
                break
    elif hasattr(item, "content"):
        for c in item.content or []:
            if hasattr(c, "text"):
                print(c.text)
                break

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log Model to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="multi_genie_langchain_agent"):
    logged_info = mlflow.pyfunc.log_model(
        python_model=agent,
        name="agent",
        artifact_path="model",
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = logged_info.model_uri
    print(f"Logged model. Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Evaluation with mlflow.genai.evaluate
# MAGIC
# MAGIC Evaluate with cross-space queries: single-space (revenue, tickets) and multi-space (customers with tickets and order values).

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

eval_dataset = [
    {"input": [{"role": "user", "content": "What was total revenue last month?"}]},
    {"input": [{"role": "user", "content": "How many open support tickets do we have?"}]},
    {"input": [{"role": "user", "content": "Which customers with open tickets have the highest order values?"}]},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda row: agent.predict(row),
    scorers=[Safety(), RelevanceToQuery()],
)

print(f"Evaluation run ID: {results.run_id}")
print(results.tables.get("eval_results_table", "N/A"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Register to Unity Catalog and Deploy

# COMMAND ----------

UC_MODEL_NAME = "sjdatabricks.agents.multi_genie_langchain"

mlflow.set_registry_uri("databricks-uc")
uc_registered = mlflow.register_model(
    model_uri=model_uri,
    name=UC_MODEL_NAME,
)
print(f"Registered model: {uc_registered.name} (version {uc_registered.version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving Endpoint

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint_name = "multi-genie-langchain-agent"

try:
    endpoint = client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": UC_MODEL_NAME,
                    "entity_version": uc_registered.version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
    )
    print(f"Endpoint created: {endpoint_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Endpoint {endpoint_name} already exists. Update if needed.")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Query Examples (Streaming and Non-Streaming)

# COMMAND ----------

from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()
input_msgs = [{"role": "user", "content": "What was total revenue last month?"}]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Streaming Query (Responses API)

# COMMAND ----------

response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    max_output_tokens=512,
)
print("Non-streaming response:")
for item in getattr(response, "output", response):
    if hasattr(item, "content"):
        for c in (item.content or []):
            if hasattr(c, "text") and c.text:
                print(c.text)
    elif hasattr(item, "text"):
        print(item.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Query (Responses API)

# COMMAND ----------

streaming_response = client.responses.create(
    model=endpoint_name,
    input=input_msgs,
    stream=True,
    max_output_tokens=512,
)
print("Streaming response:")
for chunk in streaming_response:
    if hasattr(chunk, "output"):
        for item in (chunk.output or []):
            if hasattr(item, "content"):
                for c in (item.content or []):
                    if hasattr(c, "delta") and c.delta:
                        print(c.delta, end="", flush=True)
            elif hasattr(item, "delta") and item.delta:
                print(item.delta, end="", flush=True)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local Query (Before Deployment)

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(model_uri)
local_response = loaded.predict({"input": input_msgs})
print("Local response:")
for item in local_response.output:
    if isinstance(item, dict) and item.get("content"):
        for c in item["content"]:
            if isinstance(c, dict) and c.get("text"):
                print(c["text"])
                break

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Set Model for MLflow

# COMMAND ----------

mlflow.models.set_model(agent)
print("Model set. Agent ready for deployment.")
