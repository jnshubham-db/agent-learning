# Genie DSPy Agent (WHL)

Sales Analytics Genie agent using **DSPy ReAct** (Chain-of-Thought) and **Databricks Genie**, packaged as a whl for deployment.

## Use Case

An agent that queries a Sales Analytics Genie Space using DSPy CoT. The agent uses a single `query_sales_data` tool that calls the Databricks Genie API to answer natural language questions about sales data.

## Package Structure

```
11_genie_dspy_whl/
├── src/
│   └── agent/
│       ├── __init__.py    # Exports GenieDSPyAgent, query_genie_space
│       ├── genie.py       # query_genie_space() helper
│       └── agent.py       # GenieDSPyAgent (ResponsesAgent subclass)
├── pyproject.toml
├── deploy.py              # Databricks notebook for build, log, evaluate, deploy
└── README.md
```

## Setup

1. **Replace the placeholder** in `src/agent/agent.py`:
   ```python
   SALES_GENIE_SPACE_ID = "<YOUR-SALES-GENIE-SPACE-ID>"
   ```

2. **Build and install with uv**:
   ```bash
   uv build --wheel
   uv pip install dist/genie_dspy_agent-0.1.0-py3-none-any.whl --system
   ```

## Deployment

Run `deploy.py` as a Databricks notebook. It will:

1. Install the package
2. Log the model to MLflow (`mlflow.pyfunc.log_model`)
3. Evaluate with `mlflow.genai.evaluate()` (Safety, RelevanceToQuery)
4. Register to Unity Catalog: `sjdatabricks.agents.genie_dspy_agent`
5. Provide deployment and querying examples

## Dependencies

- `dspy` – DSPy ReAct agent
- `mlflow>=3.0` – MLflow ResponsesAgent, logging, evaluation
- `databricks-sdk` – Genie API
- `pydantic>=2` – Request/response schemas
