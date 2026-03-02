# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **learning-path repository** for building AI agents on Databricks with MLflow. The specification in `prompt_v2.md` defines 14 tutorial topics covering agent development with DSPy, LangChain, LangGraph, and Genie — all using the same "Customer Order Support" scenario.

## Scenario & Data

All topics use the **sjdatabricks** catalog with three tables:
- `sjdatabricks.orders.order_details` — order id, customer name, product, quantity, status, order date
- `sjdatabricks.orders.returns` — return id, order id, reason, status, return date
- `sjdatabricks.orders.products` — product id, name, category, price, stock

Fake data setup and Genie Space creation are prerequisites for topics 7–14.

## Topic Structure

Topics 1–6 are pure DSPy/LangChain/LangGraph agents. Topics 7–14 add Genie Space integration.

- **Notebook topics** (1, 2, 4, 5, 7–10): Single Databricks notebook each
- **Whl topics** (3, 6, 11–14): Python package with `pyproject.toml`, `src/agent/` module, and a `deploy.py` script

Whl packages follow this layout:
```
<topic_folder>/
├── pyproject.toml
├── deploy.py              # mlflow log_model + UC registration
└── src/agent/
    ├── __init__.py
    ├── agent.py           # main agent / ChatModel
    ├── nodes.py           # LangGraph node functions (topics 3, 6)
    ├── tools.py           # tool definitions (topics 3, 6)
    └── genie.py           # GenieAgent wrappers (topics 11–14)
```

## Cross-Cutting Requirements

Every topic must include:
1. **MLflow >= 3** integration — log agent, parameters, metrics, artifacts
2. **Model Serving** — deployable to Databricks Model Serving
3. **Streaming output** — via `mlflow.pyfunc.ChatModel` with streaming support
4. **Evaluation** — LLM-as-a-judge cell to score agent answers
5. **Functional style** — pure functions, composition, readable notebooks

## Key Frameworks

- **DSPy**: Signatures + `ChainOfThought` for structured reasoning
- **LangChain**: `ChatDatabricks` LLM + tools via `langchain_databricks`
- **LangGraph**: `StateGraph` with classify → lookup → respond node pattern
- **Genie**: `databricks_agents.genie.GenieAgent` for natural-language SQL

## Reference

- The `sitemap.xml` indexes Databricks ML documentation pages
- The `.claude/skills/` directory contains extensive Claude Code skills for Databricks features (model serving, evaluation, asset bundles, etc.) — invoke these via slash commands when building topics
