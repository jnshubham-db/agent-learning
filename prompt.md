Identify clean usecase which should be easy to follow and understand.
For Genie Agents, create the fake data in the sjdatabricks catalog and create genies spaces on top of it. Just create a script which creates the fake data and then creates the genies spaces on top of it.

I want to learn the below topics. The idea is to create simple databricks notebook for each of the below topics.

Simple functional programming concepts.

Only where whl file is required, instead of notebooks this will be folder with multiple code files when bundled with whl.

The idea is to learn the end-to-end lifecycle of agent development with DSPy and langchain and genie.

In all the below topics I also want mlflow integration latest version.
All the below topics should also be deployable as a model serving in databricks.

Important NOte:
When we have DSPy CoT it does not exposes the streaming of the llm reasoning process. SO in all the below topics I also want to understand how can we get the streaming output from the agents on model serving to be consumed in the frontend apps.


Topics to learn:
1. Agent with DSPy without whl.
2. Agent with DSPy and langgraph with 2-3 nodes and cot withou whl.
3. Agent with DSPy and langgraph with 2-3 nodes and cot with whl.
4. Agent with langchain without whl.
5. Agent with langchain and langgraph with 2-3 nodes and cot without whl.
6. Agent with langchain and langgraph with 2-3 nodes and cot with whl.
7. Simple genie Agent with DSPy without whl.
8. Simple genie Agent with langchain without whl.
9. Agent with multiple genie agents and a main agent with DSPy without whl.
10. Agent with multiple genie agents and a main agent with langchain without whl.
11. Simple genie Agent with DSPy with whl.
12. Simple genie Agent with langchain with whl.
13. Agent with multiple genie agents and a main agent with DSPy with whl.
14. Agent with multiple genie agents and a main agent with langchain with whl.



Some very important resources to follow:
1. We need to follow mlflow best practices.
2. Follow model serving best practices.
3. Have proper evaluation setup and llm as a judge setup in the above topics.
4. Get all the data from the documentation links provided below:
https://docs.databricks.com/aws/en/machine-learning/
All the links are added in the sitemap.xml file. Get all documentation from there and create notebooks for each of the topics.

Very imp links:
https://docs.databricks.com/aws/en/generative-ai/guide/agents-dev-workflow

https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent
https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app
https://docs.databricks.com/aws/en/generative-ai/agent-framework/query-agent
https://docs.databricks.com/aws/en/mlflow3/genai/
https://docs.databricks.com/aws/en/generative-ai/dspy/

https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie