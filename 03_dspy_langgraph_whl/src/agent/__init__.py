"""
Customer Support Agent with DSPy CoT and LangGraph.
Exports the agent class and compiled graph.
"""

from .agent import CustomerSupportAgent
from .nodes import build_graph

__all__ = ["CustomerSupportAgent", "build_graph", "graph"]

# Compiled graph for direct use (lazy to avoid import-time side effects)
graph = build_graph()
