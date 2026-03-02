"""
Customer Order Support Agent - DSPy + LangGraph, packaged as a wheel.
Classify -> Lookup -> Respond pipeline using DSPy Chain-of-Thought nodes.
"""

from .agent import DSPyLangGraphAgent, build_graph

__all__ = ["DSPyLangGraphAgent", "build_graph"]
