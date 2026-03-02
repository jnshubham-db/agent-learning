"""
Customer Support Agent with LangChain + LangGraph.
Packages the classify -> lookup -> respond graph from Topic 5 into a deployable wheel.
"""

from .agent import LangChainLangGraphAgent

__all__ = ["LangChainLangGraphAgent"]
