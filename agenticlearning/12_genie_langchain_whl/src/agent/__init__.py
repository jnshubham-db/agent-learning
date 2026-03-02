"""
Genie LangChain Agent — Simple Genie Space agent with LangChain routing.
Packaged as a wheel for deployment to Databricks Model Serving.
"""

from .agent import GenieLangChainAgent

__all__ = ["GenieLangChainAgent"]
