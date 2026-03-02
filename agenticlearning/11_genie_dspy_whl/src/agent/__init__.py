"""
Genie DSPy Agent — Simple Genie Space agent with DSPy CoT routing.
Packaged as a wheel for deployment to Databricks Model Serving.
"""

from .agent import GenieDSPyAgent

__all__ = ["GenieDSPyAgent"]
