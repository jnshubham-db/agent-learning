"""
Multi-Genie DSPy Agent - Supervisor agent routing to Orders, Returns, and Products Genie spaces.
"""

from .agent import MultiGenieDSPyAgent
from .genie import route_and_query

__all__ = ["MultiGenieDSPyAgent", "route_and_query"]
