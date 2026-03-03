"""
ReAct Agent Core

This module provides the ReAct Agent Core - the reasoning engine that powers
the IC-Agent system. It implements the Thought -> Action -> Observation loop
for autonomous task resolution.

Usage:
    >>> from src.agent import ReActAgent, AgentState, AgentLogger, Action, Observation
    >>> agent = ReActAgent(llm=my_llm, tools=[...])
    >>> result = agent.run("What is the inventory for SKU ABC123?")

Feature: react-agent-core
"""

# Add libs/ai-toolkit to sys.path for ai_toolkit imports
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
_ai_toolkit_path = _project_root / "libs" / "ai-toolkit"
if _ai_toolkit_path.exists() and str(_ai_toolkit_path) not in sys.path:
    sys.path.insert(0, str(_ai_toolkit_path))

from .models import Action, AgentState, Observation
from .exceptions import AgentError, MaxIterationsError, ToolNotFoundError
from .agent_logger import AgentLogger
from .react_agent import ReActAgent

__all__ = [
    "ReActAgent",
    "AgentState",
    "AgentLogger",
    "Action",
    "Observation",
    "AgentError",
    "MaxIterationsError",
    "ToolNotFoundError",
]
