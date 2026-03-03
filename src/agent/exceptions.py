"""
Agent Custom Exceptions

This module defines exception types for the ReAct Agent Core.

Classes:
    AgentError: Base exception for all agent errors
    MaxIterationsError: Raised when max_iterations is reached without completion
    ToolNotFoundError: Raised when the LLM selects a tool not in the registry

Feature: react-agent-core
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .models import AgentState


class AgentError(Exception):
    """Base exception for all ReAct Agent errors."""

    pass


class MaxIterationsError(AgentError):
    """
    Raised when the agent reaches max_iterations without completing the task.

    Attributes:
        state: The final AgentState at the time of the error
        iterations_completed: Number of iterations completed
    """

    def __init__(self, message: str, state: "AgentState", iterations_completed: int):
        super().__init__(message)
        self.state = state
        self.iterations_completed = iterations_completed


class ToolNotFoundError(AgentError):
    """
    Raised when the LLM selects a tool not present in the Tool Registry.

    Attributes:
        requested_tool: The tool name that was requested
        available_tools: List of tool names currently in the registry
    """

    def __init__(self, requested_tool: str, available_tools: List[str]):
        super().__init__(
            f"Tool '{requested_tool}' not found. Available: {available_tools}"
        )
        self.requested_tool = requested_tool
        self.available_tools = available_tools
