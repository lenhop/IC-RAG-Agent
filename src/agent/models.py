"""
Agent Data Models

This module provides data models for the ReAct Agent Core: AgentState,
Action, and Observation. These models structure the agent's reasoning
process and enable traceability.

Classes:
    Action: Represents a selected tool invocation
    Observation: Represents the result of an executed action
    AgentState: Tracks the agent's state across iterations

Feature: react-agent-core
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Action:
    """
    Represents a selected tool invocation by the LLM.

    Attributes:
        tool_name: Name of the tool to invoke
        parameters: Parameters to pass to the tool
        reasoning: LLM's reasoning for choosing this action
    """

    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Produce a JSON-serializable dictionary."""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


@dataclass
class Observation:
    """
    Represents the result of an executed action.

    Attributes:
        tool_name: Name of the tool that produced this observation
        success: Whether the tool execution succeeded
        output: Tool output (None if failed)
        error: Error message (None if succeeded)
        metadata: Optional metadata (e.g. stub=True for stub tools)
    """

    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Produce a JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d


@dataclass
class AgentState:
    """
    Tracks the agent's state across ReAct loop iterations.

    Attributes:
        query: The original user query
        iteration: Current iteration count (starts at 0)
        is_complete: Whether the task has been resolved
        history: Ordered list of (thought, action, observation) tuples
        final_answer: The final answer text when is_complete (set by LLM)
    """

    query: str
    iteration: int = 0
    is_complete: bool = False
    history: List[Tuple[str, "Action", "Observation"]] = field(default_factory=list)
    final_answer: Optional[str] = None

    def append_history(
        self, thought: str, action: Action, observation: Observation
    ) -> None:
        """Append a new thought/action/observation tuple to history."""
        self.history.append((thought, action, observation))

    def increment_iteration(self) -> None:
        """Increment the iteration counter by 1."""
        self.iteration += 1

    def to_dict(self) -> Dict[str, Any]:
        """Produce a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "iteration": self.iteration,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
            "history": [
                {"thought": t, "action": a.to_dict(), "observation": o.to_dict()}
                for t, a, o in self.history
            ],
        }
