"""
Tests for agent data models.

Feature: react-agent-core
"""

import json
import pytest

from src.agent.models import Action, AgentState, Observation


class TestAction:
    """Tests for Action dataclass."""

    def test_create_action(self):
        action = Action(
            tool_name="calculator",
            parameters={"expression": "2+2"},
            reasoning="Need to compute",
        )
        assert action.tool_name == "calculator"
        assert action.parameters == {"expression": "2+2"}
        assert action.reasoning == "Need to compute"

    def test_to_dict(self):
        action = Action(
            tool_name="test_tool",
            parameters={"a": 1},
            reasoning="test",
        )
        d = action.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["parameters"] == {"a": 1}
        assert d["reasoning"] == "test"
        assert json.dumps(d)  # JSON-serializable


class TestObservation:
    """Tests for Observation dataclass."""

    def test_create_success_observation(self):
        obs = Observation(
            tool_name="calc",
            success=True,
            output=4,
            error=None,
        )
        assert obs.success is True
        assert obs.output == 4
        assert obs.error is None

    def test_create_failure_observation(self):
        obs = Observation(
            tool_name="calc",
            success=False,
            output=None,
            error="Division by zero",
        )
        assert obs.success is False
        assert obs.error == "Division by zero"

    def test_to_dict_with_metadata(self):
        obs = Observation(
            tool_name="stub",
            success=True,
            output={"data": 1},
            metadata={"stub": True},
        )
        d = obs.to_dict()
        assert d["metadata"] == {"stub": True}


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_initial_state(self):
        state = AgentState(query="test query")
        assert state.iteration == 0
        assert state.is_complete is False
        assert state.history == []
        assert state.final_answer is None

    def test_append_history(self):
        state = AgentState(query="q")
        action = Action("t", {}, "r")
        obs = Observation("t", True, "out", None)
        state.append_history("thought", action, obs)
        assert len(state.history) == 1
        t, a, o = state.history[0]
        assert t == "thought"
        assert a.tool_name == "t"
        assert o.output == "out"

    def test_increment_iteration(self):
        state = AgentState(query="q")
        assert state.iteration == 0
        state.increment_iteration()
        assert state.iteration == 1
        state.increment_iteration()
        assert state.iteration == 2

    def test_to_dict(self):
        state = AgentState(query="q", iteration=1)
        state.append_history(
            "t",
            Action("x", {}, "r"),
            Observation("x", True, 1, None),
        )
        d = state.to_dict()
        assert d["query"] == "q"
        assert d["iteration"] == 1
        assert len(d["history"]) == 1
        assert json.dumps(d)  # JSON-serializable
