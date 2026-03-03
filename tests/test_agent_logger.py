"""
Tests for AgentLogger.

Feature: react-agent-core
"""

import json
import io
import logging

import pytest

from src.agent.agent_logger import AgentLogger, JsonFormatter
from src.agent.models import Action, Observation


class TestAgentLogger:
    """Tests for AgentLogger."""

    def test_log_thought_output(self):
        """Verify log_thought produces JSON with required fields."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("test_thought")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        al = AgentLogger(logger_name="test_thought")
        al.logger.handlers = [handler]
        al.log_thought(iteration=1, thought="I need to use the calculator")

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert "timestamp" in data
        assert "level" in data
        assert data["event"] == "agent_thought"
        assert data["iteration"] == 1
        assert data["thought"] == "I need to use the calculator"

    def test_log_action_output(self):
        """Verify log_action produces JSON with required fields."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        al = AgentLogger(logger_name="test_action")
        al.logger.handlers = [handler]
        action = Action("calculator", {"expression": "2+2"}, "Compute")
        al.log_action(iteration=1, action=action)

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["event"] == "agent_action"
        assert data["tool_name"] == "calculator"
        assert data["parameters"] == {"expression": "2+2"}
        assert data["reasoning"] == "Compute"

    def test_log_observation_output(self):
        """Verify log_observation produces JSON with required fields."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        al = AgentLogger(logger_name="test_obs")
        al.logger.handlers = [handler]
        obs = Observation("calculator", True, "4", None)
        al.log_observation(iteration=1, observation=obs)

        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["event"] == "agent_observation"
        assert data["tool_name"] == "calculator"
        assert data["success"] is True
        assert data["output"] == "4"
