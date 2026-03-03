"""
Agent Logger

This module provides structured JSON logging for all ReAct Agent interactions.
Logs thoughts, actions, observations, and run completion summaries.

Classes:
    AgentLogger: Structured JSON logging for agent events

Feature: react-agent-core
"""

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .models import Action, AgentState, Observation


# Standard log record attributes to exclude from JSON output
_STD_RECORD_ATTRS = frozenset(
    {
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "pathname",
        "process", "processName", "relativeCreated", "stack_info",
        "exc_info", "exc_text", "thread", "threadName", "message",
    }
)


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON lines.

    Each log record is formatted as a single-line JSON object.
    Includes timestamp, level, event, iteration, and event-specific fields.

    Feature: react-agent-core, Property 13: Log entries contain required fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "unknown"),
            "iteration": getattr(record, "iteration", 0),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_ATTRS and value is not None:
                log_data[key] = value
        return json.dumps(log_data, default=str)


class AgentLogger:
    """
    Structured JSON logging for ReAct Agent events.

    Logs every thought, action, observation, and run completion with
    consistent JSON format including timestamp, level, event, and iteration.

    Attributes:
        logger: Underlying Python logger
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Feature: react-agent-core
    """

    def __init__(
        self,
        level: int = logging.INFO,
        output_path: Optional[str] = None,
        logger_name: str = "agent",
    ):
        """
        Initialize the agent logger.

        Args:
            level: Log level (default: logging.INFO)
            output_path: Optional file path for log output (stdout if None)
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = JsonFormatter()
        handler: logging.Handler
        if output_path:
            handler = logging.FileHandler(output_path, encoding="utf-8")
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def log_thought(self, iteration: int, thought: str) -> None:
        """
        Log a thought entry.

        Args:
            iteration: Current iteration number
            thought: The thought content from the LLM

        Feature: react-agent-core, Req 7.4
        """
        self.logger.info(
            thought,
            extra={"event": "agent_thought", "iteration": iteration, "thought": thought},
        )

    def log_action(self, iteration: int, action: "Action") -> None:
        """
        Log an action entry.

        Args:
            iteration: Current iteration number
            action: The Action being logged

        Feature: react-agent-core, Req 7.5
        """
        self.logger.info(
            f"Action: {action.tool_name}",
            extra={
                "event": "agent_action",
                "iteration": iteration,
                "tool_name": action.tool_name,
                "parameters": action.parameters,
                "reasoning": action.reasoning,
            },
        )

    def log_observation(self, iteration: int, observation: "Observation") -> None:
        """
        Log an observation entry.

        Args:
            iteration: Current iteration number
            observation: The Observation being logged

        Feature: react-agent-core, Req 7.6
        """
        extra: Dict[str, Any] = {
            "event": "agent_observation",
            "iteration": iteration,
            "tool_name": observation.tool_name,
            "success": observation.success,
        }
        if observation.success:
            extra["output"] = observation.output
        else:
            extra["error"] = observation.error

        self.logger.info(
            f"Observation: {observation.tool_name} success={observation.success}",
            extra=extra,
        )

    def log_run_complete(
        self, state: "AgentState", execution_time: float
    ) -> None:
        """
        Log run completion summary.

        Args:
            state: Final AgentState
            execution_time: Total execution time in seconds

        Feature: react-agent-core, Req 7.9
        """
        self.logger.info(
            "Run complete",
            extra={
                "event": "agent_run_complete",
                "iteration": state.iteration,
                "total_iterations": state.iteration,
                "is_complete": state.is_complete,
                "execution_time_s": execution_time,
            },
        )
