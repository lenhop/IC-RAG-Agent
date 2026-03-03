"""
ReAct Agent Core

This module implements the ReAct (Reasoning + Acting) agent that executes
the Thought -> Action -> Observation loop for autonomous task resolution.

Classes:
    ReActAgent: Core agent class with tool registry and ReAct loop

Feature: react-agent-core
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Import after sys.path is set in __init__
from ai_toolkit.tools import BaseTool, ToolExecutor, ToolResult

from .agent_logger import AgentLogger
from .exceptions import AgentError, MaxIterationsError, ToolNotFoundError
from .models import Action, AgentState, Observation


def _invoke_llm(llm: Any, prompt: str) -> str:
    """
    Invoke the LLM with a prompt string and return the response as a string.

    Supports ModelManager-created models (LangChain BaseChatModel) and
    simple callables.

    Args:
        llm: LLM instance (callable or object with invoke method)
        prompt: Prompt string

    Returns:
        Response string from the LLM
    """
    if callable(llm) and not hasattr(llm, "invoke"):
        return llm(prompt)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


class ReActAgent:
    """
    ReAct Agent that executes the Thought -> Action -> Observation loop.

    Uses an LLM to reason, select tools, and iteratively resolve user queries.
    Integrates with ai-toolkit ToolExecutor for tool execution.

    Feature: react-agent-core
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        logger: Optional[AgentLogger] = None,
        executor: Optional[ToolExecutor] = None,
    ):
        """
        Initialize the ReAct agent.

        Args:
            llm: LLM instance (ModelManager-compatible or callable)
            tools: Initial list of tools (default: empty)
            max_iterations: Maximum loop iterations (default: 10)
            logger: Optional AgentLogger (creates default if None)
            executor: Optional ToolExecutor (creates default if None)

        Raises:
            AgentError: If duplicate tool names are provided
        """
        self._llm = llm
        self._max_iterations = max_iterations
        self._logger = logger or AgentLogger()
        self._executor = executor or ToolExecutor()
        self._registry: Dict[str, BaseTool] = {}

        for tool in tools or []:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the registry.

        Args:
            tool: BaseTool instance to register

        Raises:
            AgentError: If a tool with the same name already exists
        """
        if tool.name in self._registry:
            raise AgentError(f"Tool '{tool.name}' is already registered")
        self._registry[tool.name] = tool

    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove
        """
        self._registry.pop(tool_name, None)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return list of tool schemas for all registered tools.

        Returns:
            List of schema dicts from each tool's to_schema()
        """
        return [tool.to_schema() for tool in self._registry.values()]

    def _build_prompt(self, state: AgentState) -> str:
        """
        Build the LLM prompt from current state and tool schemas.

        Uses the exact format specified in the design document.

        Feature: react-agent-core, Property 15: LLM prompt contains required context
        """
        tool_schemas = json.dumps(self.list_tools(), indent=2)

        history_lines = []
        for thought, action, observation in state.history:
            history_lines.append(f"Thought: {thought}")
            history_lines.append(f"Action: {action.tool_name}")
            history_lines.append(f"Parameters: {json.dumps(action.parameters)}")
            history_lines.append(
                f"Observation: {observation.output if observation.success else observation.error}"
            )
        history_str = "\n".join(history_lines) if history_lines else "(none)"

        return f"""You are a ReAct agent. Reason step by step and use tools to answer the query.

Available tools:
{tool_schemas}

Conversation history:
{history_str}

Current query: {state.query}

Respond in this exact format:
Thought: <your reasoning>
Action: <tool_name>
Parameters: <json parameters>

Or if the task is complete:
Thought: <your reasoning>
Final Answer: <your answer>
"""

    def _parse_llm_response(
        self, response: str
    ) -> Tuple[str, Optional[Action], bool, Optional[str]]:
        """
        Parse LLM output into (thought, action, is_complete, final_answer).

        Returns:
            Tuple of (thought, Action or None, is_complete, final_answer or None)

        Feature: react-agent-core, Property 16: LLM response parsing correctness
        """
        thought = ""
        action = None
        is_complete = False
        final_answer = None

        # Extract thought (use *? to allow empty thought before Action:/Final Answer:)
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=Action:|Final Answer:|\Z)", response, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if final_match:
            is_complete = True
            final_answer = final_match.group(1).strip()
            return (thought, None, True, final_answer)

        # Parse Action and Parameters
        action_match = re.search(r"Action:\s*(\w+)", response)
        params_match = re.search(r"Parameters:\s*(\{.*?\})", response, re.DOTALL)

        if action_match and params_match:
            tool_name = action_match.group(1).strip()
            try:
                params_str = params_match.group(1).strip()
                parameters = json.loads(params_str)
            except json.JSONDecodeError:
                parameters = {}
            action = Action(
                tool_name=tool_name,
                parameters=parameters,
                reasoning=thought,
            )
            return (thought, action, False, None)

        # Parse failure - create error action for observation
        if action_match and not params_match:
            tool_name = action_match.group(1).strip()
            action = Action(
                tool_name=tool_name,
                parameters={},
                reasoning=thought or "Parsing error: Parameters block missing",
            )
            return (thought, action, False, None)

        logging.getLogger(__name__).warning(
            "Could not parse LLM response: %s", response[:200]
        )
        return (thought or "Parse error", None, False, None)

    def _execute_action(self, action: Action) -> Observation:
        """
        Execute an action via ToolExecutor and return an Observation.

        Adds metadata["stub"]=True when the tool has _is_stub=True.

        Feature: react-agent-core, Property 14: Stub tools return stub metadata
        """
        if action.tool_name not in self._registry:
            raise ToolNotFoundError(
                requested_tool=action.tool_name,
                available_tools=list(self._registry.keys()),
            )

        tool = self._registry[action.tool_name]
        result: ToolResult = self._executor.execute(tool, **action.parameters)

        # Add stub metadata when tool has _is_stub (Req 8.8)
        metadata: Optional[Dict[str, Any]] = None
        if getattr(tool, "_is_stub", False):
            metadata = dict(result.metadata) if result.metadata else {}
            metadata["stub"] = True

        return Observation(
            tool_name=action.tool_name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=metadata,
        )

    def step(self, state: AgentState) -> AgentState:
        """
        Execute exactly one Thought -> Action -> Observation iteration.

        Args:
            state: Current AgentState

        Returns:
            Updated AgentState after one iteration

        Feature: react-agent-core, Property 10: step() advances state by exactly one iteration
        """
        prompt = self._build_prompt(state)
        response = _invoke_llm(self._llm, prompt)
        thought, action, is_complete, final_answer = self._parse_llm_response(response)

        self._logger.log_thought(state.iteration + 1, thought)

        if is_complete:
            state.final_answer = final_answer
            # Create a no-op action/observation for history consistency
            noop_action = Action(
                tool_name="",
                parameters={},
                reasoning=thought,
            )
            noop_obs = Observation(
                tool_name="",
                success=True,
                output="Task complete",
                error=None,
            )
            state.append_history(thought, noop_action, noop_obs)
            state.increment_iteration()
            state.is_complete = True
            return state

        if action is None:
            # Parse error - create error observation
            error_action = Action(
                tool_name="parsing_error",
                parameters={},
                reasoning=thought,
            )
            error_obs = Observation(
                tool_name="parsing_error",
                success=False,
                output=None,
                error="Could not parse LLM response",
            )
            self._logger.log_action(state.iteration + 1, error_action)
            self._logger.log_observation(state.iteration + 1, error_obs)
            state.append_history(thought, error_action, error_obs)
            state.increment_iteration()
            return state

        self._logger.log_action(state.iteration + 1, action)

        try:
            observation = self._execute_action(action)
        except ToolNotFoundError as e:
            observation = Observation(
                tool_name=action.tool_name,
                success=False,
                output=None,
                error=str(e),
            )

        self._logger.log_observation(state.iteration + 1, observation)
        state.append_history(thought, action, observation)
        state.increment_iteration()
        return state

    def run(self, query: str) -> str:
        """
        Execute the full ReAct loop and return the final response.

        Args:
            query: User query to resolve

        Returns:
            Final response string (from LLM or partial result on max_iterations)

        Feature: react-agent-core, Property 9: run() always terminates within max_iterations
        """
        start_time = time.time()
        state = AgentState(query=query)

        for _ in range(self._max_iterations):
            if state.is_complete:
                break
            state = self.step(state)

        execution_time = time.time() - start_time
        self._logger.log_run_complete(state, execution_time)

        if state.is_complete and state.final_answer:
            return state.final_answer
        if state.is_complete:
            return self._generate_final_response(state)

        # Max iterations reached - return partial result
        return (
            f"Reached maximum iterations ({self._max_iterations}) without completing. "
            f"Partial state: {len(state.history)} iterations completed."
        )

    def _generate_final_response(self, state: AgentState) -> str:
        """
        Invoke LLM to generate a final summary from the current state.

        Args:
            state: Final AgentState

        Returns:
            Summary string from the LLM
        """
        prompt = self._build_prompt(state)
        prompt += "\n\nBased on the above, provide a concise Final Answer summarizing the result."
        return _invoke_llm(self._llm, prompt)
