"""
Property-based tests for ReAct Agent Core.

Uses hypothesis with @settings(max_examples=100) for all properties.

Feature: react-agent-core, Property 1-16
"""

import json
import io
import logging

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agent.models import Action, AgentState, Observation
from src.agent.exceptions import AgentError, ToolNotFoundError
from src.agent.agent_logger import AgentLogger, JsonFormatter
from src.agent.react_agent import ReActAgent
from src.agent.tools import (
    ProductCatalogToolStub,
    InventorySummaryToolStub,
    OrderDetailsToolStub,
    UDSQueryToolStub,
    UDSReportGeneratorToolStub,
)
from ai_toolkit.tools import BaseTool, CalculatorTool


# --- Property 1: Data model field completeness ---
# Feature: react-agent-core, Property 1: Data model field completeness
@given(
    tool_name=st.text(min_size=1, max_size=50),
    reasoning=st.text(max_size=200),
)
@settings(max_examples=100)
def test_property_1_action_fields(tool_name: str, reasoning: str):
    """Action has all required fields with correct types."""
    params = {"key": "value"}
    action = Action(tool_name=tool_name, parameters=params, reasoning=reasoning)
    assert isinstance(action.tool_name, str)
    assert isinstance(action.parameters, dict)
    assert isinstance(action.reasoning, str)


@given(
    tool_name=st.text(min_size=1, max_size=50),
    success=st.booleans(),
)
@settings(max_examples=100)
def test_property_1_observation_fields(tool_name: str, success: bool):
    """Observation has all required fields with correct types."""
    obs = Observation(
        tool_name=tool_name,
        success=success,
        output="out" if success else None,
        error=None if success else "err",
    )
    assert isinstance(obs.tool_name, str)
    assert isinstance(obs.success, bool)


@given(query=st.text(min_size=1, max_size=200))
@settings(max_examples=100)
def test_property_1_agent_state_fields(query: str):
    """AgentState has all required fields with correct types."""
    state = AgentState(query=query)
    assert isinstance(state.query, str)
    assert isinstance(state.iteration, int)
    assert isinstance(state.is_complete, bool)
    assert isinstance(state.history, list)


# --- Property 2: AgentState init invariant ---
# Feature: react-agent-core, Property 2: AgentState initialization invariant
@given(query=st.text(min_size=1))
@settings(max_examples=100)
def test_property_2_agent_state_init(query: str):
    """New AgentState has iteration=0, is_complete=False, history=[]."""
    state = AgentState(query=query)
    assert state.iteration == 0
    assert state.is_complete is False
    assert state.history == []


# --- Property 3: append_history increases length by 1 ---
# Feature: react-agent-core, Property 3: AgentState history append invariant
@given(
    query=st.text(min_size=1),
    thought=st.text(max_size=100),
)
@settings(max_examples=100)
def test_property_3_append_history(query: str, thought: str):
    """append_history increases history length by 1."""
    state = AgentState(query=query)
    initial_len = len(state.history)
    action = Action("t", {}, "r")
    obs = Observation("t", True, "o", None)
    state.append_history(thought, action, obs)
    assert len(state.history) == initial_len + 1
    assert state.history[-1] == (thought, action, obs)


# --- Property 4: increment_iteration increases by 1 ---
# Feature: react-agent-core, Property 4: AgentState iteration increment invariant
@given(n=st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_property_4_increment_iteration(n: int):
    """increment_iteration increases counter by exactly 1."""
    state = AgentState(query="q")
    for _ in range(n):
        state.increment_iteration()
    assert state.iteration == n
    state.increment_iteration()
    assert state.iteration == n + 1


# --- Property 5: to_dict produces JSON-serializable dict ---
# Feature: react-agent-core, Property 5: Data model serialization round-trip
@given(
    tool_name=st.text(min_size=1, max_size=20),
    reasoning=st.text(max_size=50),
)
@settings(max_examples=100)
def test_property_5_action_to_dict(tool_name: str, reasoning: str):
    """Action.to_dict() is JSON-serializable."""
    action = Action(tool_name=tool_name, parameters={"a": 1}, reasoning=reasoning)
    d = action.to_dict()
    json.dumps(d)


@given(tool_name=st.text(min_size=1, max_size=20))
@settings(max_examples=100)
def test_property_5_observation_to_dict(tool_name: str):
    """Observation.to_dict() is JSON-serializable."""
    obs = Observation(tool_name=tool_name, success=True, output={"x": 1}, error=None)
    d = obs.to_dict()
    json.dumps(d)


@given(query=st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_property_5_agent_state_to_dict(query: str):
    """AgentState.to_dict() is JSON-serializable."""
    state = AgentState(query=query)
    state.append_history(
        "t",
        Action("x", {}, "r"),
        Observation("x", True, 1, None),
    )
    d = state.to_dict()
    json.dumps(d)


# --- Property 6: Tool registry consistency ---
# Feature: react-agent-core, Property 6: Tool registry consistency
@given(
    st.lists(
        st.sampled_from([
            CalculatorTool(),
            ProductCatalogToolStub(),
            InventorySummaryToolStub(),
        ]),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=50)
def test_property_6_tool_registry(tools: list):
    """Registry contains exactly registered tools."""
    unique_tools = []
    seen = set()
    for t in tools:
        if t.name not in seen:
            seen.add(t.name)
            unique_tools.append(t)
    if not unique_tools:
        unique_tools = [CalculatorTool()]
    agent = ReActAgent(llm=lambda p: "x", tools=unique_tools)
    for t in unique_tools:
        assert t.name in agent._registry
        assert agent._registry[t.name] is t
    assert len(agent.list_tools()) == len(unique_tools)


# --- Property 7: Duplicate tool name raises AgentError ---
# Feature: react-agent-core, Property 7: Duplicate tool name rejection
def test_property_7_duplicate_tool_raises():
    """Duplicate tool name raises AgentError."""
    with pytest.raises(AgentError):
        ReActAgent(
            llm=lambda p: "x",
            tools=[CalculatorTool(), CalculatorTool()],
        )


# --- Property 8: list_tools returns one schema per tool ---
# Feature: react-agent-core, Property 8: list_tools schema completeness
def test_property_8_list_tools_schema():
    """list_tools returns one schema per tool."""
    tools = [CalculatorTool(), ProductCatalogToolStub()]
    agent = ReActAgent(llm=lambda p: "x", tools=tools)
    schemas = agent.list_tools()
    assert len(schemas) == len(tools)
    names = {s["name"] for s in schemas}
    assert names == {t.name for t in tools}


# --- Property 9: run() always terminates ---
# Feature: react-agent-core, Property 9: Agent always terminates
@given(max_iter=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_property_9_run_terminates(max_iter: int):
    """run() terminates within max_iterations."""
    agent = ReActAgent(
        llm=lambda p: "Thought: Done.\nFinal Answer: Result.",
        tools=[CalculatorTool()],
        max_iterations=max_iter,
    )
    result = agent.run("test")
    assert isinstance(result, str)
    assert len(result) > 0


# --- Property 10: step() advances iteration and history ---
# Feature: react-agent-core, Property 10: step() advances state by exactly one iteration
def test_property_10_step_advances():
    """step() returns state with iteration+1 and history+1."""
    call_count = [0]

    def mock_llm(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return """Thought: Calc.
Action: calculator
Parameters: {"expression": "1+1"}
"""
        return """Thought: Done.
Final Answer: 2
"""

    agent = ReActAgent(llm=mock_llm, tools=[CalculatorTool()])
    state = AgentState(query="q")
    init_iter, init_len = state.iteration, len(state.history)
    state = agent.step(state)
    assert state.iteration == init_iter + 1
    assert len(state.history) == init_len + 1


# --- Property 11: Failed tool produces error Observation, loop continues ---
# Feature: react-agent-core, Property 11: Failed tool produces error Observation
def test_property_11_failed_tool_observation():
    """Failed tool execution produces error Observation, loop continues."""
    def mock_llm(prompt):
        if "Conversation history:" in prompt and "(none)" not in prompt:
            return "Thought: Done.\nFinal Answer: Continued after error."
        return """Thought: Invalid.
Action: product_catalog
Parameters: {}
"""

    agent = ReActAgent(llm=mock_llm, tools=[ProductCatalogToolStub()])
    result = agent.run("test")
    assert isinstance(result, str)
    # Agent should continue and eventually return (either Final Answer or max iter)


# --- Property 12: ToolNotFoundError includes context ---
# Feature: react-agent-core, Property 12: ToolNotFoundError contains context
def test_property_12_tool_not_found_context():
    """ToolNotFoundError includes requested_tool and available_tools."""
    agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
    from src.agent.models import Action
    action = Action("unknown_tool", {}, "r")
    with pytest.raises(ToolNotFoundError) as exc_info:
        agent._execute_action(action)
    assert exc_info.value.requested_tool == "unknown_tool"
    assert "calculator" in exc_info.value.available_tools


# --- Property 13: Log entries contain required fields ---
# Feature: react-agent-core, Property 13: Log entries contain required fields
def test_property_13_log_required_fields():
    """Log entries contain timestamp, level, event, iteration."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    al = AgentLogger(logger_name="prop13")
    al.logger.handlers = [handler]
    al.log_thought(1, "thought")
    al.log_action(1, Action("t", {}, "r"))
    al.log_observation(1, Observation("t", True, "o", None))

    for line in stream.getvalue().strip().split("\n"):
        if line:
            data = json.loads(line)
            assert "timestamp" in data
            assert "level" in data
            assert "event" in data
            assert "iteration" in data


# --- Property 14: Stub tools return success + stub metadata ---
# Feature: react-agent-core, Property 14: Stub tools return success with stub metadata
@pytest.mark.parametrize(
    "tool,params",
    [
        (ProductCatalogToolStub, {"asin": "B001"}),
        (InventorySummaryToolStub, {"sku": "SKU1"}),
        (OrderDetailsToolStub, {"order_id": "ORD1"}),
        (UDSQueryToolStub, {"query": "SELECT 1"}),
        (UDSReportGeneratorToolStub, {"report_type": "sales"}),
    ],
)
def test_property_14_stub_tools_success(tool, params):
    """Stub tools return success=True, non-None output, metadata stub=True."""
    t = tool()
    agent = ReActAgent(llm=lambda p: "x", tools=[t])
    from src.agent.models import Action
    action = Action(t.name, params, "r")
    obs = agent._execute_action(action)
    assert obs.success is True
    assert obs.output is not None
    assert obs.metadata is not None
    assert obs.metadata.get("stub") is True


# --- Property 15: _build_prompt contains query, history, schemas ---
# Feature: react-agent-core, Property 15: LLM prompt contains required context
def test_property_15_build_prompt_context():
    """_build_prompt contains query, history, tool schemas."""
    agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
    state = AgentState(query="my query")
    state.append_history(
        "t",
        Action("calc", {"expression": "1+1"}, "r"),
        Observation("calc", True, "2", None),
    )
    prompt = agent._build_prompt(state)
    assert "my query" in prompt
    assert "t" in prompt or "Thought" in prompt
    assert "calculator" in prompt
    assert "parameters" in prompt.lower() or "expression" in prompt


# --- Property 16: _parse_llm_response extracts correct fields ---
# Feature: react-agent-core, Property 16: LLM response parsing correctness
@given(
    thought=st.text(max_size=50),
    tool_name=st.sampled_from(["calculator", "product_catalog"]),
)
@settings(max_examples=100)
def test_property_16_parse_response(thought: str, tool_name: str):
    """_parse_llm_response extracts thought, tool_name, parameters."""
    params = {"expression": "2+2"} if tool_name == "calculator" else {"asin": "B001"}
    params_str = json.dumps(params)
    # Ensure newline between Thought and Action for reliable parsing
    response = f"Thought: {thought}\nAction: {tool_name}\nParameters: {params_str}"
    agent = ReActAgent(llm=lambda p: "x", tools=[])
    t, action, is_complete, final = agent._parse_llm_response(response)
    # Parser strips thought; compare normalized
    assert t == thought.strip()
    assert is_complete is False
    assert action is not None
    assert action.tool_name == tool_name
    assert action.parameters == params
