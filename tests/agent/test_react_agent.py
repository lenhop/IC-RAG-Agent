"""
Tests for ReActAgent.

Feature: react-agent-core
"""

import pytest

from src.agent import ReActAgent, AgentState, AgentError, ToolNotFoundError
from src.agent.tools import (
    ProductCatalogToolStub,
    InventorySummaryToolStub,
    OrderDetailsToolStub,
    UDSQueryToolStub,
    UDSReportGeneratorToolStub,
)
from ai_toolkit.tools import CalculatorTool


def _mock_llm_final_answer(query: str) -> str:
    """Mock LLM that returns Final Answer immediately."""
    return f"""Thought: I can answer this directly.
Final Answer: This is the answer to: {query}
"""


def _mock_llm_action_then_final(step: int) -> str:
    """Mock LLM that returns Action first, then Final Answer on second call."""
    if step == 0:
        return """Thought: I need to look up the product.
Action: product_catalog
Parameters: {"asin": "B001"}
"""
    return """Thought: I have the data now.
Final Answer: The product is available.
"""


class TestReActAgentInit:
    """Tests for ReActAgent initialization."""

    def test_init_with_tools(self):
        llm = lambda p: "Final Answer: done"
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])
        assert len(agent.list_tools()) == 1
        assert agent.list_tools()[0]["name"] == "calculator"

    def test_duplicate_tool_raises(self):
        llm = lambda p: "x"
        with pytest.raises(AgentError):
            ReActAgent(
                llm=llm,
                tools=[CalculatorTool(), CalculatorTool()],
            )

    def test_register_tool(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[])
        agent.register_tool(CalculatorTool())
        assert len(agent.list_tools()) == 1

    def test_register_duplicate_raises(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
        with pytest.raises(AgentError):
            agent.register_tool(CalculatorTool())

    def test_unregister_tool(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
        agent.unregister_tool("calculator")
        assert len(agent.list_tools()) == 0


class TestReActAgentRun:
    """Tests for ReActAgent.run()."""

    def test_run_returns_final_answer(self):
        agent = ReActAgent(
            llm=_mock_llm_final_answer,
            tools=[ProductCatalogToolStub()],
        )
        result = agent.run("What is the product?")
        assert "answer" in result.lower() or "query" in result.lower()

    def test_run_with_stub_tools(self):
        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return """Thought: Look up product.
Action: product_catalog
Parameters: {"asin": "B001"}
"""
            return """Thought: Got it.
Final Answer: Product found.
"""

        agent = ReActAgent(
            llm=mock_llm,
            tools=[
                ProductCatalogToolStub(),
                InventorySummaryToolStub(),
            ],
        )
        result = agent.run("Get product B001")
        assert "Product found" in result or "found" in result.lower()


class TestReActAgentStep:
    """Tests for ReActAgent.step()."""

    def test_step_advances_iteration(self):
        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return """Thought: Use calculator.
Action: calculator
Parameters: {"expression": "2+2"}
"""
            return """Thought: Done.
Final Answer: 4
"""

        agent = ReActAgent(llm=mock_llm, tools=[CalculatorTool()])
        state = AgentState(query="Compute 2+2")
        state = agent.step(state)
        assert state.iteration == 1
        assert len(state.history) == 1


class TestReActAgentBuildPrompt:
    """Tests for _build_prompt."""

    def test_build_prompt_contains_query_and_schemas(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
        state = AgentState(query="test query")
        prompt = agent._build_prompt(state)
        assert "test query" in prompt
        assert "calculator" in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Final Answer:" in prompt


class TestReActAgentParseResponse:
    """Tests for _parse_llm_response."""

    def test_parse_final_answer(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[])
        thought, action, is_complete, final = agent._parse_llm_response(
            "Thought: Done.\nFinal Answer: The result is 42."
        )
        assert is_complete is True
        assert action is None
        assert final == "The result is 42."

    def test_parse_action_and_parameters(self):
        agent = ReActAgent(llm=lambda p: "x", tools=[])
        thought, action, is_complete, final = agent._parse_llm_response(
            "Thought: Need to calculate.\nAction: calculator\nParameters: {\"expression\": \"1+1\"}"
        )
        assert is_complete is False
        assert action is not None
        assert action.tool_name == "calculator"
        assert action.parameters == {"expression": "1+1"}


class TestReActAgentExecuteAction:
    """Tests for _execute_action."""

    def test_tool_not_found_raises(self):
        from src.agent.models import Action

        agent = ReActAgent(llm=lambda p: "x", tools=[CalculatorTool()])
        action = Action("nonexistent", {}, "test")
        with pytest.raises(ToolNotFoundError) as exc_info:
            agent._execute_action(action)
        assert "nonexistent" in str(exc_info.value)
        assert "calculator" in exc_info.value.available_tools

    def test_stub_tool_adds_metadata(self):
        from src.agent.models import Action

        agent = ReActAgent(llm=lambda p: "x", tools=[ProductCatalogToolStub()])
        action = Action("product_catalog", {"asin": "B001"}, "lookup")
        obs = agent._execute_action(action)
        assert obs.success is True
        assert obs.metadata is not None
        assert obs.metadata.get("stub") is True
