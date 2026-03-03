# Implementation Plan: ReAct Agent Core

## Overview

Implement the ReAct Agent Core in `IC-RAG-Agent/src/agent/` over 5 days. All tool execution infrastructure (BaseTool, ToolExecutor, ToolResult, RetryManager, ErrorHandler) is provided by `libs/ai-toolkit/` and must be imported, not reimplemented. Tasks follow the Day 1–5 plan from the implementation guide.

## Tasks

- [ ] 1. Set up project structure and data models
  - Create `src/agent/__init__.py`, `src/agent/models.py`, `src/agent/exceptions.py`
  - Create `src/agent/tools/__init__.py`
  - Create `src/agent/tests/__init__.py`
  - Implement `AgentState`, `Action`, `Observation` dataclasses with `to_dict()` and mutation methods (`append_history`, `increment_iteration`) as specified in design
  - Implement `AgentError`, `MaxIterationsError`, `ToolNotFoundError` exception classes with the fields specified in design
  - Configure `pytest.ini` or `pyproject.toml` for the agent test suite with `pytest-cov`
  - _Requirements: 1.1–1.9, 6.1–6.5_

  - [ ]* 1.1 Write property tests for data models
    - **Property 1: Data model field completeness** — for any valid inputs, all required fields present with correct types
    - **Property 2: AgentState initialization invariant** — new AgentState always has iteration=0, is_complete=False, history=[]
    - **Property 3: AgentState history append invariant** — append_history increases length by 1, last entry matches
    - **Property 4: AgentState iteration increment invariant** — increment_iteration increases counter by exactly 1
    - **Property 5: Data model serialization round-trip** — to_dict() produces JSON-serializable dict with all fields
    - Use `hypothesis` with `@settings(max_examples=100)`
    - Tag: `# Feature: react-agent-core, Property {N}: {title}`
    - _Requirements: 1.1–1.9_

- [ ] 2. Implement ReActAgent initialization and tool registry
  - Create `src/agent/react_agent.py` with `ReActAgent.__init__`, `register_tool`, `unregister_tool`, `list_tools`
  - Import `BaseTool`, `ToolExecutor` from `ai_toolkit.tools`
  - Build tool registry as `Dict[str, BaseTool]` from the tools list passed at init
  - Raise `AgentError` on duplicate tool names at init or via `register_tool`
  - Default `max_iterations=10`; accept optional `AgentLogger` and `ToolExecutor` instances
  - `list_tools()` returns `[tool.to_schema() for tool in registry.values()]`
  - _Requirements: 2.1–2.8, 4.1–4.6_

  - [ ]* 2.1 Write property tests for tool registry
    - **Property 6: Tool registry consistency** — for any list of uniquely-named tools, registry contains exactly those tools
    - **Property 7: Duplicate tool name rejection** — any duplicate name raises AgentError
    - **Property 8: list_tools schema completeness** — returns exactly one schema per tool matching to_schema()
    - _Requirements: 2.2, 2.3, 2.6, 2.7, 2.8_

- [ ] 3. Implement AgentLogger
  - Create `src/agent/agent_logger.py` with `AgentLogger` class
  - Use Python `logging` module with a custom JSON formatter
  - Implement `log_thought(iteration, thought)`, `log_action(iteration, action)`, `log_observation(iteration, observation)`, `log_run_complete(state, execution_time)`
  - Each method emits a JSON line with `timestamp` (ISO 8601 UTC), `level`, `event`, `iteration`, and event-specific fields as specified in design
  - Support `level` parameter (DEBUG/INFO/WARNING/ERROR) and optional `output_path` for file logging
  - _Requirements: 7.1–7.9_

  - [ ]* 3.1 Write property tests for AgentLogger
    - **Property 13: Log entries contain required fields** — for any thought/action/observation, JSON log entry contains timestamp, level, event, iteration, and event-specific fields
    - Capture log output using `io.StringIO` handler; parse each line as JSON and assert field presence
    - _Requirements: 7.4, 7.5, 7.6_

- [ ] 4. Implement stub tools
  - Create `src/agent/tools/sp_api_stubs.py` with `ProductCatalogToolStub`, `InventorySummaryToolStub`, `OrderDetailsToolStub`
  - Create `src/agent/tools/uds_stubs.py` with `UDSQueryToolStub`, `UDSReportGeneratorToolStub`
  - Each stub inherits from `ai_toolkit.tools.BaseTool` and implements `execute()`, `validate_parameters()`, `_get_parameters()`
  - Valid params return realistic mock data; invalid params raise `ValidationError` from `ai_toolkit.errors.exception_types`
  - Each stub's `execute()` return value must include `"stub": True` in a metadata dict (the agent layer wraps this in an Observation)
  - See design document stub table for required params and mock return shapes
  - _Requirements: 8.1–8.8_

  - [ ]* 4.1 Write property tests for stub tools
    - **Property 14: Stub tools return success with mock data for valid inputs** — for any valid params, ToolResult has success=True, non-None output, metadata["stub"]=True
    - Test all 5 stubs; use hypothesis to generate valid param strings
    - Also test invalid params raise ValidationError (error condition)
    - _Requirements: 8.1–8.8_

- [ ] 5. Implement ReAct loop core
  - Implement `ReActAgent._build_prompt(state)` — formats query, history, tool schemas, and ReAct instructions as specified in design
  - Implement `ReActAgent._parse_llm_response(response)` — extracts (thought, action, is_complete) from LLM output; logs WARNING and returns error action on parse failure
  - Implement `ReActAgent._execute_action(action)` — looks up tool in registry (raises ToolNotFoundError if missing), calls `ToolExecutor.execute(tool, **action.parameters)`, converts ToolResult to Observation
  - Implement `ReActAgent.step(state)` — runs one Thought→Action→Observation iteration, updates and returns new AgentState
  - _Requirements: 3.2–3.5, 3.9, 4.4, 5.1–5.5, 9.2–9.5_

  - [ ]* 5.1 Write property tests for loop mechanics
    - **Property 9: Agent always terminates** — for any query and max_iterations ≥ 1, run() returns a string without hanging (use mock LLM that always returns Final Answer after N steps)
    - **Property 10: step() advances state by exactly one iteration** — for any AgentState, step() returns state with iteration+1 and history+1
    - **Property 11: Failed tool execution produces error Observation** — for any failure ToolResult, Observation has success=False and non-None error, loop continues
    - **Property 12: ToolNotFoundError contains context** — for any unknown tool name, error includes requested name and available names list
    - _Requirements: 3.5, 3.7, 3.8, 4.4, 5.4, 5.5, 6.4, 6.5_

- [ ] 6. Implement ReActAgent.run() and final response generation
  - Implement `ReActAgent.run(query)` — initializes AgentState, runs the loop via `step()`, handles `MaxIterationsError` by returning a partial result string, generates final response via LLM when `is_complete=True`
  - Wire AgentLogger calls into `run()` and `step()` at the correct points (log thought, action, observation, run_complete)
  - Implement `ReActAgent._generate_final_response(state)` — invokes LLM with final state to produce a summary answer string
  - _Requirements: 3.1, 3.6–3.8, 7.1–7.3, 7.9_

- [ ] 7. Checkpoint — verify core loop with stub tools
  - Write an integration test in `tests/test_react_agent.py` that:
    - Registers all 5 stub tools
    - Uses a mock LLM that returns a scripted sequence of Thought/Action/Final Answer responses
    - Calls `run()` and asserts the final response is a non-empty string
    - Asserts AgentState history has the expected number of entries
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 3.1–3.8, 10.9_

- [ ] 8. ModelManager integration and LLM prompt wiring
  - Update `ReActAgent._build_prompt()` to use the ModelManager-compatible invocation interface (LangChain `invoke` or equivalent)
  - Test with Ollama local LLM (`qwen3:1.7b`) using a simple query and the 5 stub tools
  - Verify streaming response support: if the LLM returns a streaming iterator, accumulate chunks before parsing
  - Add an example script `examples/react_agent_example.py` demonstrating basic usage with Ollama
  - _Requirements: 9.1–9.6_

- [ ] 9. Tool chaining integration
  - Implement `ReActAgent._execute_chain(actions)` that calls `ToolExecutor.execute_chain()` from ai-toolkit with a list of (tool, params) pairs
  - Wire chain execution into the loop: when the LLM response specifies a `Chain:` block with multiple steps, parse and execute as a chain
  - Write a unit test demonstrating a 3-step chain: `product_catalog` → `inventory_summary` → `uds_query` using stub tools
  - _Requirements: 5.6, 5.7_

  - [ ]* 9.1 Write property tests for LLM prompt and response parsing
    - **Property 15: LLM prompt contains required context** — for any AgentState, _build_prompt() output contains query, history, tool schemas, and ReAct instructions
    - **Property 16: LLM response parsing correctness** — for any well-formed response string, _parse_llm_response() extracts correct thought, tool_name, and parameters
    - _Requirements: 9.2, 9.3, 9.4_

- [ ] 10. Final checkpoint — full test suite and coverage
  - Run `pytest src/agent/tests/ --cov=src/agent --cov-report=term-missing` and verify ≥ 80% coverage
  - Ensure all 16 correctness properties have corresponding hypothesis tests in `test_properties.py`
  - Fix any failing tests or coverage gaps
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 10.1–10.9_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All tool execution (retry, timeout, error classification) is delegated to `ai_toolkit.tools.ToolExecutor` — do not reimplement
- The `CalculatorTool` from ai-toolkit is available as a 6th tool for testing if needed
- Property tests use `hypothesis` with `@settings(max_examples=100)` minimum
- Each property test file comment must include: `# Feature: react-agent-core, Property {N}: {title}`
- Mock LLM responses in tests using `unittest.mock.MagicMock` or a simple stub class
