# Requirements Document

## Introduction

This document specifies the requirements for the ReAct Agent Core (Part 1 of the IC-Agent system). The ReAct Agent Core is the reasoning engine that powers all future domain-specific agents (SP-API Agent, UDS Agent). It implements the Thought → Action → Observation loop, integrates with the existing ai-toolkit infrastructure (BaseTool, ToolExecutor, ToolResult), and provides structured logging and observability for all agent interactions.

**Scope:** This document covers Part 1 only — the foundational agent engine. Domain-specific SP-API and UDS integrations (Parts 2 and 3) are out of scope except for stub tool implementations used for testing.

**Dependencies:** This feature depends on the ai-toolkit `agent-tools-infrastructure` feature being complete, providing `BaseTool`, `ToolExecutor`, `ToolResult`, `ToolParameter`, `ErrorHandler`, `RetryManager`, and related exception types.

## Glossary

- **ReAct_Agent**: The core agent class that executes the Thought → Action → Observation reasoning loop
- **AgentState**: An immutable snapshot of the agent's state at a given iteration, including query, history, and current iteration count
- **Action**: A data model representing a selected tool invocation, including tool name and parameters
- **Observation**: A data model representing the result of an executed action, including success status and output
- **Tool_Registry**: A dictionary-based mapping from tool names to BaseTool instances, maintained by the ReAct_Agent
- **AgentLogger**: A structured JSON logging component that records all thoughts, actions, and observations
- **ReAct_Loop**: The iterative Thought → Action → Observation cycle that continues until the task is resolved or max_iterations is reached
- **ModelManager**: The existing AI Toolkit component that manages LLM instances (Ollama, Deepseek, Qwen, GLM)
- **ToolExecutor**: The existing ai-toolkit component that executes tools with retry, timeout, and error handling
- **BaseTool**: The existing ai-toolkit abstract base class that all custom tools must inherit from
- **ToolResult**: The existing ai-toolkit data model representing the outcome of a tool execution
- **Stub_Tool**: A tool implementation that returns realistic mock data, used for testing before real integrations are available
- **Max_Iterations**: A configurable upper bound on the number of ReAct loop iterations to prevent infinite loops
- **Tool_Schema**: A JSON Schema representation of a tool's parameters, generated for LLM function calling

## Requirements

### Requirement 1: Agent Data Models

**User Story:** As a developer, I want well-defined data models for agent state, actions, and observations, so that the agent's reasoning process is structured and traceable.

#### Acceptance Criteria

1. THE AgentState SHALL contain a query field (string), an iteration field (integer), a history field (list of thought/action/observation tuples), and a is_complete field (boolean)
2. THE Action SHALL contain a tool_name field (string), a parameters field (dictionary), and a reasoning field (string describing why this action was chosen)
3. THE Observation SHALL contain a success field (boolean), an output field (any serializable type), an error field (optional string), and a tool_name field (string)
4. WHEN an AgentState is created, THE System SHALL initialize iteration to 0 and is_complete to False
5. THE AgentState SHALL provide a method to append a new thought/action/observation tuple to history
6. THE AgentState SHALL provide a method to increment the iteration counter
7. WHEN serialized, THE AgentState SHALL produce a JSON-serializable dictionary containing all fields
8. WHEN serialized, THE Action SHALL produce a JSON-serializable dictionary containing all fields
9. WHEN serialized, THE Observation SHALL produce a JSON-serializable dictionary containing all fields

### Requirement 2: ReAct Agent Initialization

**User Story:** As a developer, I want to initialize a ReAct Agent with a set of tools and configuration, so that I can create agents tailored to specific tasks.

#### Acceptance Criteria

1. THE ReAct_Agent SHALL accept an LLM instance (from ModelManager), a list of BaseTool instances, and a max_iterations parameter at initialization
2. THE ReAct_Agent SHALL initialize a Tool_Registry by mapping each tool's name to its instance
3. WHEN initialized with duplicate tool names, THE ReAct_Agent SHALL raise an AgentError indicating the conflict
4. THE ReAct_Agent SHALL default max_iterations to 10 if not specified
5. THE ReAct_Agent SHALL accept an optional AgentLogger instance at initialization; if not provided, THE System SHALL create a default AgentLogger
6. THE ReAct_Agent SHALL expose a register_tool method that adds a single BaseTool to the Tool_Registry after initialization
7. WHEN register_tool is called with a tool whose name already exists in the registry, THE ReAct_Agent SHALL raise an AgentError
8. THE ReAct_Agent SHALL expose a list_tools method that returns a list of tool schemas (via BaseTool.to_schema()) for all registered tools

### Requirement 3: ReAct Loop Execution

**User Story:** As a developer, I want the ReAct Agent to execute the Thought → Action → Observation loop autonomously, so that it can reason and act to resolve user queries.

#### Acceptance Criteria

1. WHEN the ReAct_Agent receives a query via the run() method, THE System SHALL initialize a new AgentState and begin the ReAct_Loop
2. WHEN the ReAct_Loop begins an iteration, THE System SHALL generate a thought by invoking the LLM with the current AgentState and available tool schemas
3. WHEN a thought is generated, THE System SHALL select an Action by parsing the LLM output to identify the tool name and parameters
4. WHEN an Action is selected, THE System SHALL execute the action using ToolExecutor and capture the result as an Observation
5. WHEN an Observation is captured, THE System SHALL append the thought, action, and observation to AgentState history and increment the iteration counter
6. WHEN the LLM indicates the task is complete (via a designated completion signal in its output), THE System SHALL set AgentState.is_complete to True and exit the loop
7. WHEN AgentState.iteration reaches max_iterations without completion, THE System SHALL exit the loop and return a partial result with an explanation
8. WHEN the ReAct_Loop exits, THE System SHALL generate a final response string summarizing the result
9. THE ReAct_Agent SHALL expose a step() method that executes exactly one Thought → Action → Observation iteration and returns the updated AgentState

### Requirement 4: Tool Registry and Selection

**User Story:** As a developer, I want the agent to dynamically manage and select tools, so that it can be extended with new capabilities without code changes.

#### Acceptance Criteria

1. THE Tool_Registry SHALL support registration of at least 5 different tools simultaneously
2. WHEN a tool is registered, THE Tool_Registry SHALL store the tool instance indexed by its name
3. WHEN the agent selects a tool by name, THE System SHALL retrieve the tool from the Tool_Registry in O(1) time
4. WHEN the LLM selects a tool name that does not exist in the Tool_Registry, THE System SHALL raise a ToolNotFoundError and allow the agent to retry with a different approach
5. THE ReAct_Agent SHALL generate a combined schema document from all registered tools and include it in every LLM prompt
6. WHEN a tool is removed from the registry (via an unregister_tool method), THE System SHALL no longer include that tool's schema in LLM prompts

### Requirement 5: Tool Execution via ai-toolkit ToolExecutor

**User Story:** As a developer, I want all tool executions to go through the ai-toolkit ToolExecutor, so that retry, timeout, and error handling are handled consistently.

#### Acceptance Criteria

1. THE ReAct_Agent SHALL use the ToolExecutor from ai-toolkit for all tool invocations
2. WHEN a tool execution fails with a transient error, THE ToolExecutor SHALL retry the execution (behavior provided by ai-toolkit)
3. WHEN a tool execution fails with a fatal error, THE ToolExecutor SHALL return a failure ToolResult without retrying (behavior provided by ai-toolkit)
4. WHEN a tool execution returns a failure ToolResult, THE System SHALL create an Observation with success=False and include the error in the Observation's error field
5. WHEN a tool execution returns a failure ToolResult, THE System SHALL allow the ReAct_Loop to continue so the LLM can retry with a different approach
6. THE ReAct_Agent SHALL support tool chaining by invoking ToolExecutor.execute_chain() when the LLM requests a multi-step tool sequence
7. WHEN a tool chain is executed, THE System SHALL capture the final ToolResult as the Observation for that iteration

### Requirement 6: Custom Exception Types

**User Story:** As a developer, I want well-defined exception types for agent-specific errors, so that I can handle failures precisely.

#### Acceptance Criteria

1. THE System SHALL define an AgentError base exception class for all agent-related errors
2. THE System SHALL define a MaxIterationsError that inherits from AgentError, raised when max_iterations is reached without task completion
3. THE System SHALL define a ToolNotFoundError that inherits from AgentError, raised when the LLM selects a tool not in the Tool_Registry
4. WHEN MaxIterationsError is raised, THE System SHALL include the final AgentState and the number of iterations completed in the exception
5. WHEN ToolNotFoundError is raised, THE System SHALL include the requested tool name and the list of available tool names in the exception

### Requirement 7: AgentLogger

**User Story:** As a developer, I want structured JSON logging of all agent interactions, so that I can debug, audit, and analyze agent behavior.

#### Acceptance Criteria

1. THE AgentLogger SHALL log every thought generated during the ReAct_Loop as a structured JSON entry
2. THE AgentLogger SHALL log every action selected (tool name, parameters, reasoning) as a structured JSON entry
3. THE AgentLogger SHALL log every observation captured (success, output, error, tool_name) as a structured JSON entry
4. WHEN logging a thought, THE AgentLogger SHALL include the fields: timestamp, level, event ("agent_thought"), iteration, and thought content
5. WHEN logging an action, THE AgentLogger SHALL include the fields: timestamp, level, event ("agent_action"), iteration, tool_name, and parameters
6. WHEN logging an observation, THE AgentLogger SHALL include the fields: timestamp, level, event ("agent_observation"), iteration, tool_name, success, and output or error
7. THE AgentLogger SHALL support log levels: DEBUG, INFO, WARNING, ERROR
8. THE AgentLogger SHALL write log entries to a configurable output (stdout by default, file path optional)
9. WHEN an agent run completes, THE AgentLogger SHALL log a summary entry including total iterations, final status, and total execution time

### Requirement 8: Domain-Specific Tool Stubs

**User Story:** As a developer, I want stub implementations of SP-API and UDS tools, so that I can test the ReAct Agent without requiring live external integrations.

#### Acceptance Criteria

1. THE System SHALL provide a ProductCatalogToolStub that inherits from BaseTool and returns mock product data given an ASIN or SKU parameter
2. THE System SHALL provide an InventorySummaryToolStub that inherits from BaseTool and returns mock inventory levels given a SKU parameter
3. THE System SHALL provide an OrderDetailsToolStub that inherits from BaseTool and returns mock order data given an order_id parameter
4. THE System SHALL provide a UDSQueryToolStub that inherits from BaseTool and returns mock query results given a query string parameter
5. THE System SHALL provide a UDSReportGeneratorToolStub that inherits from BaseTool and returns a mock report given a report_type parameter
6. WHEN any stub tool is executed with valid parameters, THE System SHALL return a ToolResult with success=True and realistic mock data
7. WHEN any stub tool is executed with invalid parameters, THE System SHALL raise a ValidationError (handled by ToolExecutor)
8. THE metadata field of every stub tool's ToolResult SHALL include a "stub": true entry to distinguish stub responses from real responses

### Requirement 9: ModelManager Integration

**User Story:** As a developer, I want the ReAct Agent to integrate with the existing AI Toolkit ModelManager, so that it can use local (Ollama) and remote (Deepseek, Qwen, GLM) LLMs interchangeably.

#### Acceptance Criteria

1. THE ReAct_Agent SHALL accept any LLM instance compatible with the AI Toolkit ModelManager interface
2. WHEN generating a thought, THE System SHALL invoke the LLM using the ModelManager's standard invocation interface
3. THE System SHALL format the LLM prompt to include the current query, conversation history, available tool schemas, and instructions for the ReAct format
4. WHEN the LLM returns a response, THE System SHALL parse the response to extract the thought, selected tool name, and tool parameters
5. IF the LLM response cannot be parsed into a valid Action, THEN THE System SHALL log a WARNING and treat the iteration as a failed action with a parsing error Observation
6. THE System SHALL support both streaming and non-streaming LLM responses

### Requirement 10: Testing Infrastructure

**User Story:** As a developer, I want comprehensive test coverage including property-based tests, so that the agent's correctness can be verified across a wide range of inputs.

#### Acceptance Criteria

1. THE System SHALL achieve at least 80% code coverage across all agent modules
2. THE System SHALL include unit tests for AgentState, Action, and Observation data models
3. THE System SHALL include unit tests for ReActAgent initialization, tool registration, and tool listing
4. THE System SHALL include unit tests for the ReAct loop using a mock LLM
5. THE System SHALL include unit tests for AgentLogger output format and content
6. THE System SHALL include unit tests for all stub tools (valid and invalid parameter scenarios)
7. THE System SHALL include property-based tests using the hypothesis library for universal correctness properties
8. WHEN running property-based tests, THE System SHALL execute a minimum of 100 iterations per property
9. THE System SHALL include integration tests that run a complete agent loop with stub tools and a mock LLM
