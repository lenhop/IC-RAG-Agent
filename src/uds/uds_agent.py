"""
UDS Agent - Main orchestration engine for business intelligence queries.

Extends ReActAgent with UDS-specific capabilities including:
- Intent classification
- Task planning
- Tool execution
- Result formatting
- Error handling
"""

import json
import os
from typing import Dict, Any, List, Optional
from src.uds.cache import UDSCache  # for type hints
from src.agent.react_agent import ReActAgent
from src.uds.uds_client import UDSClient
from src.uds.config import UDSConfig
from src.uds.intent_classifier import UDSIntentClassifier, IntentResult, IntentDomain
from src.uds.task_planner import UDSTaskPlanner, TaskPlan
from src.uds.result_formatter import UDSResultFormatter
from src.uds.tools import UDSToolRegistry
from src.uds.error_handler import (
    UDSError,
    DatabaseError,
    ToolExecutionError,
    retry_with_backoff,
    CircuitBreaker,
    handle_database_error,
    handle_tool_execution_error,
    handle_llm_error,
    handle_api_error
)


class UDSAgent(ReActAgent):
    """
    UDS Agent for business intelligence queries.
    Extends ReActAgent with UDS-specific capabilities.
    """

    def __init__(
        self,
        uds_client: Optional[UDSClient],
        llm_client,
        intent_classifier: Optional[UDSIntentClassifier] = None,
        task_planner: Optional[UDSTaskPlanner] = None,
        result_formatter: Optional[UDSResultFormatter] = None,
        cache: Optional[UDSCache] = None,
    ):
        """
        Initialize UDS Agent.

        Args:
            uds_client: ClickHouse client
            llm_client: LLM for reasoning
            intent_classifier: Optional custom classifier
            task_planner: Optional custom planner
            result_formatter: Optional custom formatter
        """
        super().__init__(llm=llm_client)

        # caching layer shared across components
        self.cache = cache

        # initialize or accept provided UDS client
        if uds_client is None:
            self.uds_client = UDSClient(cache=cache)
        else:
            self.uds_client = uds_client
            # propagate cache if possible
            try:
                self.uds_client.cache = cache
            except Exception:
                pass

        # classifier/planner/formatter
        self.intent_classifier = intent_classifier or UDSIntentClassifier(llm_client)
        if self.cache is not None:
            self.intent_classifier.cache = self.cache
        self.task_planner = task_planner or UDSTaskPlanner(llm_client)
        self.result_formatter = result_formatter or UDSResultFormatter()

        self._register_tools()
        self.schema_context = self._load_schema_context()

    def _register_tools(self):
        """Register all UDS tools."""
        for tool in UDSToolRegistry.get_query_tools():
            # Some tools (e.g., GenerateSQLTool) require an LLM instance injected at runtime.
            if hasattr(tool, "set_llm"):
                try:
                    tool.set_llm(self._llm)
                except Exception:
                    pass
            self.register_tool(tool)

    def get_tool(self, tool_name: str):
        """
        Get a tool by name from the registry.

        Args:
            tool_name: Name of the tool to get

        Returns:
            Tool instance or None if not found
        """
        return self._registry.get(tool_name)

    def _load_schema_context(self) -> Dict[str, Any]:
        """Load schema metadata for context."""
        try:
            schema_path = UDSConfig.project_path(UDSConfig.SCHEMA_METADATA_PATH)
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load schema metadata: {e}")
            return {}

    def process_query(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Main entry point for processing user queries.

        Args:
            query: Natural language business question
            max_iterations: Maximum ReAct iterations (not used, kept for API compatibility)

        Returns:
            Formatted response with insights, data, and visualizations
        """
        try:
            intent = self.intent_classifier.classify(query)
            print(f"Intent: {intent.primary_domain.value} (confidence: {intent.confidence:.2f})")
        except Exception as e:
            error = handle_llm_error(e)
            return {
                'success': False,
                'query': query,
                'error': error['error'],
                'error_type': error['error_type']
            }
        
        try:
            plan = self.task_planner.create_plan(query, intent)
            print(f"Plan: {len(plan.subtasks)} subtasks")
        except Exception as e:
            error = handle_tool_execution_error(e, "TaskPlanner")
            return {
                'success': False,
                'query': query,
                'error': error['error'],
                'error_type': error['error_type']
            }
        
        try:
            context = self._build_context(query, intent, plan)
            print("Executing ReAct loop...")
        except Exception as e:
            error = handle_tool_execution_error(e, "ContextBuilder")
            return {
                'success': False,
                'query': query,
                'error': error['error'],
                'error_type': error['error_type']
            }
        
        try:
            # Keep UDS response time bounded for API callers.
            configured_iterations = int(
                os.getenv("UDS_AGENT_MAX_ITERATIONS", str(max_iterations))
            )
            bounded_iterations = max(1, min(configured_iterations, 5))
            previous_max_iterations = self._max_iterations
            self._max_iterations = bounded_iterations
            result = self.run(query)
            self._max_iterations = previous_max_iterations
        except Exception as e:
            try:
                self._max_iterations = previous_max_iterations
            except Exception:
                pass
            error = handle_llm_error(e)
            return {
                'success': False,
                'query': query,
                'error': error['error'],
                'error_type': error['error_type']
            }
        
        try:
            formatted = self.result_formatter.format(
                agent_result={'output': result},
                intent=intent
            )
        except Exception as e:
            error = handle_tool_execution_error(e, "ResultFormatter")
            return {
                'success': False,
                'query': query,
                'error': error['error'],
                'error_type': error['error_type']
            }
        
        return {
            'success': True,
            'query': query,
            'intent': intent.primary_domain.value,
            'response': formatted
        }

    def _build_context(
        self,
        query: str,
        intent: IntentResult,
        plan: TaskPlan
    ) -> str:
        """
        Build enriched context for agent.

        Args:
            query: User query
            intent: Classified intent
            plan: Task plan

        Returns:
            Context string
        """
        context_parts = []

        context_parts.append("""You are a UDS Agent specialized in Amazon business intelligence.
You have access to 9 tables with 40.3M rows of Amazon data (October 2025).
Use the available tools to answer business questions accurately.""")

        relevant_tables = self._get_relevant_tables(intent)
        if relevant_tables:
            context_parts.append(f"\nRelevant tables: {', '.join(relevant_tables)}")

            for table in relevant_tables[:3]:
                if table in self.schema_context.get('tables', {}):
                    desc = self.schema_context['tables'][table].get('description', '')
                    context_parts.append(f"- {table}: {desc}")

        context_parts.append(f"\nExecution plan:")
        for i, task_id in enumerate(plan.execution_order, 1):
            subtask = next(t for t in plan.subtasks if t.id == task_id)
            context_parts.append(f"{i}. {subtask.description} (use {subtask.tool_name})")

        context_parts.append(f"\nYou have {len(self._registry)} tools available.")

        return "\n".join(context_parts)

    def _get_relevant_tables(self, intent: IntentResult) -> List[str]:
        """
        Get tables relevant to the intent.

        Args:
            intent: Classified intent

        Returns:
            List of table names
        """
        domain_tables = {
            IntentDomain.SALES: ['amz_order', 'amz_transaction'],
            IntentDomain.INVENTORY: ['amz_fba_inventory_all', 'amz_daily_inventory_ledger'],
            IntentDomain.FINANCIAL: ['amz_transaction', 'amz_statement', 'amz_fee'],
            IntentDomain.PRODUCT: ['amz_product', 'amz_order', 'amz_listing_item'],
            IntentDomain.COMPARISON: ['amz_order', 'amz_transaction'],
            IntentDomain.GENERAL: []
        }

        return domain_tables.get(intent.primary_domain, [])

    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Execute task plan sequentially.

        Args:
            plan: Task plan to execute

        Returns:
            Execution results
        """
        results = {}

        for task_id in plan.execution_order:
            subtask = next(t for t in plan.subtasks if t.id == task_id)

            tool = self.get_tool(subtask.tool_name)
            if not tool:
                results[task_id] = {'error': f'Tool {subtask.tool_name} not found'}
                continue

            params = self._resolve_parameters(subtask.parameters, results)

            try:
                result = tool.execute(**params)
                results[task_id] = result.output if result.success else {'error': result.error}
            except Exception as e:
                results[task_id] = {'error': str(e)}

        return results

    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve parameter references to previous results.

        Args:
            parameters: Parameter dict (may contain references)
            previous_results: Results from previous tasks

        Returns:
            Resolved parameters
        """
        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('$'):
                ref = value[1:]
                if '.' in ref:
                    task_id, field = ref.split('.', 1)
                    if task_id in previous_results:
                        resolved[key] = previous_results[task_id].get(field)
                else:
                    resolved[key] = previous_results.get(ref)
            else:
                resolved[key] = value

        return resolved
