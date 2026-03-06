"""
UDS Task Planner - Decomposes complex queries into executable subtasks.

This is the core planning component that:
- Decomposes complex queries into subtasks
- Determines tool selection and execution order
- Handles dependencies between tasks
- Extracts parameters from queries
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.uds.intent_classifier import IntentResult, IntentDomain
from src.uds.tools import UDSToolRegistry


@dataclass
class Subtask:
    """A single subtask in an execution plan."""
    id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    output_key: str = ""


@dataclass
class TaskPlan:
    """Complete execution plan for a query."""
    query: str
    intent: IntentResult
    subtasks: List[Subtask]
    execution_order: List[str]


class UDSTaskPlanner:
    """
    Decomposes complex queries into executable subtasks.
    Determines tool selection and execution order.
    """

    SIMPLE_PATTERNS = {
        IntentDomain.SALES: {
            'patterns': ['total sales', 'revenue', 'sales trend', 'daily sales'],
            'tool': 'SalesTrendTool'
        },
        IntentDomain.INVENTORY: {
            'patterns': ['inventory', 'stock', 'low stock'],
            'tool': 'InventoryAnalysisTool'
        },
        IntentDomain.FINANCIAL: {
            'patterns': ['profit', 'fees', 'financial'],
            'tool': 'FinancialSummaryTool'
        },
        IntentDomain.PRODUCT: {
            'patterns': ['top products', 'product performance', 'best selling'],
            'tool': 'ProductPerformanceTool'
        },
        IntentDomain.COMPARISON: {
            'patterns': ['compare', 'vs', 'versus'],
            'tool': 'ComparisonTool'
        },
        IntentDomain.GENERAL: {
            'patterns': ['tables', 'describe', 'schema'],
            'tool': 'ListTablesTool'
        }
    }

    def __init__(self, llm_client=None):
        """
        Initialize task planner.

        Args:
            llm_client: Optional LLM for complex planning
        """
        self.llm = llm_client
        self.tool_registry = UDSToolRegistry()

    def create_plan(self, query: str, intent: IntentResult) -> TaskPlan:
        """
        Create execution plan for query.

        Args:
            query: User's natural language question
            intent: Classified intent

        Returns:
            TaskPlan with subtasks and execution order
        """
        if self._is_simple_query(query, intent):
            return self._create_simple_plan(query, intent)

        return self._create_complex_plan(query, intent)

    def _is_simple_query(self, query: str, intent: IntentResult) -> bool:
        """
        Check if query can be handled by a single tool.

        Args:
            query: User query
            intent: Classified intent

        Returns:
            True if simple query
        """
        query_lower = query.lower()

        complexity_indicators = [
            'and', 'then', 'also', 'with', 'plus',
            'compare', 'both', 'all', 'multiple'
        ]

        if any(indicator in query_lower for indicator in complexity_indicators):
            if intent.primary_domain == IntentDomain.COMPARISON:
                return True
            return False

        if intent.primary_domain in self.SIMPLE_PATTERNS:
            patterns = self.SIMPLE_PATTERNS[intent.primary_domain]['patterns']
            if any(pattern in query_lower for pattern in patterns):
                return True

        return True

    def _create_simple_plan(self, query: str, intent: IntentResult) -> TaskPlan:
        """
        Create plan for simple query (single tool).

        Args:
            query: User query
            intent: Classified intent

        Returns:
            TaskPlan with single subtask
        """
        tool_name = self._get_tool_for_domain(intent.primary_domain)
        parameters = self._extract_parameters(query, intent)

        subtask = Subtask(
            id="task_1",
            description=f"Execute {tool_name} for: {query}",
            tool_name=tool_name,
            parameters=parameters,
            dependencies=[],
            output_key="result"
        )

        return TaskPlan(
            query=query,
            intent=intent,
            subtasks=[subtask],
            execution_order=["task_1"]
        )

    def _create_complex_plan(self, query: str, intent: IntentResult) -> TaskPlan:
        """
        Create plan for complex query (multiple tools).

        Args:
            query: User query
            intent: Classified intent

        Returns:
            TaskPlan with multiple subtasks
        """
        steps = self._decompose_query(query, intent)
        subtasks = self._map_to_tools(steps, intent)
        execution_order = self._resolve_dependencies(subtasks)

        return TaskPlan(
            query=query,
            intent=intent,
            subtasks=subtasks,
            execution_order=execution_order
        )

    def _decompose_query(self, query: str, intent: IntentResult) -> List[str]:
        """
        Decompose complex query into steps.

        Args:
            query: User query
            intent: Classified intent

        Returns:
            List of step descriptions
        """
        if 'top' in query.lower() and 'inventory' in query.lower():
            return [
                "Get top products by revenue",
                "Get inventory levels for those products",
                "Combine results"
            ]

        elif 'compare' in query.lower() and 'show' in query.lower():
            return [
                "Compare periods",
                "Create visualization"
            ]

        elif 'sales' in query.lower() and 'product' in query.lower():
            return [
                "Analyze sales trends",
                "Analyze product performance",
                "Create dashboard"
            ]

        if ' and ' in query.lower():
            parts = query.lower().split(' and ')
            return [part.strip() for part in parts]

        return [query]

    def _map_to_tools(self, steps: List[str], intent: IntentResult) -> List[Subtask]:
        """
        Map decomposed steps to tools.

        Args:
            steps: List of step descriptions
            intent: Classified intent

        Returns:
            List of Subtasks
        """
        subtasks = []

        for i, step in enumerate(steps):
            task_id = f"task_{i+1}"
            step_lower = step.lower()

            if 'chart' in step_lower or 'visualiz' in step_lower or 'dashboard' in step_lower:
                tool_name = 'CreateChartTool'
                prev_task_id = f"task_{i}" if i > 0 else "task_1"
                params = {'data': f"${prev_task_id}.result"}

            elif 'inventory' in step_lower or 'stock' in step_lower:
                tool_name = 'InventoryAnalysisTool'
                params = self._extract_parameters(step, intent)

            elif 'compare' in step_lower or 'comparison' in step_lower:
                tool_name = 'ComparisonTool'
                params = self._extract_parameters(step, intent)

            elif 'top' in step_lower or 'product' in step_lower or 'performance' in step_lower:
                tool_name = 'ProductPerformanceTool'
                params = self._extract_parameters(step, intent)

            elif 'sales' in step_lower or 'revenue' in step_lower:
                tool_name = 'SalesTrendTool'
                params = self._extract_parameters(step, intent)

            elif 'combine' in step_lower or 'join' in step_lower:
                tool_name = 'ExecuteQueryTool'
                params = {'sql': 'TO_BE_GENERATED'}

            else:
                tool_name = 'ExecuteQueryTool'
                params = self._extract_parameters(step, intent)

            dependencies = []
            if i > 0:
                dependencies.append(f"task_{i}")

            subtask = Subtask(
                id=task_id,
                description=step,
                tool_name=tool_name,
                parameters=params,
                dependencies=dependencies,
                output_key=f"result_{i+1}"
            )

            subtasks.append(subtask)

        return subtasks

    def _resolve_dependencies(self, subtasks: List[Subtask]) -> List[str]:
        """
        Determine execution order based on dependencies.

        Args:
            subtasks: List of subtasks

        Returns:
            List of task IDs in execution order
        """
        order = []
        completed = set()

        while len(order) < len(subtasks):
            for task in subtasks:
                if task.id in completed:
                    continue

                if all(dep in completed for dep in task.dependencies):
                    order.append(task.id)
                    completed.add(task.id)

        return order

    def _get_tool_for_domain(self, domain: IntentDomain) -> str:
        """Get primary tool for domain."""
        if domain in self.SIMPLE_PATTERNS:
            return self.SIMPLE_PATTERNS[domain]['tool']
        return 'ExecuteQueryTool'

    def _extract_parameters(self, query: str, intent: IntentResult) -> Dict[str, Any]:
        """
        Extract parameters from query.

        Args:
            query: Query or step description
            intent: Classified intent

        Returns:
            Dictionary of parameters
        """
        params = {}
        query_lower = query.lower()

        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, query)

        if len(dates) >= 2:
            params['start_date'] = dates[0]
            params['end_date'] = dates[1]
        elif len(dates) == 1:
            params['as_of_date'] = dates[0]
        else:
            params['start_date'] = '2025-10-01'
            params['end_date'] = '2025-10-31'

        if 'top' in query_lower:
            numbers = re.findall(r'\d+', query)
            if numbers:
                params['limit'] = int(numbers[0])
            else:
                params['limit'] = 10

        if 'revenue' in query_lower:
            params['metric'] = 'revenue'
        elif 'units' in query_lower or 'quantity' in query_lower:
            params['metric'] = 'units'

        if 'low stock' in query_lower:
            params['low_stock_threshold'] = 10

        return params
