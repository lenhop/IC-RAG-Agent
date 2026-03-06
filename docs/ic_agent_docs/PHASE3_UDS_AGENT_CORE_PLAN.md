# Phase 3: UDS Agent Core - Implementation Plan

**Project:** IC-Agent / UDS Agent  
**Phase:** Phase 3 - UDS Agent Core  
**Status:** Planning  
**Owner:** Kiro (Project Manager)  
**Created:** 2026-03-06

---

## Overview

Phase 3 builds the UDS Agent Core - the intelligent orchestration layer that uses the 16 tools from Phase 2 to answer complex business questions about Amazon data.

**Prerequisites:**
- ✅ Phase 1: Data Foundation (schema docs, query templates, client)
- ✅ Phase 2: UDS Tools (16 tools: schema, query, analysis, visualization)

**Goal:** Create an autonomous agent that can:
- Understand natural language business questions
- Decompose complex queries into subtasks
- Select and execute appropriate tools
- Synthesize results into actionable insights
- Generate visualizations and reports

---

## Architecture

```
User Question
    ↓
Intent Classifier → [sales | inventory | financial | product | general]
    ↓
Task Planner → [subtask1, subtask2, subtask3]
    ↓
UDS Agent (ReAct Loop)
    ├→ Tool Selection (from 16 tools)
    ├→ Tool Execution
    ├→ Result Synthesis
    └→ Visualization Generation
    ↓
Formatted Response (text + charts + insights)
```

---

## Components

### 3.1 Intent Classification System
**Purpose:** Classify user questions into business domains

**Domains:**
- `sales` - Revenue, orders, trends, growth
- `inventory` - Stock levels, turnover, alerts
- `financial` - Profitability, fees, costs
- `product` - Product performance, rankings
- `comparison` - Period/product/marketplace comparisons
- `general` - Schema exploration, data questions

**Implementation:**
- LLM-based classification with few-shot examples
- Keyword matching fallback
- Confidence scoring
- Multi-domain support (e.g., "sales + inventory")

**Owner:** Cursor

---

### 3.2 Task Planner
**Purpose:** Decompose complex questions into executable subtasks

**Capabilities:**
- Break down multi-step questions
- Identify data dependencies
- Determine execution order
- Select appropriate tools for each subtask

**Example:**
```
Question: "Show me top 10 products by revenue and their current inventory levels"

Plan:
1. Use ProductPerformanceTool to get top 10 products by revenue
2. Extract SKUs from results
3. Use InventoryAnalysisTool to get inventory for those SKUs
4. Join results
5. Use CreateChartTool to visualize
```

**Owner:** TRAE

---

### 3.3 UDS Agent Implementation
**Purpose:** Main agent orchestrating all components

**Features:**
- Extends ReActAgent from `src/agent/react_agent.py`
- Registers all 16 UDS tools
- Implements reasoning loop (Thought → Action → Observation)
- Context management (track intermediate results)
- Error handling and recovery
- Result formatting

**Key Methods:**
```python
class UDSAgent(ReActAgent):
    def __init__(self, uds_client: UDSClient):
        # Initialize with all tools
        
    def classify_intent(self, query: str) -> IntentResult:
        # Classify user question
        
    def plan_tasks(self, query: str, intent: IntentResult) -> TaskPlan:
        # Create execution plan
        
    def execute_plan(self, plan: TaskPlan) -> AgentResult:
        # Execute ReAct loop
        
    def format_response(self, result: AgentResult) -> FormattedResponse:
        # Format with text, charts, insights
```

**Owner:** TRAE

---

### 3.4 Context Enrichment
**Purpose:** Add relevant context to improve agent performance

**Context Types:**
- Schema context (table descriptions, relationships)
- Example queries (similar questions from history)
- Business glossary (Amazon terminology)
- Data statistics (date ranges, row counts)

**Implementation:**
- Load schema metadata on initialization
- Inject context into system prompt
- Dynamic context based on intent
- Token budget management

**Owner:** VSCode

---

### 3.5 Result Formatter
**Purpose:** Transform agent output into user-friendly format

**Output Formats:**
- Text summary (key insights, recommendations)
- Data tables (formatted with markdown)
- Charts (embedded HTML or image)
- Downloadable reports (CSV, PDF)

**Features:**
- Automatic insight generation
- Chart selection based on data type
- Multi-chart dashboards
- Export functionality

**Owner:** Cursor

---

## Detailed Component Specs

### Component 3.1: Intent Classification System

**File:** `src/uds/intent_classifier.py`

**Classes:**
```python
@dataclass
class IntentResult:
    primary_domain: str  # sales, inventory, financial, product, comparison, general
    secondary_domains: List[str]  # Additional domains
    confidence: float  # 0.0 to 1.0
    keywords: List[str]  # Extracted keywords
    suggested_tools: List[str]  # Tool names

class UDSIntentClassifier:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.examples = self._load_examples()
        
    def classify(self, query: str) -> IntentResult:
        # LLM-based classification with few-shot
        
    def _keyword_fallback(self, query: str) -> IntentResult:
        # Keyword-based classification
        
    def _load_examples(self) -> Dict[str, List[str]]:
        # Load few-shot examples
```

**Few-shot Examples:**
```python
INTENT_EXAMPLES = {
    "sales": [
        "What were total sales in October?",
        "Show me daily revenue trends",
        "Which day had the highest sales?"
    ],
    "inventory": [
        "What products are low on stock?",
        "Show me inventory levels for SKU-123",
        "Which items need reordering?"
    ],
    "financial": [
        "What's our profit margin?",
        "Show me fee breakdown",
        "Calculate net revenue after fees"
    ],
    "product": [
        "Top 10 products by revenue",
        "Which products are underperforming?",
        "Product performance comparison"
    ],
    "comparison": [
        "Compare week 1 vs week 2 sales",
        "How did October compare to September?",
        "Product A vs Product B performance"
    ],
    "general": [
        "What tables are available?",
        "Describe the amz_order table",
        "How many orders do we have?"
    ]
}
```

**Tests:** `tests/test_intent_classifier.py`

---

### Component 3.2: Task Planner

**File:** `src/uds/task_planner.py`

**Classes:**
```python
@dataclass
class Subtask:
    id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of subtasks that must complete first
    
@dataclass
class TaskPlan:
    query: str
    intent: IntentResult
    subtasks: List[Subtask]
    execution_order: List[str]  # Subtask IDs in order

class UDSTaskPlanner:
    def __init__(self, llm_client, tool_registry):
        self.llm = llm_client
        self.tools = tool_registry
        
    def create_plan(self, query: str, intent: IntentResult) -> TaskPlan:
        # Generate execution plan
        
    def _decompose_query(self, query: str) -> List[str]:
        # Break into subtasks
        
    def _map_to_tools(self, subtasks: List[str]) -> List[Subtask]:
        # Map subtasks to tools
        
    def _resolve_dependencies(self, subtasks: List[Subtask]) -> List[str]:
        # Determine execution order
```

**Planning Strategies:**
- Simple queries → Single tool execution
- Complex queries → Multi-step decomposition
- Data dependencies → Sequential execution
- Independent subtasks → Parallel execution (future)

**Tests:** `tests/test_task_planner.py`

---

### Component 3.3: UDS Agent Implementation

**File:** `src/uds/uds_agent.py`

**Main Class:**
```python
class UDSAgent(ReActAgent):
    """
    UDS Agent for business intelligence queries.
    Extends ReActAgent with UDS-specific capabilities.
    """
    
    def __init__(
        self,
        uds_client: UDSClient,
        llm_client,
        intent_classifier: UDSIntentClassifier,
        task_planner: UDSTaskPlanner,
        result_formatter: UDSResultFormatter
    ):
        super().__init__(name="UDS Agent", llm=llm_client)
        
        self.uds_client = uds_client
        self.intent_classifier = intent_classifier
        self.task_planner = task_planner
        self.result_formatter = result_formatter
        
        # Register all 16 tools
        self._register_tools()
        
        # Load schema context
        self.schema_context = self._load_schema_context()
        
    def _register_tools(self):
        """Register all UDS tools."""
        from src.uds.tools import UDSToolRegistry
        
        for tool in UDSToolRegistry.get_all_tools():
            self.register_tool(tool)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing user queries.
        
        Args:
            query: Natural language business question
            
        Returns:
            Formatted response with insights, data, and visualizations
        """
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        
        # Step 2: Create task plan
        plan = self.task_planner.create_plan(query, intent)
        
        # Step 3: Enrich context
        context = self._build_context(query, intent, plan)
        
        # Step 4: Execute ReAct loop
        result = self.run(query, context)
        
        # Step 5: Format response
        formatted = self.result_formatter.format(result, intent)
        
        return formatted
    
    def _build_context(
        self,
        query: str,
        intent: IntentResult,
        plan: TaskPlan
    ) -> str:
        """Build enriched context for the agent."""
        context_parts = []
        
        # Schema context
        if intent.primary_domain != 'general':
            relevant_tables = self._get_relevant_tables(intent)
            context_parts.append(f"Relevant tables: {', '.join(relevant_tables)}")
        
        # Business glossary
        glossary = self._get_glossary_terms(query)
        if glossary:
            context_parts.append(f"Terminology: {glossary}")
        
        # Example queries
        examples = self._get_similar_examples(query, intent)
        if examples:
            context_parts.append(f"Similar queries: {examples}")
        
        # Task plan
        context_parts.append(f"Execution plan: {plan.execution_order}")
        
        return "\n\n".join(context_parts)
    
    def _get_relevant_tables(self, intent: IntentResult) -> List[str]:
        """Get tables relevant to the intent."""
        domain_tables = {
            'sales': ['amz_order', 'amz_transaction'],
            'inventory': ['amz_fba_inventory_all', 'amz_daily_inventory_ledger'],
            'financial': ['amz_transaction', 'amz_statement', 'amz_fee'],
            'product': ['amz_product', 'amz_order', 'amz_listing_item']
        }
        return domain_tables.get(intent.primary_domain, [])
    
    def _get_glossary_terms(self, query: str) -> str:
        """Extract relevant glossary terms."""
        # Load from uds_business_glossary.json
        pass
    
    def _get_similar_examples(self, query: str, intent: IntentResult) -> str:
        """Find similar example queries."""
        # Vector similarity search (future enhancement)
        pass
```

**Tests:** `tests/test_uds_agent.py`

---

### Component 3.4: Context Enrichment

**File:** `src/uds/context_enricher.py`

**Classes:**
```python
class ContextEnricher:
    """
    Enriches agent context with relevant information.
    """
    
    def __init__(self, schema_metadata_path: str, glossary_path: str):
        self.schema = self._load_schema(schema_metadata_path)
        self.glossary = self._load_glossary(glossary_path)
        
    def enrich(
        self,
        query: str,
        intent: IntentResult,
        max_tokens: int = 2000
    ) -> str:
        """
        Build enriched context within token budget.
        
        Args:
            query: User query
            intent: Classified intent
            max_tokens: Maximum context tokens
            
        Returns:
            Enriched context string
        """
        context_parts = []
        
        # Priority 1: Schema for relevant tables
        schema_context = self._get_schema_context(intent)
        context_parts.append(schema_context)
        
        # Priority 2: Business glossary
        glossary_context = self._get_glossary_context(query)
        context_parts.append(glossary_context)
        
        # Priority 3: Data statistics
        stats_context = self._get_statistics_context(intent)
        context_parts.append(stats_context)
        
        # Truncate to fit token budget
        return self._truncate_to_budget(context_parts, max_tokens)
    
    def _get_schema_context(self, intent: IntentResult) -> str:
        """Get schema info for relevant tables."""
        pass
    
    def _get_glossary_context(self, query: str) -> str:
        """Get relevant glossary terms."""
        pass
    
    def _get_statistics_context(self, intent: IntentResult) -> str:
        """Get data statistics."""
        pass
    
    def _truncate_to_budget(self, parts: List[str], max_tokens: int) -> str:
        """Truncate context to fit token budget."""
        pass
```

**Supporting Files:**
- `src/uds/uds_business_glossary.json` - Amazon terminology
- `src/uds/uds_schema_metadata.json` - Already exists
- `src/uds/uds_statistics.json` - Data statistics

**Tests:** `tests/test_context_enricher.py`

---

### Component 3.5: Result Formatter

**File:** `src/uds/result_formatter.py`

**Classes:**
```python
@dataclass
class FormattedResponse:
    summary: str  # Text summary
    insights: List[str]  # Key insights
    data: Optional[Dict]  # Structured data
    charts: List[Dict]  # Chart configurations
    recommendations: List[str]  # Action items
    metadata: Dict  # Query metadata

class UDSResultFormatter:
    """
    Formats agent results into user-friendly output.
    """
    
    def __init__(self):
        self.chart_tool = CreateChartTool()
        self.dashboard_tool = CreateDashboardTool()
        
    def format(
        self,
        agent_result: AgentResult,
        intent: IntentResult
    ) -> FormattedResponse:
        """
        Format agent result based on intent.
        
        Args:
            agent_result: Raw agent output
            intent: Query intent
            
        Returns:
            Formatted response
        """
        # Extract data from agent result
        data = self._extract_data(agent_result)
        
        # Generate summary
        summary = self._generate_summary(data, intent)
        
        # Extract insights
        insights = self._extract_insights(data, intent)
        
        # Create visualizations
        charts = self._create_visualizations(data, intent)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, intent)
        
        return FormattedResponse(
            summary=summary,
            insights=insights,
            data=data,
            charts=charts,
            recommendations=recommendations,
            metadata={
                'intent': intent.primary_domain,
                'confidence': intent.confidence,
                'tools_used': agent_result.tools_used
            }
        )
    
    def _extract_data(self, result: AgentResult) -> Dict:
        """Extract structured data from agent result."""
        pass
    
    def _generate_summary(self, data: Dict, intent: IntentResult) -> str:
        """Generate text summary."""
        pass
    
    def _extract_insights(self, data: Dict, intent: IntentResult) -> List[str]:
        """Extract key insights."""
        pass
    
    def _create_visualizations(self, data: Dict, intent: IntentResult) -> List[Dict]:
        """Create appropriate charts."""
        # Use CreateChartTool and CreateDashboardTool
        pass
    
    def _generate_recommendations(self, data: Dict, intent: IntentResult) -> List[str]:
        """Generate action items."""
        pass
```

**Chart Selection Logic:**
```python
CHART_MAPPING = {
    'sales': {
        'trend': 'line',  # Time series
        'comparison': 'bar',  # Category comparison
        'distribution': 'pie'  # Proportions
    },
    'inventory': {
        'levels': 'bar',
        'alerts': 'table',
        'trend': 'line'
    },
    'financial': {
        'breakdown': 'pie',
        'trend': 'line',
        'comparison': 'bar'
    },
    'product': {
        'ranking': 'bar',
        'performance': 'scatter',
        'trend': 'line'
    }
}
```

**Tests:** `tests/test_result_formatter.py`

---

## Integration Points

### With Phase 1 Components
- Uses `UDSClient` for database access
- Uses `QueryTemplateRegistry` for common queries
- Uses schema metadata and statistics

### With Phase 2 Tools
- Registers all 16 tools in UDSAgent
- Uses tools through ReAct loop
- Formats tool results

### With Existing Agent Infrastructure
- Extends `ReActAgent` from `src/agent/react_agent.py`
- Uses `ToolExecutor` from ai-toolkit
- Uses `AgentLogger` for observability

---

## Testing Strategy

### Unit Tests
- Intent classification accuracy (>90%)
- Task planning correctness
- Context enrichment quality
- Result formatting

### Integration Tests
- End-to-end query processing
- Multi-tool workflows
- Error handling and recovery
- Performance benchmarks

### Test Queries
```python
TEST_QUERIES = [
    # Simple queries (single tool)
    "What were total sales in October?",
    "Show me current inventory levels",
    "List all available tables",
    
    # Medium complexity (2-3 tools)
    "Top 10 products by revenue with their inventory",
    "Daily sales trend with growth rate",
    "Financial summary with fee breakdown",
    
    # Complex queries (4+ tools)
    "Compare week 1 vs week 2 sales, show top products, and create dashboard",
    "Analyze product performance, identify low stock items, and recommend actions",
    "Full business report: sales, inventory, financial, with visualizations"
]
```

---

## Timeline

**Total Duration:** 2 weeks (Week 5-6)

### Week 5
- **Days 1-2:** Intent Classification + Context Enrichment (Cursor + VSCode)
- **Days 3-5:** Task Planner + UDS Agent Core (TRAE)

### Week 6
- **Days 1-2:** Result Formatter (Cursor)
- **Days 3-4:** Integration Testing (All)
- **Day 5:** Documentation and Refinement (All)

---

## Success Criteria

✅ **Intent Classification:**
- Accuracy >90% on test queries
- Handles multi-domain queries
- Provides confidence scores

✅ **Task Planning:**
- Correctly decomposes complex queries
- Identifies dependencies
- Selects appropriate tools

✅ **UDS Agent:**
- Processes queries end-to-end
- Uses tools effectively
- Handles errors gracefully
- Response time <10 seconds for simple queries

✅ **Context Enrichment:**
- Provides relevant schema info
- Stays within token budget
- Improves agent performance

✅ **Result Formatting:**
- Clear, actionable summaries
- Appropriate visualizations
- Useful recommendations

---

## Deliverables

### Code
- [ ] `src/uds/intent_classifier.py`
- [ ] `src/uds/task_planner.py`
- [ ] `src/uds/uds_agent.py`
- [ ] `src/uds/context_enricher.py`
- [ ] `src/uds/result_formatter.py`
- [ ] `src/uds/uds_business_glossary.json`

### Tests
- [ ] `tests/test_intent_classifier.py`
- [ ] `tests/test_task_planner.py`
- [ ] `tests/test_uds_agent.py`
- [ ] `tests/test_context_enricher.py`
- [ ] `tests/test_result_formatter.py`
- [ ] `tests/test_uds_integration.py`

### Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Architecture diagrams
- [ ] Performance benchmarks

---

## Next Phase

**Phase 4:** Integration & Testing
- RAG integration for documentation
- REST API development
- Comprehensive testing
- Performance optimization

---

**Document Owner:** Kiro (AI Project Manager)  
**Last Updated:** 2026-03-06  
**Status:** Ready for team assignment
