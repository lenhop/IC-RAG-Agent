# Phase 3: UDS Agent Core - Executive Summary

**Project:** IC-Agent / UDS Agent  
**Phase:** Phase 3 - UDS Agent Core  
**Created:** 2026-03-06  
**Status:** Ready for Execution

---

## Overview

Phase 3 builds the intelligent orchestration layer that transforms 16 individual tools into a cohesive UDS Agent capable of answering complex business questions.

**Duration:** 2 weeks (Week 5-6)  
**Team:** Cursor, TRAE, VSCode  
**Deliverables:** 5 core components + integration

---

## What We're Building

```
User: "Show me top 10 products by revenue with their inventory levels"

    ↓
[Intent Classifier] → "product + inventory" (Cursor)
    ↓
[Task Planner] → "1. Get top products, 2. Get inventory, 3. Join" (TRAE)
    ↓
[Context Enricher] → "Tables: amz_order, amz_fba_inventory_all..." (VSCode)
    ↓
[UDS Agent] → Executes tools via ReAct loop (TRAE)
    ↓
[Result Formatter] → Summary + Chart + Insights (Cursor)
    ↓
Response: "Top 10 Products with Inventory
1. Product A: $50K revenue, 150 units in stock
2. Product B: $45K revenue, 80 units in stock
..."
```

---

## Components

### 1. Intent Classification (Cursor)
**File:** `src/uds/intent_classifier.py`

Classifies user queries into 6 business domains:
- Sales - Revenue, orders, trends
- Inventory - Stock levels, alerts
- Financial - Profitability, fees
- Product - Performance, rankings
- Comparison - Period/product comparisons
- General - Schema exploration

**Features:**
- LLM-based classification with few-shot examples
- Keyword fallback for reliability
- Confidence scoring
- Tool suggestions per domain

**Target:** >90% accuracy

---

### 2. Task Planner (TRAE)
**File:** `src/uds/task_planner.py`

Decomposes complex queries into executable subtasks:

**Example:**
```
Query: "Top products with inventory"

Plan:
1. ProductPerformanceTool(limit=10, metric='revenue')
2. InventoryAnalysisTool(skus=<from_step_1>)
3. Join results
```

**Features:**
- Simple vs complex query detection
- Query decomposition
- Tool mapping
- Dependency resolution
- Parameter extraction

---

### 3. Context Enricher (VSCode)
**Files:** 
- `src/uds/context_enricher.py`
- `src/uds/uds_business_glossary.json`
- `src/uds/uds_statistics.json`

Provides relevant context to improve agent performance:

**Context Types:**
- Schema information (table descriptions, columns)
- Business glossary (Amazon terminology)
- Data statistics (row counts, date ranges)
- Example queries

**Features:**
- Token budget management
- Priority-based context selection
- Domain-specific context

---

### 4. UDS Agent (TRAE)
**File:** `src/uds/uds_agent.py`

Main orchestration engine extending ReActAgent:

**Workflow:**
1. Classify intent
2. Create task plan
3. Enrich context
4. Execute ReAct loop (Thought → Action → Observation)
5. Format results

**Features:**
- Registers all 16 tools
- Reasoning loop
- Error handling
- Context management

**Target:** 
- >95% success on simple queries
- >80% success on complex queries
- <10s response time for simple queries

---

### 5. Result Formatter (Cursor)
**File:** `src/uds/result_formatter.py`

Transforms agent output into user-friendly format:

**Output Components:**
- Text summary (key metrics, trends)
- Insights (growth rates, alerts, patterns)
- Visualizations (charts, dashboards)
- Recommendations (action items)

**Features:**
- Domain-specific formatting
- Automatic chart selection
- Insight extraction
- Recommendation generation

---

## Team Assignments

| Component | Owner | Duration | Priority |
|-----------|-------|----------|----------|
| Intent Classifier | Cursor | 2 days | High |
| Result Formatter | Cursor | 2-3 days | High |
| Task Planner | TRAE | 3 days | High |
| UDS Agent | TRAE | 3 days | High |
| Context Enricher | VSCode | 2 days | High |
| Business Glossary | VSCode | 1 day | High |
| Data Statistics | VSCode | 1 day | Medium |

**Total:** Cursor (4-5 days), TRAE (5-6 days), VSCode (3-4 days)

---

## Timeline

### Week 5: Component Development

**Days 1-2:**
- Cursor: Intent Classifier
- TRAE: Task Planner
- VSCode: Business Glossary

**Days 3-4:**
- Cursor: Intent testing + Result Formatter start
- TRAE: Task Planner testing + UDS Agent start
- VSCode: Context Enricher

**Day 5:**
- Cursor: Result Formatter
- TRAE: UDS Agent
- VSCode: Data Statistics + testing

### Week 6: Integration & Testing

**Days 1-2:**
- Cursor: Result Formatter testing
- TRAE: UDS Agent testing
- VSCode: Integration support

**Days 3-4:**
- All: Integration testing with real queries
- All: Bug fixes and refinements

**Day 5:**
- All: Documentation
- All: Final testing
- All: Completion reports

---

## Test Strategy

### Test Queries

**Simple (Single Tool):**
1. "What were total sales in October?"
2. "Show me current inventory levels"
3. "Top 10 products by revenue"

**Medium (2-3 Tools):**
4. "Top 10 products with their inventory"
5. "Daily sales trend with growth rate"
6. "Compare week 1 vs week 2 sales"

**Complex (4+ Tools):**
7. "Compare sales, show top products, create dashboard"
8. "Analyze performance, identify low stock, recommend actions"
9. "Full business report with visualizations"

### Success Metrics

**Component Level:**
- Intent classification: >90% accuracy
- Task planning: Correct decomposition
- Context enrichment: Relevant and within budget
- UDS Agent: >95% success on simple queries
- Result formatting: Clear and actionable

**Integration Level:**
- All test queries processed successfully
- Accurate and useful responses
- Visualizations render correctly
- Graceful error handling

---

## Technical Architecture

### Class Hierarchy

```
ReActAgent (from src/agent/react_agent.py)
    ↓
UDSAgent (extends ReActAgent)
    ├── UDSIntentClassifier
    ├── UDSTaskPlanner
    ├── ContextEnricher
    ├── UDSResultFormatter
    └── UDSToolRegistry (16 tools)
```

### Data Flow

```
1. User Query
   ↓
2. UDSAgent.process_query()
   ├→ intent_classifier.classify()
   ├→ task_planner.create_plan()
   ├→ context_enricher.enrich()
   ├→ self.run() [ReAct loop]
   └→ result_formatter.format()
   ↓
3. Formatted Response
```

### Integration Points

- **Phase 1:** Uses UDSClient, QueryTemplateRegistry, schema metadata
- **Phase 2:** Uses all 16 tools via UDSToolRegistry
- **Existing:** Extends ReActAgent, uses ToolExecutor

---

## Deliverables

### Code Files
- [ ] `src/uds/intent_classifier.py`
- [ ] `src/uds/task_planner.py`
- [ ] `src/uds/uds_agent.py`
- [ ] `src/uds/context_enricher.py`
- [ ] `src/uds/result_formatter.py`
- [ ] `src/uds/uds_business_glossary.json`
- [ ] `src/uds/uds_statistics.json`

### Test Files
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

## Risk Management

### Identified Risks

1. **Integration Complexity**
   - Risk: Components don't integrate smoothly
   - Mitigation: Early integration testing, clear interfaces
   - Owner: All

2. **LLM Performance**
   - Risk: Intent classification or planning inaccurate
   - Mitigation: Keyword fallbacks, extensive testing
   - Owner: Cursor, TRAE

3. **Token Budget**
   - Risk: Context exceeds LLM limits
   - Mitigation: Token budget management, prioritization
   - Owner: VSCode

4. **Query Complexity**
   - Risk: Agent can't handle complex queries
   - Mitigation: Incremental complexity, fallback strategies
   - Owner: TRAE

### Contingency Plans

- LLM classification fails → Keyword-based fallback
- Task planning fails → Single-tool execution
- Context too large → Prioritize and truncate
- Agent fails → Error message with helpful guidance

---

## Success Criteria

### Phase 3 Complete When:

✅ **All Components Implemented:**
- [ ] Intent Classifier working with >90% accuracy
- [ ] Task Planner correctly decomposing queries
- [ ] Context Enricher providing relevant context
- [ ] UDS Agent processing queries end-to-end
- [ ] Result Formatter generating beautiful responses

✅ **Integration Successful:**
- [ ] All 15 test queries pass
- [ ] Simple queries: >95% success rate
- [ ] Complex queries: >80% success rate
- [ ] Response time: <10s for simple, <30s for complex

✅ **Quality Standards Met:**
- [ ] Test coverage >80%
- [ ] All edge cases handled
- [ ] Error messages are helpful
- [ ] Documentation complete

✅ **Team Deliverables:**
- [ ] All code files committed
- [ ] All tests passing
- [ ] Completion reports filed
- [ ] Integration verified

---

## Next Phase

**Phase 4:** Integration & Testing (Week 7-8)

Components:
- RAG integration for documentation retrieval
- REST API development (query endpoints, streaming)
- Comprehensive testing (unit, integration, performance)
- Performance optimization

---

## Task Files

### For Team Members
- Cursor: `tasks/20260306-000019-intent-classification-cursor.md`
- TRAE: `tasks/20260306-000020-task-planner-agent-trae.md`
- VSCode: `tasks/20260306-000021-context-enrichment-vscode.md`

### For Project Manager
- Team Assignments: `tasks/20260306-phase3-team-assignments.md`
- Kickoff Prompts: `tasks/20260306-phase3-kickoff-prompts.md`
- This Summary: `docs/ic_agent_docs/PHASE3_SUMMARY.md`

### Reference
- Detailed Plan: `docs/ic_agent_docs/PHASE3_UDS_AGENT_CORE_PLAN.md`
- Foundation Plan: `docs/ic_agent_docs/UDS_DATA_FOUNDATION_PLAN.md`

---

## Key Decisions

1. **Extend ReActAgent:** UDS Agent extends existing ReActAgent infrastructure
2. **LLM + Fallback:** Use LLM for intelligence with keyword fallbacks for reliability
3. **Token Budget:** Manage context size to stay within LLM limits
4. **Incremental Complexity:** Start with simple queries, add complexity gradually
5. **Domain-Based:** Organize by business domains (sales, inventory, etc.)

---

## Communication

### Daily Updates
Team members post daily updates in task files

### Integration Checkpoints
- Week 5, Day 3: Intent Classifier demo
- Week 5, Day 4: Task Planner demo
- Week 5, Day 5: Context Enricher demo
- Week 6, Day 1: UDS Agent demo
- Week 6, Day 2: Result Formatter demo

### Completion Reports
Filed when components are complete

---

**Document Owner:** Kiro (AI Project Manager)  
**Last Updated:** 2026-03-06  
**Status:** Ready for Execution

---

## Quick Start

**For Cursor:**
```bash
# Read your task
cat tasks/20260306-000019-intent-classification-cursor.md

# Start with Intent Classifier
# File: src/uds/intent_classifier.py
```

**For TRAE:**
```bash
# Read your task
cat tasks/20260306-000020-task-planner-agent-trae.md

# Start with Task Planner
# File: src/uds/task_planner.py
```

**For VSCode:**
```bash
# Read your task
cat tasks/20260306-000021-context-enrichment-vscode.md

# Start with Business Glossary
# File: src/uds/uds_business_glossary.json
```

Let's build something amazing! 🚀
