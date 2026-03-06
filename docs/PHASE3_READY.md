# Phase 3: UDS Agent Core - READY TO START

**Duration:** 2 weeks (Week 5-6)  
**Status:** 🚀 Ready  
**Date:** 2026-03-06

---

## What We're Building

The intelligent brain that orchestrates all 16 tools to answer business questions.

```
User Question → Intent Classifier → Task Planner → UDS Agent → Result Formatter → Response
```

---

## Team Tasks

### Cursor (4-5 days)
**Task:** `tasks/20260306-phase3-000019-intent-classification-cursor.md`

Build:
1. Intent Classifier - Classify queries into 6 domains (sales, inventory, financial, product, comparison, general)
2. Result Formatter - Generate summaries, insights, charts, recommendations

Files: `src/uds/intent_classifier.py`, `src/uds/result_formatter.py`

---

### TRAE (5-6 days)
**Task:** `tasks/20260306-phase3-000020-task-planner-agent-trae.md`

Build:
1. Task Planner - Decompose complex queries into subtasks
2. UDS Agent - Main orchestration engine (extends ReActAgent)

Files: `src/uds/task_planner.py`, `src/uds/uds_agent.py`

---

### VSCode (3-4 days)
**Task:** `tasks/20260306-phase3-000021-context-enrichment-vscode.md`

Build:
1. Business Glossary - Amazon terminology (18+ terms)
2. Context Enricher - Provide relevant context to agent
3. Data Statistics - Row counts, date ranges, metrics

Files: `src/uds/uds_business_glossary.json`, `src/uds/context_enricher.py`, `src/uds/uds_statistics.json`

---

## Timeline

**Week 5:** Build components
- Days 1-2: Intent Classifier, Task Planner, Business Glossary
- Days 3-5: Result Formatter, UDS Agent, Context Enricher

**Week 6:** Integration & testing
- Days 1-2: Component testing
- Days 3-4: Integration testing
- Day 5: Documentation

---

## Test Queries

**Simple:** "What were total sales in October?"  
**Medium:** "Top 10 products with their inventory"  
**Complex:** "Compare sales, show top products, create dashboard"

---

## Success Criteria

- Intent classification >90% accurate
- Query decomposition correct
- End-to-end processing works
- Response time <10s for simple queries
- Clear, actionable responses

---

## Detailed Plans

- Architecture: `docs/ic_agent_docs/PHASE3_UDS_AGENT_CORE_PLAN.md`
- Summary: `docs/ic_agent_docs/PHASE3_SUMMARY.md`

---

**Ready to start!** 🚀
