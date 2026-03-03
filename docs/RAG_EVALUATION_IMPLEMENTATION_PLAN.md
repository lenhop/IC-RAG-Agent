# RAG Evaluation Implementation Plan

**Project:** IC-RAG-Agent Evaluation Framework  
**Based on:** RAG_EVALUATION_BEST_PRACTICES_FINAL.md  
**Dataset:** 10 FAQ from amazon_fqa.csv (expandable to 50+)  
**Timeline:** 5 days  
**Status:** Implementation Complete (Phases 1-4)  
**Last Review:** 2025-02-23 (implementation notes integrated)

---

## Implementation Notes (Review)

- **Retrieval API:** Use `ai_toolkit.chroma.get_chroma_collection(vector_store)` and `query_collection()` for retrieval. Do NOT use `pipeline.vector_store._collection` (private API).
- **Contexts field:** For reliable Recall/Precision/MRR, `contexts` (ideal retrieval chunks) is required. `add_relevant_contexts()` auto-retrieves "what was retrieved" - use as fallback only. Prefer manual `contexts` in CSV for serious evaluation.
- **Dataset path:** Default `--dataset` to `RAG_FAQ_CSV` or `data/intent_classification/fqa/amazon_fqa.csv` for consistency with FAQ loader.
- **Code reuse:** Centralize `load_test_dataset()` in `dataset_loader.py`; remove duplicates from `test_retrieval_metrics.py` and `test_generation_llm_judge.py`.
- **Existing code:** `tests/evaluation/test_retrieval_metrics.py` and `test_generation_llm_judge.py` have metric logic and dataset loading - migrate to `src/rag/evaluation/`, keep tests for unit/integration only.

---

## Project Overview

Implement comprehensive RAG evaluation framework with:
- **Retrieval metrics:** Recall@5, Precision@5, MRR
- **Generation metrics:** Faithfulness (≥85%), Relevance (≥4/5)
- **Visualization:** UMAP embedding space
- **Automation:** Single-command evaluation pipeline

---

## Phase 1: Core Evaluation Tools (Priority: High)

### 1.1 Retrieval Metrics Module
**File:** `src/rag/evaluation/retrieval_metrics.py`

**Tools to build:**
- `calculate_recall_at_k(retrieved_docs, relevant_contexts, k=5)` → float
  - Formula: (relevant chunks in top-k) / (total relevant chunks)
  - Target: Higher is better
  
- `calculate_precision_at_k(retrieved_docs, relevant_contexts, k=5)` → float
  - Formula: (relevant chunks in top-k) / k
  - Target: Higher is better
  
- `calculate_mrr(retrieved_docs, relevant_contexts)` → float
  - Formula: 1 / rank of first relevant document
  - Target: Higher is better
  
- `RetrievalEvaluator` class
  - Method: `evaluate_batch(pipeline, test_cases, k=5)` → Dict
  - Retrieve via `get_chroma_collection(pipeline.vector_store)` + `query_collection` (public API)
  - Output: Per-case metrics + aggregated averages
  - JSON export for analysis

**Deliverables:**
- ✅ Reusable metric functions
- ✅ Batch evaluation on test dataset
- ✅ JSON output with per-case and aggregate results
- ✅ Integration with existing RAGPipeline

**Testing:**
- Unit tests with mock retrieved_docs and relevant_contexts
- Verify edge cases: empty results, no relevant docs, all relevant

**Estimated Time:** 4 hours

---

### 1.2 Generation Metrics Module
**File:** `src/rag/evaluation/generation_metrics.py`

**Tools to build:**
- `evaluate_faithfulness(answer, contexts, judge_llm)` → Dict
  - Prompt: "Is answer faithful to contexts? No hallucination?"
  - Output: `{"is_faithful": bool, "reasoning": str}`
  - Target: ≥85% faithful across test set
  
- `evaluate_relevance(question, answer, judge_llm)` → Dict
  - Prompt: "Rate answer quality 1-5 for question"
  - Output: `{"relevance_score": int, "reasoning": str}`
  - Target: ≥4.0 average score
  
- `GenerationEvaluator` class
  - Method: `evaluate_batch(pipeline, test_cases, mode="hybrid")` → Dict
  - Configurable judge LLM (default: pipeline.llm_general)
  - Parallel evaluation with progress bar

**Deliverables:**
- ✅ Structured JSON prompts for LLM judges
- ✅ Robust parsing logic for judge responses
- ✅ Fallback handling for malformed LLM outputs
- ✅ Summary statistics: faithfulness rate, avg relevance

**Testing:**
- Mock LLM responses (faithful/unfaithful, high/low relevance)
- Verify JSON parsing with edge cases
- Test with actual Deepseek/Qwen API

**Estimated Time:** 6 hours

---

### 1.3 Dataset Loader
**File:** `src/rag/evaluation/dataset_loader.py`

**Tools to build:**
- `load_fqa_dataset(csv_path, limit=None)` → List[Dict]
  - Load from amazon_fqa.csv
  - Fields: id, question, ground_truth (answer), category, source
  - Support limit for quick testing
  
- `validate_dataset(test_cases)` → bool
  - Check required fields present
  - Warn if ground_truth or contexts missing
  
- `add_relevant_contexts(test_cases, pipeline)` → List[Dict]
  - Fallback when `contexts` missing: auto-retrieve top-k for each question
  - If CSV has `contexts` column (chunk IDs or content), use it for ground-truth retrieval eval
  - Manual annotation preferred for reliable Recall/Precision/MRR

**Deliverables:**
- ✅ Standardized dataset format
- ✅ Support for 10 FAQ initial set
- ✅ Expandable to 50+ with same interface
- ✅ Validation warnings for incomplete data

**Testing:**
- Load amazon_fqa.csv and verify structure
- Test with missing fields
- Verify limit parameter works

**Estimated Time:** 2 hours

---

## Phase 2: Visualization Tools (Priority: Medium)

### 2.1 UMAP Embedding Visualization
**File:** `src/rag/evaluation/visualize_umap.py`

**Tools to build:**
- `generate_umap_plot(test_cases, pipeline, output_path)` → str
  - Embed all queries and relevant chunks using pipeline.embedder
  - UMAP reduce to 2D (n_neighbors=15, min_dist=0.1)
  - Color coding:
    - Red: Query embeddings
    - Green: Relevant chunks (from ground truth)
    - Gray: Irrelevant chunks (sampled from collection)
  - Interactive Plotly HTML + static PNG
  
- `annotate_outliers(plot, test_cases, threshold=2.0)` → None
  - Highlight queries far from relevant chunks
  - Add hover text with question ID and distance

**Deliverables:**
- ✅ UMAP scatter plot (HTML + PNG)
- ✅ Annotated outliers for manual inspection
- ✅ Legend and axis labels
- ✅ Saved to `tests/evaluation/umap_visualization.html`

**Dependencies:**
- `pip install umap-learn plotly`

**Testing:**
- Generate plot with 10 FAQ
- Verify color coding correct
- Check outlier detection logic

**Estimated Time:** 4 hours

---

### 2.2 Evaluation Report Generator
**File:** `src/rag/evaluation/report_generator.py`

**Tools to build:**
- `generate_html_report(retrieval_results, generation_results, umap_path, output_path)` → str
  - HTML template with Bootstrap CSS
  - Sections:
    1. Executive Summary (pass/fail against targets)
    2. Retrieval Metrics Table (per-case + averages)
    3. Generation Metrics Table (faithfulness + relevance)
    4. UMAP Visualization (embedded iframe)
    5. Issue List (categorized by failure type)
  - Export to HTML and optionally PDF (via weasyprint)

**Deliverables:**
- ✅ Professional HTML report with charts
- ✅ Actionable issue list:
  - "Retrieval failed: faq_003 (Recall@5 = 0.0)"
  - "Generation unfaithful: faq_007 (hallucinated FBA fee)"
  - "Low relevance: faq_009 (score 2/5, incomplete answer)"
- ✅ Timestamp and configuration metadata

**Testing:**
- Generate report with sample results
- Verify all sections render correctly
- Test PDF export (optional)

**Estimated Time:** 5 hours

---

## Phase 3: Integration & Automation (Priority: Medium)

### 3.1 End-to-End Evaluation Script
**File:** `scripts/run_evaluation.py`

**Features:**
- CLI arguments:
  - `--dataset`: Path to CSV (default: `RAG_FAQ_CSV` or `data/intent_classification/fqa/amazon_fqa.csv`)
  - `--limit`: Number of test cases (default: 10)
  - `--mode`: Answer mode (default: hybrid)
  - `--output`: Output directory (default: tests/evaluation/results_{timestamp})
  - `--skip-umap`: Skip UMAP visualization (faster)
  
- Workflow:
  1. Load dataset
  2. Build RAG pipeline
  3. Run retrieval evaluation
  4. Run generation evaluation
  5. Generate UMAP plot (unless skipped)
  6. Generate HTML report
  7. Print summary to console

**Usage:**
```bash
# Full evaluation with 10 FAQ
python scripts/run_evaluation.py --limit 10

# Quick test without UMAP
python scripts/run_evaluation.py --limit 5 --skip-umap

# Evaluate specific mode
python scripts/run_evaluation.py --mode documents --limit 10
```

**Deliverables:**
- ✅ Single command runs full evaluation
- ✅ Progress tracking with tqdm
- ✅ Results saved to timestamped folder
- ✅ Console summary with pass/fail status

**Testing:**
- Run with 3 FAQ for quick validation
- Verify all output files created
- Test error handling (missing dataset, Ollama down)

**Estimated Time:** 3 hours

---

### 3.2 RAGAS Integration (Optional)
**File:** `src/rag/evaluation/ragas_metrics.py`

**Tools to build:**
- `evaluate_with_ragas(test_cases, pipeline)` → Dict
  - Wrapper for RAGAS library
  - Metrics: context_precision, answer_relevancy, faithfulness
  - Compare RAGAS vs custom LLM-as-Judge
  
- `compare_metrics(custom_results, ragas_results)` → Dict
  - Correlation analysis between methods
  - Identify discrepancies for manual review

**Deliverables:**
- ✅ RAGAS metrics alongside custom metrics
- ✅ Correlation report (Pearson coefficient)
- ✅ Recommendation: which metric to trust

**Dependencies:**
- `pip install ragas`

**Testing:**
- Run RAGAS on 10 FAQ
- Compare with LLM-as-Judge results
- Analyze discrepancies

**Estimated Time:** 4 hours (optional)

---

## Phase 4: Testing & Validation (Priority: High)

### 4.1 Unit Tests
**Files:** `tests/evaluation/test_metrics.py`

**Coverage:**
- `test_recall_at_k()` - Mock retrieved_docs, verify formula
- `test_precision_at_k()` - Edge cases: k=0, empty results
- `test_mrr()` - No relevant docs, first doc relevant
- `test_faithfulness_prompt()` - Verify JSON parsing
- `test_relevance_scoring()` - Score range validation
- `test_dataset_loader()` - Load sample CSV, check fields

**Deliverables:**
- ✅ 100% coverage for metric functions
- ✅ Edge case handling verified
- ✅ Fast execution (<5s total)

**Estimated Time:** 3 hours

---

### 4.2 Integration Test
**File:** `tests/evaluation/test_end_to_end.py`

**Scenario:**
1. Load 10 FAQ from amazon_fqa.csv
2. Build RAG pipeline (mock or real)
3. Run full evaluation pipeline
4. Verify output JSON structure
5. Check metrics against expected ranges:
   - Recall@5: 0.4 - 1.0 (reasonable for 10 FAQ)
   - Faithfulness: 0.7 - 1.0 (most answers should be faithful)
   - Relevance: 3.0 - 5.0 (acceptable to excellent)

**Deliverables:**
- ✅ End-to-end test passes
- ✅ Output files validated
- ✅ Metrics within expected ranges

**Estimated Time:** 2 hours

---

## Deliverables Summary

| Component | File | Status | Priority | Time |
|-----------|------|--------|----------|------|
| Retrieval Metrics | `src/rag/evaluation/retrieval_metrics.py` | ✅ Done | High | 4h |
| Generation Metrics | `src/rag/evaluation/generation_metrics.py` | ✅ Done | High | 6h |
| Dataset Loader | `src/rag/evaluation/dataset_loader.py` | ✅ Done | High | 2h |
| UMAP Visualization | `src/rag/evaluation/visualize_umap.py` | ✅ Done | Medium | 4h |
| Report Generator | `src/rag/evaluation/report_generator.py` | ✅ Done | Medium | 5h |
| E2E Script | `scripts/run_evaluation.py` | ✅ Done | Medium | 3h |
| RAGAS Integration | `src/rag/evaluation/ragas_metrics.py` | ✅ Done | Low | 4h |
| Unit Tests | `tests/evaluation/test_metrics.py` | ✅ Done | High | 3h |
| Integration Test | `tests/evaluation/test_end_to_end.py` | ✅ Done | High | 2h |

**Total Estimated Time:** 33 hours (excluding RAGAS)

---

## Timeline (5 Days)

### Day 1: Core Tools Foundation
- ✅ Dataset loader (2h)
- ✅ Retrieval metrics (4h)
- ✅ Unit tests for retrieval (1.5h)

**Deliverable:** Can calculate Recall@5, Precision@5, MRR on 10 FAQ

---

### Day 2: Generation Evaluation
- ✅ Generation metrics (6h)
- ✅ Unit tests for generation (1.5h)

**Deliverable:** LLM-as-Judge evaluates faithfulness and relevance

---

### Day 3: Testing & Validation
- ✅ Integration test (2h)
- ✅ Run evaluation on 10 FAQ (1h)
- ✅ Analyze results, identify issues (2h)
- ✅ Fix bugs, iterate (3h)

**Deliverable:** Full evaluation pipeline works end-to-end

---

### Day 4: Visualization & Reporting
- ✅ UMAP visualization (4h)
- ✅ Report generator (5h)

**Deliverable:** HTML report with metrics, plots, and issue list

---

### Day 5: Automation & Documentation
- ✅ E2E evaluation script (3h)
- ✅ Update README with evaluation section (1h)
- ✅ Final testing and polish (2h)
- ⏳ RAGAS integration (optional, 4h)

**Deliverable:** Single command runs full evaluation, generates report

---

## Success Criteria

### Quantitative Targets
1. **Retrieval Metrics:**
   - Recall@5 calculated for all 10 FAQ
   - Precision@5 calculated for all 10 FAQ
   - MRR calculated for all 10 FAQ
   - Results saved to JSON

2. **Generation Metrics:**
   - Faithfulness: ≥85% of answers faithful (≥8.5/10)
   - Relevance: ≥4.0 average score (40/50 total)
   - Results saved to JSON with reasoning

3. **Visualization:**
   - UMAP plot shows query-chunk relationships
   - Outliers annotated (queries far from relevant chunks)
   - Saved as HTML and PNG

4. **Automation:**
   - Single command: `python scripts/run_evaluation.py`
   - Generates timestamped results folder
   - HTML report with all metrics and plots

5. **Testing:**
   - Unit tests: 100% coverage for metrics
   - Integration test: End-to-end pipeline passes
   - All tests run in <30s

### Qualitative Targets
- Code is modular and reusable
- Documentation clear and complete
- Report actionable (specific issues identified)
- Easy to expand from 10 to 50 FAQ

---

## Dependencies

### Python Packages
```bash
pip install umap-learn plotly tqdm ragas
```

### External Services
- Ollama (local LLM for documents)
- Deepseek API (remote LLM for general + judge)
- ChromaDB (vector store)

### Data Requirements
- amazon_fqa.csv with 10 FAQ (path: `data/intent_classification/fqa/amazon_fqa.csv` or `RAG_FAQ_CSV`)
- Fields: question, answer, source, category
- Optional: `contexts` (ideal retrieval chunk IDs or content) - required for reliable Recall/Precision/MRR; if missing, `add_relevant_contexts()` uses auto-retrieve as fallback

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing `contexts` makes retrieval metrics unreliable | High | Use manual `contexts` in CSV when possible; document that ground_truth-as-proxy is weak |
| LLM-as-Judge inconsistent | High | Run 10 samples manually, compare with LLM. Adjust prompts if >10% deviation |
| UMAP plot unclear | Medium | Add interactive hover text, annotate outliers, provide interpretation guide |
| Retrieval metrics low | High | Analyze UMAP for embedding issues, check distance thresholds, review chunk quality |
| Generation metrics low | High | Review LLM prompts, check context quality, test different answer modes |
| Ollama crashes during eval | Medium | Add retry logic, timeout handling, fallback to smaller model |

---

## Next Steps

### Immediate Actions (Day 1)
1. **User:** Add 10 FAQ to amazon_fqa.csv with proper answers
2. **Kiro:** Implement dataset loader (`src/rag/evaluation/dataset_loader.py`)
3. **Kiro:** Implement retrieval metrics (`src/rag/evaluation/retrieval_metrics.py`)
4. **Kiro:** Write unit tests for retrieval metrics
5. **Kiro:** Run initial evaluation on 10 FAQ, report results

### Monitoring & Reporting
- Daily standup: Progress update, blockers, next steps
- End of each phase: Demo working component
- End of project: Full evaluation report on 10 FAQ

---

## Output Artifacts

### Code
- `src/rag/evaluation/` - Evaluation modules
- `scripts/run_evaluation.py` - E2E script
- `tests/evaluation/` - Unit and integration tests

### Data
- `tests/evaluation/results_{timestamp}/` - Evaluation results
  - `retrieval_results.json` - Retrieval metrics
  - `generation_results.json` - Generation metrics
  - `umap_visualization.html` - UMAP plot
  - `evaluation_report.html` - Full report

### Documentation
- `docs/RAG_EVALUATION_IMPLEMENTATION_PLAN.md` - This file
- `README.md` - Updated with evaluation section
- `tests/evaluation/README.md` - How to run evaluation

---

## Appendix: Metric Formulas

### Recall@K
```
Recall@K = (Number of relevant chunks in top-K) / (Total relevant chunks)
```

### Precision@K
```
Precision@K = (Number of relevant chunks in top-K) / K
```

### Mean Reciprocal Rank (MRR)
```
MRR = 1 / (Rank of first relevant chunk)
```

### Faithfulness Rate
```
Faithfulness Rate = (Number of faithful answers) / (Total answers)
Target: ≥85%
```

### Average Relevance Score
```
Avg Relevance = Sum(relevance_scores) / (Total answers)
Target: ≥4.0 / 5.0
```

---

**Status:** Phases 1-4 complete. Optional: Phase 3.2 RAGAS integration.  
**Next Review:** As needed for RAGAS or dataset expansion to 50+ FAQ.

---

## Implementation vs Spec (Review 2025-02-23)

| Spec Item | Status | Notes |
|-----------|--------|-------|
| Retrieval API (get_chroma_collection, query_collection) | Done | No private _collection used |
| Recall@5, Precision@5, MRR | Done | retrieval_metrics.py |
| Faithfulness >=85%, Relevance >=4/5 | Done | generation_metrics.py |
| UMAP (Red/Green/Gray, outliers) | Done | visualize_umap.py; PNG if kaleido |
| HTML report + issue list | Done | report_generator.py |
| E2E script (CLI, workflow) | Done | run_evaluation.py |
| tqdm progress | Done | generation_metrics.py |
| tests/evaluation/README.md | Done | How to run evaluation |
| RAGAS integration | Optional | ragas_metrics.py exists, not wired to E2E |
| PDF export (weasyprint) | Not done | Optional per plan |
| F1/ROUGE-L (Best Practices) | Not done | Optional when ground_truth available |
| Manual LLM vs human calibration | Manual | Best practice: annotate 10, compare |
