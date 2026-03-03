# RAG System Comprehensive Improvements Plan

**Project:** IC-RAG-Agent System-Wide Enhancements  
**Scope:** Evaluation Framework + Text Generation + Workflow Optimization  
**Current Status:** Phase 1-4 Complete (Evaluation), Production RAG Running  
**Goal:** Faster, more accurate, production-ready RAG system  
**Timeline:** 6 weeks (30 days)  
**Priority:** High-impact improvements first

---

## Executive Summary

This plan covers THREE major improvement areas:

1. **RAG Evaluation Framework** (Weeks 1-3): Make evaluation 3-5x faster, more comprehensive
2. **Text Generation Efficiency** (Weeks 4-5): Reduce latency, improve quality, lower costs
3. **Workflow Framework** (Week 6): Optimize intent classification, retrieval, and orchestration

---

## Current State Assessment

### What We Have ✅

**Evaluation:**
- Retrieval metrics: Recall@5, Precision@5, MRR
- Generation metrics: Faithfulness, Relevance
- UMAP visualization, HTML reports
- 64 tests passing

**RAG Pipeline:**
- Dual-LLM architecture (local + remote)
- 4-way parallel intent classification (Documents, Keywords, FAQ, LLM)
- 3 answer modes (documents, general, hybrid)
- Query rewriting
- ChromaDB vector store

### What's Missing ❌

**Evaluation:**
- Slow evaluation (sequential, ~2min for 10 FAQ)
- No cost/latency tracking
- Static reports, no interactivity

**Generation:**
- High latency (~26s for hybrid mode)
- No response streaming
- No caching (repeated queries recompute)
- No prompt optimization

**Workflow:**
- Intent classification overhead (4 parallel calls)
- No adaptive retrieval (always k=3)
- No query routing optimization

---

## Part A: Evaluation Framework Improvements (Weeks 1-3)

### Week 1: Performance Optimization
**Goal:** 3-5x faster evaluation

#### A1.1 Parallel Evaluation (2 days)
**File:** `src/rag/evaluation/parallel_evaluator.py`
- ThreadPoolExecutor for I/O-bound tasks
- Parallel retrieval + generation evaluation
- Benchmark: 10 FAQ in <30s (vs current ~2min)

#### A1.2 Embedding Cache (1.5 days)
**File:** `src/rag/evaluation/embedding_cache.py`
- Disk-based cache with 7-day TTL
- 50% speedup on repeated evaluations

#### A1.3 Batch LLM Calls (1.5 days)
**File:** `src/rag/evaluation/batch_llm_judge.py`
- Batch 5 cases per API call
- 40% cost reduction

---

### Week 2: Enhanced Metrics
**Goal:** More comprehensive evaluation

#### A2.1 Answer Correctness (1 day)
**File:** `src/rag/evaluation/answer_correctness.py`
- Semantic similarity + token overlap
- Target: ≥0.8 for correct answers

#### A2.2 Context Relevance (1 day)
**File:** `src/rag/evaluation/context_relevance.py`
- LLM judges each retrieved chunk
- Target: ≥0.7 average relevance

#### A2.3 Performance Tracking (1.5 days)
**File:** `src/rag/evaluation/performance_tracker.py`
- Track latency, tokens, costs per query

#### A2.4 Hallucination Detection (1.5 days)
**File:** `src/rag/evaluation/hallucination_detector.py`
- Extract claims, verify against contexts
- Target: <10% hallucination rate

---

### Week 3: Dashboard & Automation
**Goal:** Production-ready evaluation

#### A3.1 Streamlit Dashboard (3 days)
**File:** `scripts/evaluation_dashboard.py`
- Interactive metric exploration
- 5 pages: Overview, Retrieval, Generation, Viz, Comparison

#### A3.2 CI/CD Integration (1 day)
**File:** `.github/workflows/evaluation.yml`
- Run on every PR, fail if metrics drop >5%

#### A3.3 Regression Detection (1 day)
**File:** `src/rag/evaluation/regression_detector.py`
- Statistical significance testing
- Baseline management

---

## Part B: Text Generation Efficiency (Weeks 4-5)

### Week 4: Latency Reduction
**Goal:** 50% faster generation (26s → 13s for hybrid)

#### B1.1 Response Streaming (2 days)
**File:** `src/rag/streaming_pipeline.py`

**Problem:** Current implementation waits for full response (20s for local LLM)

**Solution:**
- Stream tokens as they're generated
- User sees first token in <1s
- Perceived latency: 1s vs 20s

**Implementation:**
```python
class StreamingRAGPipeline:
    def query_stream(self, question: str, mode: str):
        """Yield tokens as they're generated."""
        # Step 1-5: Same as before (retrieval, prompt building)
        # Step 6: Stream LLM response
        for chunk in self.llm.stream(prompt):
            yield chunk
```

**Deliverables:**
- `StreamingRAGPipeline` class
- WebSocket support for real-time streaming
- Fallback to non-streaming for batch processing

**Testing:**
- Verify token order correctness
- Test connection interruption handling
- Benchmark: First token <1s

---

#### B1.2 Query Result Cache (1.5 days)
**File:** `src/rag/query_cache.py`

**Problem:** Repeated queries recompute everything

**Solution:**
- Cache final answers with TTL (default: 1 hour)
- Cache key: hash(question + mode + top_k_doc_ids)
- LRU eviction, max 1000 entries

**Implementation:**
```python
class QueryCache:
    def get(self, question: str, mode: str, doc_ids: list) -> dict | None
    def set(self, question: str, mode: str, doc_ids: list, result: dict)
    def invalidate_by_docs(self, doc_ids: list)  # When docs updated
```

**Deliverables:**
- Redis-based cache (production) or in-memory (dev)
- Cache hit rate logging
- Automatic invalidation on document updates

**Testing:**
- Verify cache hit/miss logic
- Test TTL expiration
- Benchmark: 95% latency reduction on cache hits

---

#### B1.3 Prompt Optimization (1.5 days)
**File:** `src/rag/optimized_prompts.py`

**Problem:** Current prompts are verbose, not optimized for token efficiency

**Solution:**
- Compress system prompts (remove redundancy)
- Use few-shot examples only when needed
- Optimize context formatting

**Current vs Optimized:**
```
Current (documents mode):
"You are a helpful assistant. Based on the following context documents, 
please answer the user's question. If the answer is not in the documents, 
say you don't know. Context: {contexts} Question: {question}"
Tokens: ~50 + contexts + question

Optimized:
"Answer using only these docs. Say 'unknown' if not found.
Docs: {contexts}
Q: {question}
A:"
Tokens: ~20 + contexts + question
```

**Deliverables:**
- Optimized prompt templates for all 3 modes
- A/B test results (quality vs token savings)
- 20-30% token reduction

**Testing:**
- Compare answer quality (must maintain ≥95% faithfulness)
- Measure token savings
- Test edge cases (empty contexts, long questions)

---

### Week 5: Quality & Cost Optimization
**Goal:** Better answers, lower costs

#### B2.1 Adaptive Retrieval (2 days)
**File:** `src/rag/adaptive_retrieval.py`

**Problem:** Always retrieve k=3 chunks, regardless of query complexity

**Solution:**
- Simple queries: k=1 (faster, cheaper)
- Complex queries: k=5 (more context)
- Adaptive based on query length + intent confidence

**Implementation:**
```python
def adaptive_k(question: str, intent_scores: dict) -> int:
    """Determine optimal k based on query characteristics."""
    # Short question + high confidence → k=1
    if len(question.split()) < 10 and max(intent_scores.values()) > 0.8:
        return 1
    # Long question or low confidence → k=5
    elif len(question.split()) > 20 or max(intent_scores.values()) < 0.5:
        return 5
    # Default
    return 3
```

**Deliverables:**
- `adaptive_k()` function
- Integration with RAGPipeline
- 15-20% average token reduction

**Testing:**
- Verify retrieval quality maintained
- Test edge cases (very short/long queries)
- Measure token savings

---

#### B2.2 Reranking (2 days)
**File:** `src/rag/reranker.py`

**Problem:** Vector similarity doesn't always capture semantic relevance

**Solution:**
- After retrieval, rerank top-k chunks using cross-encoder
- Use lightweight model (e.g., `ms-marco-MiniLM-L-6-v2`)
- Improves Recall@3 by 10-15%

**Implementation:**
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, question: str, chunks: list) -> list:
        """Rerank chunks by relevance to question."""
        pairs = [(question, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)
        return [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
```

**Deliverables:**
- `Reranker` class
- Optional reranking (enabled via env var)
- Benchmark: +10-15% Recall@3

**Testing:**
- Compare with/without reranking
- Measure latency overhead (<200ms)
- Verify quality improvement

---

#### B2.3 Model Quantization (1 day)
**File:** `scripts/quantize_models.sh`

**Problem:** Local LLM (qwen3:1.7b) is slow on CPU

**Solution:**
- Quantize to 4-bit (GGUF format)
- 2-3x faster inference, minimal quality loss
- Use `llama.cpp` for optimized inference

**Implementation:**
```bash
# Quantize qwen3:1.7b to Q4_K_M
ollama pull qwen3:1.7b
python scripts/quantize_model.py --model qwen3:1.7b --quant Q4_K_M
```

**Deliverables:**
- Quantization script
- Benchmarks: latency, quality, memory
- Documentation for model selection

**Testing:**
- Compare quantized vs full precision
- Verify faithfulness ≥95% of original
- Measure speedup (target: 2-3x)

---

## Part C: Workflow Framework Optimization (Week 6)

### Week 6: Intent Classification & Orchestration
**Goal:** Faster, smarter routing

#### C1.1 Intent Classification Optimization (2 days)
**File:** `src/rag/fast_intent_classifier.py`

**Problem:** 4-way parallel classification adds latency (even with parallelism)

**Solution:**
- **Tier 1 (Fast):** Rule-based + keyword matching (0ms)
- **Tier 2 (Medium):** Embedding similarity to FAQ (50ms)
- **Tier 3 (Slow):** LLM classification (1-2s)
- Early exit: If Tier 1 confident (>0.9), skip Tier 2-3

**Implementation:**
```python
def classify_fast(question: str) -> tuple[str, float]:
    """Fast 3-tier classification with early exit."""
    # Tier 1: Rules (0ms)
    if matches_document_keywords(question):
        return "documents", 0.95
    if matches_general_patterns(question):
        return "general", 0.95
    
    # Tier 2: FAQ similarity (50ms)
    faq_score = faq_similarity(question)
    if faq_score > 0.8:
        return "documents", faq_score
    
    # Tier 3: LLM (1-2s) - only if needed
    return llm_classify(question)
```

**Deliverables:**
- `FastIntentClassifier` class
- 70% queries skip LLM (Tier 1-2 sufficient)
- Average latency: 200ms (vs current 1-2s)

**Testing:**
- Compare accuracy with current 4-way parallel
- Measure latency distribution
- Verify early exit logic

---

#### C2.2 Query Routing Optimization (1.5 days)
**File:** `src/rag/query_router.py`

**Problem:** All queries go through same pipeline, regardless of complexity

**Solution:**
- **Fast path:** Simple FAQ-like queries → direct FAQ lookup (no LLM)
- **Standard path:** Normal queries → current pipeline
- **Complex path:** Multi-hop queries → iterative retrieval + reasoning

**Implementation:**
```python
class QueryRouter:
    def route(self, question: str) -> str:
        """Determine optimal path for query."""
        # Fast path: Exact FAQ match
        if exact_faq_match(question):
            return "fast"
        # Complex path: Multi-hop indicators
        if is_multi_hop(question):
            return "complex"
        # Standard path
        return "standard"
```

**Deliverables:**
- `QueryRouter` class
- 20% queries use fast path (50ms vs 2s)
- 5% queries use complex path (better quality)

**Testing:**
- Verify routing accuracy
- Measure latency per path
- Compare answer quality

---

#### C3.3 Pipeline Orchestration (1.5 days)
**File:** `src/rag/orchestrator.py`

**Problem:** Monolithic pipeline, hard to customize/extend

**Solution:**
- Modular pipeline with pluggable components
- DAG-based execution (skip unnecessary steps)
- Observability (trace each step)

**Implementation:**
```python
class RAGOrchestrator:
    def __init__(self):
        self.steps = [
            QueryRewriteStep(),
            IntentClassifyStep(),
            RetrievalStep(),
            RerankStep(),  # Optional
            PromptBuildStep(),
            GenerationStep(),
        ]
    
    def execute(self, question: str, config: dict) -> dict:
        """Execute pipeline with tracing."""
        context = {"question": question}
        for step in self.steps:
            if step.should_run(context, config):
                context = step.run(context)
                log_step(step.name, context)
        return context
```

**Deliverables:**
- `RAGOrchestrator` class
- Step-level tracing and metrics
- Easy to add/remove/reorder steps

**Testing:**
- Verify backward compatibility
- Test step skipping logic
- Measure overhead (<5%)

---

## Implementation Priority

### Must-Have (Weeks 1-4)
1. **Evaluation:** Parallel evaluation, embedding cache, enhanced metrics
2. **Generation:** Response streaming, query cache, prompt optimization
3. **Workflow:** Fast intent classification

### Should-Have (Weeks 5-6)
4. **Generation:** Adaptive retrieval, reranking, quantization
5. **Workflow:** Query routing, pipeline orchestration
6. **Evaluation:** Dashboard, CI/CD, regression detection

### Nice-to-Have (Future)
7. A/B testing framework
8. Production monitoring
9. Human annotation tool
10. Advanced error analysis

---

## Success Metrics

### Evaluation
- Evaluation time: <30s for 10 FAQ (vs ~2min)
- Cache hit rate: >70%
- API cost: -40%

### Generation
- Hybrid mode latency: <13s (vs 26s)
- First token latency: <1s (streaming)
- Cache hit rate: >30% (production)
- Token usage: -20-30% (prompt optimization + adaptive retrieval)

### Workflow
- Intent classification: <200ms average (vs 1-2s)
- Fast path usage: 20% of queries
- Recall@3: +10-15% (reranking)

### Quality
- Faithfulness: ≥85% (maintained)
- Relevance: ≥4.0/5 (maintained)
- Answer correctness: ≥0.8 (new metric)

---

## Resource Requirements

### Development
- 6 weeks full-time (1 developer)
- Or 12 weeks part-time (50% allocation)

### Infrastructure
- Redis for caching (production)
- GitHub Actions (CI/CD)
- Streamlit Cloud (dashboard)
- Optional: Prometheus/Grafana (monitoring)

### Dependencies
```bash
# Evaluation
pip install streamlit plotly scipy concurrent-futures

# Generation
pip install redis sentence-transformers

# Workflow
pip install networkx  # For DAG orchestration
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Streaming breaks existing clients | High | Feature flag, backward compatibility |
| Cache invalidation bugs | Medium | Conservative TTL, manual clear |
| Quantization quality loss | High | A/B test, rollback if faithfulness <95% |
| Fast intent classifier accuracy drop | High | Extensive testing, fallback to full pipeline |
| Reranking latency overhead | Medium | Make optional, benchmark before deploy |

---

## Rollout Plan

### Weeks 1-3: Evaluation (Part A)
- Parallel evaluation, caching, enhanced metrics
- Dashboard, CI/CD, regression detection
- **Milestone:** 3x faster evaluation, production-ready

### Weeks 4-5: Generation (Part B)
- Streaming, caching, prompt optimization
- Adaptive retrieval, reranking, quantization
- **Milestone:** 50% faster generation, 30% cost reduction

### Week 6: Workflow (Part C)
- Fast intent classification, query routing
- Pipeline orchestration
- **Milestone:** 80% latency reduction for simple queries

---

## Code Structure

```
src/rag/
├── query_pipeline.py              # ✅ Existing
├── query_rewriting.py             # ✅ Existing
├── intent_methods.py              # ✅ Existing
├── intent_aggregator.py           # ✅ Existing
├── streaming_pipeline.py          # 🆕 B1.1
├── query_cache.py                 # 🆕 B1.2
├── optimized_prompts.py           # 🆕 B1.3
├── adaptive_retrieval.py          # 🆕 B2.1
├── reranker.py                    # 🆕 B2.2
├── fast_intent_classifier.py      # 🆕 C1.1
├── query_router.py                # 🆕 C2.2
└── orchestrator.py                # 🆕 C3.3

src/rag/evaluation/
├── parallel_evaluator.py          # 🆕 A1.1
├── embedding_cache.py             # 🆕 A1.2
├── batch_llm_judge.py             # 🆕 A1.3
├── answer_correctness.py          # 🆕 A2.1
├── context_relevance.py           # 🆕 A2.2
├── performance_tracker.py         # 🆕 A2.3
├── hallucination_detector.py      # 🆕 A2.4
└── regression_detector.py         # 🆕 A3.3

scripts/
├── evaluation_dashboard.py        # 🆕 A3.1
├── quantize_models.sh             # 🆕 B2.3
└── run_evaluation.py              # ✅ Existing

.github/workflows/
└── evaluation.yml                 # 🆕 A3.2
```

---

**Status:** Ready for review and approval  
**Last Updated:** 2026-03-01  
**Owner:** Kiro (Project Manager)  
**Scope:** Evaluation + Generation + Workflow (Complete System)
