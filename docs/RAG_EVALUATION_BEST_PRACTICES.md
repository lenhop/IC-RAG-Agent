# RAG Evaluation Best Practices

How to evaluate Retrieval Augmented Generation (RAG) systems: metrics, methods, tools, and a practical evaluation plan for IC-RAG-Agent.

---

## 1. What to Evaluate

RAG evaluation typically has three layers:

| Layer | What It Measures |
|-------|------------------|
| **Retrieval** | Whether the right documents are retrieved |
| **Generation** | Whether the answer is correct, relevant, and well-formed |
| **End-to-end** | Whether the final answer satisfies the user's need |

---

## 2. Retrieval Evaluation

### 2.1 Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of relevant docs in top-K results |
| **Precision@K** | Fraction of top-K results that are relevant |
| **MRR** (Mean Reciprocal Rank) | Rank of first relevant doc |
| **NDCG** | Ranking quality with graded relevance |

### 2.2 Ground Truth

- **Human-labeled**: Annotate each doc as relevant/irrelevant per query
- **LLM-as-judge**: Use LLM to score relevance (0–5)
- **Synthetic**: Generate Q&A from docs, treat as ground truth

### 2.3 Practical Notes

- For IC-RAG-Agent (Chroma, all-MiniLM-L6-v2), start with 20–50 labeled queries
- Use same embedding model for indexing and retrieval when measuring retrieval quality

---

## 3. Generation Evaluation

### 3.1 Dimensions

| Dimension | Meaning | How to Measure |
|-----------|---------|----------------|
| **Faithfulness** | Answer grounded in retrieved context | LLM-as-judge, NLI models |
| **Relevance** | Answer addresses the question | LLM-as-judge, semantic similarity |
| **Correctness** | Factual accuracy vs ground truth | Exact match, F1, BLEU, ROUGE |
| **Completeness** | Covers key aspects | LLM-as-judge, checklist |
| **Coherence** | Readable and logical | LLM-as-judge, fluency metrics |

### 3.2 Metrics

- **Faithfulness**: % of claims supported by context
- **Answer Relevancy**: Semantic similarity between question and answer
- **Context Precision/Recall**: Overlap between used context and ideal context

---

## 4. End-to-End Evaluation

### 4.1 LLM-as-Judge

Use an LLM to score answers (e.g., 1–5) on:

- Relevance
- Faithfulness
- Helpfulness

**Example prompt:**

```
Rate 1-5: Does this answer correctly address the question using only the provided context?
Question: {question}
Context: {context}
Answer: {answer}
```

### 4.2 Human Evaluation

- **A/B testing**: Compare two RAG variants
- **Side-by-side**: Rank answers from different configs
- **Task-based**: "Can you complete task X with this answer?"

---

## 5. Frameworks and Tools

| Tool | Focus | Notes |
|------|--------|------|
| **RAGAS** | Faithfulness, Answer Relevancy, Context Precision/Recall | Popular, open-source |
| **TruLens** | Faithfulness, Relevance, Groundedness | LangChain integration |
| **LangSmith** | Tracing, debugging, eval | Commercial, LangChain ecosystem |
| **BEIR** | Retrieval benchmarks | Standard retrieval eval |
| **MTEB** | Embedding model benchmarks | For embedding choice |

### 5.1 RAGAS Example

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

dataset = ...  # question, answer, contexts, ground_truth
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

---

## 6. Practical Evaluation Plan for IC-RAG-Agent

### 6.1 Phase 1: Minimal Eval

1. **Test set**
   - 20–50 labeled Q&A pairs (question, expected answer or key facts)
   - Mix of documents / general / hybrid

2. **Retrieval**
   - For each question, check if top-5 results contain at least one relevant doc
   - Compute Recall@5

3. **Generation**
   - LLM-as-judge: "Is this answer faithful to the context?" (yes/no)
   - LLM-as-judge: "Does it answer the question?" (1–5)

### 6.2 Phase 2: Rigorous Eval

1. **RAGAS**
   - Faithfulness, Answer Relevancy, Context Precision/Recall

2. **A/B tests**
   - With vs without query rewriting
   - Different answer modes (documents vs hybrid)

3. **Regression tests**
   - Golden set of Q&A pairs
   - Run after changes to catch regressions

### 6.3 Phase 3: Production Monitoring

1. **Implicit signals**
   - Thumbs up/down, follow-up questions, session length

2. **Latency**
   - P50, P95, P99

3. **Error rates**
   - Retrieval empty, timeout, API errors

---

## 7. Quick Reference

| Goal | Approach |
|------|----------|
| Measure retrieval quality | Recall@K, Precision@K, labeled test set |
| Measure answer quality | RAGAS (faithfulness, relevancy) or LLM-as-judge |
| Compare configs | A/B test or side-by-side human eval |
| Catch regressions | Golden Q&A set + automated metrics |
| Production monitoring | User feedback, latency, error rates |

---

## 8. Recommended Starting Point

1. Build a small labeled test set (20–50 queries)
2. Add RAGAS (or similar) for faithfulness and relevancy
3. Use LLM-as-judge for quick qualitative checks
4. Add a golden regression suite and run it in CI or before releases

---

## 9. References

- [RAGAS: Evaluation of RAG pipelines](https://github.com/explodinggradients/ragas)
- [TruLens](https://www.trulens.org/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
