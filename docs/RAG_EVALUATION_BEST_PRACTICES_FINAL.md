# IC-RAG-Agent RAG 评估方案（最终版）

整合 prompts 思路、优化方案与 RAG 最佳实践，形成可落地的评估方案。

---

## 1. 数据集构建（50 条）

### 1.1 数据来源与占比

| 来源 | 占比 | 采集规则 |
|------|------|----------|
| Amazon SellerCentral FAQ | 60%（30 条） | 高频、高客诉、易出错问题（如退款、FBA 仓储） |
| Seller Assistant | 20%（10 条） | 典型问答，含边界场景（如多店铺政策） |
| 官方 Documents | 20%（10 条） | 从手册/政策拆解具体问答（税率、物流等） |

### 1.2 数据分层

- **基础场景**（30 条）：单文档、单答案、无歧义
- **复杂场景**（20 条）：多文档拼接、条件判断、易混淆（如不同站点退货政策）

### 1.3 单条数据字段

- `question`：用户问题
- `ground_truth`：标准答案或核心要点
- `contexts`：理想检索文档 chunk（含 chunk ID/位置）

### 1.4 标注方式（Ground Truth）

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **人工标注** | 每条 query 人工标注相关/无关文档 | 高质量评估，小规模 |
| **LLM-as-Judge** | 用 LLM 对文档相关性打分（0–5） | 快速扩展 |
| **合成数据** | 从文档生成 Q&A，作为 ground truth | 大规模预评估 |

---

## 2. 评估维度

RAG 评估通常分为三层：

| 层级 | 评估内容 | 英文对应 |
|------|----------|----------|
| **检索层** | 是否召回正确文档 | Retrieval |
| **生成层** | 答案是否忠实、相关、完整 | Generation |
| **端到端** | 是否满足用户需求 | End-to-end |

---

## 3. 检索层评估

### 3.1 核心指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Recall@5** | top-5 中相关 chunk 数 / 总相关数 | 越高越好 |
| **Precision@5** | top-5 中相关 chunk 数 / 5 | 越高越好 |
| **MRR** | 首个相关结果的排名倒数 | 越高越好 |
| **NDCG**（可选） | 考虑排序质量的归一化折损累积增益 | 有分级相关性时使用 |

### 3.2 UMAP 可视化

- **步骤**：用 all-MiniLM-L6-v2 提取 query 与 chunk 的 embedding → UMAP 降维至 2D → 绘制散点图
- **配色**：问题（红）、相关 chunk（绿）、无关 chunk（灰）
- **用途**：定性分析检索偏差（如问题与无关 chunk 过近、相关 chunk 分散）
- **注意**：仅选测试集相关 chunk，降低计算量

### 3.3 实践要点

- IC-RAG-Agent（Chroma、all-MiniLM-L6-v2）建议 20–50 条标注 query 起步
- 评估检索质量时，索引与检索必须使用同一 embedding 模型

---

## 4. 生成层评估

### 4.1 LLM-as-Judge（必做）

| 维度 | Prompt 要点 | 评分 |
|------|-------------|------|
| **忠实度** | 答案是否完全基于 context，无编造/夸大 | 是/否，目标「是」≥85% |
| **回答相关性** | 是否完整、准确回答问题 | 1–5 分，目标≥4 |

- 建议用 gpt-4o/minimax 等中端 LLM，保留打分理由便于复盘
- 人工标注 10 条作对照，LLM 与人工偏差≤10% 即可

**示例 Prompt：**

```
Rate 1-5: Does this answer correctly address the question using only the provided context?
Question: {question}
Context: {context}
Answer: {answer}
```

### 4.2 生成维度扩展（可选）

| 维度 | 含义 | 测量方式 |
|------|------|----------|
| **Correctness** | 与 ground truth 的事实一致性 | F1、BLEU、ROUGE |
| **Completeness** | 是否覆盖关键要点 | LLM-as-Judge、检查清单 |
| **Coherence** | 可读性、逻辑性 | LLM-as-Judge、流畅度指标 |

### 4.3 客观指标（可选）

- **RAGAS**：`context_precision`、`answer_relevancy`、faithfulness
- **有 ground_truth 时**：F1、ROUGE-L

### 4.4 人工评估

- **A/B 测试**：对比两种 RAG 配置
- **并排对比**：对不同配置的答案排序
- **任务型**：「能否用该答案完成任务 X？」

---

## 5. 工具与实现

| 工具 | 用途 | 备注 |
|------|------|------|
| **RAGAS** | Faithfulness、Answer Relevancy、Context Precision | 开源，常用 |
| **UMAP** | 检索可视化 | 定性分析 |
| **LLM** | 忠实度、相关性打分 | LLM-as-Judge |
| **TruLens** | Faithfulness、Relevance、Groundedness | LangChain 集成 |
| **LangSmith** | 追踪、调试、评估 | 商业，LangChain 生态 |
| **BEIR** | 检索基准 | 标准检索评估 |
| **MTEB** | Embedding 模型基准 | 选型参考 |

### 5.1 RAGAS 示例

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

dataset = ...  # question, answer, contexts, ground_truth
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

---

## 6. 落地节奏

| 阶段 | 周期 | 内容 |
|------|------|------|
| **阶段 1** | 1–2 天 | 20 条核心用例；Recall@5 + UMAP；LLM 忠实度+相关性 |
| **阶段 2** | 3–5 天 | 补足 50 条；Recall@5/Precision@5；RAGAS；UMAP 异常与 LLM 低分对照分析；A/B 测试（如 query 改写 vs 不改写、documents vs hybrid）；黄金回归集，变更后自动跑 |
| **阶段 3** | 持续 | 黄金测试集；每次迭代回归评估；针对低分场景调 embedding、top-K、prompt |
| **阶段 4** | 生产 | 生产监控：用户反馈（点赞/点踩、追问）、延迟（P50/P95/P99）、错误率（检索空、超时、API 错误） |

---

## 7. 注意事项

1. **Embedding**：统一用 all-MiniLM-L6-v2，与线上一致
2. **UMAP**：定性分析用，不替代量化指标
3. **LLM 打分**：需人工对照验证一致性
4. **结论**：产出可执行结论（如「Recall@5 仅 60%，FBA 仓储类检索失效」），而非只列数字

---

## 8. 输出物

1. 测试集（50 条，含标注）
2. 检索层：Recall@5/Precision@5 统计表 + UMAP 图（标异常点）
3. 生成层：LLM 打分表 + RAGAS 报告
4. 问题清单：按「检索失效 / 生成不忠实 / 回答不相关」分类，附优化建议

---

## 9. 快速参考

| 目标 | 方法 |
|------|------|
| 检索质量 | Recall@5、Precision@5、UMAP |
| 生成质量 | LLM 忠实度+相关性、RAGAS |
| 配置对比 | A/B 测试或人工并排评估 |
| 回归验证 | 黄金测试集 + 自动化指标 |
| 生产监控 | 用户反馈、延迟、错误率 |

---

## 10. 推荐起步步骤

1. 构建 20–50 条标注测试集
2. 接入 RAGAS（或类似工具）做 faithfulness 与 relevancy
3. 用 LLM-as-Judge 做快速定性检查
4. 建立黄金回归集，在 CI 或发版前自动运行

---

## 11. 参考资料

- [RAGAS: Evaluation of RAG pipelines](https://github.com/explodinggradients/ragas)
- [TruLens](https://www.trulens.org/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
