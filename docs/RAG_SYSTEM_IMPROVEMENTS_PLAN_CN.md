# RAG 系统全面改进计划

**项目：** IC-RAG-Agent 系统级增强  
**范围：** 评估框架 + 文本生成 + 工作流优化  
**当前状态：** 第1-4阶段完成（评估），生产RAG运行中  
**目标：** 更快、更准确、生产就绪的RAG系统  
**时间线：** 6周（30天）  
**优先级：** 高影响力改进优先

---

## 执行摘要

本计划涵盖三个主要改进领域：

1. **RAG评估框架**（第1-3周）：使评估速度提升3-5倍，更全面
2. **文本生成效率**（第4-5周）：降低延迟，提高质量，降低成本
3. **工作流框架**（第6周）：优化意图分类、检索和编排

---

## 当前状态评估

### 已完成功能 ✅

**评估：**
- 检索指标：Recall@5、Precision@5、MRR
- 生成指标：忠实度、相关性
- UMAP可视化、HTML报告
- 64个测试通过

**RAG管道：**
- 双LLM架构（本地 + 远程）
- 4路并行意图分类（Documents、Keywords、FAQ、LLM）
- 3种答案模式（documents、general、hybrid）
- 查询重写
- ChromaDB向量存储

### 缺失功能 ❌

**评估：**
- 评估速度慢（顺序处理，10个FAQ约2分钟）
- 无成本/延迟跟踪
- 静态报告，无交互性

**生成：**
- 高延迟（hybrid模式约26秒）
- 无响应流式传输
- 无缓存（重复查询重新计算）
- 无提示词优化

**工作流：**
- 意图分类开销（4个并行调用）
- 无自适应检索（始终k=3）
- 无查询路由优化

---

## A部分：评估框架改进（第1-3周）

### 第1周：性能优化
**目标：** 3-5倍速度提升

#### A1.1 并行评估（2天）
**文件：** `src/rag/evaluation/parallel_evaluator.py`
- ThreadPoolExecutor处理I/O密集型任务
- 并行检索 + 生成评估
- 基准测试：10个FAQ <30秒（当前约2分钟）

#### A1.2 嵌入缓存（1.5天）
**文件：** `src/rag/evaluation/embedding_cache.py`
- 基于磁盘的缓存，7天TTL
- 重复评估速度提升50%

#### A1.3 批量LLM调用（1.5天）
**文件：** `src/rag/evaluation/batch_llm_judge.py`
- 每次API调用批处理5个案例
- 成本降低40%

---

### 第2周：增强指标
**目标：** 更全面的评估

#### A2.1 答案正确性（1天）
**文件：** `src/rag/evaluation/answer_correctness.py`
- 语义相似度 + 词元重叠
- 目标：正确答案≥0.8

#### A2.2 上下文相关性（1天）
**文件：** `src/rag/evaluation/context_relevance.py`
- LLM判断每个检索片段
- 目标：平均相关性≥0.7

#### A2.3 性能跟踪（1.5天）
**文件：** `src/rag/evaluation/performance_tracker.py`
- 跟踪每个查询的延迟、词元、成本

#### A2.4 幻觉检测（1.5天）
**文件：** `src/rag/evaluation/hallucination_detector.py`
- 提取声明，根据上下文验证
- 目标：幻觉率<10%

---

### 第3周：仪表板与自动化
**目标：** 生产就绪评估

#### A3.1 Streamlit仪表板（3天）
**文件：** `scripts/evaluation_dashboard.py`
- 交互式指标探索
- 5个页面：概览、检索、生成、可视化、对比

#### A3.2 CI/CD集成（1天）
**文件：** `.github/workflows/evaluation.yml`
- 每个PR运行，指标下降>5%则失败

#### A3.3 回归检测（1天）
**文件：** `src/rag/evaluation/regression_detector.py`
- 统计显著性检验
- 基线管理

---

## B部分：文本生成效率（第4-5周）

### 第4周：延迟降低
**目标：** 生成速度提升50%（hybrid模式 26秒 → 13秒）

#### B1.1 响应流式传输（2天）
**文件：** `src/rag/streaming_pipeline.py`

**问题：** 当前实现等待完整响应（本地LLM 20秒）

**解决方案：**
- 生成时流式传输词元
- 用户在<1秒内看到第一个词元
- 感知延迟：1秒 vs 20秒

**实现：**
```python
class StreamingRAGPipeline:
    def query_stream(self, question: str, mode: str):
        """生成时产出词元。"""
        # 步骤1-5：与之前相同（检索、提示构建）
        # 步骤6：流式LLM响应
        for chunk in self.llm.stream(prompt):
            yield chunk
```

**交付物：**
- `StreamingRAGPipeline` 类
- WebSocket支持实时流式传输
- 批处理的非流式回退

**测试：**
- 验证词元顺序正确性
- 测试连接中断处理
- 基准测试：首个词元<1秒

---

#### B1.2 查询结果缓存（1.5天）
**文件：** `src/rag/query_cache.py`

**问题：** 重复查询重新计算所有内容

**解决方案：**
- 缓存最终答案，TTL（默认：1小时）
- 缓存键：hash(问题 + 模式 + top_k_doc_ids)
- LRU淘汰，最多1000条

**实现：**
```python
class QueryCache:
    def get(self, question: str, mode: str, doc_ids: list) -> dict | None
    def set(self, question: str, mode: str, doc_ids: list, result: dict)
    def invalidate_by_docs(self, doc_ids: list)  # 文档更新时
```

**交付物：**
- 基于Redis的缓存（生产）或内存（开发）
- 缓存命中率日志
- 文档更新时自动失效

**测试：**
- 验证缓存命中/未命中逻辑
- 测试TTL过期
- 基准测试：缓存命中时延迟降低95%

---

#### B1.3 提示词优化（1.5天）
**文件：** `src/rag/optimized_prompts.py`

**问题：** 当前提示词冗长，未针对词元效率优化

**解决方案：**
- 压缩系统提示词（删除冗余）
- 仅在需要时使用少样本示例
- 优化上下文格式

**当前 vs 优化：**
```
当前（documents模式）：
"你是一个有帮助的助手。根据以下上下文文档，
请回答用户的问题。如果答案不在文档中，
说你不知道。上下文：{contexts} 问题：{question}"
词元：约50 + contexts + question

优化后：
"仅使用这些文档回答。如果未找到则说'未知'。
文档：{contexts}
问：{question}
答："
词元：约20 + contexts + question
```

**交付物：**
- 所有3种模式的优化提示词模板
- A/B测试结果（质量 vs 词元节省）
- 词元减少20-30%

**测试：**
- 比较答案质量（必须保持≥95%忠实度）
- 测量词元节省
- 测试边缘情况（空上下文、长问题）

---

### 第5周：质量与成本优化
**目标：** 更好的答案，更低的成本

#### B2.1 自适应检索（2天）
**文件：** `src/rag/adaptive_retrieval.py`

**问题：** 始终检索k=3个片段，无论查询复杂度

**解决方案：**
- 简单查询：k=1（更快、更便宜）
- 复杂查询：k=5（更多上下文）
- 基于查询长度 + 意图置信度自适应

**实现：**
```python
def adaptive_k(question: str, intent_scores: dict) -> int:
    """根据查询特征确定最优k。"""
    # 短问题 + 高置信度 → k=1
    if len(question.split()) < 10 and max(intent_scores.values()) > 0.8:
        return 1
    # 长问题或低置信度 → k=5
    elif len(question.split()) > 20 or max(intent_scores.values()) < 0.5:
        return 5
    # 默认
    return 3
```

**交付物：**
- `adaptive_k()` 函数
- 与RAGPipeline集成
- 平均词元减少15-20%

**测试：**
- 验证检索质量保持
- 测试边缘情况（非常短/长的查询）
- 测量词元节省

---

#### B2.2 重排序（2天）
**文件：** `src/rag/reranker.py`

**问题：** 向量相似度并不总能捕获语义相关性

**解决方案：**
- 检索后，使用交叉编码器重排top-k片段
- 使用轻量级模型（如 `ms-marco-MiniLM-L-6-v2`）
- Recall@3提升10-15%

**实现：**
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, question: str, chunks: list) -> list:
        """按与问题的相关性重排片段。"""
        pairs = [(question, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)
        return [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
```

**交付物：**
- `Reranker` 类
- 可选重排序（通过环境变量启用）
- 基准测试：Recall@3 +10-15%

**测试：**
- 比较有/无重排序
- 测量延迟开销（<200ms）
- 验证质量提升

---

#### B2.3 模型量化（1天）
**文件：** `scripts/quantize_models.sh`

**问题：** 本地LLM（qwen3:1.7b）在CPU上很慢

**解决方案：**
- 量化为4位（GGUF格式）
- 推理速度提升2-3倍，质量损失最小
- 使用 `llama.cpp` 进行优化推理

**实现：**
```bash
# 将qwen3:1.7b量化为Q4_K_M
ollama pull qwen3:1.7b
python scripts/quantize_model.py --model qwen3:1.7b --quant Q4_K_M
```

**交付物：**
- 量化脚本
- 基准测试：延迟、质量、内存
- 模型选择文档

**测试：**
- 比较量化 vs 全精度
- 验证忠实度≥原始的95%
- 测量加速（目标：2-3倍）

---

## C部分：工作流框架优化（第6周）

### 第6周：意图分类与编排
**目标：** 更快、更智能的路由

#### C1.1 意图分类优化（2天）
**文件：** `src/rag/fast_intent_classifier.py`

**问题：** 4路并行分类增加延迟（即使并行）

**解决方案：**
- **第1层（快速）：** 基于规则 + 关键词匹配（0ms）
- **第2层（中等）：** 与FAQ的嵌入相似度（50ms）
- **第3层（慢速）：** LLM分类（1-2秒）
- 早期退出：如果第1层置信度高（>0.9），跳过第2-3层

**实现：**
```python
def classify_fast(question: str) -> tuple[str, float]:
    """快速3层分类，早期退出。"""
    # 第1层：规则（0ms）
    if matches_document_keywords(question):
        return "documents", 0.95
    if matches_general_patterns(question):
        return "general", 0.95
    
    # 第2层：FAQ相似度（50ms）
    faq_score = faq_similarity(question)
    if faq_score > 0.8:
        return "documents", faq_score
    
    # 第3层：LLM（1-2秒）- 仅在需要时
    return llm_classify(question)
```

**交付物：**
- `FastIntentClassifier` 类
- 70%的查询跳过LLM（第1-2层足够）
- 平均延迟：200ms（当前1-2秒）

**测试：**
- 与当前4路并行比较准确率
- 测量延迟分布
- 验证早期退出逻辑

---

#### C2.2 查询路由优化（1.5天）
**文件：** `src/rag/query_router.py`

**问题：** 所有查询都经过相同管道，无论复杂度

**解决方案：**
- **快速路径：** 简单的FAQ类查询 → 直接FAQ查找（无LLM）
- **标准路径：** 正常查询 → 当前管道
- **复杂路径：** 多跳查询 → 迭代检索 + 推理

**实现：**
```python
class QueryRouter:
    def route(self, question: str) -> str:
        """确定查询的最优路径。"""
        # 快速路径：精确FAQ匹配
        if exact_faq_match(question):
            return "fast"
        # 复杂路径：多跳指标
        if is_multi_hop(question):
            return "complex"
        # 标准路径
        return "standard"
```

**交付物：**
- `QueryRouter` 类
- 20%的查询使用快速路径（50ms vs 2秒）
- 5%的查询使用复杂路径（更好的质量）

**测试：**
- 验证路由准确性
- 测量每条路径的延迟
- 比较答案质量

---

#### C3.3 管道编排（1.5天）
**文件：** `src/rag/orchestrator.py`

**问题：** 单体管道，难以定制/扩展

**解决方案：**
- 具有可插拔组件的模块化管道
- 基于DAG的执行（跳过不必要的步骤）
- 可观察性（跟踪每个步骤）

**实现：**
```python
class RAGOrchestrator:
    def __init__(self):
        self.steps = [
            QueryRewriteStep(),
            IntentClassifyStep(),
            RetrievalStep(),
            RerankStep(),  # 可选
            PromptBuildStep(),
            GenerationStep(),
        ]
    
    def execute(self, question: str, config: dict) -> dict:
        """执行带跟踪的管道。"""
        context = {"question": question}
        for step in self.steps:
            if step.should_run(context, config):
                context = step.run(context)
                log_step(step.name, context)
        return context
```

**交付物：**
- `RAGOrchestrator` 类
- 步骤级跟踪和指标
- 易于添加/删除/重排步骤

**测试：**
- 验证向后兼容性
- 测试步骤跳过逻辑
- 测量开销（<5%）

---

## 实施优先级

### 必须有（第1-4周）
1. **评估：** 并行评估、嵌入缓存、增强指标
2. **生成：** 响应流式传输、查询缓存、提示词优化
3. **工作流：** 快速意图分类

### 应该有（第5-6周）
4. **生成：** 自适应检索、重排序、量化
5. **工作流：** 查询路由、管道编排
6. **评估：** 仪表板、CI/CD、回归检测

### 最好有（未来）
7. A/B测试框架
8. 生产监控
9. 人工标注工具
10. 高级错误分析

---

## 成功指标

### 评估
- 评估时间：10个FAQ <30秒（当前约2分钟）
- 缓存命中率：>70%
- API成本：-40%

### 生成
- Hybrid模式延迟：<13秒（当前26秒）
- 首个词元延迟：<1秒（流式传输）
- 缓存命中率：>30%（生产）
- 词元使用：-20-30%（提示词优化 + 自适应检索）

### 工作流
- 意图分类：平均<200ms（当前1-2秒）
- 快速路径使用：20%的查询
- Recall@3：+10-15%（重排序）

### 质量
- 忠实度：≥85%（保持）
- 相关性：≥4.0/5（保持）
- 答案正确性：≥0.8（新指标）

---

## 资源需求

### 开发
- 6周全职（1名开发人员）
- 或12周兼职（50%分配）

### 基础设施
- Redis用于缓存（生产）
- GitHub Actions（CI/CD）
- Streamlit Cloud（仪表板）
- 可选：Prometheus/Grafana（监控）

### 依赖项
```bash
# 评估
pip install streamlit plotly scipy concurrent-futures

# 生成
pip install redis sentence-transformers

# 工作流
pip install networkx  # 用于DAG编排
```

---

## 风险缓解

| 风险 | 影响 | 缓解措施 |
|------|--------|------------|
| 流式传输破坏现有客户端 | 高 | 功能标志，向后兼容 |
| 缓存失效错误 | 中 | 保守TTL，手动清除 |
| 量化质量损失 | 高 | A/B测试，忠实度<95%则回滚 |
| 快速意图分类器准确率下降 | 高 | 广泛测试，回退到完整管道 |
| 重排序延迟开销 | 中 | 设为可选，部署前基准测试 |

---

## 推出计划

### 第1-3周：评估（A部分）
- 并行评估、缓存、增强指标
- 仪表板、CI/CD、回归检测
- **里程碑：** 评估速度提升3倍，生产就绪

### 第4-5周：生成（B部分）
- 流式传输、缓存、提示词优化
- 自适应检索、重排序、量化
- **里程碑：** 生成速度提升50%，成本降低30%

### 第6周：工作流（C部分）
- 快速意图分类、查询路由
- 管道编排
- **里程碑：** 简单查询延迟降低80%

---

## 代码结构

```
src/rag/
├── query_pipeline.py              # ✅ 现有
├── query_rewriting.py             # ✅ 现有
├── intent_methods.py              # ✅ 现有
├── intent_aggregator.py           # ✅ 现有
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
└── run_evaluation.py              # ✅ 现有

.github/workflows/
└── evaluation.yml                 # 🆕 A3.2
```

---

**状态：** 待审查和批准  
**最后更新：** 2026-03-01  
**负责人：** Kiro（项目经理）  
**范围：** 评估 + 生成 + 工作流（完整系统）
