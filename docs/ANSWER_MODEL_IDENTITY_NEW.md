# 答案模式自动识别方案（整合优化版）

整合豆包 AI 原始方案与 RAG 实战最佳实践，按**由小到大、由粗到精**迭代，每阶段 1~2 个可选方案。

---

## 1. 豆包原始方案概览与评估

| 方案 | 思路 | 评估 |
|------|------|------|
| **1. 关键词识别** | 从 documents 提取关键词（TF-IDF、KeyBERT），query 命中则 documents | 适合冷启动；需离线构建词表，对同义表达弱 |
| **2. FAQ 相似度** | 收集 SellerCentral FAQ，LLM 算 query 与 FAQ 的相似度 | 高准确率；依赖 FAQ 质量与覆盖，有 LLM 延迟 |
| **3. 检索结果判断** | Chroma retrieve + 距离阈值，results>0→documents，=0→general；可选 LLM 二次判断 | 与现有 RAG 流程一致，易落地；阈值需调优 |
| **4. 直接 LLM 判断** | 用 LLM 直接输出 documents/general/hybrid | 灵活；每次 +1~2s 延迟，资源占用高 |

**结论**：四方案统一为**并行四策略**架构（见第 2 节）：查询重写后，FAQ、Documents、Keywords、LLM 四路并行，每路返回 Yes/No，聚合后决定答案模式。

---

## 2. 架构与流程图

### 2.1 总体架构（并行四策略）

用户 Query 经查询重写后，四路并行执行，每路返回 Yes/No，聚合后决定最终答案模式。

```mermaid
flowchart TB
    subgraph input [输入]
        Q[用户 Query]
    end

    subgraph preprocess [预处理]
        QR[查询重写]
    end

    subgraph parallel [四路并行]
        M1[方法1: Documents]
        M2[方法2: Keywords]
        M3[方法3: FAQ]
        M4[方法4: LLM]
    end

    subgraph signals [每路输出]
        S1[Yes/No]
        S2[Yes/No]
        S3[Yes/No]
        S4[Yes/No]
    end

    subgraph output [聚合与输出]
        AGG[聚合]
        GEN[general]
        DOC[documents / hybrid]
    end

    Q --> QR
    QR --> M1
    QR --> M2
    QR --> M3
    QR --> M4
    M1 --> S1
    M2 --> S2
    M3 --> S3
    M4 --> S4
    S1 --> AGG
    S2 --> AGG
    S3 --> AGG
    S4 --> AGG
    AGG --> GEN
    AGG --> DOC
```

### 2.2 主流程图

```mermaid
flowchart TD
    A[User Query]
    B[Query Rewriting]
    A --> B

    B --> L1
    B --> L2
    B --> L3

    L1[Line 1: Embed Query]
    L2[Line 2: Match Keywords]
    L3[Line 3: LLM Classify]

    L1 --> M1[方法1: FAQ 相似度]
    L1 --> M2[方法2: Doc 检索距离]
    M1 --> R1[Yes/No]
    M2 --> R2[Yes/No]

    L2 --> M3[方法3: Keywords]
    M3 --> R3[Yes/No]

    L3 --> M4[方法4: LLM]
    M4 --> ZS[Zero-shot NLI]
    ZS --> MAP[documents=Yes, general=No]
    MAP --> R4[Yes/No]

    R1 --> AGG
    R2 --> AGG
    R3 --> AGG
    R4 --> AGG

    AGG[聚合]
    AGG --> CHECK{All No?}
    CHECK -->|Yes| GEN[general]
    CHECK -->|No| MODE[documents or hybrid]
```

**说明**：查询重写后分三路，其中 Embed 路含两个子方法（FAQ、Docs），共四路独立方法，每路返回 Yes/No。

### 2.3 四路方法及 Yes 条件

| 方法 | 说明 | Yes 条件 | No 条件 |
|------|------|----------|---------|
| **1. Documents** | Chroma 检索文档，计算向量距离 | min_dist ≤ 阈值（有相关文档） | min_dist > 阈值 |
| **2. Keywords** | 匹配业务关键词/短语 | query 命中关键词 | 未命中 |
| **3. FAQ** | 与 Chroma 中 FAQ 问题算相似度 | faq_min_dist < 阈值 | faq_min_dist ≥ 阈值 |
| **4. LLM** | Zero-shot NLI 二分类 | 输出 documents | 输出 general |

**LLM 分支**：使用 zero-shot NLI 模型（如 `distilbert-base-uncased-finetuned-mnli`），仅输出 documents 或 general（无 hybrid），将 documents 映射为 Yes，general 映射为 No。

### 2.4 聚合规则

| 聚合结果 | 最终模式 |
|----------|----------|
| 四路全为 No | **general** |
| 至少一路为 Yes | **documents** 或 **hybrid**（由配置决定） |

### 2.5 简化流程图（高层）

```mermaid
flowchart LR
    A[User Query] --> B[Query Rewrite]
    B --> C[4 Parallel Methods]
    C --> D[Aggregate Yes/No]
    D --> E{All No?}
    E -->|Yes| F[general]
    E -->|No| G[documents or hybrid]
```

---

## 3. 三种答案模式

| 模式 | 含义 |
|------|------|
| documents | 仅基于已入库文档 |
| general | 仅 LLM 通用知识 |
| hybrid | 文档 + 通用知识 |

---

## 4. 分阶段迭代方案

### 阶段 1：MVP（最小投入，快速验证）

**目标**：用现有检索流程实现自动识别，无新依赖。

| 方案 | 实现 | 适用 |
|------|------|------|
| **A. 查询重写（轻量版）** | 术语补全（FBA→Amazon FBA fee）+ 标准化（cost→fulfillment fee）；长度<5词必改写，>10词不改写 | **必做**，提升检索召回率 |
| **B. 检索结果数量** | Chroma retrieve，results>0→documents/hybrid，=0→general | 首选，零改动成本 |
| **C. 简单关键词** | query 含业务词（FBA、Amazon 等）→documents，否则→general | 无 FAQ/检索时兜底 |

**推荐顺序**：先做 A（查询重写），再做 B（检索数量判断），最后可选 C（关键词快速路径）。

**查询重写必要性**：
- 用户常问短/模糊查询（"FBA fee"、"cost?"、"how much"），直接检索召回差
- 轻量版（术语映射）零延迟、零依赖，必做
- 不做 LLM 重写、多轮改写、复杂 HyDe（资源受限场景）

---

### 阶段 2：增强（提升准确率）

**目标**：引入距离阈值，减少误判。

| 方案 | 实现 | 适用 |
|------|------|------|
| **A. 检索 + 距离阈值** | 在 `query_collection` 中 `include=["distances"]`；min(distance)>阈值→general；≤阈值→documents/hybrid | 首选，与阶段 1 兼容 |
| **B. 关键词 + 检索组合** | 规则快速路径：通用前缀（what is、define）且无业务词→general；否则走检索 | 减少对明显通用问题的检索 |

**配置**：`RAG_MODE_DISTANCE_THRESHOLD_GENERAL=1.0`（all-MiniLM-L6-v2 下 L2 距离，需按数据调优）

---

### 阶段 3：精细化（高准确率）

**目标**：引入 FAQ 或 LLM，处理边界与复杂意图。

| 方案 | 实现 | 适用 |
|------|------|------|
| **A. FAQ 相似度** | 收集 Amazon SellerCentral FAQ，embed 后与 query 算相似度；高相似→documents | 有稳定 FAQ 来源时 |
| **B. LLM 二次判断** | 当检索 results>0 且距离在阈值附近时，用 zero-shot NLI 做二分类（documents/general） | 边界 case 多时 |

**注意**：方案 B 增加 ~1s 延迟，建议默认关闭，按需开启。

---

### 阶段 4：并行四策略融合

**目标**：采用并行四策略架构（见 2.1），四路并行、聚合 Yes/No 决定最终模式。

```
query → 查询重写 → [FAQ | Documents | Keywords | LLM] 四路并行 → 聚合 → general / documents / hybrid
```

---

## 5. 实施要点

### 5.1 代码改动（阶段 1~2）

**阶段 1：查询重写**
- `query_pipeline.py`：新增 `rewrite_query_lightweight(question: str) -> str`
- `_step4_embed_question()`：在 embed 前调用重写函数
- 实现：术语补全映射表（FBA→Amazon FBA fee）+ 标准化（cost→fulfillment fee）

**阶段 1~2：模式识别**
- `query_pipeline.py`：`_step5_retrieve_docs` 中 `include` 增加 `"distances"`
- 新增 `classify_answer_mode(question, distances, ...)`，实现阈值逻辑
- API/Gradio：支持 `mode="auto"` 时调用 `classify_answer_mode`

### 5.2 配置项（.env）

```ini
# 查询重写（阶段 1）
RAG_QUERY_REWRITE_ENABLED=true
RAG_QUERY_REWRITE_MIN_LENGTH=5  # <5词必改写
RAG_QUERY_REWRITE_MAX_LENGTH=10  # >10词不改写

# 答案模式自动识别
RAG_AUTO_MODE_ENABLED=true
RAG_MODE_DISTANCE_THRESHOLD_GENERAL=1.0
RAG_GENERAL_PREFIXES=what is,define,什么是
RAG_DOMAIN_KEYWORDS=FBA,FBM,Amazon,eBay,库存,政策
```

### 5.3 查询重写实现示例

```python
def rewrite_query_lightweight(question: str) -> str:
    """阶段 1：轻量查询重写 - 术语补全 + 标准化"""
    words = question.split()
    if len(words) > 10:
        return question  # 跳过改写
    
    # 术语补全映射
    term_map = {
        "FBA": "Amazon FBA",
        "FBM": "Amazon FBM",
        "fee": "Amazon FBA fee",
        "cost": "fulfillment fee",
        "charge": "fulfillment fee",
    }
    
    # 标准化映射（仅短查询）
    std_map = {
        "cost": "fulfillment fee",
        "charge": "fulfillment fee",
        "price": "fulfillment fee",
    } if len(words) < 5 else {}
    
    rewritten = question
    for term, expansion in {**term_map, **std_map}.items():
        if term.lower() in rewritten.lower():
            rewritten = rewritten.replace(term, expansion)
    
    return rewritten
```

### 5.4 关键词方案（方案 1）补充

- **TF-IDF**：从 documents 语料计算，取 top-N 作为业务词表
- **KeyBERT**：`pip install keybert`，从文档提取语义关键词
- 词表可离线生成，存入配置或 JSON，供规则快速路径使用

---

## 6. 总结

| 方法 | 说明 | 优先级 |
|------|------|--------|
| 查询重写 | 术语补全、标准化 | **必做** |
| Documents | Chroma 检索 + 距离阈值 | 必做 |
| Keywords | 业务关键词/短语匹配 | 推荐 |
| FAQ | FAQ 相似度（Chroma 中 FAQ 向量） | 有 FAQ 时 |
| LLM | Zero-shot NLI 二分类 | 可选 |

**架构**：并行四策略（见 2.1）。四路全 No → general；至少一路 Yes → documents 或 hybrid。

**落地顺序**：查询重写 → Documents + Keywords → 按需加 FAQ、LLM。

**查询重写重要性**：
- 短/模糊查询（"FBA fee"、"cost?"）直接检索召回差，必须做轻量重写
- 轻量版（术语映射）零延迟、零依赖，适合资源受限场景
- 不做 LLM 重写、多轮改写、复杂 HyDe（避免过度工程）
