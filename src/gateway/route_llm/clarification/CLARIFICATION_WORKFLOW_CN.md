# Clarification 澄清流程说明

本文档说明 `clarification.py` 的完整工作流程，以及各函数在流程中的角色。

---

## 一、整体流程概览

```
用户查询 (query)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 前置过滤                                              │
│   - 空查询 → 返回 needs_clarification=False                    │
│   - 文档/政策/合规类问题 → 返回 needs_clarification=False       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 启发式路径 (heuristic)                                │
│   - _heuristic_needs_clarification() 匹配 库存/订单/费用/销售   │
│   - 命中且无上下文 → 直接返回澄清 (可选 LLM 生成问题)           │
│   - 命中且有上下文 → 调用 LLM 决定 (用户可能已提供信息)         │
└─────────────────────────────────────────────────────────────┘
    │ 未命中
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 完整 LLM 路径                                         │
│   - _call_clarification_ollama / _call_clarification_deepseek  │
│   - 使用 clarification_detect_ambiguity.txt                  │
│   - 解析 JSON，返回 needs_clarification + clarification_question │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、函数与流程对应关系

| 流程步骤 | 函数 | 行号 | 作用 |
|----------|------|------|------|
| **Step 1a** | `_is_concrete_documentation_query` | 35-55 | 判断是否为文档/政策/合规类问题，此类问题不需澄清 |
| **Step 2a** | `_heuristic_needs_clarification` | 134-151 | 启发式匹配：库存/订单/费用/销售 且缺少 ASIN/OrderID/日期/费用类型 等 |
| **Step 2b** | `_generate_clarification_question_ollama` | 93-132 | 启发式命中时，调用 Ollama 生成更自然的澄清问题（使用 clarification_generate_question.txt） |
| **Step 3** | `_call_clarification_ollama` | 191-216 | 调用 Ollama 做歧义检测（使用 clarification_detect_ambiguity.txt） |
| **Step 3** | `_call_clarification_deepseek` | 219-252 | 调用 DeepSeek 做歧义检测（同上 prompt） |
| **辅助** | `_build_user_input` | 84-90 | 构建 LLM 输入：对话历史 + 当前查询 |
| **辅助** | `_get_timeout` | 152-158 | 读取超时配置 |
| **辅助** | `_strip_markdown_fences` | 161-169 | 去除 LLM 输出中的 ``` 标记 |
| **辅助** | `_extract_first_json_object` | 171-188 | 从文本提取第一个 {...} JSON 对象 |
| **入口** | `check_ambiguity` | 255-374 | 主入口，串联上述步骤 |

---

## 三、启发式规则 (_HEURISTIC_AMBIGUOUS)

| 主题模式 | 必需标识符 | 缺省澄清问题 |
|----------|------------|--------------|
| inventory, stock | ASIN, store, SKU, marketplace | Which store, ASIN, or SKU do you want inventory for? |
| order, check order | Order ID (如 112-1234567-8901234) | Please provide the Order ID... |
| fees, charges, breakdown | FBA/storage/referral, 日期, Q1-Q4 | Which fees do you mean? (FBA, storage, or referral) And for which time period? |
| sales, trends, metrics | 日期范围, 月份名 | Which date range or time period do you want the data for? |

**逻辑**：若查询命中「主题」且**不**包含「必需标识符」，则需澄清。

---

## 四、Prompt 文件作用

### clarification_detect_ambiguity.txt

- **用途**：让 LLM 判断查询是否歧义、是否需要澄清。
- **调用位置**：`_call_clarification_ollama`、`_call_clarification_deepseek`（Step 3 完整路径）；以及启发式命中且有上下文时。
- **输入**：系统 prompt + 对话历史（可选）+ 用户查询。
- **输出**：`{"needs_clarification": true/false, "clarification_question": "..."}`

### clarification_generate_question.txt

- **用途**：在已确定需澄清时，让 LLM 生成具体的澄清问题。
- **调用位置**：`_generate_clarification_question_ollama`（Step 2b 启发式路径）。
- **输入**：系统 prompt + 用户查询 + 对话历史（可选）。
- **输出**：`{"clarification_question": "..."}`

---

## 五、阅读顺序建议

1. **入口**：`check_ambiguity` (L255) — 看整体分支逻辑。
2. **Step 1**：L275-281 — 空查询、文档类跳过。
3. **Step 2**：L283-330 — 启发式匹配及有/无上下文分支。
4. **Step 3**：L332-374 — 完整 LLM 歧义检测。
5. **辅助函数**：按需查看 `_build_user_input`、`_strip_markdown_fences`、`_extract_first_json_object`。

---

## 六、调用链（从 api.py）

```
api.py: check_ambiguity(original_query, backend, conversation_context=clarification_context)
    │
    ├─ /api/v1/rewrite (L596)  — 预览时先做澄清检查
    └─ /api/v1/query  (L818)  — 正式查询时做澄清检查
```

当 `check_ambiguity` 返回 `needs_clarification=True` 时，API 会直接返回澄清问题，不执行后续的 rewrite 和 intent classification。
