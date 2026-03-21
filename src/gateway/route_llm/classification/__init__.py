"""Gateway intent classification package.

Architecture (强制):
  classification.py — 对外门面（公共函数 + 校验器；委托 implement_methods）
  implement_methods.py — 实现层（IntentResult、keyword/vector/LLM 管线）
  __init__.py         — 包入口（re-export 公共 API）

Sub-queries are produced by the unified rewrite stage (route_llm.rewriting).

所有下游模块必须通过包入口导入分类 API：
    from src.gateway.route_llm.classification import (
        classify_intent, classify_intents_batch,
        validate_intents, IntentResult,
    )

禁止直接导入 implement_methods 内部实现细节（优先使用包入口的公共 API）。
"""

from .classification import (
    IntentResult,
    classify_intent,
    classify_intents_batch,
    validate_intents,
)

__all__ = [
    "IntentResult",
    "classify_intent",
    "classify_intents_batch",
    "validate_intents",
]
