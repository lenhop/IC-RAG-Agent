"""Gateway intent classification package.

Architecture (强制):
  classification.py — 对外门面（公共函数 + 校验器；委托 implement_methods）
  implement_methods.py — 实现层（IntentResult、拆分、keyword/vector/LLM 管线）
  __init__.py         — 包入口（re-export 公共 API）

所有下游模块必须通过包入口导入：
    from src.gateway.route_llm.classification import (
        split_intents, classify_intent, classify_intents_batch,
        validate_intents, IntentResult,
    )

禁止直接导入 implement_methods 内部实现细节（优先使用包入口的公共 API）。
"""

from .classification import (
    IntentResult,
    classify_intent,
    classify_intents_batch,
    split_intents,
    validate_intents,
)

__all__ = [
    "IntentResult",
    "classify_intent",
    "classify_intents_batch",
    "split_intents",
    "validate_intents",
]
