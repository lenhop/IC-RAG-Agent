"""Gateway intent classification package.

Architecture (强制):
  classification.py — 公共接口模块（唯一统一公共接口，包含全部逻辑）
  __init__.py       — 包入口（re-export 公共 API）

所有下游模块必须通过包入口导入：
    from src.gateway.route_llm.classification import (
        split_intents, classify_intent, classify_intents_batch,
        validate_intents, IntentResult,
    )

禁止直接导入内部类。
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
