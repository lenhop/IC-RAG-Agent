"""
Intent Retrieval — Public API Package (公共接口包)

Architecture (强制):
  __init__.py         = 包入口，唯一对外统一接口；从 keyword_retrieval / vector_retrieval re-export 公共 API
  keyword_retrieval   = Layer 1 关键字检索（公共接口见该模块 docstring）
  vector_retrieval    = Layer 2 向量检索（公共接口见该模块 docstring）

  所有下游业务模块（gateway、classification、dispatcher 等）禁止直接导入
  keyword_retrieval / vector_retrieval 内部实现，必须通过本包完成检索相关操作。

Public API (exported by this package):
  KeywordMatchResult                  — 关键字匹配结果数据类
  KeywordRetrieval                    — Layer 1 主类；match(query) → Optional[KeywordMatchResult]
  keyword_retrieve(query)             — Layer 1 一次性便捷函数
  VectorCandidate                     — 向量检索单条候选数据类
  VectorRetrieval                     — Layer 2 主类；retrieve(query, ...) → List[VectorCandidate]
  vector_retrieve(query, ...)         — Layer 2 一次性便捷函数
"""

# ---------------------------------------------------------------------------
# Public API — 公共接口（唯一对外入口）
#
# 下游业务模块（gateway、classification、dispatcher 等）必须通过本包访问
# 关键字检索与向量检索能力，禁止直接导入子模块内部实现。
# ---------------------------------------------------------------------------

from __future__ import annotations

from src.retrieval.keyword_retrieval import (
    KeywordMatchResult,
    KeywordRetrieval,
    keyword_retrieve,
)
from src.retrieval.vector_retrieval import (
    VectorCandidate,
    VectorRetrieval,
    vector_retrieve,
)

__all__ = [
    "KeywordMatchResult",
    "KeywordRetrieval",
    "VectorCandidate",
    "VectorRetrieval",
    "keyword_retrieve",
    "vector_retrieve",
]
