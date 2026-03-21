"""Rewriting stage: unified JSON rewrite+split (single LLM), optional route."""

from __future__ import annotations

from .rewrite_implement import (
    IntentSplitMethod,
    JsonRewriteParser,
    RewriteSplitMethod,
    RouterEnvConfig,
    UnifiedRewriteResult,
    split_intents as split_intents_core,
)
from .rewriters import (
    build_merged_context_for_rewrite,
    rewrite_and_route,
    split_intents,
)

__all__ = [
    "IntentSplitMethod",
    "JsonRewriteParser",
    "RewriteSplitMethod",
    "RouterEnvConfig",
    "UnifiedRewriteResult",
    "build_merged_context_for_rewrite",
    "rewrite_and_route",
    "split_intents",
    "split_intents_core",
]
