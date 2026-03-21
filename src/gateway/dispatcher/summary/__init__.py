"""Dispatcher result summarization (rule merge and optional DeepSeek)."""

from .merge import ResultAggregator
from .rule_merge import RuleMergeFacade
from .summarize_llm import SummaryLlmFacade

__all__ = [
    "ResultAggregator",
    "RuleMergeFacade",
    "SummaryLlmFacade",
]
