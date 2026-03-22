"""
Tests for GATEWAY_TEXT_GENERATION_BACKEND, RAG merge composer, and SP-API gateway formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_resolve_text_generation_backend_ollama_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_TEXT_GENERATION_BACKEND", "ollama")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from src.llm.text_generation_backend import resolve_text_generation_backend

    assert resolve_text_generation_backend() == "ollama"


def test_resolve_text_generation_backend_deepseek_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_TEXT_GENERATION_BACKEND", "deepseek")
    from src.llm.text_generation_backend import resolve_text_generation_backend

    assert resolve_text_generation_backend() == "deepseek"


def test_merge_composer_calls_complete_chat_with_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str, str]] = []

    def fake_complete(backend, system_prompt, user_content, **kwargs):
        captured.append((backend, system_prompt[:40], user_content[:30]))
        return "synthesized answer"

    monkeypatch.setattr("src.agent.rag.merge_compose.complete_chat", fake_complete)
    from src.agent.rag.merge_compose import MergeComposer

    out = MergeComposer.final_answer(
        "What is FBA?",
        "chroma excerpt here",
        "bullet evidence",
        text_generation_backend="ollama",
    )
    assert out == "synthesized answer"
    assert len(captured) == 1
    assert captured[0][0] == "ollama"


def test_result_aggregator_sp_api_uses_format_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_SP_API_FORMAT_LLM_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_TEXT_GENERATION_BACKEND", "deepseek")

    def fake_format(worker_answer, **kwargs):
        return "FORMATTED: " + worker_answer

    monkeypatch.setattr(
        "src.gateway.dispatcher.summary.sp_api_format_llm.format_sp_api_worker_answer",
        fake_format,
    )

    from src.gateway.dispatcher.summary.merge import ResultAggregator
    from src.gateway.schemas import RewritePlan, TaskExecutionResult

    plan = RewritePlan(plan_type="single_domain", merge_strategy="concat", task_groups=[])
    results = [
        TaskExecutionResult(
            task_id="t1",
            workflow="sp_api",
            query="order status?",
            status="completed",
            answer="```yaml\norders: []\n```",
        )
    ]
    merged = ResultAggregator.merge(plan, results)
    assert merged.startswith("FORMATTED:")


def test_result_aggregator_sp_api_skips_llm_for_authoritative_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-built getOrder YAML must not pass through a summarizing LLM."""
    monkeypatch.setenv("GATEWAY_SP_API_FORMAT_LLM_ENABLED", "true")

    def should_not_run(*_a, **_k):
        raise AssertionError("format_sp_api_worker_answer must not be called")

    monkeypatch.setattr(
        "src.gateway.dispatcher.summary.sp_api_format_llm.format_sp_api_worker_answer",
        should_not_run,
    )

    from src.gateway.dispatcher.summary.merge import ResultAggregator
    from src.gateway.schemas import RewritePlan, TaskExecutionResult

    plan = RewritePlan(plan_type="single_domain", merge_strategy="concat", task_groups=[])
    raw = (
        "Below is the Amazon Selling Partner API getOrder response, formatted as YAML. "
        "This data comes from the API only (not from the language model).\n\n"
        "```yaml\n"
        "orders:\n"
        "  - ok: true\n"
        "    order_id: 111-2886487-4917844\n"
        "    sp_api_response:\n"
        "      OrderStatus: Shipped\n"
        "```"
    )
    results = [
        TaskExecutionResult(
            task_id="t1",
            workflow="sp_api",
            query="status?",
            status="completed",
            answer=raw,
        )
    ]
    merged = ResultAggregator.merge(plan, results)
    assert merged == raw.strip()
    assert "Shipped" in merged
    assert "Processing" not in merged


def test_result_aggregator_sp_api_skips_llm_yaml_fence_without_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authoritative payload is detected by fenced YAML plus sp_api_response key."""
    monkeypatch.setenv("GATEWAY_SP_API_FORMAT_LLM_ENABLED", "true")

    def should_not_run(*_a, **_k):
        raise AssertionError("format_sp_api_worker_answer must not be called")

    monkeypatch.setattr(
        "src.gateway.dispatcher.summary.sp_api_format_llm.format_sp_api_worker_answer",
        should_not_run,
    )

    from src.gateway.dispatcher.summary.merge import ResultAggregator
    from src.gateway.schemas import RewritePlan, TaskExecutionResult

    plan = RewritePlan(plan_type="single_domain", merge_strategy="concat", task_groups=[])
    raw = "```yaml\norders:\n  - ok: true\n    sp_api_response:\n      OrderStatus: Shipped\n```"
    results = [
        TaskExecutionResult(
            task_id="t1",
            workflow="sp_api",
            query="q",
            status="completed",
            answer=raw,
        )
    ]
    assert ResultAggregator.merge(plan, results) == raw.strip()


def test_result_aggregator_sp_api_fallback_on_format_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_SP_API_FORMAT_LLM_ENABLED", "true")

    def boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        "src.gateway.dispatcher.summary.sp_api_format_llm.format_sp_api_worker_answer",
        boom,
    )

    from src.gateway.dispatcher.summary.merge import ResultAggregator
    from src.gateway.schemas import RewritePlan, TaskExecutionResult

    plan = RewritePlan(plan_type="single_domain", merge_strategy="concat", task_groups=[])
    raw = "raw worker payload only"
    results = [
        TaskExecutionResult(
            task_id="t1",
            workflow="sp_api",
            query="q",
            status="completed",
            answer=raw,
        )
    ]
    merged = ResultAggregator.merge(plan, results)
    assert merged == raw
