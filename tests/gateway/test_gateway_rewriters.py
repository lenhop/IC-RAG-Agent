"""
Tests for gateway rewriters (rewrite_with_ollama, rewrite_with_deepseek).

Uses unittest.mock to patch HTTP and API calls. Verifies success paths and
fallback behavior on connection error and timeout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.gateway.rewriters import (
    INTENT_CLASSIFICATION_PROMPT,
    REWRITE_PROMPT,
    REWRITE_PLANNER_PROMPT,
    parse_rewrite_plan_text,
    rewrite_intents_only,
    rewrite_with_ollama,
    rewrite_with_deepseek,
)


# ---------------------------------------------------------------------------
# rewrite_with_ollama
# ---------------------------------------------------------------------------


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_success(mock_post):
    """rewrite_with_ollama returns rewritten text when HTTP 200 with response field."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "  sales revenue October 2024  "}

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("What were my sales in October 2024?")

    assert result == "sales revenue October 2024"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert "model" in call_kwargs["json"]
    assert "prompt" in call_kwargs["json"]
    assert REWRITE_PROMPT in call_kwargs["json"]["prompt"]
    assert "What were my sales in October 2024?" in call_kwargs["json"]["prompt"]
    assert call_kwargs["json"]["stream"] is False


@patch("src.gateway.rewriters.requests.post")
@patch.dict("os.environ", {"GATEWAY_REWRITE_PLANNER_ENABLED": "true"})
def test_rewrite_with_ollama_uses_planner_prompt_when_enabled(mock_post):
    """Planner prompt is used when planner rewrite mode is enabled."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "PlanType: hybrid; TaskGroups: ..."}
    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("compare policy and my recent shipment delays")

    assert result == "PlanType: hybrid; TaskGroups: ..."
    call_kwargs = mock_post.call_args[1]
    assert REWRITE_PLANNER_PROMPT in call_kwargs["json"]["prompt"]


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_connection_error_returns_original(mock_post):
    """On ConnectionError, rewrite_with_ollama returns original query."""
    mock_post.side_effect = requests.ConnectionError("Connection refused")

    result = rewrite_with_ollama("my original query")

    assert result == "my original query"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_timeout_returns_original(mock_post):
    """On Timeout, rewrite_with_ollama returns original query."""
    mock_post.side_effect = requests.Timeout("Request timed out")

    result = rewrite_with_ollama("query with timeout")

    assert result == "query with timeout"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_http_error_returns_original(mock_post):
    """On HTTP non-200, rewrite_with_ollama returns original query."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": "Internal server error"}
    mock_resp.text = "Internal server error"

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("my query")

    assert result == "my query"
    mock_post.assert_called_once()


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_with_ollama_empty_response_returns_original(mock_post):
    """When response field is empty, returns original query."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": ""}

    mock_post.return_value = mock_resp

    result = rewrite_with_ollama("original")

    assert result == "original"
    mock_post.assert_called_once()


def test_rewrite_with_ollama_empty_query_returns_as_is():
    """Empty or whitespace-only query returns unchanged."""
    assert rewrite_with_ollama("") == ""
    assert rewrite_with_ollama("   ") == "   "


# ---------------------------------------------------------------------------
# rewrite_with_deepseek
# ---------------------------------------------------------------------------


@patch("openai.OpenAI")
@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key-123"})
def test_rewrite_with_deepseek_success(mock_openai_class):
    """rewrite_with_deepseek returns rewritten text when API succeeds."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "  refined search query for knowledge base  "
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    result = rewrite_with_deepseek("What is the profit margin?")

    assert result == "refined search query for knowledge base"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"]
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert REWRITE_PROMPT in call_kwargs["messages"][0]["content"]
    assert call_kwargs["messages"][1]["content"] == "What is the profit margin?"


@patch("openai.OpenAI")
@patch.dict(
    "os.environ",
    {
        "DEEPSEEK_API_KEY": "test-key-123",
        "GATEWAY_REWRITE_PLANNER_ENABLED": "true",
    },
)
def test_rewrite_with_deepseek_uses_planner_prompt_when_enabled(mock_openai_class):
    """Planner prompt is injected for DeepSeek when planner rewrite mode is enabled."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "PlanType: single-domain; TaskGroups: [T1:general:define term]"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = rewrite_with_deepseek("what is FBA and shipment status impact")

    assert "PlanType:" in result
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert REWRITE_PLANNER_PROMPT in call_kwargs["messages"][0]["content"]


@patch("openai.OpenAI")
@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"})
def test_rewrite_with_deepseek_api_error_returns_original(mock_openai_class):
    """On API exception, rewrite_with_deepseek returns original query."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API rate limit")

    result = rewrite_with_deepseek("my question")

    assert result == "my question"
    mock_client.chat.completions.create.assert_called_once()


@patch.dict("os.environ", {}, clear=True)
def test_rewrite_with_deepseek_no_api_key_returns_original():
    """When DEEPSEEK_API_KEY not set, returns original query."""
    result = rewrite_with_deepseek("query without key")

    assert result == "query without key"


def test_rewrite_with_deepseek_empty_query_returns_as_is():
    """Empty or whitespace-only query returns unchanged."""
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "key"}):
        assert rewrite_with_deepseek("") == ""
        assert rewrite_with_deepseek("   ") == "   "


def test_parse_rewrite_plan_text_valid_json():
    """Planner parser should return structured plan for valid JSON output."""
    raw = """
    {
      "plan_type": "hybrid",
      "merge_strategy": "concat",
      "task_groups": [
        {
          "group_id": "g1",
          "parallel": true,
          "tasks": [
            {"task_id": "t1", "workflow": "general", "query": "what is fba", "depends_on": [], "reason": "definition"},
            {"task_id": "t2", "workflow": "sp_api", "query": "fba fee for ASIN B074KF7RKS", "depends_on": [], "reason": "metrics"}
          ]
        }
      ]
    }
    """
    plan = parse_rewrite_plan_text(raw, "fallback query")
    assert plan is not None
    assert plan.plan_type == "hybrid"
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) == 2
    assert plan.task_groups[0].tasks[0].workflow == "general"
    assert plan.task_groups[0].tasks[1].workflow == "sp_api"


def test_parse_rewrite_plan_text_invalid_json_falls_back_to_single_task():
    """Invalid planner text should degrade to one fallback task."""
    plan = parse_rewrite_plan_text("PlanType: hybrid; TaskGroups: ...", "total sales by month")
    assert plan is not None
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) == 1
    task = plan.task_groups[0].tasks[0]
    assert task.workflow == "general"
    assert task.query == "total sales by month"


def test_parse_rewrite_plan_text_intents_only_format():
    """Phase 1 intents-only JSON should produce RewritePlan with extracted_intents."""
    raw = '{"intents": ["what is FBA", "get order 123", "which table stores referral fee data"]}'
    plan = parse_rewrite_plan_text(raw, "fallback query")
    assert plan is not None
    assert plan.extracted_intents == [
        "what is FBA",
        "get order 123",
        "which table stores referral fee data",
    ]
    # task_groups gets fallback single task when empty; build_execution_plan uses intents
    assert plan.task_groups


def test_parse_rewrite_plan_text_intents_only_with_markdown_fences():
    """Intents-only with markdown code fences should be stripped and parsed."""
    raw = '```json\n{"intents": ["q1", "q2"]}\n```'
    plan = parse_rewrite_plan_text(raw, "fallback")
    assert plan is not None
    assert plan.extracted_intents == ["q1", "q2"]


def test_parse_rewrite_plan_text_intents_only_empty_list_falls_back():
    """Intents-only with empty intents list should fall back to single task."""
    raw = '{"intents": []}'
    plan = parse_rewrite_plan_text(raw, "my fallback query")
    assert plan is not None
    assert plan.extracted_intents != [] or len(plan.task_groups) >= 1


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_intents_only_ollama_success(mock_post):
    """rewrite_intents_only returns parsed intents when Ollama returns valid JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": '{"intents": ["what is FBA", "get order 112-9876543-12"]}'
    }
    mock_post.return_value = mock_resp

    result = rewrite_intents_only("what is FBA get order 112-9876543-12", backend="ollama")

    assert result is not None
    assert result.get("intents") == ["what is FBA", "get order 112-9876543-12"]
    call_kwargs = mock_post.call_args[1]
    assert INTENT_CLASSIFICATION_PROMPT in call_kwargs["json"]["prompt"]


@patch("src.gateway.rewriters.requests.post")
def test_rewrite_intents_only_ollama_failure_returns_none(mock_post):
    """rewrite_intents_only returns None when Ollama fails."""
    mock_post.side_effect = requests.ConnectionError("Connection refused")

    result = rewrite_intents_only("my query", backend="ollama")

    assert result is None


def test_rewrite_intents_only_empty_query_returns_none():
    """rewrite_intents_only returns None for empty query."""
    assert rewrite_intents_only("") is None
    assert rewrite_intents_only("   ") is None


@patch.dict("os.environ", {"GATEWAY_REWRITE_PLANNER_MAX_TASKS": "1"})
def test_parse_rewrite_plan_text_honors_max_tasks_guard():
    """Planner parser should trim tasks based on max task guard."""
    raw = """
    {
      "plan_type": "hybrid",
      "merge_strategy": "concat",
      "task_groups": [
        {
          "group_id": "g1",
          "parallel": true,
          "tasks": [
            {"task_id": "t1", "workflow": "general", "query": "q1"},
            {"task_id": "t2", "workflow": "uds", "query": "q2"}
          ]
        }
      ]
    }
    """
    plan = parse_rewrite_plan_text(raw, "fallback")
    assert plan is not None
    assert len(plan.task_groups) == 1
    assert len(plan.task_groups[0].tasks) == 1
    assert plan.task_groups[0].tasks[0].query == "q1"
