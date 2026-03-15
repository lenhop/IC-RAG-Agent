"""Tests for unified DeepSeek client (Route LLM)."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.call_deepseek import DeepSeekChat, get_deepseek_config


@patch("src.llm.call_deepseek.OpenAI")
@patch.dict(
    "os.environ",
    {
        "DEEPSEEK_API_KEY": "k",
        "DEEPSEEK_LLM_MODEL": "deepseek-chat",
        "DEEPSEEK_REQUEST_TIMEOUT": "20",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    },
)
def test_deep_seek_chat_complete(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_choice = MagicMock()
    mock_choice.message.content = '{"needs_clarification": false}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    out = DeepSeekChat().complete("sys", "user")
    assert "needs_clarification" in out
    mock_client.chat.completions.create.assert_called_once()
    assert mock_client.chat.completions.create.call_args[1]["model"] == "deepseek-chat"


@patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=True)
def test_get_deepseek_config_requires_key():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_deepseek_config()


@patch.dict(
    "os.environ",
    {
        "DEEPSEEK_API_KEY": "k",
        "DEEPSEEK_LLM_MODEL": "deepseek-chat",
        "DEEPSEEK_REQUEST_TIMEOUT": "10",
    },
)
def test_get_deepseek_config_defaults():
    cfg = get_deepseek_config()
    assert cfg.llm_model == "deepseek-chat"
    assert cfg.request_timeout == 10
    assert cfg.base_url == "https://api.deepseek.com"
