"""Tests for unified Ollama client (four env vars only)."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.call_ollama import OllamaClient, get_ollama_config

_OLLAMA_ENV = {
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_GENERATE_MODEL": "qwen3:1.7b",
    "OLLAMA_REQUEST_TIMEOUT": "15",
    "OLLAMA_EMBED_MODEL": "all-minilm:latest",
}


@patch.dict("os.environ", _OLLAMA_ENV, clear=False)
def test_get_ollama_config():
    cfg = get_ollama_config()
    assert cfg.generate_url.endswith("/api/generate")
    assert cfg.model == "qwen3:1.7b"
    assert cfg.timeout == 15
    assert cfg.embed_model == "all-minilm:latest"


@patch.dict("os.environ", {}, clear=True)
def test_get_ollama_config_requires_env():
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL is not set"):
        get_ollama_config()


@patch.dict("os.environ", _OLLAMA_ENV, clear=False)
@patch("src.llm.call_ollama.requests.post")
def test_ollama_client_generate(mock_post):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"response": "  hello  "})
    out = OllamaClient().generate("prompt")
    assert out == "hello"
    mock_post.assert_called_once()


@patch.dict("os.environ", _OLLAMA_ENV, clear=False)
@patch("src.llm.call_ollama.requests.post")
def test_ollama_client_generate_http_error(mock_post):
    mock_post.return_value = MagicMock(status_code=500, json=lambda: {"error": "x"}, text="x")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        OllamaClient().generate("p")
