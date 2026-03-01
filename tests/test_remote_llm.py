"""
Unit tests for remote LLM provider support (Deepseek, Qwen, GLM).

Tests _step3_create_llm remote path and _step7_generate_answer response normalization.
Uses mocks for ModelManager and LLM to avoid API calls.

Run:
  pytest tests/test_remote_llm.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TestStep3CreateLlmRemote:
    """_step3_create_llm with remote provider."""

    @patch.dict(os.environ, {"RAG_LLM_PROVIDER": "deepseek", "RAG_LLM_MODEL": "deepseek-chat"})
    @patch("ai_toolkit.models.ModelManager")
    def test_remote_provider_uses_model_manager(self, mock_model_manager_cls):
        """When RAG_LLM_PROVIDER=deepseek, uses ModelManager.create_model."""
        from src.rag.query_pipeline import _step3_create_llm

        mock_manager = MagicMock()
        mock_model_manager_cls.return_value = mock_manager
        mock_llm = MagicMock()
        mock_manager.create_model.return_value = mock_llm

        llm = _step3_create_llm(provider="deepseek")

        mock_manager.create_model.assert_called_once()
        call_kwargs = mock_manager.create_model.call_args[1]
        assert call_kwargs["provider"] == "deepseek"
        assert call_kwargs["model"] == "deepseek-chat"
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs
        assert llm is mock_llm

    @patch.dict(os.environ, {"RAG_LLM_PROVIDER": "ollama"})
    def test_ollama_provider_uses_ollama_llm(self):
        """When RAG_LLM_PROVIDER=ollama, uses OllamaLLM (no ModelManager)."""
        from src.rag.query_pipeline import _step3_create_llm

        with patch("langchain_ollama.OllamaLLM") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            llm = _step3_create_llm(provider="ollama")

            mock_ollama.assert_called_once()
            call_kwargs = mock_ollama.call_args[1]
            assert "model" in call_kwargs
            assert "temperature" in call_kwargs
            assert "num_ctx" in call_kwargs
            assert "num_predict" in call_kwargs


class TestStep7GenerateAnswerNormalize:
    """_step7_generate_answer response normalization."""

    def test_ollama_str_return_unchanged(self):
        """OllamaLLM returns str; passes through unchanged."""
        from src.rag.query_pipeline import _step7_generate_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "This is the answer."

        result = _step7_generate_answer(mock_llm, "What is FBA?")

        assert result == "This is the answer."

    def test_chat_openai_aimessage_normalized(self):
        """ChatOpenAI returns AIMessage; _step7 extracts .content."""
        from src.rag.query_pipeline import _step7_generate_answer

        mock_llm = MagicMock()
        mock_aimessage = MagicMock()
        mock_aimessage.content = "Answer from remote API."
        mock_llm.invoke.return_value = mock_aimessage

        result = _step7_generate_answer(mock_llm, "What is FBA?")

        assert result == "Answer from remote API."

    def test_empty_response_returns_empty_string(self):
        """None or empty response returns empty string."""
        from src.rag.query_pipeline import _step7_generate_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = None

        result = _step7_generate_answer(mock_llm, "Question")

        assert result == ""


class TestStep7HybridDualResponse:
    """_step7_hybrid_dual_response: 2-step flow, documents never sent to remote."""

    def test_remote_gets_question_only_documents_stay_local(self):
        """Remote LLM receives question only; local LLM gets documents + general."""
        from langchain_core.documents import Document
        from src.rag.query_pipeline import _step7_hybrid_dual_response

        mock_remote = MagicMock()
        mock_remote.invoke.return_value = "FBA is a logistics service for sellers."
        mock_local = MagicMock()
        mock_local.invoke.return_value = "Based on documents and general: FBA is Fulfillment by Amazon."

        retrieved_docs = [
            Document(page_content="FBA means Fulfillment by Amazon.", metadata={}),
        ]
        result = _step7_hybrid_dual_response(
            mock_local, mock_remote, retrieved_docs, "What is FBA?", verbose=False
        )

        assert mock_remote.invoke.call_count == 1
        remote_prompt = mock_remote.invoke.call_args[0][0]
        assert "FBA means Fulfillment by Amazon" not in remote_prompt
        assert "What is FBA?" in remote_prompt

        assert mock_local.invoke.call_count == 1
        local_prompt = mock_local.invoke.call_args[0][0]
        assert "FBA means Fulfillment by Amazon" in local_prompt
        assert "FBA is a logistics service for sellers" in local_prompt
        assert result == "Based on documents and general: FBA is Fulfillment by Amazon."

    def test_verbose_does_not_raise(self):
        """verbose=True does not raise; output is still correct."""
        from langchain_core.documents import Document
        from src.rag.query_pipeline import _step7_hybrid_dual_response

        mock_remote = MagicMock()
        mock_remote.invoke.return_value = "General answer."
        mock_local = MagicMock()
        mock_local.invoke.return_value = "Synthesized answer."

        retrieved_docs = [Document(page_content="Context.", metadata={})]
        result = _step7_hybrid_dual_response(
            mock_local, mock_remote, retrieved_docs, "Question?", verbose=True
        )

        assert result == "Synthesized answer."
