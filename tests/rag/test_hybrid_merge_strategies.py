"""
Test hybrid mode: 2-step flow, documents never sent to remote LLM.

Step 1: Remote LLM general (question only).
Step 2: Local LLM synthesizes (documents + remote response + question).
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


class TestHybridTwoStepFlow:
    """Test hybrid mode 2-step flow with data security."""

    def test_documents_never_sent_to_remote(self):
        """Remote LLM receives question only; documents stay with local LLM."""
        from src.rag.query_pipeline import _step7_hybrid_dual_response

        mock_remote = MagicMock()
        mock_remote.invoke.return_value = "General knowledge: FBA helps sellers."
        mock_local = MagicMock()
        mock_local.invoke.return_value = "Combined: FBA is Fulfillment by Amazon."

        retrieved_docs = [Document(page_content="FBA info here.", metadata={})]
        _step7_hybrid_dual_response(
            mock_local, mock_remote, retrieved_docs, "What is FBA?", verbose=False
        )

        remote_prompt = mock_remote.invoke.call_args[0][0]
        assert "FBA info here" not in remote_prompt
        assert "What is FBA?" in remote_prompt

        local_prompt = mock_local.invoke.call_args[0][0]
        assert "FBA info here" in local_prompt
        assert "General knowledge: FBA helps sellers" in local_prompt

    def test_synthesis_prioritizes_documents(self):
        """Local LLM synthesis prompt instructs to prioritize document facts."""
        from src.rag.query_pipeline import _step6_build_rag_prompt

        docs = [Document(page_content="Doc: FBA = Fulfillment by Amazon.", metadata={})]
        prompt = _step6_build_rag_prompt(
            docs, "What is FBA?", mode="synthesis", general_response="General: FBA is a service."
        )
        assert "Prioritize document facts" in prompt
        assert "Doc: FBA = Fulfillment by Amazon" in prompt
        assert "General: FBA is a service" in prompt
