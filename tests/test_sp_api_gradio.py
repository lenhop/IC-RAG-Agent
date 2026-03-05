"""
Unit tests for SP-API Gradio UI helper functions.

Tests the following functions without making real HTTP calls:
- _get_or_create_session_id()
- _clear_session_id()
- _check_health()
- _fetch_tools()
- _query_sync()
- _query_stream_generator()

Feature: sp-api-gradio-ui
"""

import json
from unittest.mock import patch, MagicMock, call
from uuid import UUID

import pytest


# ============================================================================
# Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Tests for session ID creation and clearing."""

    def test_get_or_create_session_id_creates_new_on_first_call(self):
        """_get_or_create_session_id creates new UUID when _session_id is empty."""
        # Reset global state
        import scripts.run_sp_api_gradio as gradio_module
        gradio_module._session_id = ""
        
        sid = gradio_module._get_or_create_session_id()
        assert sid, "Session ID should not be empty"
        # Verify it's a valid UUID
        try:
            UUID(sid)
        except ValueError:
            pytest.fail(f"Session ID '{sid}' is not a valid UUID")

    def test_get_or_create_session_id_returns_existing(self):
        """_get_or_create_session_id returns existing ID if already set."""
        import scripts.run_sp_api_gradio as gradio_module
        gradio_module._session_id = "existing-session-12345"
        
        sid = gradio_module._get_or_create_session_id()
        assert sid == "existing-session-12345"

    def test_clear_session_id_generates_new(self):
        """_clear_session_id generates and returns new session ID."""
        import scripts.run_sp_api_gradio as gradio_module
        gradio_module._session_id = "old-session-id"
        
        new_id = gradio_module._clear_session_id()
        assert new_id != "old-session-id"
        try:
            UUID(new_id)
        except ValueError:
            pytest.fail(f"New session ID '{new_id}' is not a valid UUID")

    @patch("scripts.run_sp_api_gradio.requests.delete")
    def test_clear_session_id_calls_delete_on_old_session(self, mock_delete):
        """_clear_session_id calls DELETE on old session ID."""
        import scripts.run_sp_api_gradio as gradio_module
        old_id = "old-session-abc"
        gradio_module._session_id = old_id
        
        new_id = gradio_module._clear_session_id()
        
        mock_delete.assert_called_once()
        call_args = mock_delete.call_args
        assert old_id in call_args[0][0]  # Check URL contains old ID
        assert gradio_module._session_id == new_id

    @patch("scripts.run_sp_api_gradio.requests.delete")
    def test_clear_session_id_handles_delete_error(self, mock_delete):
        """_clear_session_id handles DELETE errors gracefully."""
        import scripts.run_sp_api_gradio as gradio_module
        mock_delete.side_effect = Exception("Connection refused")
        gradio_module._session_id = "old-id"
        
        new_id = gradio_module._clear_session_id()
        
        assert new_id  # Should still return new ID
        assert gradio_module._session_id == new_id


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthCheck:
    """Tests for _check_health()."""

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_check_health_returns_true_on_200(self, mock_get):
        """_check_health returns (True, msg) on 200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response
        
        ok, msg = gradio_module._check_health()
        
        assert ok is True
        assert "ok" in msg.lower() or "status" in msg.lower()

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_check_health_returns_false_on_non_200(self, mock_get):
        """_check_health returns (False, msg) on non-200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        ok, msg = gradio_module._check_health()
        
        assert ok is False
        assert "500" in msg

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_check_health_handles_connection_error(self, mock_get):
        """_check_health returns (False, msg) on ConnectionError."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_get.side_effect = gradio_module.requests.ConnectionError("Refused")
        
        ok, msg = gradio_module._check_health()
        
        assert ok is False
        assert "reachable" in msg.lower() or "connection" in msg.lower()

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_check_health_handles_generic_exception(self, mock_get):
        """_check_health returns (False, msg) on generic exception."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_get.side_effect = ValueError("Bad value")
        
        ok, msg = gradio_module._check_health()
        
        assert ok is False
        assert msg  # Should have some error message


# ============================================================================
# Fetch Tools Tests
# ============================================================================

class TestFetchTools:
    """Tests for _fetch_tools()."""

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_fetch_tools_returns_list_on_200(self, mock_get):
        """_fetch_tools returns tool list on 200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        tools = [
            {"name": "product_catalog", "description": "Look up products"},
            {"name": "inventory_summary", "description": "Check inventory"},
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = tools
        mock_get.return_value = mock_response
        
        result = gradio_module._fetch_tools()
        
        assert result == tools
        assert len(result) == 2
        assert result[0]["name"] == "product_catalog"

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_fetch_tools_returns_empty_on_non_200(self, mock_get):
        """_fetch_tools returns empty list on non-200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = gradio_module._fetch_tools()
        
        assert result == []

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_fetch_tools_returns_empty_on_connection_error(self, mock_get):
        """_fetch_tools returns empty list on ConnectionError."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_get.side_effect = gradio_module.requests.ConnectionError()
        
        result = gradio_module._fetch_tools()
        
        assert result == []

    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_fetch_tools_returns_empty_on_json_error(self, mock_get):
        """_fetch_tools returns empty list when JSON parsing fails."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        result = gradio_module._fetch_tools()
        
        assert result == []


# ============================================================================
# Sync Query Tests
# ============================================================================

class TestQuerySync:
    """Tests for _query_sync()."""

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_returns_response_on_200(self, mock_post):
        """_query_sync returns response text on 200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Here is the inventory data",
            "iterations": 2,
        }
        mock_post.return_value = mock_response
        
        result = gradio_module._query_sync("What is my inventory?", "session-1")
        
        assert "inventory data" in result
        assert "Iterations: 2" in result

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_handles_connection_error(self, mock_post):
        """_query_sync returns error message on ConnectionError."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_post.side_effect = gradio_module.requests.ConnectionError()
        
        result = gradio_module._query_sync("test", "session-1")
        
        assert isinstance(result, str)
        assert "cannot connect" in result.lower() or "running" in result.lower()

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_handles_timeout(self, mock_post):
        """_query_sync returns error message on Timeout."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_post.side_effect = gradio_module.requests.Timeout()
        
        result = gradio_module._query_sync("test", "session-1")
        
        assert isinstance(result, str)
        assert "timeout" in result.lower() or "took too long" in result.lower()

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_handles_503_error(self, mock_post):
        """_query_sync returns specific message on 503 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response
        
        result = gradio_module._query_sync("test", "session-1")
        
        assert "not initialized" in result.lower() or "503" in result

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_handles_other_http_errors(self, mock_post):
        """_query_sync returns error message on other HTTP errors."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Bad request"}
        mock_post.return_value = mock_response
        
        result = gradio_module._query_sync("test", "session-1")
        
        assert "Error 400" in result
        assert "Bad request" in result

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_handles_invalid_json_response(self, mock_post):
        """_query_sync handles invalid JSON response."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response
        
        result = gradio_module._query_sync("test", "session-1")
        
        assert "Invalid response" in result

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_query_sync_uses_correct_timeout(self, mock_post):
        """_query_sync passes the configured timeout to requests.post."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok", "iterations": 0}
        mock_post.return_value = mock_response
        
        gradio_module._query_sync("test", "session-1")
        
        call_kwargs = mock_post.call_args[1]
        assert "timeout" in call_kwargs


# ============================================================================
# Stream Generator Tests
# ============================================================================

class TestQueryStreamGenerator:
    """Tests for _query_stream_generator()."""

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_parses_sse_chunks(self, mock_post):
        """_query_stream_generator parses SSE thought/observation/final chunks."""
        import scripts.run_sp_api_gradio as gradio_module
        
        sse_data = [
            'data: {"type": "thought", "content": "I will check inventory"}',
            'data: {"type": "observation", "content": "Found 150 units"}',
            'data: {"type": "final", "response": "Your inventory is 150 units"}',
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_data
        mock_post.return_value = mock_response
        
        chunks = list(gradio_module._query_stream_generator("test", "session-1"))
        
        assert len(chunks) > 0
        final_chunk = chunks[-1]
        assert "150 units" in final_chunk or "inventory" in final_chunk

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_falls_back_to_sync_on_connection_error(self, mock_post):
        """_query_stream_generator falls back to sync on ConnectionError."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_post.side_effect = gradio_module.requests.ConnectionError()
        
        chunks = list(gradio_module._query_stream_generator("test", "session-1"))
        
        assert len(chunks) > 0
        # Should call _query_sync, which returns a fallback message
        assert isinstance(chunks[-1], str)

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_falls_back_to_sync_on_non_200(self, mock_post):
        """_query_stream_generator falls back to sync on non-200 status."""
        import scripts.run_sp_api_gradio as gradio_module
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        chunks = list(gradio_module._query_stream_generator("test", "session-1"))
        
        assert len(chunks) > 0

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_handles_json_decode_error(self, mock_post):
        """_query_stream_generator skips invalid JSON in SSE lines."""
        import scripts.run_sp_api_gradio as gradio_module
        
        sse_data = [
            'data: {"type": "thought", "content": "Start"}',
            'data: {invalid json}',  # Malformed
            'data: {"type": "final", "response": "End"}',
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_data
        mock_post.return_value = mock_response
        
        chunks = list(gradio_module._query_stream_generator("test", "session-1"))
        
        # Should not raise, just skip malformed line
        assert len(chunks) > 0

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_accumulates_chunks(self, mock_post):
        """_query_stream_generator accumulates chunks during streaming."""
        import scripts.run_sp_api_gradio as gradio_module
        
        sse_data = [
            'data: {"type": "thought", "content": "Analyzing"}',
            'data: {"type": "observation", "content": "Found data"}',
            'data: {"type": "final", "response": "Complete"}',
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = sse_data
        mock_post.return_value = mock_response
        
        chunks = list(gradio_module._query_stream_generator("query", "session-1"))
        
        # Final chunk should contain accumulated thoughts/observations
        final = chunks[-1]
        assert "Analyzing" in final or "data" in final or "Complete" in final

    @patch("scripts.run_sp_api_gradio.requests.post")
    def test_stream_generator_handles_stream_parse_error(self, mock_post):
        """_query_stream_generator handles stream parse errors gracefully."""
        import scripts.run_sp_api_gradio as gradio_module
        
        # First call (streaming) raises error
        stream_response = MagicMock()
        stream_response.status_code = 200
        stream_response.iter_lines.side_effect = RuntimeError("Parse error")
        
        # Fallback call (sync) returns valid response
        sync_response = MagicMock()
        sync_response.status_code = 200
        sync_response.json.return_value = {"response": "Fallback response", "iterations": 0}
        
        mock_post.side_effect = [stream_response, sync_response]
        
        chunks = list(gradio_module._query_stream_generator("test", "session-1"))
        
        # Should fall back and return something
        assert len(chunks) > 0
        assert any("Fallback" in chunk or "error" in chunk.lower() for chunk in chunks)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    @patch("scripts.run_sp_api_gradio.requests.delete")
    @patch("scripts.run_sp_api_gradio.requests.post")
    @patch("scripts.run_sp_api_gradio.requests.get")
    def test_session_lifecycle(self, mock_get, mock_post, mock_delete):
        """Test full session lifecycle: create → health check → query → clear."""
        import scripts.run_sp_api_gradio as gradio_module
        
        # Reset state
        gradio_module._session_id = ""
        
        # 1. Create session
        sid1 = gradio_module._get_or_create_session_id()
        assert sid1
        
        # 2. Health check
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        ok, msg = gradio_module._check_health()
        assert ok
        
        # 3. Query
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "Answer", "iterations": 1}
        )
        result = gradio_module._query_sync("query", sid1)
        assert "Answer" in result
        
        # 4. Clear session
        mock_delete.return_value = MagicMock(status_code=200)
        sid2 = gradio_module._clear_session_id()
        assert sid2 != sid1
