"""
Tests for UDS Error Handler.

Tests error handling patterns including:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Error message handling
- Error type detection
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.uds.error_handler import (
    UDSError,
    DatabaseError,
    ToolExecutionError,
    LLMError,
    APIError,
    retry_with_backoff,
    CircuitBreaker,
    handle_database_error,
    handle_tool_execution_error,
    handle_llm_error,
    handle_api_error
)


class TestRetryWithBackoff:
    """Test retry decorator with exponential backoff."""

    def test_successful_execution(self):
        """Test successful execution without retries."""
        @retry_with_backoff(max_retries=3, initial_delay=0.1, backoff_factor=2)
        def always_succeed():
            return "success"
        
        result = always_succeed()
        assert result == "success"

    def test_retry_on_failure(self):
        """Test retry logic on first failure."""
        call_count = [0]
        
        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=2)
        def fail_once():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            return "success"
        
        result = fail_once()
        assert result == "success"
        assert call_count[0] == 2

    def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        call_count = [0]
        
        @retry_with_backoff(max_retries=2, initial_delay=0.01, backoff_factor=2)
        def always_fail():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fail()
        
        assert call_count[0] == 2  # 1 initial + 1 retry (max_retries=2 means 2 total attempts)

    def test_specific_exception_type(self):
        """Test that only specified exceptions trigger retry."""
        call_count = [0]
        
        @retry_with_backoff(max_retries=3, exceptions=(ValueError,))
        def raise_value_error():
            call_count[0] += 1
            raise ValueError("Value error")
        
        with pytest.raises(ValueError, match="Value error"):
            raise_value_error()
        
        assert call_count[0] == 3  # 1 initial + 2 retries

    def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        delays = []
        
        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=2)
        def track_delays():
            delays.append(time.time())
            if len(delays) < 2:
                raise ValueError("Fail")
            return "success"
        
        start_time = time.time()
        track_delays()
        
        assert len(delays) == 2
        expected_delay_1 = 0.01
        expected_delay_2 = 0.02
        assert abs(delays[0] - start_time - expected_delay_1) < 0.01
        assert abs(delays[1] - start_time - expected_delay_2) < 0.01


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        assert breaker.state == "closed"
        assert breaker.failure_count == 0

    def test_success_resets_failure_count(self):
        """Test successful execution resets failure count."""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        def always_succeed():
            return "success"
        
        result = breaker.call(always_succeed)
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    def test_failure_increases_count(self):
        """Test failure increases failure count."""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            breaker.call(always_fail)
        
        assert breaker.failure_count == 1
        assert breaker.state == "closed"

    def test_threshold_opens_circuit(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        def always_fail():
            raise ValueError("Always fails")
        
        # Trigger failures
        for _ in range(3):
            try:
                breaker.call(always_fail)
            except ValueError:
                pass
        
        assert breaker.state == "open"
        assert breaker.failure_count == 3

    def test_half_open_state(self):
        """Test half-open state after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        def always_fail():
            raise ValueError("Always fails")
        
        # Open circuit
        for _ in range(3):
            try:
                breaker.call(always_fail)
            except ValueError:
                pass
        
        assert breaker.state == "open"
        
        # Wait for recovery timeout (6.0s > 1.0s, ensuring half-open state)
        time.sleep(6.0)
        
        # Should be in half-open state now (allowing one attempt)
        # Call a function that succeeds to trigger half-open state transition
        result = breaker.call(lambda: "success")
        assert result == "success"  # Should succeed and close the circuit
        assert breaker.state == "closed"  # Circuit should close after successful call

    def test_closed_after_recovery_timeout(self):
        """Test circuit closes after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        def always_fail():
            raise ValueError("Always fails")
        
        # Open circuit
        for _ in range(3):
            try:
                breaker.call(always_fail)
            except ValueError:
                pass
        
        assert breaker.state == "open"
        
        # Wait for recovery timeout
        time.sleep(1.5)
        
        # Try again - should fail
        with pytest.raises(ValueError):
            breaker.call(always_fail)
        
        # Wait for recovery timeout
        time.sleep(1.5)
        
        # Should be closed now
        def succeed():
            return "success"
        
        result = breaker.call(succeed)
        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0


class TestDatabaseErrorHandler:
    """Test database error handling."""

    def test_timeout_error(self):
        """Test timeout error handling."""
        error = TimeoutError("Query timeout")
        
        result = handle_database_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'timeout'
        assert 'timeout' in result['error'].lower()

    def test_connection_error(self):
        """Test connection error handling."""
        error = ConnectionError("Connection failed")
        
        result = handle_database_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'connection'
        assert 'connection' in result['error'].lower()

    def test_syntax_error(self):
        """Test syntax error handling."""
        error = Exception("Syntax error: invalid SQL")
        
        result = handle_database_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'syntax'
        assert 'syntax' in result['error'].lower()

    def test_permission_error(self):
        """Test permission error handling."""
        error = PermissionError("Access denied")
        
        result = handle_database_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'permission'
        assert 'permission' in result['error'].lower()

    def test_generic_database_error(self):
        """Test generic database error handling."""
        error = Exception("Generic database error")
        
        result = handle_database_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'database'
        assert 'database error' in result['error'].lower()


class TestToolExecutionErrorHandler:
    """Test tool execution error handling."""

    def test_timeout_error(self):
        """Test tool timeout error."""
        error = TimeoutError("Tool timeout")
        
        result = handle_tool_execution_error(error, "TestTool")
        
        assert result['success'] is False
        assert result['error_type'] == 'timeout'
        assert 'TestTool' in result['error']
        assert 'timed out' in result['error'].lower()

    def test_not_found_error(self):
        """Test resource not found error."""
        error = FileNotFoundError("Resource not found")
        
        result = handle_tool_execution_error(error, "QueryTool")
        
        assert result['success'] is False
        assert result['error_type'] == 'not_found'
        assert 'QueryTool' in result['error']
        assert 'not found' in result['error'].lower()

    def test_generic_tool_error(self):
        """Test generic tool error."""
        error = Exception("Tool execution failed")
        
        result = handle_tool_execution_error(error, "AnalysisTool")
        
        assert result['success'] is False
        assert result['error_type'] == 'tool_execution'
        assert 'AnalysisTool' in result['error']


class TestLLMErrorHandler:
    """Test LLM error handling."""

    def test_timeout_error(self):
        """Test LLM timeout error."""
        error = TimeoutError("LLM timeout")
        
        result = handle_llm_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'timeout'
        assert 'timeout' in result['error'].lower()

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = Exception("Rate limit exceeded")
        
        result = handle_llm_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'rate_limit'
        assert 'rate limit' in result['error'].lower()

    def test_authentication_error(self):
        """Test authentication error."""
        error = Exception("API key invalid")
        
        result = handle_llm_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'authentication'
        assert 'api key' in result['error'].lower()

    def test_generic_llm_error(self):
        """Test generic LLM error."""
        error = Exception("LLM request failed")
        
        result = handle_llm_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'llm'
        assert 'llm' in result['error'].lower()


class TestAPIErrorHandler:
    """Test API error handling."""

    def test_timeout_error(self):
        """Test API timeout error."""
        error = TimeoutError("API timeout")
        
        result = handle_api_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'timeout'
        assert 'timeout' in result['error'].lower()

    def test_unauthorized_error(self):
        """Test unauthorized error."""
        error = Exception("401 Unauthorized")
        
        result = handle_api_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'authentication'
        assert 'authentication' in result['error'].lower()

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = Exception("429 Rate limit")
        
        result = handle_api_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'rate_limit'
        assert 'rate limit' in result['error'].lower()

    def test_server_error(self):
        """Test server error."""
        error = Exception("500 Internal Server Error")
        
        result = handle_api_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'server'
        assert 'server' in result['error'].lower()

    def test_generic_api_error(self):
        """Test generic API error."""
        error = Exception("API request failed")
        
        result = handle_api_error(error)
        
        assert result['success'] is False
        assert result['error_type'] == 'api'
        assert 'api' in result['error'].lower()


class TestErrorClasses:
    """Test custom error classes."""

    def test_database_error_inheritance(self):
        """Test DatabaseError inherits from UDSError."""
        error = DatabaseError("Database error")
        
        assert isinstance(error, UDSError)
        assert isinstance(error, Exception)

    def test_tool_execution_error_inheritance(self):
        """Test ToolExecutionError inherits from UDSError."""
        error = ToolExecutionError("Tool error")
        
        assert isinstance(error, UDSError)
        assert isinstance(error, Exception)

    def test_llm_error_inheritance(self):
        """Test LLMError inherits from UDSError."""
        error = LLMError("LLM error")
        
        assert isinstance(error, UDSError)
        assert isinstance(error, Exception)

    def test_api_error_inheritance(self):
        """Test APIError inherits from UDSError."""
        error = APIError("API error")
        
        assert isinstance(error, UDSError)
        assert isinstance(error, Exception)
