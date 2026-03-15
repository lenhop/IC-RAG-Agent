"""
UDS Error Handler - Robust error handling and resilience patterns.

Provides:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Graceful error messages
- Handle database, agent, and API errors
"""

import logging
import threading
import time
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class UDSError(Exception):
    """Base exception for UDS Agent errors."""
    pass


class DatabaseError(UDSError):
    """Database connection or query errors."""
    pass


class ToolExecutionError(UDSError):
    """Tool execution failures."""
    pass


class LLMError(UDSError):
    """LLM API failures."""
    pass


class APIError(UDSError):
    """API request errors."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch

    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by stopping calls to failing services.
    Thread-safe via threading.Lock for concurrent use.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again after circuit opens
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function execution

        Raises:
            UDSError if circuit is open
        """
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker transitioning to half_open")
                else:
                    raise UDSError("Circuit breaker is open")

        # Execute outside lock to avoid blocking other threads
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful call")
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(
                        "Circuit breaker opened after %d failures",
                        self.failure_count,
                    )
            raise e


def handle_database_error(error: Exception) -> dict:
    """
    Handle database errors gracefully.

    Args:
        error: Database error exception

    Returns:
        Error response dictionary
    """
    error_str = str(error).lower()
    
    if "timeout" in error_str:
        return {
            'success': False,
            'error': 'Query timeout. Try simplifying your question.',
            'error_type': 'timeout'
        }
    elif "connection" in error_str:
        return {
            'success': False,
            'error': 'Database connection failed. Please try again.',
            'error_type': 'connection'
        }
    elif "syntax" in error_str or "invalid" in error_str:
        return {
            'success': False,
            'error': 'Invalid SQL syntax. Please rephrase your question.',
            'error_type': 'syntax'
        }
    elif "permission" in error_str or "access" in error_str:
        return {
            'success': False,
            'error': 'Permission denied. Please contact administrator.',
            'error_type': 'permission'
        }
    else:
        return {
            'success': False,
            'error': 'Database error occurred.',
            'error_type': 'database'
        }


def handle_tool_execution_error(error: Exception, tool_name: str) -> dict:
    """
    Handle tool execution errors gracefully.

    Args:
        error: Tool execution error
        tool_name: Name of the tool that failed

    Returns:
        Error response dictionary
    """
    error_str = str(error).lower()
    
    if "timeout" in error_str:
        return {
            'success': False,
            'error': f'{tool_name} timed out. Try a simpler query.',
            'error_type': 'timeout'
        }
    elif "not found" in error_str or "does not exist" in error_str:
        return {
            'success': False,
            'error': f'{tool_name} failed: Resource not found.',
            'error_type': 'not_found'
        }
    else:
        return {
            'success': False,
            'error': f'{tool_name} execution failed.',
            'error_type': 'tool_execution'
        }


def handle_llm_error(error: Exception) -> dict:
    """
    Handle LLM API errors gracefully.

    Args:
        error: LLM API error

    Returns:
        Error response dictionary
    """
    error_str = str(error).lower()
    
    if "timeout" in error_str:
        return {
            'success': False,
            'error': 'LLM request timeout. Please try again.',
            'error_type': 'timeout'
        }
    elif "rate limit" in error_str or "quota" in error_str:
        return {
            'success': False,
            'error': 'Rate limit exceeded. Please wait and try again.',
            'error_type': 'rate_limit'
        }
    elif "authentication" in error_str or "api key" in error_str:
        return {
            'success': False,
            'error': 'LLM authentication failed. Check API key.',
            'error_type': 'authentication'
        }
    elif (
        "connection refused" in error_str
        or "failed to connect" in error_str
        or "couldn't connect" in error_str
        or "max retries exceeded" in error_str
    ):
        return {
            'success': False,
            'error': 'Local LLM is unavailable. Start Ollama (ollama serve) or configure remote LLM API key.',
            'error_type': 'connection'
        }
    elif "model" in error_str and "not found" in error_str:
        return {
            'success': False,
            'error': 'LLM model is not available locally. Pull it with: ollama pull qwen3:1.7b',
            'error_type': 'model_not_found'
        }
    else:
        # Include concise detail so users can self-diagnose provider/auth/network issues.
        detail = str(error).strip()
        if len(detail) > 240:
            detail = detail[:240] + "..."
        return {
            'success': False,
            'error': f'LLM request failed: {detail}' if detail else 'LLM request failed.',
            'error_type': 'llm'
        }


def handle_api_error(error: Exception) -> dict:
    """
    Handle API request errors gracefully.

    Args:
        error: API request error

    Returns:
        Error response dictionary
    """
    error_str = str(error).lower()
    
    if "timeout" in error_str:
        return {
            'success': False,
            'error': 'API request timeout.',
            'error_type': 'timeout'
        }
    elif "401" in error_str or "unauthorized" in error_str:
        return {
            'success': False,
            'error': 'API authentication failed.',
            'error_type': 'authentication'
        }
    elif "429" in error_str or "rate limit" in error_str:
        return {
            'success': False,
            'error': 'API rate limit exceeded.',
            'error_type': 'rate_limit'
        }
    elif "500" in error_str or "503" in error_str:
        return {
            'success': False,
            'error': 'API server error. Please try again.',
            'error_type': 'server'
        }
    else:
        return {
            'success': False,
            'error': 'API request failed.',
            'error_type': 'api'
        }
