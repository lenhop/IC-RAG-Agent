"""
API Streaming Tests - SSE validation.

Tests the streaming endpoint (POST /api/v1/uds/query/stream) with
Server-Sent Events. Validates event sequence and structure.

Requires: API server running (uvicorn src.uds.api:app --port 8000)
          ClickHouse, LLM

Run: pytest tests/test_api_streaming.py -v
"""

import json
import os

import pytest
import requests

BASE_URL = os.getenv("UDS_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("UDS_API_TIMEOUT", "90"))


@pytest.fixture(scope="module")
def api_available():
    """Check API is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            return True
    except requests.exceptions.RequestException:
        pass
    pytest.skip("API server not running")


def _parse_sse_events(response: requests.Response):
    """Parse SSE events from streaming response."""
    events = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                pass
    return events


def test_streaming_query_basic(api_available):
    """Test streaming query endpoint - basic structure."""
    response = requests.post(
        f"{BASE_URL}/api/v1/uds/query/stream",
        json={"query": "What were total sales in October?"},
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type

    events = _parse_sse_events(response)

    assert len(events) >= 1, "Expected at least one SSE event"

    event_types = [e.get("event") for e in events if "event" in e]
    assert "start" in event_types, "Expected 'start' event"

    # Current implementation: start -> complete or error
    assert "complete" in event_types or "error" in event_types, (
        "Expected 'complete' or 'error' event"
    )


def test_streaming_query_event_sequence(api_available):
    """Test streaming event sequence: start -> complete/error."""
    response = requests.post(
        f"{BASE_URL}/api/v1/uds/query/stream",
        json={"query": "List all available tables"},
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    events = _parse_sse_events(response)

    assert len(events) >= 1

    first = events[0]
    assert first.get("event") == "start"
    assert "query_id" in first

    last = events[-1]
    assert last.get("event") in ("complete", "error")
    if last.get("event") == "complete":
        assert "data" in last or "query_id" in last


def test_streaming_query_complex(api_available):
    """Test streaming with complex query."""
    response = requests.post(
        f"{BASE_URL}/api/v1/uds/query/stream",
        json={
            "query": "Top 10 products by revenue with their inventory levels"
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    events = _parse_sse_events(response)

    assert len(events) >= 1
    event_types = [e.get("event") for e in events if "event" in e]
    assert "start" in event_types
    assert "complete" in event_types or "error" in event_types


def test_streaming_query_error_handling(api_available):
    """Test streaming sends error event on failure."""
    # Empty or malformed query might fail
    response = requests.post(
        f"{BASE_URL}/api/v1/uds/query/stream",
        json={"query": ""},
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    events = _parse_sse_events(response)

    assert len(events) >= 1
    event_types = [e.get("event") for e in events if "event" in e]
    assert "start" in event_types
    # May get complete (empty) or error
    assert "complete" in event_types or "error" in event_types


def test_streaming_content_type(api_available):
    """Test streaming response has correct headers."""
    response = requests.post(
        f"{BASE_URL}/api/v1/uds/query/stream",
        json={"query": "What were total sales in October?"},
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "Cache-Control" in response.headers or "cache-control" in str(
        response.headers
    ).lower()
