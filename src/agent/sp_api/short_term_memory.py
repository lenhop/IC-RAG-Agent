"""Conversation Memory - Redis-backed session history (short-term)."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

SESSION_TTL = 86400


class ConversationMemory:
    """Redis-backed conversation memory."""

    def __init__(self, redis_client):
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    def save_turn(self, session_id: str, query: str, response: str, agent_state: Optional[Any] = None) -> None:
        iterations = getattr(agent_state, "iteration", 0) if agent_state else 0
        turn = {"query": query, "response": response, "timestamp": datetime.utcnow().isoformat() + "Z", "iterations": iterations}
        key = self._key(session_id)
        try:
            self._redis.rpush(key, json.dumps(turn))
            self._redis.expire(key, SESSION_TTL)
        except Exception:
            pass

    def get_history(self, session_id: str, last_n: int = 10) -> List[Dict[str, Any]]:
        key = self._key(session_id)
        try:
            raw = self._redis.lrange(key, -last_n, -1)
            return [json.loads(r) for r in raw] if raw else []
        except Exception:
            return []

    def clear_session(self, session_id: str) -> None:
        try:
            self._redis.delete(self._key(session_id))
        except Exception:
            pass
