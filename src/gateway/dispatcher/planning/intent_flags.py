"""Feature flags for dispatcher planning (environment-driven)."""

from __future__ import annotations

import os


class IntentClassificationFlags:
    """Gateway intent-classification toggles (classmethod facade)."""

    @classmethod
    def vector_intent_enabled(cls) -> bool:
        """Return True when gateway vector/LLM intent classification is enabled."""
        v = os.getenv("GATEWAY_VECTOR_INTENT_ENABLED", "false").strip().lower()
        return v in ("1", "true", "yes", "on")
