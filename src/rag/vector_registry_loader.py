"""
Intent registry Chroma loader — re-exports ``src.chroma.intent_registry``.

Prefer ``from src.chroma import load_vector_registry_local, ...`` in new code.
"""

from __future__ import annotations

from src.chroma.intent_registry import (
    load_vector_registry_local,
    resolve_registry_chroma_path,
    resolve_registry_csv_path,
)

__all__ = [
    "load_vector_registry_local",
    "resolve_registry_csv_path",
    "resolve_registry_chroma_path",
]
