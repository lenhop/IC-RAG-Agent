"""
Shared utilities (project bootstrap, path helpers, etc.).

Chroma ingest scripts should call ``bootstrap_project`` from here; ``src.chroma``
re-exports it for backward compatibility.
"""

from __future__ import annotations

from .bootstrap import bootstrap_project, resolve_path

__all__ = ["bootstrap_project", "resolve_path"]
