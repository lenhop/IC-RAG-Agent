"""
Chroma loaders — backward-compatible facade over ``src.chroma``.

New code should use ``src.utils`` for ``bootstrap_project`` / ``resolve_path`` and
``src.chroma`` for ingest. This module re-exports the same symbols and keeps
RAG-specific helpers (e.g. FAQ CSV) that are not part of the generic Chroma API.
"""

from __future__ import annotations

import csv as csv_module
import os
from pathlib import Path
from typing import List

# Project root for resolving relative paths when callers omit project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.chroma import (
    load_csv_column_from_file,
    load_csv_column_to_chroma,
    load_documents_to_chroma,
)
from src.utils import bootstrap_project, resolve_path

__all__ = [
    "bootstrap_project",
    "resolve_path",
    "load_faq_questions",
    "load_csv_column_from_file",
    "load_csv_column_to_chroma",
    "load_documents_to_chroma",
]


def load_faq_questions(
    project_root: Path | None = None,
    csv_path: str | None = None,
) -> List[str]:
    """
    Load FAQ questions from CSV at RAG_FAQ_CSV or csv_path.

    Reads the "question" column only. Returns empty list if path not set,
    file missing, or parse error.

    Args:
        project_root: Optional project root for resolving relative paths.
        csv_path: Optional explicit CSV path; overrides RAG_FAQ_CSV when set.

    Returns:
        List of question strings (stripped, non-empty).
    """
    root = project_root or _PROJECT_ROOT
    path_val = csv_path or os.getenv(
        "RAG_FAQ_CSV", "data/intent_classification/fqa/amazon_fqa.csv"
    )
    if not path_val:
        return []

    path = Path(path_val)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        return []

    try:
        return load_csv_column_from_file(path, "question")
    except (csv_module.Error, OSError):
        return []
