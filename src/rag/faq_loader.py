"""
FAQ question loader for intent classification.

Loads question column from FAQ CSV (e.g. amazon_fqa.csv) for embedding
and similarity comparison with user queries.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_faq_questions(project_root: Path | None = None) -> List[str]:
    """
    Load FAQ questions from CSV at RAG_FAQ_CSV.

    Reads the "question" column only. Returns empty list if path not set,
    file missing, or parse error. Handles multi-line CSV cells (quoted).

    Args:
        project_root: Optional project root for resolving relative paths.

    Returns:
        List of question strings (stripped, non-empty).
    """
    root = project_root or PROJECT_ROOT
    path_val = os.getenv("RAG_FAQ_CSV", "data/intent_classification/fqa/amazon_fqa.csv")
    if not path_val:
        return []

    path = Path(path_val)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        return []

    questions: List[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                if q:
                    questions.append(q)
    except (csv.Error, OSError):
        pass
    return questions
