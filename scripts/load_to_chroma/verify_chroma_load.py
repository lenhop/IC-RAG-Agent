#!/usr/bin/env python3
"""
Verify local ChromaDB after load_documents_to_chroma + load_intent_registry_to_chroma.

Exits 0 only if documents and intent_registry collections have expected minimum counts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.chroma import verify_default_project_chroma_stores
from src.utils import bootstrap_project

PROJECT_ROOT = bootstrap_project()


def main() -> int:
    """Delegate verification to src.chroma.verify (single implementation)."""
    ok = verify_default_project_chroma_stores(PROJECT_ROOT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
