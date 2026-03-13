"""
CLI tests for standalone Chroma loading scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_SCRIPT = PROJECT_ROOT / "scripts" / "load_to_chroma" / "load_documents_to_chroma.py"
REG_SCRIPT = PROJECT_ROOT / "scripts" / "load_to_chroma" / "load_intent_registry_to_chroma.py"


def test_load_documents_script_help():
    """Documents script should expose help and exit 0."""
    result = subprocess.run(
        [sys.executable, str(DOC_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "pdf" in result.stdout.lower()
    assert "--doc-root" in result.stdout


def test_load_registry_script_help():
    """Intent registry script should expose help and exit 0."""
    result = subprocess.run(
        [sys.executable, str(REG_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "intent registry" in result.stdout.lower()
    assert "--csv-path" in result.stdout
