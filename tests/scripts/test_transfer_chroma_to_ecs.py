"""
Tests for scripts/transfer_chroma_to_ecs.py.

CLI tests only; no real local/remote ChromaDB required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "transfer_chroma_to_ecs.py"


def test_transfer_script_help():
    """Script should run and show help."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "Transfer ChromaDB" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--all" in result.stdout
    assert "documents" in result.stdout
    assert "fqa" in result.stdout
    assert "keywords" in result.stdout


def test_transfer_script_requires_collections_or_all():
    """Script should error when no collections specified."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode != 0
    err = result.stderr + result.stdout
    assert (
        "Specify collections or --all" in err
        or "required" in err.lower()
        or "invalid choice" in err.lower()
    )


def test_transfer_script_dry_run_exits_zero():
    """Dry-run on non-existent collection should exit 0 (skip) or report."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "fqa", "keywords", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    # Dry-run does not connect to remote; exits 0
    assert result.returncode == 0
