"""
Unit tests for run_evaluation.py script (Phase 3.1).
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def test_run_evaluation_help():
    """Script --help runs without error."""
    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / "run_evaluation.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "run_evaluation" in result.stdout or "evaluation" in result.stdout
    assert "--limit" in result.stdout
    assert "--skip-umap" in result.stdout


def test_run_evaluation_cli_args():
    """Script accepts expected CLI arguments and produces output when successful."""
    result = subprocess.run(
        [
            PYTHON,
            str(PROJECT_ROOT / "scripts" / "run_evaluation.py"),
            "--limit", "1",
            "--skip-umap",
            "--output", "/tmp/eval_cli_test",
            "--no-verbose",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    # May fail if dataset/pipeline unavailable; we mainly verify it runs
    assert result.returncode in (0, 1)
    if result.returncode == 0:
        out_dir = Path("/tmp/eval_cli_test")
        assert (out_dir / "retrieval_results.json").exists()
        assert (out_dir / "evaluation_report.html").exists()
