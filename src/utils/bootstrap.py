"""
Project bootstrap for CLI scripts and offline loaders.

Sets telemetry default, sys.path for project and optional ai-toolkit, loads .env.
Lives under ``src.utils`` so Chroma and other tool layers do not own environment setup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# src/utils/bootstrap.py -> parents[2] == repository root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_project() -> Path:
    """
    Set up project environment for load scripts.

    - Disable Chroma telemetry (ANONYMIZED_TELEMETRY=FALSE)
    - Add PROJECT_ROOT to sys.path
    - Locate and add ai-toolkit to sys.path when present
    - Load .env from project root

    Returns:
        PROJECT_ROOT Path.
    """
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
    project_root = _PROJECT_ROOT
    sys.path.insert(0, str(project_root))
    for _path in (
        project_root.parent / "ai-toolkit",
        project_root / "src" / "ai-toolkit",
        project_root / "libs" / "ai-toolkit",
        project_root / "external" / "ai-toolkit",
    ):
        if _path.exists():
            sys.path.insert(0, str(_path))
            break

    try:
        from dotenv import load_dotenv

        load_dotenv(project_root / ".env")
    except ImportError:
        pass

    return project_root


def resolve_path(env_key: str, default: str, project_root: Path) -> str:
    """
    Resolve path from env or default; if relative, join with project_root.

    Args:
        env_key: Environment variable name.
        default: Default path string (may be relative to project_root).
        project_root: Repository root for relative resolution.

    Returns:
        Absolute resolved path string.
    """
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())
