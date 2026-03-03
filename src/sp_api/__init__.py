"""SP-API Seller Operations module."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
_ai_toolkit_path = _project_root / "libs" / "ai-toolkit"
if _ai_toolkit_path.exists() and str(_ai_toolkit_path) not in sys.path:
    sys.path.insert(0, str(_ai_toolkit_path))

from .sp_api_client import SPAPIClient, SPAPICredentials

__all__ = ["SPAPIClient", "SPAPICredentials"]
