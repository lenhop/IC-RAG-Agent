"""
File search and filtering for RAG pipeline.

Layer 2: Filesystem operations - search and filter files by extension,
size, create/modify time, directory, etc.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional


def search_files(
    root: str | Path,
    *,
    extensions: Optional[List[str]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    modified_after: Optional[datetime] = None,
    modified_before: Optional[datetime] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    recursive: bool = True,
    limit: Optional[int] = None,
) -> List[Path]:
    """
    Search and filter files by criteria.

    Args:
        root: Root directory to search.
        extensions: File extensions to include (e.g. [".pdf", ".txt"]). None = all.
        min_size: Minimum file size in bytes.
        max_size: Maximum file size in bytes.
        created_after: Include only files created after this datetime.
        created_before: Include only files created before this datetime.
        modified_after: Include only files modified after this datetime.
        modified_before: Include only files modified before this datetime.
        include_dirs: Include only files under these subdir names (relative to root).
        exclude_dirs: Exclude files under these dir names (e.g. ["__pycache__", ".git"]).
        recursive: Search recursively.
        limit: Maximum number of files to return.

    Returns:
        Sorted list of file paths.
    """
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    excluded = set(exclude_dirs or [])
    exclude_dirs_lower = {d.lower() for d in excluded}

    def _collect() -> List[Path]:
        paths: List[Path] = []
        if recursive:
            for p in root_path.rglob("*"):
                if not p.is_file():
                    continue
                paths.append(p)
        else:
            for p in root_path.iterdir():
                if not p.is_file():
                    continue
                paths.append(p)
        return paths

    candidates = _collect()

    def _matches(p: Path) -> bool:
        # Extension filter
        if extensions:
            ext = p.suffix.lower()
            if ext not in {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}:
                return False

        # Exclude dirs: check if any parent dir name in path is in exclude list
        try:
            rel = p.relative_to(root_path)
            for part in rel.parts[:-1]:
                if part.lower() in exclude_dirs_lower:
                    return False
        except ValueError:
            pass

        # Include dirs: if specified, file must be under one of them
        if include_dirs:
            rel = p.relative_to(root_path)
            if not any(rel.parts[0] == d for d in include_dirs):
                return False

        if min_size is not None or max_size is not None:
            try:
                size = p.stat().st_size
                if min_size is not None and size < min_size:
                    return False
                if max_size is not None and size > max_size:
                    return False
            except OSError:
                return False

        if created_after is not None or created_before is not None:
            try:
                ctime = datetime.fromtimestamp(p.stat().st_ctime)
                if created_after is not None and ctime < created_after:
                    return False
                if created_before is not None and ctime > created_before:
                    return False
            except OSError:
                return False

        if modified_after is not None or modified_before is not None:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                if modified_after is not None and mtime < modified_after:
                    return False
                if modified_before is not None and mtime > modified_before:
                    return False
            except OSError:
                return False

        return True

    filtered = [p for p in candidates if _matches(p)]
    filtered.sort(key=lambda x: str(x))

    if limit is not None:
        filtered = filtered[:limit]

    return filtered
