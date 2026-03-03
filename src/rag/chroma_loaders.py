"""
Chroma loaders - Layer 2 toolkit for ingesting data into Chroma.

Provides: bootstrap_project, resolve_path, load_faq_questions, load_csv_column_from_file,
load_csv_column_to_chroma, load_documents_to_chroma. Uses ai-toolkit and src/rag components.
"""

import csv as csv_module
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

# Project root for resolving relative paths (src/rag/chroma_loaders.py -> parents[2])
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_project() -> Path:
    """Set up project environment for load scripts.

    - Disable Chroma telemetry (ANONYMIZED_TELEMETRY=FALSE)
    - Add PROJECT_ROOT to sys.path
    - Locate and add ai-toolkit to sys.path
    - Load .env from project root

    Returns:
        PROJECT_ROOT Path.
    """
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
    import sys

    project_root = _PROJECT_ROOT
    sys.path.insert(0, str(project_root))
    for _path in (
        project_root.parent / "ai-toolkit",
        project_root / "src" / "ai-toolkit",
        project_root / "libs" / "ai-toolkit",
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
    """Resolve path from env or default; if relative, join with project_root."""
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def load_faq_questions(
    project_root: Path | None = None,
    csv_path: str | None = None,
) -> List[str]:
    """Load FAQ questions from CSV at RAG_FAQ_CSV or csv_path.

    Reads the "question" column only. Returns empty list if path not set,
    file missing, or parse error. Handles multi-line CSV cells (quoted).

    Args:
        project_root: Optional project root for resolving relative paths.
        csv_path: Optional explicit CSV path; overrides RAG_FAQ_CSV when set.

    Returns:
        List of question strings (stripped, non-empty).
    """
    root = project_root or _PROJECT_ROOT
    path_val = csv_path or os.getenv("RAG_FAQ_CSV", "data/intent_classification/fqa/amazon_fqa.csv")
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
            reader = csv_module.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                if q:
                    questions.append(q)
    except (csv_module.Error, OSError):
        pass
    return questions


def load_csv_column_from_file(csv_path: Path, column: str) -> List[str]:
    """Load non-empty values from a CSV column.

    Args:
        csv_path: Path to CSV file.
        column: Column name to extract.

    Returns:
        List of stripped non-empty strings.
    """
    items: List[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            val = (row.get(column) or "").strip()
            if val:
                items.append(val)
    return items


def load_csv_column_to_chroma(
    csv_path: Path,
    column: str,
    collection_name: str,
    chroma_path: Path,
    *,
    project_root: Path,
    embed_model: str = "minilm",
    batch_size: int = 16,
    reset_db: bool = False,
    source_metadata: Optional[str] = None,
    items_loader: Optional[Callable[[Path, str], List[str]]] = None,
) -> int:
    """Load a CSV column into Chroma with embeddings.

    Args:
        csv_path: Path to CSV file.
        column: Column name to load.
        collection_name: Chroma collection name.
        chroma_path: Chroma persist directory.
        project_root: Project root for model paths.
        embed_model: minilm, ollama, or qwen3.
        batch_size: Embedding batch size.
        reset_db: Remove Chroma DB before run.
        source_metadata: Optional source string for metadata (default: csv_path.name).
        items_loader: Optional callable(csv_path, column) -> list[str]; overrides default loader.

    Returns:
        Number of items stored.
    """
    from .embeddings import create_embeddings

    import chromadb
    from chromadb.config import Settings

    if items_loader:
        items = items_loader(csv_path, column)
    else:
        items = load_csv_column_from_file(csv_path, column)

    if not items:
        return 0

    if reset_db and chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.create_collection(name=collection_name)
    collection = client.get_collection(name=collection_name)

    embeddings = create_embeddings(
        model_type=embed_model,
        project_root=project_root,
    )

    source = source_metadata or csv_path.name
    stored = 0
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        batch = items[start:end]
        ids = [f"{collection_name}_{start + i}" for i in range(len(batch))]
        vectors = embeddings.embed_documents(batch)
        metadatas = [{"source": source} for _ in batch]
        collection.add(
            documents=batch,
            metadatas=metadatas,
            ids=ids,
            embeddings=vectors,
        )
        stored += len(batch)

    return stored


def load_documents_to_chroma(
    doc_root: str | Path,
    chroma_path: str | Path,
    collection_name: str = "documents",
    *,
    project_root: Path,
    extensions: Optional[List[str]] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    min_chunk_length: int = 20,
    embed_model: str = "minilm",
    batch_size: int = 16,
    reset_db: bool = False,
    limit_files: Optional[int] = None,
) -> int:
    """Load documents (PDF, TXT, MD, CSV, JSON) into Chroma via RAG ingest pipeline.

    Args:
        doc_root: Root directory to search.
        chroma_path: Chroma persist path.
        collection_name: Collection name.
        project_root: Project root for model paths.
        extensions: File extensions (default: .pdf, .txt, .md, .csv, .json).
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap.
        min_chunk_length: Minimum chunk length to keep.
        embed_model: minilm, ollama, or qwen3.
        batch_size: Embedding batch size.
        reset_db: Remove Chroma DB before run.
        limit_files: Limit number of files (for testing).

    Returns:
        Number of chunks stored.
    """
    from . import rag_ingest_pipeline

    exts = extensions or [".pdf", ".txt", ".md", ".csv", ".json"]
    exts = [e if e.startswith(".") else f".{e}" for e in exts]

    return rag_ingest_pipeline(
        root=doc_root,
        chroma_path=chroma_path,
        collection_name=collection_name,
        search_kwargs={"extensions": exts},
        split_kwargs={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        clean_kwargs={"min_length": min_chunk_length},
        embed_kwargs={"model_type": embed_model},
        embed_batch_size=batch_size,
        reset_db=reset_db,
        limit_files=limit_files,
        project_root=project_root,
    )
