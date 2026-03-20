"""
Load an arbitrary CSV text column into Chroma with embeddings.

Includes ``load_csv_column_from_file`` for reading a single UTF-8 column.
Embedding storage uses ``src.rag.embeddings`` (factory) inside
``load_csv_column_to_chroma``.
"""

from __future__ import annotations

import csv as csv_module
import shutil
from pathlib import Path
from typing import Callable, List, Optional


def load_csv_column_from_file(csv_path: Path, column: str) -> List[str]:
    """
    Load non-empty string values from a single CSV column.

    Args:
        csv_path: Path to UTF-8 CSV file.
        column: Header name to read.

    Returns:
        List of stripped non-empty cell values.

    Raises:
        OSError: If the file cannot be read.
        csv.Error: If CSV parsing fails.
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
    """
    Load a CSV column into Chroma with embeddings.

    Args:
        csv_path: Path to CSV file.
        column: Column name to load.
        collection_name: Chroma collection name.
        chroma_path: Chroma persist directory.
        project_root: Project root for embedding model paths.
        embed_model: minilm, ollama, or qwen3.
        batch_size: Embedding batch size.
        reset_db: Remove Chroma DB directory before run.
        source_metadata: Optional source string for metadata (default: csv_path.name).
        items_loader: Optional callable(csv_path, column) -> list[str].

    Returns:
        Number of items stored.
    """
    from src.rag.embeddings import create_embeddings

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
