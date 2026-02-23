"""
Document loading from files for RAG pipeline.

Layer 2: Dispatches by file extension to ai-toolkit loaders.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


def _default_loader_map() -> dict:
    """Build default extension -> loader mapping using ai-toolkit loaders."""
    from ai_toolkit.rag.loaders import (
        load_csv_document,
        load_json_document,
        load_markdown_document,
        load_pdf_document,
        load_txt_document,
    )

    return {
        ".pdf": lambda p: load_pdf_document(str(p)),
        ".json": lambda p: load_json_document(str(p)),
        ".csv": lambda p: load_csv_document(str(p)),
        ".txt": lambda p: load_txt_document(str(p)),
        ".md": lambda p: load_markdown_document(str(p)),
        ".markdown": lambda p: load_markdown_document(str(p)),
    }


def load_documents_from_files(
    file_paths: List[Path],
    *,
    loader_map: Optional[dict] = None,
    add_source_metadata: bool = True,
    skip_unsupported: bool = True,
) -> List[Document]:
    """
    Load documents from files by dispatching on extension.

    Args:
        file_paths: List of file paths.
        loader_map: Extension -> loader callable. None = use default (pdf, json, csv, txt).
        add_source_metadata: Add "source" metadata to each document.
        skip_unsupported: Skip files with unsupported extension instead of raising.

    Returns:
        List of Document objects (merged from all files).
    """
    if loader_map is None:
        loader_map = _default_loader_map()

    normalized_map = {}
    for k, v in loader_map.items():
        ext = k.lower() if k.startswith(".") else f".{k.lower()}"
        normalized_map[ext] = v

    all_docs: List[Document] = []

    for fp in file_paths:
        path = Path(fp)
        ext = path.suffix.lower()

        loader = normalized_map.get(ext)
        if loader is None:
            if skip_unsupported:
                continue
            raise ValueError(f"No loader for extension: {ext} (file: {path})")

        try:
            docs = loader(path)
            if not docs:
                continue
            if add_source_metadata:
                for d in docs:
                    meta = dict(d.metadata)
                    meta["source"] = str(path)
                    all_docs.append(Document(page_content=d.page_content, metadata=meta))
            else:
                all_docs.extend(docs)
        except Exception:
            raise

    return all_docs
