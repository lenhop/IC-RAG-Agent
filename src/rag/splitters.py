"""
Document splitting for RAG pipeline.

Layer 2: Wraps ai-toolkit splitters with a unified entry point.
Supports English and Chinese-optimized splitting.
"""

from typing import List, Optional

from langchain_core.documents import Document


def split_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    language: str = "en",
    add_start_index: bool = True,
    **kwargs,
) -> List[Document]:
    """
    Unified entry point for document splitting.

    Args:
        documents: List of LangChain Documents.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks.
        language: "en" for English (default separators), "zh" for Chinese.
        add_start_index: Track start index in original document.
        **kwargs: Passed to ai-toolkit splitter.

    Returns:
        List of chunk Documents.
    """
    from ai_toolkit.rag.splitters import split_document_recursive, split_for_chinese

    if language == "zh":
        return split_for_chinese(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return split_document_recursive(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
        **kwargs,
    )
