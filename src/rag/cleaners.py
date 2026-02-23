"""
Document cleaning and preprocessing for RAG pipeline.

Layer 2: Data preprocessing before embedding. DocumentCleaner class
provides normalize_whitespace, remove_control_chars, and batch cleaning.
"""

import re
from typing import List, Optional

from langchain_core.documents import Document


class DocumentCleaner:
    """
    Data preprocessing for documents before embedding.

    Reduces noise from PDF/HTML extraction and improves embedding quality.
    """

    def __init__(
        self,
        min_length: int = 20,
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Args:
            min_length: Minimum content length to keep (filter short noise).
            remove_control_chars: Remove non-printable characters.
            normalize_whitespace: Collapse multiple whitespace to single space.
        """
        self.min_length = min_length
        self._remove_control_chars_flag = remove_control_chars
        self._normalize_whitespace_flag = normalize_whitespace

    def normalize_whitespace(self, text: str) -> str:
        """
        Collapse multiple whitespace (spaces, newlines, tabs) to single space.

        Args:
            text: Raw text.

        Returns:
            Normalized text.
        """
        if not text or not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def remove_control_chars(self, text: str) -> str:
        """
        Keep only printable characters and whitespace.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""
        return "".join(c for c in text if c.isprintable() or c.isspace()).strip()

    def strip_headers_footers(
        self,
        text: str,
        patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Remove common header/footer patterns (regex).

        Args:
            text: Raw text.
            patterns: List of regex patterns to remove. None = use defaults.

        Returns:
            Text with matched patterns removed.
        """
        if not text or not isinstance(text, str):
            return ""
        if patterns is None:
            patterns = [
                r"^\d+\s*$",  # Page numbers alone
                r"^Page \d+ of \d+\s*$",
                r"^-\s*\d+\s*-$",
            ]
        result = text
        for pat in patterns:
            result = re.sub(pat, "", result, flags=re.MULTILINE | re.IGNORECASE)
        return re.sub(r"\n{3,}", "\n\n", result).strip()

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to raw text.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""
        result = text
        if self._normalize_whitespace_flag:
            result = self.normalize_whitespace(result)
        if self._remove_control_chars_flag:
            result = self.remove_control_chars(result)
        return result.strip()

    def clean_document(
        self,
        doc: Document,
        min_length: Optional[int] = None,
    ) -> Optional[Document]:
        """
        Clean a single document. Return None if invalid (e.g. too short).

        Args:
            doc: LangChain Document.
            min_length: Override instance min_length for this call.

        Returns:
            Cleaned Document or None.
        """
        content = self.clean_text(doc.page_content)
        threshold = min_length if min_length is not None else self.min_length
        if len(content) < threshold:
            return None
        return Document(page_content=content, metadata=dict(doc.metadata))

    def clean_documents(
        self,
        documents: List[Document],
        min_length: Optional[int] = None,
    ) -> List[Document]:
        """
        Batch clean documents. Filter by min_length.

        Args:
            documents: List of LangChain Documents.
            min_length: Override instance min_length for this call.

        Returns:
            List of cleaned Documents.
        """
        result: List[Document] = []
        for doc in documents:
            cleaned = self.clean_document(doc, min_length=min_length)
            if cleaned is not None:
                result.append(cleaned)
        return result
