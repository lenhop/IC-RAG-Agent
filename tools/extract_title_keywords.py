#!/usr/bin/env python3
"""
Extract phrases from document titles for RAG intent classification.

Scans documents under a given path, extracts titles (PDF metadata, filename,
or first line), and outputs a phrase list (JSON) for use in answer mode
identification (ANSWER_MODEL_IDENTITY_NEW.md). Phrases (multi-word expressions)
are more effective than single keywords for identifying user intention.

Usage:
    python tools/extract_title_keywords.py \\
      --doc-root /path/to/documents \\
      --output data/intent/phrases_from_titles.json \\
      --method simple \\
      --top-n 50 \\
      --prefer-filename   # use filename for PDFs when metadata is generic

Methods:
    simple  - N-gram extraction (2-4 words), frequency count (no extra deps)
    tfidf   - TF-IDF with ngram_range (requires scikit-learn)
    keybert - Semantic phrase extraction (requires: pip install keybert)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Path setup: project root and src for search_files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

# Minimal English stopwords for simple method (no NLTK dependency)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "else", "when", "where", "how",
    "your", "my", "our", "their", "this", "that", "these", "those",
}


def get_document_title(file_path: Path, prefer_filename: bool = False) -> str:
    """
    Extract title from a document file.

    PDF: metadata.title or filename stem (use prefer_filename to skip metadata).
    TXT/MD: first line (max 200 chars) or filename stem.
    JSON/CSV: filename stem.

    Args:
        file_path: Path to the document file.
        prefer_filename: If True, use filename stem for PDFs (useful when
            metadata is generic, e.g. all "Amazon").

    Returns:
        Extracted title string.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        if prefer_filename:
            return path.stem
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            meta = reader.metadata
            title = getattr(meta, "title", None) if meta else None
            if title and isinstance(title, str) and title.strip():
                return title.strip()
        except Exception:
            pass
        return path.stem

    if ext in (".txt", ".md", ".markdown"):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
            if first_line and first_line.strip():
                return first_line.strip()[:200]
        except Exception:
            pass
        return path.stem

    if ext in (".json", ".csv"):
        return path.stem

    return path.stem


def collect_titles(
    documents_path: Path,
    extensions: Optional[List[str]] = None,
    limit: Optional[int] = None,
    prefer_filename: bool = False,
) -> List[Tuple[Path, str]]:
    """
    Search documents and collect (path, title) for each file.

    Args:
        documents_path: Root directory to search.
        extensions: File extensions to include (e.g. [".pdf", ".txt"]).
        limit: Max number of files to process.
        prefer_filename: Use filename stem for PDFs instead of metadata.

    Returns:
        List of (path, title) tuples.
    """
    from src.rag.file_search import search_files

    exts = extensions or [".pdf", ".txt", ".md", ".json", ".csv"]
    exts = [e if e.startswith(".") else f".{e}" for e in exts]

    file_paths = search_files(documents_path, extensions=exts, limit=limit)
    results: List[Tuple[Path, str]] = []

    for fp in file_paths:
        title = get_document_title(fp, prefer_filename=prefer_filename)
        results.append((fp, title))

    return results


def _tokenize_title(title: str, stopwords: Set[str]) -> List[str]:
    """
    Tokenize title into words (supports English and CJK).

    Args:
        title: Raw title string.
        stopwords: Set of stopwords to filter.

    Returns:
        List of valid tokens (lowercased, non-stopword, length >= 2).
    """
    # Match: English words (2+ chars), numbers, or CJK characters
    tokens = re.findall(r"[a-zA-Z0-9]{2,}|[\u4e00-\u9fff]+", title)
    return [t.lower() if t.isascii() else t for t in tokens if t.lower() not in stopwords]


def extract_phrases_simple(
    titles: List[str],
    top_n: int = 50,
    stopwords: Optional[Set[str]] = None,
    ngram_range: Tuple[int, int] = (2, 4),
) -> List[str]:
    """
    Extract phrases by n-gram extraction and frequency count.

    Forms 2-to-4 word n-grams from tokenized titles, returns top-N by frequency.
    Phrases are more effective than single keywords for user intention identification.

    Args:
        titles: List of title strings.
        top_n: Number of phrases to return.
        stopwords: Set of stopwords to filter. Default: built-in English set.
        ngram_range: (min_n, max_n) for phrase length in words.

    Returns:
        List of phrase strings, sorted by frequency (desc).
    """
    stop = stopwords or DEFAULT_STOPWORDS
    counter: dict[str, int] = {}
    min_n, max_n = ngram_range

    for title in titles:
        if not title:
            continue
        tokens = _tokenize_title(title, stop)
        if len(tokens) < min_n:
            continue
        # Extract n-grams for each length in range
        for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                counter[phrase] = counter.get(phrase, 0) + 1

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [p for p, _ in sorted_items[:top_n]]


def extract_phrases_tfidf(
    titles: List[str],
    top_n: int = 50,
    ngram_range: Tuple[int, int] = (2, 4),
) -> List[str]:
    """
    Extract phrases using TF-IDF with n-gram range across titles.

    Each title is treated as a document. Uses ngram_range=(2,4) to extract
    multi-word phrases. Requires scikit-learn.

    Args:
        titles: List of title strings.
        top_n: Number of phrases to return.
        ngram_range: (min_n, max_n) for phrase length in words.

    Returns:
        List of phrase strings.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        raise ImportError(
            "TF-IDF method requires scikit-learn. Install: pip install scikit-learn"
        ) from e

    if not titles:
        return []

    vectorizer = TfidfVectorizer(
        max_features=top_n * 3,
        stop_words="english",
        token_pattern=r"[a-zA-Z0-9]{2,}",
        ngram_range=ngram_range,
    )
    try:
        matrix = vectorizer.fit_transform(titles)
    except ValueError:
        return []

    # Sum TF-IDF scores per phrase across all documents
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    sorted_idx = scores.argsort()[::-1]

    return [terms[i] for i in sorted_idx[:top_n] if scores[i] > 0]


def extract_phrases_keybert(
    titles: List[str],
    top_n: int = 50,
    ngram_range: Tuple[int, int] = (2, 4),
) -> List[str]:
    """
    Extract phrases using KeyBERT semantic extraction.

    KeyBERT extracts keyphrases (multi-word) with semantic relevance.
    Uses keyphrase_ngram_range=(2,4) for phrases only (no single words).
    Requires: pip install keybert.

    Args:
        titles: List of title strings.
        top_n: Number of phrases to return.
        ngram_range: (min_n, max_n) for phrase length in words.

    Returns:
        List of phrase strings.

    Raises:
        ImportError: If keybert is not installed.
    """
    try:
        from keybert import KeyBERT
    except ImportError as e:
        raise ImportError(
            "KeyBERT method requires keybert. Install: pip install keybert"
        ) from e

    if not titles:
        return []

    combined = " ".join(t for t in titles if t)
    if not combined.strip():
        return []

    kw_model = KeyBERT()
    # Extract 2-4 word phrases only (no single keywords)
    min_n, max_n = ngram_range
    results = kw_model.extract_keywords(
        combined,
        keyphrase_ngram_range=(min_n, max_n),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )

    return [kw for kw, _ in results]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract phrases from document titles for RAG intent classification"
    )
    parser.add_argument(
        "--doc-root",
        type=Path,
        default=Path("/Users/hzz/KMS/IC-RAG-Agent/data/documents"),
        help="Root directory to search for documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/hzz/KMS/IC-RAG-Agent/data/intent_classification/keywords/phrases_from_titles.csv"),
        help="Output CSV file path (default: data/intent_classification/keywords/phrases_from_titles.csv)",
    )
    parser.add_argument(
        "--method",
        choices=("simple", "tfidf", "keybert"),
        default="simple",
        help="Phrase extraction method (default: simple)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of phrases to extract (default: 50)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf", ".txt", ".md"],
        help="File extensions to include",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--with-sources",
        action="store_true",
        help="Include phrase -> titles mapping in output (for debugging)",
    )
    parser.add_argument(
        "--prefer-filename",
        action="store_true",
        help="Use filename stem for PDFs instead of metadata (recommended when metadata is generic, e.g. all 'Amazon')",
    )
    args = parser.parse_args()

    doc_root = args.doc_root
    if not doc_root.exists():
        print(f"[ERROR] Document root does not exist: {doc_root}", file=sys.stderr)
        return 1
    if not doc_root.is_dir():
        print(f"[ERROR] Document root is not a directory: {doc_root}", file=sys.stderr)
        return 1

    # Resolve output path relative to project root if needed
    out_path = args.output
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f"[1/3] Collecting titles from {doc_root}...")
    title_pairs = collect_titles(
        doc_root,
        extensions=args.extensions,
        limit=args.limit,
        prefer_filename=args.prefer_filename,
    )
    titles = [t for _, t in title_pairs]

    if not titles:
        print("[WARN] No documents found. Output will have empty phrases.")
        phrases = []
    else:
        print(f"[2/3] Extracting phrases ({args.method}, top_n={args.top_n})...")
        if args.method == "simple":
            phrases = extract_phrases_simple(titles, top_n=args.top_n)
        elif args.method == "tfidf":
            try:
                phrases = extract_phrases_tfidf(titles, top_n=args.top_n)
            except ImportError as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                return 1
        else:
            try:
                phrases = extract_phrases_keybert(titles, top_n=args.top_n)
            except ImportError as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                return 1

    # Sort phrases alphabetically
    phrases = sorted(phrases)

    # Build CSV output (phrase column only)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phrase"])
        for phrase in phrases:
            writer.writerow([phrase])

    print(f"[3/3] Wrote {len(phrases)} phrases to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
