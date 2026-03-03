#!/usr/bin/env python3
"""
Load data into Chroma - single entry point (Layer 3 application).

Subcommands:
  documents  - PDF, TXT, MD, CSV, JSON -> Chroma (RAG documents)
  fqa        - FAQ questions from amazon_fqa.csv -> Chroma (intent classification)
  keywords   - Phrases from phrases_from_titles.csv -> Chroma (intent classification)
  csv        - Generic CSV column -> Chroma (--csv-path, --column, --collection)

Uses src/rag (Layer 2) toolkits: chroma_loaders, load_utils.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root for src.rag import
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from src.rag.chroma_loaders import bootstrap_project, resolve_path

PROJECT_ROOT = bootstrap_project()

# Preset configs for fqa and keywords - paths from env vars (see .env.example)
PRESETS = {
    "fqa": {
        "csv_env": "RAG_FAQ_CSV",
        "csv_default": "data/intent_classification/fqa/amazon_fqa.csv",
        "column": "question",
        "collection": "fqa_question",
        "chroma_env": "CHROMA_FQA_PATH",
        "chroma_default": "data/chroma_db/fqa_question",
        "use_faq_loader": True,
    },
    "keywords": {
        "csv_env": "RAG_TITLE_PHRASES_CSV",
        "csv_default": "data/intent_classification/keywords/phrases_from_titles.csv",
        "column": "phrase",
        "collection": "keyword",
        "chroma_env": "CHROMA_KEYWORD_PATH",
        "chroma_default": "data/chroma_db/keyword",
        "use_faq_loader": False,
    },
}

EMBED_BATCH_SIZE = int(os.getenv("DOC_LOAD_EMBED_BATCH_SIZE", "16"))
CHROMA_PERSIST_PATH = resolve_path(
    "CHROMA_DOCUMENTS_PATH",
    str(PROJECT_ROOT / "data" / "chroma_db" / "documents"),
    PROJECT_ROOT,
)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
DOC_ROOT = resolve_path("DOC_LOAD_ROOT", str(PROJECT_ROOT / "data" / "documents"), PROJECT_ROOT)
DEFAULT_EXTENSIONS = [
    e.strip() for e in os.getenv("DOC_LOAD_EXTENSIONS", ".pdf,.txt,.md,.csv,.json").split(",")
]
CHUNK_SIZE = int(os.getenv("DOC_LOAD_CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("DOC_LOAD_CHUNK_OVERLAP", "100"))
MIN_CHUNK_LENGTH = int(os.getenv("DOC_LOAD_MIN_CHUNK_LENGTH", "20"))


def _cmd_documents(args) -> int:
    """Run documents subcommand."""
    from src.rag.chroma_loaders import load_documents_to_chroma

    exts = [e if e.startswith(".") else f".{e}" for e in args.extensions]
    print(f"[OK] Doc root: {args.doc_root}")
    print(f"[OK] Extensions: {exts}")
    print(f"[OK] Embed model: {args.embed_model}")

    start_time = datetime.now()
    stored = load_documents_to_chroma(
        doc_root=args.doc_root,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        project_root=PROJECT_ROOT,
        extensions=exts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_chunk_length,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        reset_db=args.reset_db,
        limit_files=args.limit,
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[OK] Time taken: {elapsed:.1f}s")
    if stored == 0:
        print("[WARN] No chunks stored")
        return 1
    print(f"[OK] Stored {stored} chunks")
    return 0


def _cmd_csv(args, preset: str | None) -> int:
    """Run fqa, keywords, or generic csv subcommand."""
    from src.rag.chroma_loaders import load_csv_column_to_chroma
    from src.rag.chroma_loaders import load_faq_questions

    if preset:
        cfg = PRESETS[preset]
        csv_path_str = args.csv_path or resolve_path(
            cfg["csv_env"], cfg["csv_default"], PROJECT_ROOT
        )
        column = cfg["column"]
        collection_name = cfg["collection"]
        chroma_path_str = args.chroma_path or resolve_path(
            cfg["chroma_env"], cfg["chroma_default"], PROJECT_ROOT
        )
        use_faq_loader = cfg["use_faq_loader"]
    else:
        if not args.csv_path or not args.column or not args.collection:
            return 1  # Caller should have validated
        csv_path_str = args.csv_path
        column = args.column
        collection_name = args.collection
        chroma_path_str = args.chroma_path or str(
            PROJECT_ROOT / "data" / "chroma_db" / collection_name
        )
        use_faq_loader = False

    csv_path = Path(csv_path_str)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    def _items_loader(p: Path, col: str):
        if use_faq_loader and col == "question":
            return load_faq_questions(project_root=PROJECT_ROOT, csv_path=str(p))
        from src.rag.chroma_loaders import load_csv_column_from_file
        return load_csv_column_from_file(p, col)

    items = _items_loader(csv_path, column)
    if not items:
        print("[WARN] No items found in CSV")
        return 1

    print(f"[1/3] Loaded {len(items)} items from {csv_path}")

    chroma_path = Path(chroma_path_str)
    start_time = datetime.now()
    stored = load_csv_column_to_chroma(
        csv_path=csv_path,
        column=column,
        collection_name=collection_name,
        chroma_path=chroma_path,
        project_root=PROJECT_ROOT,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        reset_db=args.reset_db,
        items_loader=_items_loader,
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[3/3] Stored {stored} chunks to {chroma_path} ({elapsed:.1f}s)")
    return 0 if stored > 0 else 1


def _add_common_args(parser):
    """Add common embedding/reset args to a subparser."""
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-model",
        choices=("minilm", "ollama", "qwen3"),
        default="minilm",
        help="Embedding model",
    )
    parser.add_argument("--reset-db", action="store_true", help="Remove Chroma DB before run")


def main() -> int:
    """Parse subcommand and dispatch."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load data into Chroma (documents, fqa, keywords, csv)"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # documents
    p_doc = subparsers.add_parser("documents", help="Load PDF, TXT, MD, CSV, JSON into Chroma")
    p_doc.add_argument("--doc-root", default=DOC_ROOT, help="Root directory to search")
    p_doc.add_argument("--chroma-path", default=CHROMA_PERSIST_PATH, help="Chroma persist path")
    p_doc.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")
    p_doc.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS, help="File extensions")
    p_doc.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size")
    p_doc.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap")
    p_doc.add_argument("--min-chunk-length", type=int, default=MIN_CHUNK_LENGTH, help="Min chunk length")
    p_doc.add_argument("--limit", type=int, default=None, metavar="N", help="Limit files (testing)")
    _add_common_args(p_doc)
    p_doc.set_defaults(handler=lambda a: _cmd_documents(a))

    # fqa
    p_fqa = subparsers.add_parser("fqa", help="Load FAQ questions into Chroma")
    p_fqa.add_argument("--csv-path", help="Override default CSV path")
    p_fqa.add_argument("--chroma-path", help="Override Chroma path")
    _add_common_args(p_fqa)
    p_fqa.set_defaults(handler=lambda a: _cmd_csv(a, "fqa"))

    # keywords
    p_kw = subparsers.add_parser("keywords", help="Load keywords/phrases into Chroma")
    p_kw.add_argument("--csv-path", help="Override default CSV path")
    p_kw.add_argument("--chroma-path", help="Override Chroma path")
    _add_common_args(p_kw)
    p_kw.set_defaults(handler=lambda a: _cmd_csv(a, "keywords"))

    # csv (generic)
    p_csv = subparsers.add_parser("csv", help="Load generic CSV column into Chroma")
    p_csv.add_argument("--csv-path", required=True, help="CSV file path")
    p_csv.add_argument("--column", required=True, help="Column name")
    p_csv.add_argument("--collection", required=True, help="Collection name")
    p_csv.add_argument("--chroma-path", help="Chroma path (default: data/chroma_db/<collection>)")
    _add_common_args(p_csv)
    p_csv.set_defaults(handler=lambda a: _cmd_csv(a, None))

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
