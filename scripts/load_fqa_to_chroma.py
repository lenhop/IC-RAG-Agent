#!/usr/bin/env python3
"""
Load FAQ questions from amazon_fqa.csv into Chroma for intent classification.

Reads the question column only (first column, header skipped). One question = one chunk.
Uses same embedding model as RAG pipeline (minilm/ollama/qwen3) for consistency.

Flow:
  1. Load questions from CSV (load_faq_questions)
  2. Embed with local model
  3. Store in Chroma collection fqa_question
"""

# [ANNOTATION] Disable Chroma telemetry before any chromadb import.
# Prevents "Failed to send telemetry event" errors (PostHog API compatibility issue).
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Path setup: project root and ai-toolkit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT.parent / "ai-toolkit",
    PROJECT_ROOT / "src" / "ai-toolkit",
    PROJECT_ROOT / "libs" / "ai-toolkit",
):
    if _path.exists():
        sys.path.insert(0, str(_path))
        break

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _resolve_path(env_key: str, default: str) -> str:
    """Resolve path from env; if relative, join with PROJECT_ROOT."""
    val = os.getenv(env_key, default)
    p = Path(val)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


DEFAULT_CSV_PATH = str(PROJECT_ROOT / "data" / "intent_classification" / "fqa" / "amazon_fqa.csv")
CHROMA_PATH = _resolve_path(
    "CHROMA_FQA_PATH", str(PROJECT_ROOT / "data" / "chroma_db" / "fqa_question")
)
COLLECTION_NAME = "fqa_question"
EMBED_BATCH_SIZE = int(os.getenv("DOC_LOAD_EMBED_BATCH_SIZE", "16"))


def main() -> int:
    """Load FAQ questions to Chroma."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load FAQ questions from amazon_fqa.csv into Chroma"
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to FAQ CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_PATH,
        help="Chroma persist path",
    )
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
        help="Embedding model: minilm, ollama, or qwen3",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Remove Chroma DB before run",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # Load questions (one question = one chunk, skip header)
    import csv as csv_module
    questions: list[str] = []
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                if q:
                    questions.append(q)
    except (csv_module.Error, OSError) as e:
        print(f"[ERROR] Failed to read CSV: {e}", file=sys.stderr)
        return 1
    if not questions:
        print("[WARN] No questions found in CSV")
        return 1

    print(f"[1/3] Loaded {len(questions)} questions from {csv_path}")

    # Initialize Chroma and embeddings
    chroma_path = Path(args.chroma_path)
    if args.reset_db and chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.parent.mkdir(parents=True, exist_ok=True)

    import chromadb
    from chromadb.config import Settings
    from src.rag.embeddings import create_embeddings

    # [ANNOTATION] Disable Chroma telemetry to avoid "Failed to send telemetry event" errors.
    # Alternative: set env ANONYMIZED_TELEMETRY=FALSE before running.
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(name=COLLECTION_NAME)
    collection = client.get_collection(name=COLLECTION_NAME)

    embeddings = create_embeddings(
        model_type=args.embed_model,
        project_root=PROJECT_ROOT,
    )
    print(f"[2/3] Embedding model ready ({args.embed_model})")

    # Embed and store in batches
    stored = 0
    start_time = datetime.now()
    for start in range(0, len(questions), args.batch_size):
        end = min(start + args.batch_size, len(questions))
        batch = questions[start:end]
        ids = [f"{COLLECTION_NAME}_{start + i}" for i in range(len(batch))]
        vectors = embeddings.embed_documents(batch)
        metadatas = [{"source": csv_path.name} for _ in batch]
        collection.add(
            documents=batch,
            metadatas=metadatas,
            ids=ids,
            embeddings=vectors,
        )
        stored += len(batch)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[3/3] Stored {stored} chunks to {chroma_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
