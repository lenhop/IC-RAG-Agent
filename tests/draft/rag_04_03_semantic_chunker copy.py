"""
RAG semantic chunker using LangChain SemanticChunker with local all-MiniLM-L6-v2.

Semantic chunking splits text by meaning (embedding similarity) instead of fixed size.
Flow: sentences -> group by buffer -> embed -> cosine distance -> split at topic breaks.
"""

import os
from pathlib import Path

# [ANNOTATION] Path setup: resolve project root for model path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")

# [ANNOTATION] SemanticChunker params: breakpoint_threshold_type controls how splits are chosen
# - "percentile" (default 95): split where distance > 95th percentile of all distances
# - "standard_deviation": split where distance > mean + N*std
# - "interquartile": uses IQR
# - "gradient": uses gradient of distance array
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 95.0
BUFFER_SIZE = 1  # sentences before/after to combine (1 -> groups of 3 for embedding)
MIN_CHUNK_SIZE = 50  # merge chunks smaller than this (avoids tiny fragments)


def load_embeddings(model_path: str):
    """
    Load LangChain Embeddings for local all-MiniLM-L6-v2.
    [ANNOTATION] SemanticChunker requires Embeddings with embed_documents() for batch embedding.
    HuggingFaceEmbeddings wraps sentence-transformers and satisfies LangChain Embeddings interface.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_semantic_chunker(embeddings, **kwargs):
    """
    Initialize SemanticChunker.
    [ANNOTATION] Key params:
      - embeddings: used to embed sentence groups; cosine distance between neighbors drives splits
      - buffer_size: each "sentence" is actually [prev, curr, next]; larger = more context per embedding
      - min_chunk_size: skip emitting chunks below this char length (merge into next)
    """
    from langchain_experimental.text_splitter import SemanticChunker

    return SemanticChunker(
        embeddings=embeddings,
        buffer_size=kwargs.get("buffer_size", BUFFER_SIZE),
        breakpoint_threshold_type=kwargs.get(
            "breakpoint_threshold_type", BREAKPOINT_THRESHOLD_TYPE
        ),
        breakpoint_threshold_amount=kwargs.get(
            "breakpoint_threshold_amount", BREAKPOINT_THRESHOLD_AMOUNT
        ),
        min_chunk_size=kwargs.get("min_chunk_size", MIN_CHUNK_SIZE),
    )


# ===================== Main: run semantic chunking on sample text =====================
if __name__ == "__main__":
    # [ANNOTATION] Step 1: Load embeddings (same model used for similarity in chunker)
    try:
        embeddings = load_embeddings(LOCAL_MODEL_PATH)
        print(f"[OK] Embeddings loaded: {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"[FAIL] Embeddings load error: {e}")
        exit(1)

    # [ANNOTATION] Step 2: Create SemanticChunker; it will call embeddings.embed_documents()
    # internally for each sentence group to compute cosine distances
    chunker = init_semantic_chunker(embeddings)

    # [ANNOTATION] Step 3: Sample text with multiple topics (fee table vs. calculation rules)
    # Semantic chunker should split at topic boundaries (e.g. table vs. paragraph)
    amazon_text = """
    FBA Non-peak Fulfillment Fees (excluding apparel)
    January 15, 2025 – October 14, 2025 Starting January 15, 2026
    Size tier Shipping weight <$10 $10-$50 >$50 <$10 $10-$50 >$50
    Small standard 2 oz or less $2.29 $3.06 $3.06 $2.43 $3.32 $3.58
    Small standard 2+ to 4 oz $2.38 $3.15 $3.15 $2.49 $3.42 $3.68
    Large standard 4 oz or less $2.91 $3.68 $3.68 $2.91 $3.73 $3.99

    Fee calculations for Large standard, Small Bulky, Large Bulky, and Extra-Large units
    will be based on the greater of unit weight or dimensional weight (details on how to
    calculate can be found on dimensional weight), with the exception of small standard
    and extra-large 150+ lb items, which will use unit weight only. For detailed size tiers,
    go to Product size tiers.
    """

    # [ANNOTATION] Step 4: split_text() returns list of strings; split_documents() for Document objects
    # Use split_text for raw text; use split_documents when feeding into Chroma/RAG pipeline
    chunks = chunker.split_text(amazon_text)

    # [ANNOTATION] Step 5: Print results; semantic chunks may vary in size (meaning-based, not fixed)
    print(f"\n=== Semantic chunking result ({len(chunks)} chunks) ===")
    for i, chunk in enumerate(chunks, 1):
        char_count = len(chunk)
        print(f"\n[Chunk {i}] (chars: {char_count})")
        print(f"Content: {chunk}")
