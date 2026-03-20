"""
RAG text splitter using SentenceTransformersTokenTextSplitter with local all-MiniLM-L6-v2.
Splits text by token count using the model's tokenizer (token-based, not character-based).
"""

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
import os

# ===================== Core Configuration =====================
# Local all-MiniLM-L6-v2 model path (replace with your actual absolute path)
LOCAL_MODEL_PATH = "/Users/hzz/KMS/IC-RAG-Agent/models/all-MiniLM-L6-v2"
# Chunk parameters (all-MiniLM-L6-v2 max seq length 256, use smaller for overlap)
CHUNK_SIZE = 256  # tokens per chunk
CHUNK_OVERLAP = 50  # token overlap between chunks

# ===================== Step 1: Load local all-MiniLM-L6-v2 model =====================
def load_local_embedding_model(model_path):
    """Load local SentenceTransformer model (all-MiniLM-L6-v2)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    model = SentenceTransformer(
        model_name_or_path=model_path,
        device="cpu"  # Use "cuda" for GPU, "mps" for Mac M-series
    )
    return model


# ===================== Step 2: Initialize SentenceTransformersTokenTextSplitter =====================
def init_text_splitter(model_path, tokens_per_chunk, chunk_overlap):
    """
    Initialize SentenceTransformersTokenTextSplitter for local all-MiniLM-L6-v2.
    Note: Token-based splitter does not support separators (unlike character-based splitters).
    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap,
        model_name=model_path
    )
    return text_splitter


# ===================== Step 3: Run chunking (Amazon FBA sample) =====================
if __name__ == "__main__":
    # 1. Load local model (verify availability)
    try:
        embedding_model = load_local_embedding_model(LOCAL_MODEL_PATH)
        print(f"[OK] Local model loaded: {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"[FAIL] Model load error: {e}")
        exit(1)

    # 2. Initialize splitter
    text_splitter = init_text_splitter(
        model_path=LOCAL_MODEL_PATH,
        tokens_per_chunk=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # 3. Sample Amazon FBA text (tables + paragraphs)
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

    # 4. Split text
    split_chunks = text_splitter.split_text(amazon_text)

    # 5. Print results (with token counts)
    print(f"\n=== all-MiniLM-L6-v2 chunking result ({len(split_chunks)} chunks) ===")
    for i, chunk in enumerate(split_chunks, 1):
        token_count = len(embedding_model.tokenizer.encode(chunk, add_special_tokens=False))
        print(f"\n[Chunk {i}] (tokens: {token_count})")
        print(f"Content: {chunk}")
