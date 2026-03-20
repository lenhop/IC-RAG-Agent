"""
RAG LLM-based semantic chunker (Doubao AI case).

Uses a local LLM (Ollama) to split text by semantic units. The LLM decides chunk boundaries
based on rules (table integrity, token limits, complete semantic units). Token counting
uses the embedding model's tokenizer for RAG pipeline consistency.
"""

import os
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# [ANNOTATION] Path setup: resolve project root (src/draft -> IC-RAG-Agent)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# [ANNOTATION] Tokenizer path: use all-MiniLM-L6-v2 to match embedding model (rag_04_02, rag_04_03)
# Token counts align with embedding model limits (max 256 tokens)
TOKENIZER_PATH = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")

# [ANNOTATION] Ollama LLM config: align with rag_03_05, rag_03_04 (qwen3:1.7b)
LOCAL_LLM_MODEL = "qwen3:1.7b"
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0

# [ANNOTATION] Chunk limits: max_tokens should not exceed embedding model max (all-MiniLM-L6-v2 = 256)
MAX_TOKEN_COUNT = 256
MIN_CHUNK_TOKENS = 50


def load_tokenizer(model_path: str):
    """
    Load tokenizer for token counting. Uses SentenceTransformer to match embedding model.
    [ANNOTATION] AutoTokenizer does not accept 'device' param; tokenizers run on CPU.
    """
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = SentenceTransformer(model_name_or_path=model_path, device="cpu")
    return model.tokenizer


def llm_semantic_chunk(text: str, llm, tokenizer):
    """
    Chunk text via LLM based on semantic rules (Doubao AI case).
    [ANNOTATION] LLM outputs chunks separated by ===; we parse and validate token counts.
    """
    prompt = PromptTemplate.from_template("""
You are an expert in semantic chunking for single-topic Amazon Sellercentral PDF documents. Please strictly split the text according to the following rules:
Rule 1: Splitting Principle — Split by "complete semantic units", and each chunk must be an independent semantic sub-unit (such as tables, core calculation rules, or exceptional clauses);
Rule 2: Table Protection — Preserve the integrity of tables; do not split any row of a table;
Rule 3: Token Limit — The number of tokens in each chunk must not exceed {max_tokens} (character count ≈ token count × 4);
Rule 4: Small Chunk Filtering — Do not generate chunks with fewer than {min_tokens} tokens;
Rule 5: Output Format — Only return the chunking results; separate each chunk with === and do not add any extra explanation.

The text to be chunked:
{text}
""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "text": text,
        "max_tokens": MAX_TOKEN_COUNT,
        "min_tokens": MIN_CHUNK_TOKENS,
    })

    # [ANNOTATION] Parse chunks by === separator; filter and split oversized chunks
    chunks = [chunk.strip() for chunk in result.split("===") if chunk.strip()]
    final_chunks = []
    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        if token_count < MIN_CHUNK_TOKENS:
            continue
        if token_count > MAX_TOKEN_COUNT:
            # [ANNOTATION] Fallback: split by paragraph when LLM output exceeds limit
            sub_chunks = chunk.split("\n\n")
            sub_current = ""
            sub_tokens = 0
            for sub in sub_chunks:
                sub_t = len(tokenizer.encode(sub, add_special_tokens=False))
                if sub_tokens + sub_t > MAX_TOKEN_COUNT and sub_current:
                    final_chunks.append(sub_current.strip())
                    sub_current = sub
                    sub_tokens = sub_t
                else:
                    sub_current += "\n\n" + sub if sub_current else sub
                    sub_tokens += sub_t
            if sub_current:
                final_chunks.append(sub_current.strip())
        else:
            final_chunks.append(chunk)
    return final_chunks


# ===================== Main =====================
if __name__ == "__main__":
    # [ANNOTATION] Step 1: Load tokenizer (matches embedding model for RAG consistency)
    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        print(f"[OK] Tokenizer loaded: {TOKENIZER_PATH}")
    except Exception as e:
        print(f"[FAIL] Tokenizer load error: {e}")
        exit(1)

    # [ANNOTATION] Step 2: Init Ollama LLM; use langchain_ollama for project consistency
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=LOCAL_LLM_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_ctx=OLLAMA_NUM_CTX,
        )
        print(f"[OK] LLM initialized: {LOCAL_LLM_MODEL}")
    except ImportError:
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(
            model=LOCAL_LLM_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_ctx=OLLAMA_NUM_CTX,
        )
        print(f"[OK] LLM initialized (langchain_community): {LOCAL_LLM_MODEL}")

    # [ANNOTATION] Step 3: Sample Amazon FBA text (tables + paragraphs)
    amazon_text = """
    FBA Non-peak Fulfillment Fees (excluding apparel)
    January 15, 2025 – October 14, 2025 Starting January 15, 2026

    Size tier Shipping weight <$10 $10-$50 >$50 <$10 $10-$50 >$50
    Small standard 2 oz or less $2.29 $3.06 $3.06 $2.43 $3.32 $3.58

    Fee calculations for Large standard units will be based on the greater of unit weight or dimensional weight.
    """

    # [ANNOTATION] Step 4: Chunk and output
    chunks = llm_semantic_chunk(amazon_text, llm, tokenizer)
    print(f"\n=== LLM semantic chunking result ({len(chunks)} chunks) ===")
    for i, chunk in enumerate(chunks, 1):
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        print(f"\n[Chunk {i}] (tokens: {token_count})")
        print(f"Content: {chunk}")
