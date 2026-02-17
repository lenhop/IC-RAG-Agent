from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Load local PDF
import os

# ===================== Step 1: Load local PDF (Amazon business docs) =====================
def load_amazon_pdf(pdf_path):
    """
    Load PDF document saved from Amazon Seller Central.
    :param pdf_path: Local PDF file path
    :return: List of LangChain Document objects (page content + metadata)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Load PDF (split by page)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Returns List[Document], one Document per PDF page
    
    # Merge all pages (optional: chunk by page or merge then chunk)
    combined_text = "\n".join([doc.page_content for doc in documents])
    # Wrap as single Document (for unified chunking)
    from langchain_core.documents import Document
    combined_doc = Document(
        page_content=combined_text,
        metadata={"source": pdf_path, "total_pages": len(documents)}
    )
    return [combined_doc]

# ===================== Step 2: Init LangChain recursive splitter (for Amazon docs) =====================
def init_amazon_text_splitter():
    """
    Initialize recursive splitter for Amazon business docs (English, tables, long paragraphs).
    """
    # Separator priority: paragraph -> line -> space -> sentence (for table text)
    separators = [
        "\n\n",  # First: blank lines (paragraph boundary)
        "\n",    # Second: newlines (line boundary, e.g. table rows)
        " ",     # Third: spaces (word boundary, fallback)
        ".",     # Sentence boundary (English)
        ",",     # Comma boundary (fallback)
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,          # Max chars per chunk (for 16GB CPU, avoid long context)
        chunk_overlap=50,        # Overlap between chunks (preserve table/paragraph context)
        separators=separators,   # Custom separator priority
        length_function=len,     # Length function (char count; can use token count)
        is_separator_regex=False # Disable regex separators (faster)
    )
    return text_splitter

# ===================== Step 3: Run recursive splitting =====================
if __name__ == "__main__":
    # 1. Configure path (replace with your Amazon PDF path)
    AMAZON_PDF_PATH = "amazon_fba_fee.pdf"  # PDF saved from Seller Central
    
    # 2. Load PDF document
    docs = load_amazon_pdf(AMAZON_PDF_PATH)
    print(f"Loaded PDF, total text length: {len(docs[0].page_content)} chars")
    
    # 3. Initialize recursive splitter
    text_splitter = init_amazon_text_splitter()
    
    # 4. Run splitting (core: LangChain recursive split)
    split_docs = text_splitter.split_documents(docs)
    
    # 5. Print split results
    print(f"\n=== Amazon doc recursive split result ({len(split_docs)} chunks) ===")
    for i, doc in enumerate(split_docs, 1):
        print(f"\nChunk {i} (length: {len(doc.page_content)} chars)")
        print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        