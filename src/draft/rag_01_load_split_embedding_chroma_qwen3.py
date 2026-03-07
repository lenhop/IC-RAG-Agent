

import sys 
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
import chromadb 

load_dotenv()

__file__ = '/Users/hzz/KMS/IC-RAG-Agent/src/draft/load_split_pdf.py'
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'external/ai-toolkit/'))


def _patch_torch_autocast_for_older_pytorch() -> None:
    """
    On PyTorch < 2.3, torch.is_autocast_enabled() takes no args; transformers
    4.57+ calls torch.is_autocast_enabled(device_type). Patch to accept one
    arg and ignore it, while preserving behavior on newer versions.
    """

    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return

    try:
        # Newer PyTorch supports the device_type argument.
        torch.is_autocast_enabled("cpu")
    except TypeError:
        # Older PyTorch: only no-arg version is available.
        original = torch.is_autocast_enabled

        def _patched(device_type=None):
            return original()

        _patched._qwen_patched = True  # type: ignore[attr-defined]
        torch.is_autocast_enabled = _patched  # type: ignore[assignment]


_patch_torch_autocast_for_older_pytorch()


# 1.load pdf document
from ai_toolkit.rag.loaders import load_pdf_document
pdf_root_path = '/Users/hzz/KMS/IC-RAG-Agent/data/documents/sales_platform/amazon/fba'
pdf_path = '/Users/hzz/KMS/IC-RAG-Agent/data/documents/sales_platform/amazon/fba/FBA features/FBA Global Selling.pdf'
pdf_docs = load_pdf_document(pdf_path)


# 2.split pdf document
from ai_toolkit.rag.splitters import split_document_recursive
split_docs = split_document_recursive(pdf_docs)


# 3.add embedding models - Using local Qwen3-VL-Embedding-2B
from typing import List
from langchain_core.embeddings import Embeddings
from ai_toolkit.rag.retrievers import create_vector_store

# Add model scripts to path
model_scripts_path = project_root / "models/Qwen3-VL-Embedding-2B/scripts"
if model_scripts_path.is_dir():
    sys.path.insert(0, str(model_scripts_path))

from qwen3_vl_embedding import Qwen3VLEmbedder


class LocalQwenEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for local Qwen3-VL-Embedding-2B (text-only)."""

    def __init__(self, model_path: str, max_length: int = 8192):
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=max_length,
        )
        print(f"Loaded local Qwen3-VL-Embedding-2B from {model_path}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        inputs = [{"text": t} for t in texts]
        out = self.embedder.process(inputs, normalize=True)
        # Qwen3-VL may return bfloat16 tensors; convert to float32 for NumPy/langchain.
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 1:
            return [arr.tolist()]
        return [arr[i].tolist() for i in range(len(arr))]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        inputs = [{"text": text}]
        out = self.embedder.process(inputs, normalize=True)
        # Qwen3-VL may return bfloat16 tensors; convert to float32 for NumPy/langchain.
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 2:
            return arr[0].tolist()
        return arr.tolist()


# Initialize local Qwen embeddings and create vector store via ai_toolkit
model_path = str(project_root / "models/Qwen3-VL-Embedding-2B")
embeddings = LocalQwenEmbeddings(model_path)
print("Local Qwen embeddings initialized")

print(f"Creating vector store with {len(split_docs)} document chunks...")
vector_store = create_vector_store(split_docs, embeddings, store_type="inmemory")
print("Vector store created successfully")

# Test similarity search
query = "What is FBA Global Selling?"
print(f"Testing similarity search for: '{query}'")
results = vector_store.similarity_search(query, k=3)
print(f"Found {len(results)} relevant chunks")
if results:
    print("Top result preview:")
    print(f"   {results[0].page_content[:200]}...")
    print(f"   Metadata: {results[0].metadata}")


# ===================== 新增：将分块文档存入Chroma向量库 =====================
def store_docs_to_chroma(split_docs, embeddings, chroma_persist_path, collection_name):
    """
    将分块后的文档向量化并存储到Chroma
    :param split_docs: 分块后的Document列表
    :param embeddings: 初始化后的LocalQwenEmbeddings对象
    :param chroma_persist_path: Chroma持久化路径
    :param collection_name: Chroma集合名称
    """
    # 1. 初始化Chroma持久化客户端（数据落地到指定路径）
    chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
    # 2. 创建/获取集合（不存在则创建，存在则复用）
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # 3. 提取文档核心信息
    texts = [doc.page_content for doc in split_docs]  # 分块文本内容
    metadatas = [doc.metadata for doc in split_docs]  # 元数据（PDF路径、页码等）
    ids = [f"{collection_name}_{i}" for i in range(len(split_docs))]  # 唯一ID
    
    # 4. 生成向量并批量写入Chroma
    vectors = embeddings.embed_documents(texts)  # 调用Qwen模型生成向量
    collection.add(
        documents=texts,       # 原始文本（用于检索后展示）
        metadatas=metadatas,   # 溯源元数据
        ids=ids,               # 唯一标识
        embeddings=vectors     # Qwen生成的向量
    )
    
    print(f"\n✅ 成功将{len(texts)}个文档分块存入Chroma")
    print(f"📌 Chroma数据持久化路径：{chroma_persist_path}")
    print(f"📌 Chroma集合名称：{collection_name}")
    print(f"📌 向量库总数据量：{collection.count()}条")

# 配置Chroma存储参数
CHROMA_PERSIST_PATH = str(project_root / "data/chroma_db/amazon/fba")  # 按业务维度划分路径
COLLECTION_NAME = "amazon_fba_features" 

# 执行存储操作
store_docs_to_chroma(
    split_docs=split_docs,
    embeddings=embeddings,
    chroma_persist_path=CHROMA_PERSIST_PATH,
    collection_name=COLLECTION_NAME
)


chroma_persist_path = CHROMA_PERSIST_PATH
collection_name = COLLECTION_NAME
query = 'what method is included in the FBA Global Selling?'

# 可选：测试Chroma检索（验证存储结果）
def test_chroma_retrieval(chroma_persist_path, collection_name, query, k=3):
    """Test Chroma vector database retrieval"""
    chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
    collection = chroma_client.get_collection(name=collection_name)
    
    # Generate query vector
    query_vector = embeddings.embed_query(query)
    # Semantic search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n🔍 Chroma retrieval results (query: {query}):")
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        print(f"\nResult {i+1} (similarity distance: {dist:.4f}):")
        print(f"Source: {meta.get('source', 'unknown')}, Page: {meta.get('page', 'unknown')}")
        print(f"Content preview: {doc[:200]}...")

# Execute Chroma retrieval test
test_chroma_retrieval(
    chroma_persist_path=CHROMA_PERSIST_PATH,
    collection_name=COLLECTION_NAME,
    query="What is FBA Global Selling?"
)

