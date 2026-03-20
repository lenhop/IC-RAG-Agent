
import sys 
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
import chromadb 
from chromadb.utils.embedding_functions import EmbeddingFunction


load_dotenv()

__file__ = '/Users/hzz/KMS/IC-RAG-Agent/src/draft/load_split_pdf.py'
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'external/ai-toolkit/'))


# 配置Chroma存储参数
CHROMA_PERSIST_PATH = str(project_root / "data/chroma_db/amazon/fba")  # 按业务维度划分路径
COLLECTION_NAME = "amazon_fba_features" 

# 初始化Qwen3-VL-Embedding-2B模型
from typing import List
from langchain_core.embeddings import Embeddings

# Add model scripts to path
model_scripts_path = project_root / "models/Qwen3-VL-Embedding-2B/scripts"
if model_scripts_path.is_dir():
    sys.path.insert(0, str(model_scripts_path))

def _patch_torch_autocast_for_older_pytorch() -> None:
    """Patch torch.is_autocast_enabled for older PyTorch versions"""
    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return
    try:
        torch.is_autocast_enabled("cpu")
    except TypeError:
        original = torch.is_autocast_enabled
        def _patched(device_type=None):
            return original()
        _patched._qwen_patched = True
        torch.is_autocast_enabled = _patched

_patch_torch_autocast_for_older_pytorch()

from qwen3_vl_embedding import Qwen3VLEmbedder

class LocalQwenEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for local Qwen3-VL-Embedding-2B"""
    def __init__(self, model_path: str, max_length: int = 8192):
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=max_length,
        )
        print(f"✅ 已加载 Qwen3-VL-Embedding-2B 模型")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        inputs = [{"text": t} for t in texts]
        out = self.embedder.process(inputs, normalize=True)
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 1:
            return [arr.tolist()]
        return [arr[i].tolist() for i in range(len(arr))]
    
    def embed_query(self, text: str) -> List[float]:
        inputs = [{"text": text}]
        out = self.embedder.process(inputs, normalize=True)
        out = out.to(dtype=torch.float32)
        arr = out.cpu().numpy()
        if arr.ndim == 2:
            return arr[0].tolist()
        return arr.tolist()


class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        self.qwen_ef = LocalQwenEmbeddings(model_path)
    def __call__(self, texts):
        vectors = self.qwen_ef.embed_documents(texts)
        return [np.array(v).astype(np.float32) for v in vectors]
    def name(self):
        return "Qwen3-VL-Embedding-2B"  # 必须定义name方法



model_path = str(project_root / "models/Qwen3-VL-Embedding-2B")
embeddings = LocalQwenEmbeddings(model_path)


# 1. 初始化客户端
# 初始化Chroma持久化客户端（数据落地到指定路径）
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

# 2. 获取/创建集合
# 若不指定 embedding_function, 则默认使用Chroma的默认embedding模型，这里我们使用Qwen3-VL-Embedding-2B模型
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embeddings)


# 3.统计数据量, 或测试集合是否存在
collection.count()

# 4.查询 collection 内容
# 4.1 返回所有ids 
# include=["ids", "metadatas", "documents", "embeddings"]：返回所有字段（按需选择，避免加载不必要的向量数据）；
ids = collection.get(include=[])["ids"]  # include=[] 表示只获取ids，不加载其他数据
print(ids)

# 4.2 简单获取前 N 条数据（默认按插入顺序）
all_items_preview = collection.get(
    limit=5,  # only fetch first 5 items for preview
    include=["metadatas", "documents"]
)
print("🔍 Preview first 5 items in collection:")
print(all_items_preview)

# 4.3 按 ID 精确获取指定条目
sample_ids = ids[:2] if len(ids) >= 2 else ids  # reuse ids we just inserted if available
if sample_ids:
    items_by_id = collection.get(
        ids=sample_ids,
        include=["documents", "metadatas"]
    )
    print(f"🔍 Fetch items by ids={sample_ids}:")
    print(items_by_id)

# 4.3 按元数据过滤（例如：按 source 文件路径过滤）
"""
{'page_label': '1',
   'source': '/Users/hzz/KMS/IC-RAG-Agent/data/documents/sales_platform/amazon/fba/FBA features/FBA Global Selling.pdf',
   'creationdate': '2026-02-02T13:37:42+00:00',
   'creator': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
   'total_pages': 3,
   'title': 'Amazon',
   'start_index': 0,
   'moddate': '2026-02-02T13:37:42+00:00',
   'page': 0,
   'producer': 'Skia/PDF m144'}
"""

items_by_metadata = collection.get(
    where={"page": 0},  # metadata key 'source' is added by loader
    limit=3,
    include=["metadatas", "documents"]
)
print(f"🔍 Fetch items where page == '0':")
print(items_by_metadata)

# 4.4 按文档内容过滤（例如：包含某个关键词）
items_by_document = collection.get(
    where_document={"$contains": "customer service"},
    limit=3,
    include=["metadatas", "documents"]
)
print("🔍 Fetch items where document contains 'customer service':")
print(items_by_document)

# 4.5 使用 limit / offset 做简单分页
page_size = 5
page_2_items = collection.get(
    limit=page_size,
    offset=page_size,  # skip first page_size items
    include=["ids", "metadatas"]
)
print(f"🔍 Fetch page 2 items (limit={page_size}, offset={page_size}):")
print(page_2_items)


# 5. 新增数据
# 5.1 加载并准备文档数据
from ai_toolkit.rag.loaders import load_pdf_document
from ai_toolkit.rag.splitters import split_document_recursive

pdf_path = '/Users/hzz/KMS/IC-RAG-Agent/data/documents/sales_platform/amazon/fba/FBA features/FBA customer service.pdf'
pdf_docs = load_pdf_document(pdf_path)
split_docs = split_document_recursive(pdf_docs)
print(f"📄 已加载并分割文档，共 {len(split_docs)} 个分块")

# 5.2 提取文档信息
current_ids = collection.get(include=[])["ids"]
texts = [doc.page_content for doc in split_docs]
metadatas = [doc.metadata for doc in split_docs]
new_ids = [f"{COLLECTION_NAME}_{i}" for i in range(len(current_ids), len(current_ids) + len(split_docs))]
print(f"📝 准备新增 {len(texts)} 条数据到集合 '{COLLECTION_NAME}'")

# 5.3 生成向量
print("🔄 正在生成文档向量...")
vectors = embeddings.embed_documents(texts)
print(f"✅ 向量生成完成，维度: {len(vectors[0])}")

# 5.4 批量写入Chroma
collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=new_ids,
    embeddings=vectors
)
print(f"✅ 成功将 {len(texts)} 个文档分块存入 Chroma")
print(f"📊 当前集合总数据量: {collection.count()} 条")


# 6. 语义检索
## 6.1 use words 
words = "what's the FBA customer service"
res1 = collection.query(query_texts=[words], n_results=2)

## 6.2 use vector
query_vector = embeddings.embed_query(words)
res2 = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )


# 7. 删除数据
collection.delete(ids=["fba_001"])


