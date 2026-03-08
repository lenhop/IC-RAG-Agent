"""
测试 Chroma 工具包功能
验证 Chroma 工具包是否正确安装和可用
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_chroma_module_import():
    """测试 Chroma 模块导入"""
    try:
        import ai_toolkit.chroma
        print("✅ Chroma 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Chroma 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_chroma_client_creation():
    """测试 Chroma 客户端创建"""
    try:
        from ai_toolkit.chroma import create_chroma_client
        
        # 测试内存模式
        client = create_chroma_client(mode="memory")
        print("✅ 内存模式客户端创建成功")
        assert client is not None
        
        # 测试持久化模式
        persist_dir = "./data/chroma_test"
        client = create_chroma_client(
            mode="persistent",
            persist_directory=persist_dir
        )
        print("✅ 持久化模式客户端创建成功")
        assert client is not None
        
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        assert False, f"创建失败: {e}"


def test_collection_management():
    """测试集合管理"""
    try:
        from ai_toolkit.chroma import create_chroma_client, get_or_create_collection
        
        client = create_chroma_client(mode="memory")
        
        # 创建集合
        collection = get_or_create_collection(
            client=client,
            name="test_collection",
            metadata={"description": "Test collection"}
        )
        print("✅ 集合创建成功")
        assert collection is not None
        
        # 再次获取（应该返回已有集合）
        collection2 = get_or_create_collection(
            client=client,
            name="test_collection"
        )
        print("✅ 集合获取成功")
        assert collection.name == collection2.name
        
    except Exception as e:
        print(f"❌ 集合管理测试失败: {e}")
        assert False, f"失败: {e}"


def test_chroma_store_creation():
    """测试 Chroma VectorStore 创建"""
    try:
        from ai_toolkit.chroma import create_chroma_store
        from langchain_core.documents import Document
        from langchain_community.embeddings import FakeEmbeddings
        
        # 创建测试文档
        documents = [
            Document(page_content="测试文档1", metadata={"source": "test1"}),
            Document(page_content="测试文档2", metadata={"source": "test2"}),
        ]
        
        # 使用 FakeEmbeddings（不需要真实 API）
        embeddings = FakeEmbeddings(size=128)
        
        # 创建内存模式的 VectorStore
        store = create_chroma_store(
            documents=documents,
            embeddings=embeddings,
            collection_name="test_store",
            persist_directory=None  # 内存模式
        )
        print("✅ Chroma VectorStore 创建成功")
        assert store is not None
        
        # 测试搜索
        results = store.similarity_search("测试", k=2)
        print(f"✅ 搜索成功: 找到 {len(results)} 个结果")
        assert len(results) > 0
        
    except Exception as e:
        print(f"❌ VectorStore 创建测试失败: {e}")
        assert False, f"失败: {e}"


def test_chroma_store_class():
    """测试 ChromaStore 类"""
    try:
        from ai_toolkit.chroma import ChromaStore
        from langchain_core.documents import Document
        from langchain_community.embeddings import FakeEmbeddings
        
        embeddings = FakeEmbeddings(size=128)
        
        # 创建 ChromaStore
        store = ChromaStore(
            collection_name="test_store_class",
            embeddings=embeddings,
            persist_directory=None  # 内存模式
        )
        print("✅ ChromaStore 类创建成功")
        
        # 添加文档
        documents = [
            Document(page_content="文档1", metadata={"platform": "amazon"}),
            Document(page_content="文档2", metadata={"platform": "ebay"}),
        ]
        
        ids = store.add_documents(documents)
        print(f"✅ 文档添加成功: {len(ids)} 个文档")
        assert len(ids) == 2
        
        # 测试搜索
        results = store.search("文档", k=2)
        print(f"✅ 搜索成功: {len(results)} 个结果")
        assert len(results) > 0
        
        # 测试带过滤搜索
        results = store.search_with_filter(
            query="文档",
            filter={"platform": "amazon"},
            k=1
        )
        print(f"✅ 带过滤搜索成功: {len(results)} 个结果")
        
        # 测试统计信息
        stats = store.get_stats()
        print(f"✅ 统计信息: {stats}")
        assert "document_count" in stats
        
    except Exception as e:
        print(f"❌ ChromaStore 类测试失败: {e}")
        assert False, f"失败: {e}"


def test_rag_integration():
    """测试与 RAG 模块集成"""
    try:
        from ai_toolkit.rag import create_vector_store
        from langchain_core.documents import Document
        from langchain_community.embeddings import FakeEmbeddings
        
        documents = [
            Document(page_content="测试文档", metadata={"source": "test"}),
        ]
        embeddings = FakeEmbeddings(size=128)
        
        # 测试通过 RAG 模块创建 Chroma 存储
        store = create_vector_store(
            documents=documents,
            embeddings=embeddings,
            store_type="chroma",
            collection_name="test_rag_integration",
            persist_directory=None  # 内存模式
        )
        print("✅ RAG 集成测试成功")
        assert store is not None
        
        # 测试搜索
        results = store.similarity_search("测试", k=1)
        assert len(results) > 0
        
    except Exception as e:
        print(f"❌ RAG 集成测试失败: {e}")
        assert False, f"失败: {e}"


def test_utils_functions():
    """测试工具函数"""
    try:
        from ai_toolkit.chroma import (
            validate_collection_name,
            generate_document_ids,
            validate_metadata
        )
        
        # 测试集合名称验证
        validate_collection_name("test_collection")
        print("✅ 集合名称验证成功")
        
        # 测试无效名称
        try:
            validate_collection_name("123invalid")
            assert False, "应该抛出异常"
        except ValueError:
            print("✅ 无效名称验证成功")
        
        # 测试 ID 生成
        ids = generate_document_ids(5, prefix="doc")
        print(f"✅ ID 生成成功: {ids}")
        assert len(ids) == 5
        
        # 测试元数据验证
        validate_metadata({"platform": "amazon", "count": 10})
        print("✅ 元数据验证成功")
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        assert False, f"失败: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 Chroma 工具包")
    print("=" * 60)
    
    tests = [
        test_chroma_module_import,
        test_chroma_client_creation,
        test_collection_management,
        test_chroma_store_creation,
        test_chroma_store_class,
        test_rag_integration,
        test_utils_functions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ 测试失败: {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ 测试异常: {test.__name__}: {e}")
    
    print("=" * 60)
    print(f"测试完成: 通过 {passed}, 失败 {failed}, 总计 {len(tests)}")
    print("=" * 60)
