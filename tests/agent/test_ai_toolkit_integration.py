"""
测试 ai-toolkit 集成
验证 ai-toolkit 是否正确安装和导入
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ai_toolkit_import():
    """测试 ai-toolkit 基本导入"""
    try:
        import ai_toolkit
        print("✅ ai-toolkit 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ ai-toolkit 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_model_manager_import():
    """测试 ModelManager 导入"""
    try:
        from ai_toolkit.models import ModelManager
        print("✅ ModelManager 导入成功")
        assert True
    except ImportError as e:
        print(f"❌ ModelManager 导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_model_manager_creation():
    """测试 ModelManager 创建"""
    try:
        from ai_toolkit.models import ModelManager
        
        manager = ModelManager()
        print("✅ ModelManager 创建成功")
        
        # 测试列出可用提供商
        providers = manager.list_providers()
        print(f"✅ 可用模型提供商: {providers}")
        assert isinstance(providers, list)
        assert len(providers) > 0
        
    except Exception as e:
        print(f"❌ ModelManager 创建失败: {e}")
        assert False, f"创建失败: {e}"


def test_rag_modules_import():
    """测试 RAG 相关模块导入"""
    try:
        from ai_toolkit.rag import loaders, splitters, retrievers
        print("✅ RAG 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ RAG 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_agent_modules_import():
    """测试 Agent 相关模块导入"""
    try:
        from ai_toolkit.agents import agent_helpers, middleware_utils
        print("✅ Agent 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Agent 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_tools_modules_import():
    """测试 Tools 相关模块导入"""
    try:
        from ai_toolkit.tools import tool_utils
        print("✅ Tools 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Tools 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_memory_modules_import():
    """测试 Memory 相关模块导入"""
    try:
        from ai_toolkit.memory import memory_manager
        print("✅ Memory 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Memory 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_messages_modules_import():
    """测试 Messages 相关模块导入"""
    try:
        from ai_toolkit.messages import message_builder
        print("✅ Messages 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Messages 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_streaming_modules_import():
    """测试 Streaming 相关模块导入"""
    try:
        from ai_toolkit.streaming import stream_handler
        print("✅ Streaming 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Streaming 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


def test_config_modules_import():
    """测试 Config 相关模块导入"""
    try:
        from ai_toolkit.config import config_loader
        print("✅ Config 模块导入成功")
        assert True
    except ImportError as e:
        print(f"❌ Config 模块导入失败: {e}")
        assert False, f"导入失败: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 ai-toolkit 集成")
    print("=" * 60)
    
    tests = [
        test_ai_toolkit_import,
        test_model_manager_import,
        test_model_manager_creation,
        test_rag_modules_import,
        test_agent_modules_import,
        test_tools_modules_import,
        test_memory_modules_import,
        test_messages_modules_import,
        test_streaming_modules_import,
        test_config_modules_import,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ 测试失败: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ 测试异常: {test.__name__}: {e}")
    
    print("=" * 60)
    print(f"测试完成: 通过 {passed}, 失败 {failed}, 总计 {len(tests)}")
    print("=" * 60)
