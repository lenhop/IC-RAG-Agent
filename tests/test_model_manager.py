"""
测试 ModelManager 功能
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


def test_create_deepseek_model():
    """测试创建 DeepSeek 模型"""
    try:
        from ai_toolkit.models import ModelManager
        
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        print("✅ DeepSeek 模型创建成功")
        assert model is not None
        
        # 测试简单调用（如果 API key 配置了）
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                response = model.invoke("Say hello in 5 words")
                print(f"✅ 模型响应: {response.content[:50]}...")
            except Exception as e:
                print(f"⚠️  模型调用失败（可能是 API key 问题）: {e}")
        
    except Exception as e:
        print(f"❌ DeepSeek 模型创建失败: {e}")
        assert False, f"创建失败: {e}"


def test_create_qwen_model():
    """测试创建 Qwen 模型"""
    try:
        from ai_toolkit.models import ModelManager
        
        manager = ModelManager()
        model = manager.create_model("qwen")
        
        print("✅ Qwen 模型创建成功")
        assert model is not None
        
    except Exception as e:
        print(f"❌ Qwen 模型创建失败: {e}")
        assert False, f"创建失败: {e}"


def test_list_providers():
    """测试列出可用提供商"""
    try:
        from ai_toolkit.models import ModelManager
        
        manager = ModelManager()
        providers = manager.list_providers()
        
        print(f"✅ 可用提供商: {providers}")
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "deepseek" in providers or "qwen" in providers
        
    except Exception as e:
        print(f"❌ 列出提供商失败: {e}")
        assert False, f"失败: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 ModelManager")
    print("=" * 60)
    
    tests = [
        test_list_providers,
        test_create_deepseek_model,
        test_create_qwen_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
        except Exception as e:
            failed += 1
            print(f"❌ 测试异常: {test.__name__}: {e}")
    
    print("=" * 60)
    print(f"测试完成: 通过 {passed}, 失败 {failed}, 总计 {len(tests)}")
    print("=" * 60)
