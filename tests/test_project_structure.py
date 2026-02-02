"""
测试项目结构
验证项目目录结构是否正确
"""
import os
from pathlib import Path


def test_project_root_exists():
    """测试项目根目录存在"""
    project_root = Path(__file__).parent.parent
    assert project_root.exists(), "项目根目录不存在"
    print(f"✅ 项目根目录: {project_root}")


def test_required_directories():
    """测试必需的目录是否存在"""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "config",
        "docs",
        "scripts",
        "tests",
        "data",
        "docker",
        "libs",
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✅ 目录存在: {dir_name}")
    
    if missing_dirs:
        print(f"⚠️  缺失目录: {missing_dirs}")
    else:
        print("✅ 所有必需目录都存在")


def test_ai_toolkit_submodule():
    """测试 ai-toolkit submodule 是否存在"""
    project_root = Path(__file__).parent.parent
    ai_toolkit_path = project_root / "libs" / "ai-toolkit"
    
    if ai_toolkit_path.exists():
        print(f"✅ ai-toolkit submodule 存在: {ai_toolkit_path}")
        
        # 检查关键文件
        key_files = [
            "ai_toolkit/__init__.py",
            "setup.py",
            "requirements.txt",
        ]
        
        for file_name in key_files:
            file_path = ai_toolkit_path / file_name
            if file_path.exists():
                print(f"✅ 文件存在: {file_name}")
            else:
                print(f"⚠️  文件缺失: {file_name}")
    else:
        print("⚠️  ai-toolkit submodule 不存在")


def test_requirements_file():
    """测试 requirements.txt 是否存在"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    if requirements_file.exists():
        print(f"✅ requirements.txt 存在")
        
        # 检查是否包含 ai-toolkit
        content = requirements_file.read_text()
        if "ai-toolkit" in content or "libs/ai-toolkit" in content:
            print("✅ requirements.txt 包含 ai-toolkit")
        else:
            print("⚠️  requirements.txt 不包含 ai-toolkit")
    else:
        print("❌ requirements.txt 不存在")


def test_env_file():
    """测试 .env 文件是否存在"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example_file = project_root / ".env.example"
    
    if env_file.exists():
        print("✅ .env 文件存在")
    else:
        print("⚠️  .env 文件不存在（需要从 .env.example 创建）")
    
    if env_example_file.exists():
        print("✅ .env.example 文件存在")
    else:
        print("⚠️  .env.example 文件不存在")


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试项目结构")
    print("=" * 60)
    
    tests = [
        test_project_root_exists,
        test_required_directories,
        test_ai_toolkit_submodule,
        test_requirements_file,
        test_env_file,
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
