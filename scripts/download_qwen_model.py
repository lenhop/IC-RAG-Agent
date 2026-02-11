#!/usr/bin/env python3
"""
Download Qwen3-VL-Embedding-8B model

This script downloads the Qwen3-VL-Embedding-8B model from HuggingFace.
Model size: ~16GB, so ensure you have sufficient disk space and bandwidth.
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_model(
    repo_id: str = "Qwen/Qwen3-VL-Embedding-8B",
    local_dir: str = "./models/Qwen3-VL-Embedding-8B"
):
    """
    Download Qwen3-VL-Embedding-8B model from HuggingFace.
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save the model
    """
    model_dir = Path(local_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Qwen3-VL-Embedding-8B Model")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Local directory: {model_dir.absolute()}")
    print(f"Estimated size: ~16GB")
    print("\nThis may take a while depending on your internet speed...")
    print("=" * 60)
    
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True  # Resume if interrupted
        )
        
        print("\n" + "=" * 60)
        print("✅ Model downloaded successfully!")
        print(f"Location: {Path(downloaded_path).absolute()}")
        print("=" * 60)
        
        return downloaded_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~20GB)")
        print("3. Try again - the download will resume from where it stopped")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Qwen3-VL-Embedding-8B model")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-8B",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./models/Qwen3-VL-Embedding-8B",
        help="Local directory to save the model"
    )
    
    args = parser.parse_args()
    
    download_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )
