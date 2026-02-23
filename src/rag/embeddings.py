"""
Embedding model factory for RAG pipeline.

Layer 2: Integrates HuggingFace (sentence-transformers) and Ollama
local embedding models. Use same model for indexing and query.
"""

from pathlib import Path
from typing import Any, Optional

# LangChain Embeddings protocol
Embeddings = Any


def _patch_torch_autocast() -> None:
    """Patch torch.is_autocast_enabled for PyTorch < 2.3 (Qwen compatibility)."""
    import torch

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


def create_embeddings(
    model_type: str,
    *,
    # HuggingFace options
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = "cpu",
    # Ollama options
    ollama_model: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434",
    # Qwen3 (ai-toolkit LocalQwenEmbeddings)
    qwen3_path: Optional[str] = None,
    # Project root for default paths
    project_root: Optional[Path] = None,
) -> Embeddings:
    """
    Create LangChain Embeddings for HuggingFace or Ollama.

    Args:
        model_type: "huggingface" | "ollama" | "qwen3".
        model_path: Local path for HuggingFace model (sentence-transformers).
        model_name: HuggingFace hub model name if not using local path.
        device: "cpu" or "cuda" for HuggingFace.
        ollama_model: Ollama model name (e.g. "all-minilm").
        ollama_base_url: Ollama API base URL.
        qwen3_path: Path to Qwen3-VL-Embedding-2B for model_type="qwen3".
        project_root: Used to resolve default model paths when None.

    Returns:
        LangChain Embeddings instance (embed_documents, embed_query).
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]

    if model_type == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = ollama_model or "all-minilm"
        return OllamaEmbeddings(model=model, base_url=ollama_base_url)

    if model_type == "minilm":
        model_path = model_path or str(root / "models" / "all-MiniLM-L6-v2")
        model_type = "huggingface"

    if model_type == "qwen3":
        _patch_torch_autocast()
        from ai_toolkit.models import LocalQwenEmbeddings

        path = qwen3_path or str(root / "models" / "Qwen3-VL-Embedding-2B")
        return LocalQwenEmbeddings(path)

    # HuggingFace (sentence-transformers)
    from langchain_community.embeddings import HuggingFaceEmbeddings

    if model_path:
        path = str(Path(model_path).expanduser().resolve())
    elif model_name:
        path = model_name
    else:
        path = str(root / "models" / "all-MiniLM-L6-v2")

    return HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
