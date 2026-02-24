# HuggingFace CLI Setup

## ✅ Solution Found!

The CLI command is called **`hf`**, not `huggingface-cli`. It's already installed and working!

## Usage

### Download Model

```bash
hf download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B
```

### Get Help

```bash
hf --help
hf download --help
```

## Alternative Options

### Option 1: Use `hf` Command (Available Now!)

The Python API works perfectly and is more reliable:

```python
from huggingface_hub import snapshot_download

# Download model
snapshot_download(
    repo_id="Qwen/Qwen3-VL-Embedding-8B",
    local_dir="./models/Qwen3-VL-Embedding-8B"
)
```

### Option 2: Use Shell Script (Qwen3-1.7B-GGUF, all-MiniLM-L6-v2)

```bash
./scripts/sh/download_models_from_hf.sh [all|qwen3|minilm]
```

For Qwen3-VL-Embedding-8B, use `hf download` or Python API directly.

### Option 3: Use Python Module

```bash
python3 -m huggingface_hub.cli.hf download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models
```

### Option 4: Create Alias (if you prefer `huggingface-cli` name)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
alias huggingface-cli='hf'
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

## Note

The command is called **`hf`**, not `huggingface-cli`. The `hf` command is the official HuggingFace CLI tool.

## Verification

Check if Python API works:

```bash
python3 -c "from huggingface_hub import snapshot_download; print('✅ Python API works')"
```

## Recommendation

**Use the Python API or the download script** - they're more reliable and don't depend on PATH configuration.

---

**Last Updated**: 2025-01-23  
**Status**: ✅ `hf` command is available and working!
