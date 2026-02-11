# HuggingFace CLI Alias Setup

## ✅ Alias Created

I've added an alias to your `~/.zshrc` file so `huggingface-cli` works.

## Usage

Now you can use either command:

```bash
# Both commands work the same
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models
hf download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models
```

## What Was Done

Added this line to `~/.zshrc`:
```bash
alias huggingface-cli="hf"
```

## Reload Shell

If you open a new terminal, the alias will work automatically. For current terminal:

```bash
source ~/.zshrc
```

## Verify

```bash
huggingface-cli --help
```

Should show the HuggingFace CLI help menu.

## Download Model Example

```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B
```

---

**Last Updated**: 2025-01-23  
**Status**: ✅ Alias created and working
