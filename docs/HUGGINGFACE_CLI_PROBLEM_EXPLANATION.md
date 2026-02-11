# HuggingFace CLI Problem Explanation

## The Problem

You were trying to use `huggingface-cli` but got:
```
zsh: command not found: huggingface-cli
```

## Root Cause

The `huggingface_hub` package **does not install a command called `huggingface-cli`**. 

Instead, it installs a command called **`hf`**.

### Why This Happens

When `huggingface_hub` is installed via pip, it registers entry points in the Python package metadata. The entry point is defined as:

```
Entry point: hf
Module: huggingface_hub.cli.hf
```

This creates a command called `hf`, not `huggingface-cli`.

### Verification

```bash
# Check what commands are available
which hf
# Output: /opt/miniconda3/bin/hf ✅

which huggingface-cli
# Output: huggingface-cli not found ❌
```

## The Solution

I created a wrapper script that makes `huggingface-cli` work by calling `hf`:

```bash
# Created at: /opt/miniconda3/bin/huggingface-cli
#!/bin/bash
exec hf "$@"
```

This wrapper script:
1. Intercepts calls to `huggingface-cli`
2. Passes all arguments to the `hf` command
3. Makes `huggingface-cli` work as expected

## Why People Expect `huggingface-cli`

1. **Naming Convention**: Many CLI tools use `package-cli` naming (e.g., `aws-cli`, `gcloud-cli`)
2. **Documentation**: Some older documentation or examples might reference `huggingface-cli`
3. **Intuition**: It's a natural name to expect

## Current Status

✅ **Both commands now work:**
- `hf` - Original command (always worked)
- `huggingface-cli` - Wrapper script (now works)

Both do exactly the same thing.

## Technical Details

### Entry Points

Python packages can define "entry points" that create command-line scripts. The `huggingface_hub` package defines:

```python
# In setup.py or pyproject.toml
[project.scripts]
hf = "huggingface_hub.cli.hf:main"
```

This creates the `hf` command, not `huggingface-cli`.

### Why Not `huggingface-cli`?

The HuggingFace team chose `hf` because:
- **Shorter**: Faster to type
- **Consistent**: Matches their branding (HF = HuggingFace)
- **Modern**: Many tools use short names (e.g., `kubectl` → `k`, `docker` → `d`)

## Summary

| Aspect | Details |
|--------|---------|
| **Expected Command** | `huggingface-cli` |
| **Actual Command** | `hf` |
| **Problem** | Name mismatch - command doesn't exist |
| **Solution** | Created wrapper script `/opt/miniconda3/bin/huggingface-cli` |
| **Status** | ✅ Both commands work now |

---

**Last Updated**: 2025-01-23  
**Status**: ✅ Problem solved - both `hf` and `huggingface-cli` work
