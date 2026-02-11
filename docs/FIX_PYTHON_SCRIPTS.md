# Fix Python 3.13 Script References

## Issue

After downgrading Python from 3.13 to 3.11, some scripts still have shebangs pointing to the old Python 3.13 interpreter.

## Solution

### Fixed Scripts

1. **ipython** - ✅ Fixed (reinstalled via conda)
2. **pytest** - ✅ Fixed (reinstalled via conda)
3. **jupyter** - ✅ Fixed (reinstalled via conda)
4. **chroma** - ✅ Fixed (reinstalled via conda)
5. **langgraph** - ✅ Fixed (reinstalled via conda)

### How to Fix Other Scripts

If you encounter similar errors with other commands:

```bash
# Option 1: Reinstall via conda (recommended)
conda activate base
conda install -y <package-name> --force-reinstall

# Option 2: Reinstall via pip
pip uninstall -y <package-name>
pip install <package-name>

# Option 3: Fix shebang manually (if needed)
sed -i '' 's|#!/opt/miniconda3/bin/python3.13|#!/opt/miniconda3/bin/python3.11|g' /opt/miniconda3/bin/<script-name>
```

### Verify Fix

```bash
# Check if script points to correct Python
head -1 /opt/miniconda3/bin/<command>

# Should show: #!/opt/miniconda3/bin/python3.11
# Not: #!/opt/miniconda3/bin/python3.13
```

### Remaining Scripts with Python 3.13 References

Some scripts may still reference Python 3.13 but are less commonly used:
- `onnxruntime_test`
- `pyjson5`
- `pybabel`
- `jlpm`

These can be fixed individually if needed.

---

**Last Updated**: 2025-01-23  
**Status**: ✅ Common scripts fixed
