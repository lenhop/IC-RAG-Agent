# Claude Code Auto-Run Test

Verify that Claude Code runs commands without asking for approval.

## Config Check (instant)

```bash
# Should show bypassPermissions
grep -A1 defaultMode ~/.claude/settings.local.json
```

Expected:
```json
"defaultMode": "bypassPermissions",
```

## Manual Test

1. Open a terminal in this project.
2. Run:
   ```bash
   claude -p "Run: echo OK"
   ```
3. **If Claude runs the command and prints OK without pausing for approval** → PASS.
4. **If Claude asks "Approve this command?"** → FAIL (settings not applied).

## Alternative: Use Flag

If settings do not apply (e.g. different project root), use the flag each time:

```bash
claude -p "your prompt" --dangerously-skip-permissions
```

## Script Test

```bash
./scripts/test_claude_code_auto_run.sh
```

This checks the config and runs a quick Claude test.
