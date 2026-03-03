#!/usr/bin/env bash
# Test that Claude Code runs commands without approval prompts.
# Run: ./scripts/test_claude_code_auto_run.sh
#
# This verifies ~/.claude/settings.local.json has defaultMode: bypassPermissions
# so Claude Code executes without asking for approval.

set -e

echo "=== Claude Code Auto-Run Test ==="
echo ""

# 1. Verify settings
SETTINGS="$HOME/.claude/settings.local.json"
if [[ ! -f "$SETTINGS" ]]; then
  echo "FAIL: $SETTINGS not found"
  exit 1
fi

if grep -q '"bypassPermissions"' "$SETTINGS"; then
  echo "PASS: settings.local.json has defaultMode: bypassPermissions"
else
  echo "FAIL: settings.local.json missing defaultMode bypassPermissions"
  echo "Add: \"defaultMode\": \"bypassPermissions\" to $SETTINGS"
  exit 1
fi

# 2. Run claude WITHOUT --dangerously-skip-permissions
# Settings should make it run without approval
echo "Running: claude -p 'Run: echo OK' (no flag - using settings)"
if command -v timeout &>/dev/null; then
  OUTPUT=$(timeout 120 claude -p "Run exactly this command: echo OK. Reply with the output." 2>&1) || true
elif command -v gtimeout &>/dev/null; then
  OUTPUT=$(gtimeout 120 claude -p "Run exactly this command: echo OK. Reply with the output." 2>&1) || true
else
  OUTPUT=$(claude -p "Run exactly this command: echo OK. Reply with the output." 2>&1) || true
fi

if echo "$OUTPUT" | grep -q "OK"; then
  echo "PASS: Claude executed command without approval prompt"
  exit 0
fi

echo "Output (last 800 chars):"
echo "$OUTPUT" | tail -c 800
echo ""
echo "---"
echo "If you saw no approval prompt, settings work. Run manually to confirm:"
echo "  claude -p 'echo test'"
