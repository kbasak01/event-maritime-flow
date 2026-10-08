#!/usr/bin/env bash
# Blocks hand-edits of generated results, raw data and external checkouts.
# Results must be produced by scripts (run via Bash), never typed by hand.
set -euo pipefail
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
[ -z "$FILE" ] && exit 0
ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
REL="${FILE#"$ROOT"/}"
case "$REL" in
  results/*|data/*|external/*|splits/*.json)
    echo "Blocked: $REL is generated or third-party. Produce it with a script (see CLAUDE.md rule 3)." >&2
    exit 2 ;;
esac
exit 0
