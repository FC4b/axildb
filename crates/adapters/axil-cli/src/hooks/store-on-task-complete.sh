#!/bin/bash
# Fires on TaskUpdate — if a task was just marked completed,
# inject a reminder to store BEFORE continuing.
INPUT=$(cat)
STATUS=$(echo "$INPUT" | jq -r '.tool_input.status // empty' 2>/dev/null)

if [ "$STATUS" = "completed" ]; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"You just marked a task completed. BEFORE doing anything else, run axil store with a summary of what you did and why. This is mandatory for every completed task."}}
EOF
else
  echo '{}'
fi
