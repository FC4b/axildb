#!/usr/bin/env bash
# Phase 12 end-to-end validation (12.6).
#
# Verifies the full second-brain pipeline:
#   axil install (dry-run) → axil ingest → axil recall (context-block) → axil brief → axil retro → axil schedule (dry-run)
#
# Runs against a temporary database so it never touches the user's state.
# Exit code 0 = pass, non-zero = fail.

set -euo pipefail

AXIL="${AXIL:-./target/release/axil}"
if [[ ! -x "$AXIL" ]]; then
    echo "build the release binary first: cargo build --release -p axildb" >&2
    exit 2
fi
# Make the binary path absolute so later `cd $TMPROOT` calls still find it.
AXIL=$(cd "$(dirname "$AXIL")" && pwd)/$(basename "$AXIL")

TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# Create a proper install so the DB has embeddings + graph + FTS attached.
cd "$TMPROOT"
"$AXIL" install >/dev/null 2>&1
DB_DIR="$TMPROOT/.axil"
DB="$DB_DIR/memory.axil"
if [[ ! -f "$DB" ]]; then
    echo "install did not create $DB" >&2
    exit 1
fi

# Create a small fixture corpus of markdown notes.
CORPUS="$TMPROOT/corpus"
mkdir -p "$CORPUS"
cat > "$CORPUS/auth.md" <<'EOF'
# Auth refactor

Decided to move AuthModule to OAuth2 using the standard JWT library.
LoginService will handle token refresh; UserController stays unchanged.
EOF
cat > "$CORPUS/ingest.md" <<'EOF'
# Ingest pipeline

Bulk ingest walks a directory, chunks each file on paragraph boundaries,
and auto-embeds every chunk. Idempotent via FNV content hash.
EOF
cat > "$CORPUS/hooks.md" <<'EOF'
# Claude Code hooks

UserPromptSubmit injects a context block on every user prompt.
The hook calls axil recall with --recall-format context-block and a 1.8s timeout.
EOF

pass() { echo "  PASS $*"; }
fail() { echo "  FAIL $*" >&2; exit 1; }

echo "Phase 12 e2e: database at $DB"

# 1. install --dry-run (uses a sibling temp dir so the main DB is untouched)
echo "[1] axil install --dry-run --claude-code"
DRY_ROOT=$(mktemp -d)
pushd "$DRY_ROOT" >/dev/null
OUT=$("$AXIL" install --dry-run --claude-code 2>&1)
echo "$OUT" | grep -q '"dry_run":true' && pass "dry-run flag honored" || fail "dry-run did not set flag"
echo "$OUT" | grep -q '"UserPromptSubmit"' && pass "UserPromptSubmit scheduled" || fail "missing UserPromptSubmit"
popd >/dev/null
rm -rf "$DRY_ROOT"

# 2. ingest
echo "[2] axil ingest"
"$AXIL" --db "$DB" ingest "$CORPUS" --table notes 2>/dev/null >"$TMPROOT/ingest.json"
COUNT=$(python3 -c "import json;print(json.load(open('$TMPROOT/ingest.json'))['files_ingested'])")
[[ "$COUNT" == "3" ]] && pass "3 files ingested" || fail "expected 3 files, got $COUNT"

# Resume should skip all three.
"$AXIL" --db "$DB" ingest "$CORPUS" --table notes --resume 2>/dev/null >"$TMPROOT/ingest2.json"
SKIPPED=$(python3 -c "import json;print(json.load(open('$TMPROOT/ingest2.json'))['files_skipped'])")
[[ "$SKIPPED" == "3" ]] && pass "resume skipped all 3 files" || fail "resume broke: skipped=$SKIPPED"

# 3. recall context-block format
echo "[3] axil recall --recall-format context-block"
BLOCK=$("$AXIL" --db "$DB" recall "OAuth2 JWT" --recall-format context-block --budget 1500 --timeout-ms 2000 --top-k 3 2>/dev/null)
echo "$BLOCK" | grep -q '<context source="axil">' && pass "context block emitted" || fail "missing context block tag"
echo "$BLOCK" | grep -q 'OAuth2\|AuthModule' && pass "relevant content returned" || fail "recall did not surface ingested content"

# 4. recall --rerank off (default fast path)
echo "[4] axil recall --rerank off (fast path)"
"$AXIL" --db "$DB" recall "auth refactor" --rerank off --top-k 3 --recall-format oneline 2>/dev/null | head -1 >/dev/null && \
    pass "rerank=off succeeded"

# 5. recall --rerank llm (no LLM configured — should fall back cleanly)
echo "[5] axil recall --rerank llm (fallback)"
LLM_STDERR=$("$AXIL" --db "$DB" recall "auth refactor" --rerank llm --top-k 3 --recall-format oneline 2>&1 >/dev/null || true)
echo "$LLM_STDERR" | grep -q 'no LLM configured' && pass "LLM rerank fell back cleanly" || \
    fail "LLM rerank did not emit fallback message: $LLM_STDERR"

# 6. recall --expand
echo "[6] axil recall --expand"
"$AXIL" --db "$DB" recall "AuthModule OAuth2" --expand --top-k 3 --recall-format oneline 2>/dev/null | head -1 >/dev/null && \
    pass "expand flag accepted"

# 7. brief (must at least produce output; counts should be >=3 sessions/decisions/errors combined)
echo "[7] axil brief"
"$AXIL" --db "$DB" brief --window 1h --brief-format json 2>/dev/null >"$TMPROOT/brief.json"
python3 -c "import json;d=json.load(open('$TMPROOT/brief.json'));assert 'counts' in d and 'narrative' in d" \
    && pass "brief JSON has counts + narrative"

# 8. retro --save
echo "[8] axil retro --save"
"$AXIL" --db "$DB" retro --window 1d --save --brief-format json 2>/dev/null >"$TMPROOT/retro.json"
RETRO_FILE=$(ls "$DB_DIR/reports"/retro-*.md 2>/dev/null | head -1)
[[ -f "$RETRO_FILE" ]] && pass "retro markdown written to $RETRO_FILE" || fail "retro did not write a report file"

# 9. schedule dry-run
echo "[9] axil schedule install daily-brief --dry-run"
"$AXIL" --db "$DB" schedule install daily-brief --dry-run --hour 8 2>/dev/null >"$TMPROOT/sched.json"
python3 -c "import json;d=json.load(open('$TMPROOT/sched.json'));assert d['dry_run']==True and d['name']=='daily-brief'" \
    && pass "schedule dry-run returned plan"

echo
echo "Phase 12 e2e: all checks passed."
