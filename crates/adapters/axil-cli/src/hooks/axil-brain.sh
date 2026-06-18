#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Axil Brain — Claude Code cognitive memory hook
#
# Events:
#   UserPromptSubmit           — inject <context> block from recall (12.1)
#   PreToolUse (first call)    — boot with context-aware push
#   PreToolUse (periodic)      — nudge to store after significant edits
#   PostToolUse (Edit/Write)   — log file + push relevant memories
#   PostToolUse (Bash)         — auto-capture errors from failed commands
#   Stop                       — session summary + worker + beliefs
# ──────────────────────────────────────────────────────────────────────
set -e

INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

[ -z "$SESSION_ID" ] && exit 0

# ── Shared: find axil binary ─────────────────────────────────────────
find_axil() {
    for candidate in \
        "${CLAUDE_PROJECT_DIR}/target/release/axil" \
        "${CLAUDE_PROJECT_DIR}/target/debug/axil" \
        "$(command -v axil 2>/dev/null)"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
}

# ── Shared: find .axil/memory.axil ───────────────────────────────────
find_db() {
    DIR="${CLAUDE_PROJECT_DIR:-.}"
    while [ -n "$DIR" ] && [ "$DIR" != "/" ]; do
        if [ -f "$DIR/.axil/memory.axil" ]; then
            echo "$DIR/.axil/memory.axil"
            return
        fi
        DIR=$(dirname "$DIR")
    done
}

# ── Shared: cached binary+db lookup (memoized for the script lifetime) ─
# Use as `ensure_axil_db || exit 0` or `if ensure_axil_db; then …; fi`.
# Bare call would abort under `set -e` on miss.
AXIL=""
DB=""
ensure_axil_db() {
    [ -z "$AXIL" ] && AXIL=$(find_axil)
    [ -z "$DB" ] && DB=$(find_db)
    [ -n "$AXIL" ] && [ -n "$DB" ]
}

# ── Shared: skip non-code paths (logs, lockfiles, build artifacts) ───
is_skipped_path() {
    case "$1" in
        *.axil*|*.lock|*.log|*/node_modules/*|*/target/*|*/.git/*|*/.axil/*) return 0 ;;
        *) return 1 ;;
    esac
}

# ── Shared: narrative-table list (single source of truth) ────────────
# If you add a new narrative table, update both constants.
NARRATIVE_TABLES_JSON='["decisions","errors","context","commits","_checkpoint_records"]'
NARRATIVE_TABLES_TEXT="decisions/errors/context/commits/checkpoint"

# Count narrative records stored in the last hour. Requires ensure_axil_db.
count_recent_narrative() {
    "$AXIL" --db "$DB" since 1h 2>/dev/null \
        | jq --argjson t "$NARRATIVE_TABLES_JSON" \
            '[.[] | select(.table as $x | $t | index($x))] | length' 2>/dev/null \
        || echo 0
}

# Returns 0 if HEAD has a commit within the last hour, 1 otherwise.
# Used by the Stop guard to satisfy the narrative requirement when the
# agent committed but the async PostToolUse `commits` row hasn't landed
# in the DB yet (race between async hook + sync Stop guard).
has_recent_git_commit() {
    local repo="${CLAUDE_PROJECT_DIR:-.}"
    local commit_ts
    commit_ts=$(git -C "$repo" log -1 --pretty=%ct 2>/dev/null || echo 0)
    [ "$commit_ts" = "0" ] && return 1
    local now=$(date +%s)
    [ $((now - commit_ts)) -lt 3600 ]
}

# ── Temp files for this session ──────────────────────────────────────
MANIFEST="/tmp/axil-session-${SESSION_ID}.manifest"
CONTENT="/tmp/axil-session-${SESSION_ID}.content"
BOOTED="/tmp/axil-session-${SESSION_ID}.booted"
COUNTS="/tmp/axil-session-${SESSION_ID}.counts"
PROBLEMS="/tmp/axil-session-${SESSION_ID}.problems"
# Cap to keep a runaway session (hundreds of empty recalls) from bloating
# the queue and the eventual session-heal load.
PROBLEMS_MAX_BYTES=262144

# Sweep every per-session temp file. The sentinel files
# (.recalled-<hash>, .searched-<hash>) accumulate one-per-file-or-query and
# weren't cleaned by name-based rm; the glob covers them too.
cleanup_session_files() {
    rm -f /tmp/axil-session-"${SESSION_ID}".* 2>/dev/null || true
}

log_problem() {
    if [ -f "$PROBLEMS" ] && [ "$(wc -c <"$PROBLEMS" 2>/dev/null || echo 0)" -ge "$PROBLEMS_MAX_BYTES" ]; then
        return 0
    fi
    printf '%s\n' "$1" >> "$PROBLEMS" 2>/dev/null || true
}

# Empty-result sniffer for axil read commands. Handles the common forms:
# blank stdout, JSON empty array, and the textual sentinels axil prints.
# BSD-grep safe (no \b, no PCRE).
is_empty_axil_output() {
    case "$1" in
        ""|"[]"|"(no results)"|"(no code proxies matched"*|"(no matches)"*) return 0 ;;
    esac
    printf '%s' "$1" | head -c 200 | grep -q '^[[:space:]]*\[[[:space:]]*\][[:space:]]*$'
}

# Best-effort axil-subcommand extractor. BSD sed (macOS) lacks \b, so use
# awk to find the "axil" token and return the next non-flag arg.
extract_axil_subcmd() {
    echo "$1" | awk '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /(^|\/)axil$/) {
                for (j=i+1; j<=NF; j++) {
                    if ($j !~ /^-/) { print $j; exit }
                }
            }
        }
    }'
}

# Run session-heal and act on the hints it reports. The single user-visible
# auto-fix today: when session-heal flags `stale_structural_index` (because
# code-search/fts returned no results for queries the agent ran), kick off
# `axil index` in the background under nohup so the next session's queries
# actually hit the index. Lock check at .axil/index-refresh.lock throttles
# back-to-back stops within the 5-minute stale window.
run_session_heal_with_autofix() {
    local args=("session-heal" "--session" "$SESSION_ID")
    [ -f "$PROBLEMS" ] && args+=("--problems-file" "$PROBLEMS")
    local heal_out
    heal_out=$("$AXIL" --db "$DB" "${args[@]}" 2>/dev/null || true)
    [ -z "$heal_out" ] && return 0

    if echo "$heal_out" | jq -e '.hints[]? | select(.kind == "stale_structural_index")' >/dev/null 2>&1; then
        local repo="${CLAUDE_PROJECT_DIR:-$(pwd)}"
        local axil_dir="$repo/.axil"
        local lock="$axil_dir/index-refresh.lock"
        local log="$axil_dir/index-refresh.log"
        # Skip if a recent (<5 min) refresh is already running.
        if [ -f "$lock" ]; then
            local age
            age=$(( $(date +%s) - $(stat -f %m "$lock" 2>/dev/null || stat -c %Y "$lock" 2>/dev/null || echo 0) ))
            if [ "$age" -lt 300 ]; then
                return 0
            fi
        fi
        mkdir -p "$axil_dir" 2>/dev/null || true
        date +%s > "$lock"
        # nohup detaches from the parent so the agent's Stop completes
        # immediately; the child writes its own log, removes the lock when
        # done, and survives parent exit (matches the scip-refresh pattern).
        nohup sh -c "
            trap 'rm -f \"$lock\"' EXIT
            \"$AXIL\" --db \"$DB\" index \"$repo\" >> \"$log\" 2>&1
        " </dev/null >/dev/null 2>&1 &
        disown 2>/dev/null || true
        echo "🧠 Axil session-heal: stale structural index → spawned 'axil index' in background (log: .axil/index-refresh.log)" >&2
    fi
}

# ── Heartbeat counters ───────────────────────────────────────────────
# One JSON file per session: { stores, recalls, tools, errors }.
# Races on concurrent hook runs at worst drop a count; never corrupts.
bump_count() {
    local key="$1"
    local cur=0
    if [ -f "$COUNTS" ]; then
        cur=$(jq -r ".${key} // 0" "$COUNTS" 2>/dev/null || echo 0)
    fi
    local next=$((cur + 1))
    if [ -f "$COUNTS" ]; then
        jq ".${key} = ${next}" "$COUNTS" > "${COUNTS}.tmp" 2>/dev/null && mv "${COUNTS}.tmp" "$COUNTS"
    else
        echo "{\"${key}\": ${next}}" > "$COUNTS"
    fi
}

read_counts_compact() {
    if [ -f "$COUNTS" ]; then
        jq -r '"\(.stores // 0)s ∙ \(.recalls // 0)r ∙ \(.tools // 0)t"' "$COUNTS" 2>/dev/null || echo "0s ∙ 0r ∙ 0t"
    else
        echo "0s ∙ 0r ∙ 0t"
    fi
}

# ── UserPromptSubmit: inject <context> block from Axil recall (12.1) ─
# This runs on every user prompt. Must stay under 2s wall-clock.
if [ "$EVENT" = "UserPromptSubmit" ]; then
    PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
    [ -z "$PROMPT" ] && exit 0

    ensure_axil_db || exit 0

    # Skip trivially short prompts (acks, confirmations) — no useful recall.
    if [ "${#PROMPT}" -lt 8 ]; then
        exit 0
    fi

    CTX=$("$AXIL" --db "$DB" recall "$PROMPT" \
        --recall-format context-block \
        --budget 2000 \
        --timeout-ms 1800 \
        --top-k 5 2>/dev/null || true)

    if [ -n "$CTX" ]; then
        # Claude Code injects stdout from UserPromptSubmit hooks back into the prompt.
        echo "$CTX"
    fi
    exit 0
fi

# ── Every PreToolUse: bump tool counter ──────────────────────────────
if [ "$EVENT" = "PreToolUse" ]; then
    bump_count tools
fi

# ── PreToolUse: context-aware boot on first tool call ────────────────
if [ "$EVENT" = "PreToolUse" ] && [ ! -f "$BOOTED" ]; then
    touch "$BOOTED"

    ensure_axil_db || exit 0

    # Build context-aware boot flags from previous session's manifest
    BOOT_FLAGS="--boot-format narrative --budget 800"
    PREV_MANIFEST="/tmp/axil-session-prev.manifest"
    if [ -f "$PREV_MANIFEST" ]; then
        FILE_LIST=$(sort -u "$PREV_MANIFEST" | head -5 | tr '\n' ',' | sed 's/,$//')
        if [ -n "$FILE_LIST" ]; then
            BOOT_FLAGS="$BOOT_FLAGS --files $FILE_LIST"
        fi
    fi

    # Show branded brain banner with session stats
    $AXIL --db "$DB" brain-banner 2>/dev/null || true

    BOOT=$($AXIL --db "$DB" boot $BOOT_FLAGS 2>/dev/null || echo "")
    if [ -n "$BOOT" ]; then
        echo "$BOOT" >&2
    fi

    # Opportunistic SCIP refresh — only if the existing index is older
    # than 14 days (matches doctor's warning threshold) or missing.
    # Runs in the background so the user isn't blocked by rust-analyzer.
    # The lock at .axil/scip-refresh.lock prevents concurrent spawns.
    # Errors are silent: the brain hook must never break the agent loop.
    $AXIL --db "$DB" scip refresh --if-stale --in-background --quiet >/dev/null 2>&1 || true

    # Opportunistic time-gated maintenance — runs `axil snapshot` and
    # `health-report --save` only when their cadence ([maintenance] in
    # axil.toml) has elapsed. Cheap when fresh; detached so it never blocks;
    # lock at .axil/maintain.lock. Only additive tasks auto-run; destructive
    # downsampling and reindex stay explicit (`axil heal`).
    $AXIL --db "$DB" maintain --if-stale --in-background --quiet >/dev/null 2>&1 || true

    # Fall through to per-tool handlers below — if the first tool of the
    # session is Edit/Write, the PreToolUse Edit/Write block needs to run
    # so file-recall context is injected before that first edit.
fi

# ── PreToolUse (Edit/Write): inject relevant memories + edit-count nudge ──
# Surfaces past memories about the file BEFORE the agent edits it, via
# hookSpecificOutput.additionalContext (the only PreToolUse channel the model
# actually sees). Combines the file-recall and the 5-edit "have you stored
# anything?" nudge into one JSON output so they don't fight for stdout.
if [ "$EVENT" = "PreToolUse" ] && { [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; }; then
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    if [ -n "$FILE_PATH" ] && ! is_skipped_path "$FILE_PATH" && ensure_axil_db; then
        REL_PATH="${FILE_PATH#${CLAUDE_PROJECT_DIR:-$(pwd)}/}"
        # Per-file sentinel: only run recall-for-file once per file per session.
        # Avoids repeated vector+graph queries on multi-edit refactors of the
        # same file (each query was 50-200ms blocking the tool call).
        PATH_HASH=$(echo "$REL_PATH" | cksum | cut -d' ' -f1)
        RECALLED_SENTINEL="/tmp/axil-session-${SESSION_ID}.recalled-${PATH_HASH}"
        CTX=""

        if [ ! -f "$RECALLED_SENTINEL" ]; then
            RECALL=$("$AXIL" --db "$DB" recall-for-file "$REL_PATH" --top-k 3 2>/dev/null || echo "")
            MATCH_COUNT=$(echo "$RECALL" | jq -r '.matches // 0' 2>/dev/null || echo 0)
            if [ "$MATCH_COUNT" -gt 0 ]; then
                SUMMARIES=$(echo "$RECALL" | jq -r '.results[] | "  • [\(.table)] \(.summary)"' 2>/dev/null || echo "")
                if [ -n "$SUMMARIES" ]; then
                    CTX="📎 AXIL — past memories about ${REL_PATH} (read these before editing):
$SUMMARIES"
                fi
            fi
            touch "$RECALLED_SENTINEL"
        fi

        # 5-edit nudge: append if 5+ files touched and no narrative yet.
        if [ -f "$MANIFEST" ]; then
            EDIT_COUNT=$(wc -l < "$MANIFEST" 2>/dev/null | tr -d ' ')
            if [ "$EDIT_COUNT" -ge 5 ] && [ $(( EDIT_COUNT % 5 )) -eq 0 ] && [ "$(count_recent_narrative)" = "0" ]; then
                NUDGE="⚠️ AXIL — ${EDIT_COUNT} files edited this session, no ${NARRATIVE_TABLES_TEXT} stored. Store inline (axil store …) or commit; don't batch at the end."
                if [ -n "$CTX" ]; then
                    CTX="$CTX

$NUDGE"
                else
                    CTX="$NUDGE"
                fi
            fi
        fi

        if [ -n "$CTX" ]; then
            jq -n --arg ctx "$CTX" \
                '{hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$ctx}}'
        fi
    fi
fi

# ── Shared: read heartbeat counter ───────────────────────────────────
read_count() {
    local key="$1"
    if [ -f "$COUNTS" ]; then
        jq -r ".${key} // 0" "$COUNTS" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# ── PreToolUse (Bash): pair repo search with Axil recall/search context ─
# When the agent reaches for broad repo search, run Axil's structural or
# full-text index first and inject the compact result. If no query can be
# extracted, inject a gate reminder when the session has not recalled yet.
if [ "$EVENT" = "PreToolUse" ] && [ "$TOOL_NAME" = "Bash" ]; then
    BASH_CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
    IS_REPO_SEARCH=0
    QUERY=""
    case "$BASH_CMD" in
        *"rg "*|*"grep "*|*"git grep "*|*"fd "*|*"find "*)
            IS_REPO_SEARCH=1
            # Try double-quoted arg first, then single-quoted.
            QUERY=$(echo "$BASH_CMD" | grep -oE '"[^"]+"' | head -1 | tr -d '"' || true)
            if [ -z "$QUERY" ]; then
                QUERY=$(echo "$BASH_CMD" | grep -oE "'[^']+'" | head -1 | tr -d "'" || true)
            fi
            if [ -z "$QUERY" ]; then
                QUERY=$(printf "%s\n" "$BASH_CMD" \
                    | sed -E 's/.*(rg|grep|git grep|fd|find)[[:space:]]+//' \
                    | awk '{for (i=1; i<=NF; i++) if ($i !~ /^-/ && $i != "." && $i != "./") {print $i; exit}}' \
                    | sed 's/[;&|].*$//' || true)
            fi
            ;;
        *"ls "*|*"tree "*)
            IS_REPO_SEARCH=1
            ;;
    esac
    if [ -n "$QUERY" ] && [ "${#QUERY}" -lt 3 ]; then
        QUERY=""
    fi

    if [ "$IS_REPO_SEARCH" = "1" ] && [ -z "$QUERY" ] && [ "$(read_count recalls)" = "0" ]; then
        CTX="⚠️ AXIL search gate — this session has not used Axil recall yet. Before broad repo discovery, run one of:
  axil recall \"<what you need>\" --top-k 5
  axil code-search \"<symbol/module/API>\" --top-k 5
  axil fts \"<exact term>\" --limit 5

Then open the files Axil returns and verify current code."
        jq -n --arg ctx "$CTX" \
            '{hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$ctx}}'
    elif [ -n "$QUERY" ] && [ "${#QUERY}" -ge 3 ] && ensure_axil_db; then
        MODE="fts"
        SEARCH_CMD="fts"
        if echo "$QUERY" | grep -qE '^(fn |impl |struct |trait |pub |async |mod |use )|_|::|[a-z][A-Z]'; then
            MODE="code-search"
            SEARCH_CMD="code-search"
        fi

        QUERY_HASH=$(printf "%s:%s" "$MODE" "$QUERY" | cksum | cut -d' ' -f1)
        SEARCHED_SENTINEL="/tmp/axil-session-${SESSION_ID}.searched-${QUERY_HASH}"
        if [ ! -f "$SEARCHED_SENTINEL" ]; then
            if [ "$MODE" = "code-search" ]; then
                HITS=$("$AXIL" --db "$DB" code-search "$QUERY" --top-k 3 --format pretty 2>/dev/null || echo "")
            else
                HITS=$("$AXIL" --db "$DB" fts "$QUERY" --limit 3 --format table 2>/dev/null || echo "")
            fi

            HAS_HITS=1
            case "$HITS" in
                ""|"[]"|"(no results)"|"(no code proxies matched"*) HAS_HITS=0 ;;
            esac

            if [ "$HAS_HITS" = "1" ]; then
                touch "$SEARCHED_SENTINEL"
                CTX="📎 AXIL ${MODE}('${QUERY}') — check this before spending tokens on repo-wide search:
$HITS

For broad repo lookups, prefer 'axil ${SEARCH_CMD} <query>' first; use rg/grep after Axil points you at files or when verifying current text."
                jq -n --arg ctx "$CTX" \
                    '{hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$ctx}}'
            fi
        fi
    fi
fi

# ── PostToolUse (Read): fallback-capture after a recent empty_result ─
# When the agent reads a file shortly after an Axil recall/code-search/fts
# returned empty, attach a low-importance `context` row to the missed query
# pointing at the exact line range the agent opened. The next time the
# agent asks the same question, recall (Phase 13b structural pass) hits the
# proxy under that path:line range and surfaces the row — closing the
# miss→fallback loop.
#
# Tagged `_origin: fallback_capture` and `_importance: 0.2` so these rows
# sit below the default recall floor; they only surface for closely-matched
# follow-up queries. Tier promotion (#3) and dedup-by-task (#4) come later.
if [ "$EVENT" = "PostToolUse" ] && [ "$TOOL_NAME" = "Read" ]; then
    [ ! -f "$PROBLEMS" ] && exit 0

    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
    [ -z "$FILE_PATH" ] && exit 0
    is_skipped_path "$FILE_PATH" && exit 0

    # Find the most recent empty_result event in the last 5 minutes. ISO
    # 8601 strings sort lexically so a string compare on `at` is enough —
    # avoids macOS-vs-GNU date parsing differences.
    CUTOFF_ISO=$(date -u -v -5M +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null \
                 || date -u -d "5 minutes ago" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null \
                 || echo "")
    [ -z "$CUTOFF_ISO" ] && exit 0

    RECENT_MISS=$(awk -v cutoff="$CUTOFF_ISO" '
        /"kind":"empty_result"/ {
            if (match($0, /"at":"[^"]*"/) > 0) {
                at = substr($0, RSTART+6, RLENGTH-7)
                if (at >= cutoff) last = $0
            }
        }
        END { if (last) print last }
    ' "$PROBLEMS" 2>/dev/null)
    [ -z "$RECENT_MISS" ] && exit 0

    MISSED_QUERY=$(echo "$RECENT_MISS" | jq -r '.query // empty' 2>/dev/null)
    [ -z "$MISSED_QUERY" ] && exit 0

    OFFSET=$(echo "$INPUT" | jq -r '.tool_input.offset // 1' 2>/dev/null)
    LIMIT=$(echo "$INPUT" | jq -r '.tool_input.limit // 2000' 2>/dev/null)
    LINE_START=$OFFSET
    LINE_END=$((OFFSET + LIMIT - 1))
    REL_PATH="${FILE_PATH#${CLAUDE_PROJECT_DIR:-$(pwd)}/}"

    # Dedup: skip if (query, path, line range) already captured this session.
    KEY=$(printf "%s:%s:%s:%s" "$MISSED_QUERY" "$REL_PATH" "$LINE_START" "$LINE_END" \
          | cksum | cut -d' ' -f1)
    CAPTURED_SENTINEL="/tmp/axil-session-${SESSION_ID}.fallback-${KEY}"
    [ -f "$CAPTURED_SENTINEL" ] && exit 0

    if ensure_axil_db; then
        PAYLOAD=$(jq -cn \
            --arg type "fallback_capture" \
            --arg q "$MISSED_QUERY" \
            --arg path "$REL_PATH" \
            --argjson ls "$LINE_START" \
            --argjson le "$LINE_END" \
            --arg origin "fallback_capture" \
            --argjson imp 0.2 \
            '{type:$type, summary:("Fallback capture for query: " + $q), query:$q, code_refs:[{path:$path, line_start:$ls, line_end:$le}], _origin:$origin, _importance:$imp}' 2>/dev/null)
        if [ -n "$PAYLOAD" ]; then
            "$AXIL" --db "$DB" store context "$PAYLOAD" >/dev/null 2>&1 \
                && touch "$CAPTURED_SENTINEL" \
                && echo "🧠 Axil captured fallback: '${MISSED_QUERY}' → ${REL_PATH}:${LINE_START}-${LINE_END}" >&2
        fi
    fi
    exit 0
fi

# ── PostToolUse (Edit/Write): log file path + content snippet ────────
# Tracks the manifest and accumulates content for end-of-session entity
# extraction. File-recall context push lives in PreToolUse Edit/Write.
if [ "$EVENT" = "PostToolUse" ] && { [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; }; then
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    [ -z "$FILE_PATH" ] && exit 0
    is_skipped_path "$FILE_PATH" && exit 0

    REL_PATH="${FILE_PATH#${CLAUDE_PROJECT_DIR:-$(pwd)}/}"
    echo "$REL_PATH" >> "$MANIFEST"

    if [ "$TOOL_NAME" = "Edit" ]; then
        NEW_STR=$(echo "$INPUT" | jq -r '.tool_input.new_string // empty' 2>/dev/null)
        if [ -n "$NEW_STR" ]; then
            echo "$NEW_STR" | head -c 500 >> "$CONTENT"
            echo "" >> "$CONTENT"
        fi
    elif [ "$TOOL_NAME" = "Write" ]; then
        SNIPPET=$(echo "$INPUT" | jq -r '.tool_input.content // empty' 2>/dev/null | head -c 500)
        if [ -n "$SNIPPET" ]; then
            echo "$SNIPPET" >> "$CONTENT"
            echo "" >> "$CONTENT"
        fi
    fi

    exit 0
fi

# ── PostToolUse (Bash): heartbeat on axil store/recall + auto-capture on failures ─
if [ "$EVENT" = "PostToolUse" ] && [ "$TOOL_NAME" = "Bash" ]; then
    EXIT_CODE=$(echo "$INPUT" | jq -r '.tool_response.exitCode // .tool_response.exit_code // "0"' 2>/dev/null)
    BASH_CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

    # Heartbeat: the agent just interacted with its own brain.
    if [ "$EXIT_CODE" = "0" ] && [ -n "$BASH_CMD" ]; then
        case "$BASH_CMD" in
            *"axil store "*|*"axil observe "*|*"axil believe "*)
                bump_count stores
                echo "🧠 Axil stored (session: $(read_counts_compact))" >&2
                ;;
            *"axil recall"*|*"axil boot"*|*"axil recall-for-"*)
                bump_count recalls
                ;;
        esac

        # Capture git commits as narrative records — a commit message IS a
        # decision/summary the agent already wrote. Counts toward the Stop
        # guard so the agent doesn't have to re-state what's already in the
        # commit. Matches `git commit -m`, `--amend`, `-F`, etc.
        case "$BASH_CMD" in
            *"git commit "*|*"git commit -"*)
                if ensure_axil_db; then
                    REPO="${CLAUDE_PROJECT_DIR:-.}"
                    # One git log call with US-separator splits the headers;
                    # body fetched separately because it can contain newlines.
                    HEADERS=$(git -C "$REPO" log -1 --pretty=$'%H\x1f%s\x1f%an\x1f%cI' 2>/dev/null)
                    SHA=$(echo "$HEADERS" | cut -d$'\x1f' -f1)
                    if [ -n "$SHA" ]; then
                        SUBJECT=$(echo "$HEADERS" | cut -d$'\x1f' -f2)
                        AUTHOR=$(echo "$HEADERS" | cut -d$'\x1f' -f3)
                        COMMITTED_AT=$(echo "$HEADERS" | cut -d$'\x1f' -f4)
                        BODY=$(git -C "$REPO" log -1 --pretty=%b 2>/dev/null)
                        FILES_JSON=$(git -C "$REPO" diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null \
                            | jq -R -s 'split("\n") | map(select(length > 0))' 2>/dev/null || echo "[]")

                        PAYLOAD=$(jq -n \
                            --arg sha "$SHA" \
                            --arg subject "$SUBJECT" \
                            --arg body "$BODY" \
                            --arg author "$AUTHOR" \
                            --arg ts "$COMMITTED_AT" \
                            --argjson files "$FILES_JSON" \
                            '{sha:$sha, subject:$subject, body:$body, author:$author, committed_at:$ts, files:$files}' 2>/dev/null)

                        if [ -n "$PAYLOAD" ]; then
                            "$AXIL" --db "$DB" store commits "$PAYLOAD" >/dev/null 2>&1 || true
                            bump_count stores
                            echo "🧠 Axil captured commit ${SHA:0:7}: ${SUBJECT}" >&2
                        fi
                    fi
                fi
                ;;
        esac
    fi

    if [ "$EXIT_CODE" != "0" ]; then
        bump_count errors
        if ensure_axil_db; then
            # Extract stdout/stderr from the tool response (first 2000 chars)
            OUTPUT=$(echo "$INPUT" | jq -r '.tool_response.stdout // .tool_response.output // empty' 2>/dev/null | head -c 2000)
            if [ -n "$OUTPUT" ]; then
                # Auto-capture with high threshold to avoid noise
                echo "$OUTPUT" | "$AXIL" --db "$DB" auto-capture - --min-confidence 0.8 --source bash 2>/dev/null || true
            fi
        fi

        # Generic build/test failures already flow through auto-capture above;
        # only record axil-specific failures here so session-heal can act on them.
        case "$BASH_CMD" in
            *axil\ *)
                STDERR=$(echo "$INPUT" | jq -r '.tool_response.stderr // empty' 2>/dev/null | head -c 500)
                SUBCMD=$(extract_axil_subcmd "$BASH_CMD")
                TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                EVENT_JSON=$(jq -cn \
                    --arg kind "command_failure" \
                    --arg sub "$SUBCMD" \
                    --arg q "$BASH_CMD" \
                    --argjson exit "${EXIT_CODE:-1}" \
                    --arg err "$STDERR" \
                    --arg at "$TS" \
                    '{kind:$kind, subcommand:$sub, query:$q, exit_code:$exit, stderr:$err, at:$at}' 2>/dev/null)
                [ -n "$EVENT_JSON" ] && log_problem "$EVENT_JSON"
                ;;
        esac
    else
        # axil read commands return 0 with empty output when nothing matched.
        # A session full of these tells session-heal that the index is stale
        # or memory is sparse for the topics being asked.
        case "$BASH_CMD" in
            *"axil recall "*|*"axil code-search "*|*"axil fts "*|*"axil recall-for-file "*|*"axil recall-for-entity "*)
                STDOUT=$(echo "$INPUT" | jq -r '.tool_response.stdout // .tool_response.output // empty' 2>/dev/null)
                if is_empty_axil_output "$STDOUT"; then
                    SUBCMD=$(extract_axil_subcmd "$BASH_CMD")
                    # Pull the first quoted argument as the query (best-effort).
                    QRY=$(echo "$BASH_CMD" | grep -oE '"[^"]+"' | head -1 | tr -d '"' || true)
                    [ -z "$QRY" ] && QRY=$(echo "$BASH_CMD" | grep -oE "'[^']+'" | head -1 | tr -d "'" || true)
                    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                    EVENT_JSON=$(jq -cn \
                        --arg kind "empty_result" \
                        --arg sub "$SUBCMD" \
                        --arg q "$QRY" \
                        --arg at "$TS" \
                        '{kind:$kind, subcommand:$sub, query:$q, at:$at}' 2>/dev/null)
                    [ -n "$EVENT_JSON" ] && log_problem "$EVENT_JSON"
                fi
                ;;
        esac
    fi
    exit 0
fi

# ── Stop: enforce narrative-store guard, then session summary ────────
if [ "$EVENT" = "Stop" ]; then
    # If the harness already forced us to continue once for this stop event,
    # don't block again — let the agent through to avoid an infinite loop.
    STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)

    # Read-only session: no edits this turn, but the agent may still have
    # accumulated empty-result misses or axil command failures in $PROBLEMS.
    # Run session-heal on those before cleanup so recall/code-search-only
    # sessions still drive auto-fix and _heal_log entries.
    if [ ! -f "$MANIFEST" ]; then
        if [ -f "$PROBLEMS" ] && ensure_axil_db; then
            run_session_heal_with_autofix
        fi
        cleanup_session_files
        exit 0
    fi

    if ! ensure_axil_db; then
        cleanup_session_files
        exit 0
    fi

    # Deduplicate and format the file list (needed by both guard and summary).
    FILES=$(sort -u "$MANIFEST" | jq -R -s 'split("\n") | map(select(length > 0))')
    FILE_COUNT=$(echo "$FILES" | jq 'length')

    # ── Guard: block stop if substantive work happened with no narrative ──
    # Substantive = >2 distinct files edited this turn AND no narrative row
    # stored in the last hour. Returning {"decision":"block","reason":...}
    # on stdout is the only channel the harness re-injects into the model
    # on the next turn — stderr is invisible.
    if [ "$STOP_HOOK_ACTIVE" != "true" ] && [ "$FILE_COUNT" -gt 2 ] && [ "$(count_recent_narrative)" = "0" ] && ! has_recent_git_commit; then
        REASON="Axil brain: ${FILE_COUNT} files were edited this turn but no ${NARRATIVE_TABLES_TEXT} row was stored in the last hour (and no git commit). Before stopping, either: (a) commit the work — the commit message is captured as narrative — or (b) run: axil checkpoint '{\"state\":\"<where things stand>\",\"next_steps\":[\"<remaining work>\"],\"references\":[{\"kind\":\"file\",\"ref\":\"<path>\"}]}' (files touched this turn: ${FILES}). After storing, you may stop."
        jq -n --arg reason "$REASON" '{decision:"block", reason:$reason}'
        # Fall through to cleanup. If the harness honors block (sync mode,
        # the default after axil install ≥0.7.1), the agent continues and a
        # second Stop fires later — we'll cleanup again then. If the harness
        # ignores block (stale settings.json with async:true, or hand-edited),
        # this fall-through prevents temp-file leak.
    fi

    # ── Actually stopping: write _sessions, run worker, cleanup ──
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Extract entities from accumulated content
    ENTITIES="[]"
    if [ -f "$CONTENT" ] && [ -s "$CONTENT" ]; then
        ENTITIES=$(head -c 4000 "$CONTENT" | $AXIL extract-entities - 2>/dev/null || echo "[]")
    fi
    ENTITY_COUNT=$(echo "$ENTITIES" | jq 'length' 2>/dev/null || echo 0)

    # Write session record
    STORE_RESULT=$($AXIL --db "$DB" store _sessions "{
        \"session\": \"$SESSION_ID\",
        \"files_changed\": $FILES,
        \"file_count\": $FILE_COUNT,
        \"entities\": $ENTITIES,
        \"entity_count\": $ENTITY_COUNT,
        \"ended_at\": \"$TS\"
    }" 2>/dev/null || echo "{}")

    # Auto-link session record
    SESSION_RECORD_ID=$(echo "$STORE_RESULT" | jq -r '.id // empty' 2>/dev/null)
    if [ -n "$SESSION_RECORD_ID" ]; then
        $AXIL --db "$DB" auto-link "$SESSION_RECORD_ID" 2>/dev/null || true
    fi

    # Run worker (consolidation, connections, inference, decay)
    $AXIL --db "$DB" worker run 2>/dev/null || true

    # Auto-generate beliefs from high-importance facts
    $AXIL --db "$DB" beliefs --generate 2>/dev/null || true

    # Replay session failures + run auto-fixes. session-heal always inspects
    # detect_problems() even if PROBLEMS is missing/empty, so the brain
    # opportunistically heals every session, not just ones with explicit misses.
    # The helper also auto-spawns `axil index` in background when the structural
    # index looks stale based on session misses.
    run_session_heal_with_autofix

    # Save manifest for next session's context-aware boot. cleanup_session_files
    # below would remove it, so copy first.
    cp "$MANIFEST" /tmp/axil-session-prev.manifest 2>/dev/null || true

    # Session heartbeat summary (only when we actually interacted with the brain).
    if [ -f "$COUNTS" ]; then
        STORES=$(jq -r '.stores // 0' "$COUNTS" 2>/dev/null || echo 0)
        RECALLS=$(jq -r '.recalls // 0' "$COUNTS" 2>/dev/null || echo 0)
        TOOLS=$(jq -r '.tools // 0' "$COUNTS" 2>/dev/null || echo 0)
        ERRORS=$(jq -r '.errors // 0' "$COUNTS" 2>/dev/null || echo 0)
        if [ "$STORES" != "0" ] || [ "$RECALLS" != "0" ]; then
            echo "🧠 Axil session: ${STORES} stored ∙ ${RECALLS} recalled ∙ ${TOOLS} tools ∙ ${ERRORS} errors" >&2
        fi
    fi

    cleanup_session_files
fi

exit 0
