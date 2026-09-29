#!/usr/bin/env bash
# scripts/daemon-latency.sh — wall-clock latency of the agent hooks.
#
# Feeds synthetic Claude Code hook payloads to `axil hook run --dialect claude`
# and times each whole hook process, the way the harness waits on it:
#
#   user_prompt_recall   UserPromptSubmit  -> recall --recall-format context-block
#   session_start_boot   SessionStart      -> boot --boot-format narrative
#   pre_edit_file_recall PreToolUse Edit   -> recall-for-file
#   pre_bash_code_search PreToolUse Bash   -> code-search   (identifier query)
#   pre_bash_fts         PreToolUse Bash   -> fts           (phrase query)
#   stop_since           Stop              -> since 1h      (3 files edited)
#   hook_floor           UserPromptSubmit with a 3-char prompt: no lookup, so
#                        this is the cost of the hook process alone
#
# Every sample uses a fresh session id, so no per-session sentinel skips a
# lookup, and every lookup is checked to have run (a busy database would
# otherwise look fast). A prompt recall that hits the hook's own 1800 ms
# deadline injects nothing; it still counts, as a `deadline_hits` sample. The database is a COPY of --snapshot inside a scratch
# project whose tree is `git archive HEAD` of this checkout, so recall's
# freshness scan walks and hashes a real source tree. The snapshot is only
# read. SessionStart fires `scip refresh` and `maintain` in the background;
# their lock files are refreshed before each sample so those children skip at
# once instead of indexing or holding the database into the next sample (the
# hook never waits on them, so this does not shorten what is measured).
#
# With the freshness_timing example built, it also times the freshness scan
# every recall and boot run (stale_file_paths / check_freshness) on the same
# copy and tree.
#
# Usage:
#   scripts/daemon-latency.sh [-n 50] [--out results.json] [--snapshot DIR]
#                             [--label TEXT] [--cases name,name]
#   AXIL_BIN=target/release/axil scripts/daemon-latency.sh ...
#
# Build first (the default binary and example are the release builds):
#   cargo build --release -p axildb --bin axil
#   cargo build --release -p axil-indexer --example freshness_timing
#
# Exit: 0 all samples ran their lookup; 1 setup error; 3 a lookup failed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N=50
OUT=""
LABEL=""
ONLY=""
SNAPSHOT="$ROOT/benchmarks/dogfood-recall/data/snap-prefix"
AXIL_BIN="${AXIL_BIN:-$ROOT/target/release/axil}"
FRESHNESS_BIN="${FRESHNESS_BIN:-$ROOT/target/release/examples/freshness_timing}"

while [ $# -gt 0 ]; do
    case "$1" in
        -n) N="$2"; shift 2 ;;
        --cases) ONLY="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --snapshot) SNAPSHOT="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "[daemon-latency] unknown argument: $1" >&2; exit 1 ;;
    esac
done

die() { echo "[daemon-latency] ERROR: $*" >&2; exit 1; }
[ -x "$AXIL_BIN" ] || die "AXIL_BIN=$AXIL_BIN is not executable (build it first)"
[ -f "$SNAPSHOT/memory.axil" ] || die "no memory.axil in snapshot $SNAPSHOT"
command -v python3 >/dev/null || die "python3 is required"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/axil-hook-latency.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
WORK="$(cd "$WORK" && pwd -P)"
PROJ="$WORK/proj"
mkdir -p "$PROJ/.axil" "$WORK/tmp"

echo "[daemon-latency] project tree: git archive HEAD -> $PROJ" >&2
git -C "$ROOT" archive HEAD | tar -x -C "$PROJ"
echo "[daemon-latency] copying snapshot $SNAPSHOT" >&2
for f in "$SNAPSHOT"/memory.axil*; do
    cp -R "$f" "$PROJ/.axil/"
done

AXIL_BIN="$AXIL_BIN" FRESHNESS_BIN="$FRESHNESS_BIN" PROJ="$PROJ" \
HOOK_TMP="$WORK/tmp" N="$N" ONLY="$ONLY" OUT="$OUT" LABEL="$LABEL" SNAPSHOT="$SNAPSHOT" \
ROOT="$ROOT" python3 <<'PY'
import json, math, os, platform, statistics, subprocess, sys, time
from datetime import datetime, timezone

axil = os.environ["AXIL_BIN"]
proj = os.environ["PROJ"]
hook_tmp = os.environ["HOOK_TMP"]
n = int(os.environ["N"])
root = os.environ["ROOT"]
snapshot = os.environ["SNAPSHOT"]
db = os.path.join(proj, ".axil", "memory.axil")

env = dict(os.environ)
env["CLAUDE_PROJECT_DIR"] = proj
env["TMPDIR"] = hook_tmp  # the hook's per-session files land here
for k in ("AXIL_DB", "AXIL_BIN", "FRESHNESS_BIN"):
    env.pop(k, None)


def fnv1a(s):
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "x")


def sfile(sid, suffix):
    return os.path.join(hook_tmp, f"axil-session-{sid}.{suffix}")


def touch(path):
    with open(path, "a"):
        pass
    os.utime(path, None)


PROMPTS = [
    "Why is the published crate named axildb when the binary is called axil?",
    "How do we make local builds link onnxruntime statically like the release binaries?",
    "What does the hook drainer do when the database is busy?",
    "Why does vector search scan exactly instead of building the HNSW graph?",
    "How does recall decide which files are stale in the index?",
]
# The hook passes `--timeout-ms 1800` to its prompt recall.
RECALL_DEADLINE_MS = 1800
EDIT_FILE = os.path.join(proj, "crates/adapters/axil-cli/src/hook_brain.rs")
EDIT_REL = "crates/adapters/axil-cli/src/hook_brain.rs"
CODE_QUERY = "stale_file_paths"
FTS_QUERY = "drain lock"
STOP_FILES = [
    "crates/adapters/axil-cli/src/main.rs",
    "crates/adapters/axil-cli/src/hook_brain.rs",
    "crates/extensions/axil-indexer/src/freshness.rs",
]


def case_prompt(sid, i):
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": sid,
               "cwd": proj, "prompt": PROMPTS[i % len(PROMPTS)]}
    return payload, lambda out: bool(out.strip())


def case_floor(sid, i):
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": sid,
               "cwd": proj, "prompt": "ok!"}
    return payload, lambda out: out.strip() == ""


def case_boot(sid, i):
    # Keep the background scip/maintain children to a lock-check skip.
    for lock in ("maintain.lock", "scip-refresh.lock"):
        touch(os.path.join(proj, ".axil", lock))
    payload = {"hook_event_name": "SessionStart", "session_id": sid,
               "cwd": proj, "source": "startup"}
    return payload, lambda out: "additionalContext" in out and "Boot Context" in out


def pre_tool(sid, tool_name, tool_input):
    touch(sfile(sid, "booted"))  # past the session's first-tool boot
    return {"hook_event_name": "PreToolUse", "session_id": sid, "cwd": proj,
            "tool_name": tool_name, "tool_input": tool_input}


def case_edit(sid, i):
    payload = pre_tool(sid, "Edit", {"file_path": EDIT_FILE,
                                     "old_string": "a", "new_string": "b"})
    sentinel = sfile(sid, f"recalled-{fnv1a(EDIT_REL)}")
    return payload, lambda out: os.path.exists(sentinel)


def case_code_search(sid, i):
    payload = pre_tool(sid, "Bash", {"command": f"rg -n {CODE_QUERY} crates/"})
    sentinel = sfile(sid, f"searched-{fnv1a('code-search:' + CODE_QUERY)}")
    return payload, lambda out: os.path.exists(sentinel)


def case_fts(sid, i):
    payload = pre_tool(sid, "Bash", {"command": f"grep -rn '{FTS_QUERY}' docs/"})
    sentinel = sfile(sid, f"searched-{fnv1a('fts:' + FTS_QUERY)}")
    return payload, lambda out: os.path.exists(sentinel)


def case_stop(sid, i):
    # Three files edited this turn and no narrative in the last hour (the
    # snapshot is older than that): the guard runs `since 1h` and blocks.
    with open(sfile(sid, "manifest"), "w") as f:
        f.write("\n".join(STOP_FILES) + "\n")
    payload = {"hook_event_name": "Stop", "session_id": sid, "cwd": proj,
               "stop_hook_active": False}
    return payload, lambda out: '"decision":"block"' in out


ALL_CASES = [
    ("hook_floor", "UserPromptSubmit (no lookup)", case_floor),
    ("user_prompt_recall", "UserPromptSubmit -> recall context-block", case_prompt),
    ("session_start_boot", "SessionStart -> boot narrative", case_boot),
    ("pre_edit_file_recall", "PreToolUse Edit -> recall-for-file", case_edit),
    ("pre_bash_code_search", "PreToolUse Bash -> code-search", case_code_search),
    ("pre_bash_fts", "PreToolUse Bash -> fts", case_fts),
    ("stop_since", "Stop -> since 1h", case_stop),
]
only = [c for c in os.environ.get("ONLY", "").split(",") if c]
CASES = [c for c in ALL_CASES if not only or c[0] in only]


def run_hook(payload):
    body = json.dumps(payload).encode()
    t = time.perf_counter()
    p = subprocess.run([axil, "hook", "run", "--dialect", "claude"], input=body,
                       capture_output=True, env=env, cwd=proj)
    ms = (time.perf_counter() - t) * 1000.0
    return ms, p.returncode, p.stdout.decode(errors="replace")


def pct(sorted_vals, q):
    """Nearest-rank percentile."""
    idx = min(len(sorted_vals), max(1, math.ceil(len(sorted_vals) * q)))
    return sorted_vals[idx - 1]


# Warm-up: first opens of the copy pay one-time costs (page cache, any
# migration); two unmeasured rounds of every case take them out.
seq = 0
for _ in range(2):
    for name, _, make in CASES:
        seq += 1
        payload, _ = make(f"warm-{name}-{seq}", seq)
        run_hook(payload)

load_before = os.getloadavg()
results = {}
failures = 0
for name, desc, make in CASES:
    samples, bad, deadline_hits = [], [], 0
    for i in range(n):
        seq += 1
        payload, ok = make(f"bench-{name}-{seq}", i)
        ms, code, out = run_hook(payload)
        if (name == "user_prompt_recall" and code == 0 and not out.strip()
                and ms >= RECALL_DEADLINE_MS):
            # The recall ran into its own --timeout-ms and the hook injected
            # nothing. That is a real outcome the user waited for: keep it.
            deadline_hits += 1
            samples.append(ms)
            continue
        if code != 0 or not ok(out):
            bad.append({"sample": i, "ms": round(ms, 1), "exit": code})
            continue
        samples.append(ms)
    failures += len(bad)
    s = sorted(samples)
    results[name] = {
        "path": desc,
        "ok": len(samples),
        "failed": len(bad),
        "deadline_hits": deadline_hits,
        "p50_ms": round(pct(s, 0.50), 1) if s else None,
        "p95_ms": round(pct(s, 0.95), 1) if s else None,
        "mean_ms": round(statistics.fmean(s), 1) if s else None,
        "min_ms": round(s[0], 1) if s else None,
        "max_ms": round(s[-1], 1) if s else None,
    }
    if bad:
        results[name]["failed_samples"] = bad
    r = results[name]
    print(f"[daemon-latency] {name:22s} p50={r['p50_ms']}ms p95={r['p95_ms']}ms "
          f"ok={r['ok']} deadline_hits={deadline_hits} failed={len(bad)}",
          file=sys.stderr)
load_after = os.getloadavg()

freshness = None
fbin = os.environ.get("FRESHNESS_BIN", "")
if fbin and os.access(fbin, os.X_OK):
    p = subprocess.run([fbin, db, proj, "20"], capture_output=True, env=env)
    if p.returncode == 0:
        freshness = json.loads(p.stdout)
        freshness["method"] = ("crates/extensions/axil-indexer/examples/freshness_timing.rs "
                               "on the same copy and tree: the calls recall "
                               "(stale_file_paths) and boot (check_freshness) make")
    else:
        print(f"[daemon-latency] freshness_timing failed: {p.stderr.decode()}",
              file=sys.stderr)


recall_p50 = results.get("user_prompt_recall", {}).get("p50_ms")
if freshness and recall_p50:
    scan = freshness["stale_file_paths_ms"]["p50"]
    freshness["share_of_prompt_hook_p50"] = round(scan / recall_p50, 3)


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


machine = platform.platform()
if platform.system() == "Darwin":
    machine = (f"{sh('sysctl', '-n', 'machdep.cpu.brand_string')}, "
               f"{sh('sysctl', '-n', 'hw.ncpu')} cores, "
               f"{int(sh('sysctl', '-n', 'hw.memsize') or 0) // 2**30} GiB, "
               f"macOS {platform.mac_ver()[0]}")
tree_files = 0
for dirpath, dirnames, filenames in os.walk(proj):
    dirnames[:] = [d for d in dirnames if d != ".axil"]
    tree_files += len(filenames)
snap_bytes = {f: os.path.getsize(os.path.join(snapshot, f))
              for f in sorted(os.listdir(snapshot))
              if os.path.isfile(os.path.join(snapshot, f))}
snap_real = os.path.realpath(snapshot)
snap_label = ("benchmarks/" + snap_real.rsplit("/benchmarks/", 1)[1]
              if "/benchmarks/" in snap_real else snap_real)
build = "release" if "/release/" in os.path.realpath(axil) else (
    "debug" if "/debug/" in os.path.realpath(axil) else "unknown")

report = {
    "benchmark": "agent hook wall-clock latency, one-shot (every lookup is a fresh axil process)",
    "label": os.environ.get("LABEL") or "one-shot baseline (no daemon)",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": sh("git", "-C", root, "rev-parse", "--short", "HEAD"),
    "axil_version": sh(axil, "--version"),
    "build": build,
    "machine": machine,
    "load_average_1m": {"before": round(load_before[0], 2), "after": round(load_after[0], 2)},
    "corpus": {
        "database": f"copy of {snap_label} (frozen copy of this repo's .axil)",
        "files_bytes": snap_bytes,
        "project_tree": f"git archive of HEAD, {tree_files} files",
    },
    "method": {
        "command": "axil hook run --dialect claude, payload on stdin",
        "samples_per_case": n,
        "timing": "whole hook process wall clock, python perf_counter around subprocess.run",
        "warmup": "2 unmeasured rounds of every case",
        "session_ids": "fresh per sample, so no per-session sentinel skips a lookup",
        "validation": ("a sample counts only if its lookup ran (output or sentinel "
                       "checked); a prompt recall that returns nothing after its "
                       "1800 ms --timeout-ms counts, as a deadline_hit"),
    },
    "cases": results,
    "freshness_scan": freshness,
}
text = json.dumps(report, indent=2) + "\n"
out = os.environ.get("OUT")
if out:
    with open(out, "w") as f:
        f.write(text)
    print(f"[daemon-latency] wrote {out}", file=sys.stderr)
else:
    sys.stdout.write(text)
sys.exit(3 if failures else 0)
PY
