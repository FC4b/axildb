#!/usr/bin/env python3
"""Harvest the U3 (capture classification) texts: what auto-capture is shown.

Writes sets/u3_texts.jsonl: about 200 unlabeled texts. Gold labels (class
decision / error / context / rule / none, plus an importance bucket) are
added later by the annotators, following validation/u3-guidelines.md.

`axil auto-capture` (auto_capture::analyze) is fed by the brain hook: when a
shell command exits non-zero, Claude Code's PostToolUseFailure event carries
the failure text ("Exit code N" and the command's output) in `tool_error`,
and the hook queues it, cut to 2,000 bytes, for
`axil auto-capture - --min-confidence 0.8 --source bash`. The live DB holds
no auto-captured record (none passed that bar), so the hook's input is
rebuilt from the Claude Code session transcripts of this repo, where every
tool result is kept. Five sources, one stratum each:

  bash_failure    Bash results that start "Exit code N": the tool ran and
                  exited non-zero, so this is exactly the text the hook pipes
                  to auto-capture (refusals by a PreToolUse hook never ran
                  and are skipped)
  bash_output     Bash results of commands that exited 0: tool output and
                  status lines, the negatives a regex over raw output meets
  agent_text      the agent's own prose (assistant text blocks), one
                  paragraph each: chatter, status lines, findings, decisions
  agent_brief     paragraphs of the briefs the agent wrote for subagents
                  (Agent tool prompts): constraints, plans, background
  commit_subject  subjects in the live DB `commits` table, which the hook
                  fills when the agent runs `git commit`

The user's own prompts are never read; workflow-computed tasks and relayed
user requests are skipped. Texts that mention the closed commercial product
(Atlas) are skipped, as in the U5 set.

Shaping. Shell output is first cut to the hook's 2,000 bytes, then, when it
is still longer than MAX_OUTPUT, to its first lines and last lines around one
"[... N lines cut ...]" marker line, so a failure printed at the end stays
visible; single lines are capped at MAX_LINE. Prose is cut at a sentence
boundary (the U5 `shorten`). Email addresses and token-shaped strings become
[REDACTED], home directories (/Users/<name>/) become ~/, and Claude Code
session temp paths (scratchpad, task output) become $SCRATCH/.

Sampling is deterministic. Only transcript entries timestamped before CUTOFF
and commits created before it are read, so a rerun on grown transcripts picks
the same pool. Within a source, half the quota goes to texts that contain one
of the production patterns (auto_capture.rs ERROR_PATTERNS or
DECISION_PATTERNS, matched as lowercase substrings) and half to texts that
contain none, each in sha256 order of the source id; a half with too few
candidates leaves its room to the other. This over-samples the texts the
regex fires on, so the set measures its false captures as well as its misses;
rates on it are not rates over all hook traffic. Neutral ids (u3-001 ...)
follow sha256 order of the source id; `split` is 40% train / 60% test from
sha256 of the neutral id, the rule the U1/U2/U5 sets use.

Claude Code deletes old transcripts, so the candidate pool can be saved and
the set rebuilt from it:

  python repo_sets_u3.py --db <path/to/memory.axil> [--save-pool out/u3_pool.jsonl]
  python repo_sets_u3.py --pool out/u3_pool.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from repo_sets_u5 import end_sentence, redact as redact_u5, shorten

HERE = Path(__file__).resolve().parent
OUT = HERE / "sets" / "u3_texts.jsonl"
TRANSCRIPTS = Path.home() / ".claude" / "projects" / "-Users-sek-Documents-GitHub-axil-project-axildb"

# Transcript entries and commits after this are ignored, so the pool is fixed.
CUTOFF = "2026-10-14T10:00:00Z"
HOOK_BYTES = 2000
MAX_OUTPUT = 700
MAX_LINE = 240
MAX_PROSE = 600
MIN_CHARS = 20

QUOTAS = {"bash_failure": 56, "bash_output": 30, "agent_text": 56, "agent_brief": 22, "commit_subject": 36}
SOURCES = list(QUOTAS)

# auto_capture.rs, matched against lowercased text.
ERROR_PATTERNS = ["error[e", "error:", "failed", "panicked at", "panic!", "exception:", "traceback",
                  "fatal", "segfault", "cannot find", "not found", "permission denied",
                  "connection refused", "timed out", "out of memory"]
DECISION_PATTERNS = ["decided to", "switching to", "adopted ", "going with", "let's use", "chose to ",
                     "decision:", "we picked ", "we settled on", "moved from "]

PRIVATE = re.compile(r"\batlas\b|axil-atlas|enterprise-design|\bpricing\b", re.I)
SCRATCH = re.compile(r"/private/tmp/claude-\d+/[^/\s]+/(?:[0-9a-f-]{36}/)?(?:scratchpad/)?")
HARNESS = re.compile(r"^\s*(?:\[Workflow harness|<system-reminder>|<task-notification>|Another Claude session)")
EXIT_LINE = re.compile(r"^Exit code (\d+)\b")


def split_of(item_id: str) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 4 else "test"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def triggers(text: str) -> list:
    low = text.lower()
    return [p for p in ERROR_PATTERNS + DECISION_PATTERNS if p in low]


# ── Shaping ─────────────────────────────────────────────────────────


def hook_cut(text: str) -> str:
    """The brain hook's truncate_utf8(stdout, 2000)."""
    raw = text.encode()
    if len(raw) <= HOOK_BYTES:
        return text
    return raw[:HOOK_BYTES].decode(errors="ignore")


def cap_line(line: str) -> str:
    return line if len(line) <= MAX_LINE else line[: MAX_LINE - 1].rstrip() + "…"


def shape_output(text: str) -> str:
    """Hook cut, then head and tail lines around one marker when still too long."""
    lines = [cap_line(ln.rstrip()) for ln in hook_cut(text).strip("\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if sum(len(ln) + 1 for ln in lines) <= MAX_OUTPUT:
        return "\n".join(lines)
    head, tail, used = [], [], 0
    budget_head, budget_tail = int(MAX_OUTPUT * 0.6), int(MAX_OUTPUT * 0.4)
    for ln in lines:
        if used + len(ln) + 1 > budget_head and head:
            break
        head.append(ln)
        used += len(ln) + 1
    used = 0
    for ln in reversed(lines[len(head):]):
        if used + len(ln) + 1 > budget_tail and tail:
            break
        tail.insert(0, ln)
        used += len(ln) + 1
    cut = len(lines) - len(head) - len(tail)
    if cut <= 0:
        return "\n".join(head + tail)
    return "\n".join(head + [f"[... {cut} lines cut ...]"] + tail)


def paragraphs(text: str) -> list:
    """Blank-line separated blocks; a code fence stays one block."""
    out, cur, fence = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
        if not line.strip() and not fence:
            if cur:
                out.append("\n".join(cur).strip())
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur).strip())
    return [p for p in out if p]


def redact(text: str, counter: Counter) -> str:
    text, n = SCRATCH.subn("$SCRATCH/", text)
    counter["scratch_path"] += n
    return redact_u5(text, counter)


# ── Reading ─────────────────────────────────────────────────────────


def block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def read_transcripts(root: Path) -> list:
    """Candidate texts from every transcript (sessions and subagents) under root."""
    cands = []
    for path in sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True)):
        stem = os.path.relpath(path, root)
        tools = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp") or ""
                msg = entry.get("message")
                if not ts or ts >= CUTOFF or not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                uuid = entry.get("uuid") or sha(line)[:16]
                for k, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if entry.get("type") == "assistant" and kind == "tool_use":
                        tools[block.get("id")] = block.get("name")
                        if block.get("name") == "Agent":
                            prompt = (block.get("input") or {}).get("prompt") or ""
                            for j, para in enumerate(paragraphs(prompt)):
                                cands.append(("agent_brief", f"{stem}:{uuid}:{k}:{j}", ts, para))
                    elif entry.get("type") == "assistant" and kind == "text":
                        for j, para in enumerate(paragraphs(block.get("text") or "")):
                            cands.append(("agent_text", f"{stem}:{uuid}:{k}:{j}", ts, para))
                    elif entry.get("type") == "user" and kind == "tool_result":
                        if tools.get(block.get("tool_use_id")) != "Bash":
                            continue
                        text = block_text(block.get("content"))
                        if block.get("is_error"):
                            if EXIT_LINE.match(text):
                                cands.append(("bash_failure", f"{stem}:{uuid}:{k}", ts, text))
                        elif text.strip():
                            cands.append(("bash_output", f"{stem}:{uuid}:{k}", ts, text))
    return cands


def read_commits(args) -> list:
    db = args.db or os.environ.get("AXIL_DB")
    if not db:
        sys.exit("pass --db <memory.axil> (or --pool <file>)")
    cmd = [args.axil, "export", "--db", db, "--tables", "commits", "--quiet"]
    lines = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    out = []
    for line in lines:
        r = json.loads(line) if line.strip() else {}
        if r.get("kind") != "record" or r.get("table") != "commits":
            continue
        created = (r.get("created_at") or "").replace("+00:00", "Z")
        subject = (r["data"].get("subject") or "").strip()
        if created < CUTOFF and subject:
            out.append(("commit_subject", r["id"], created, subject))
    return out


def shape(source: str, text: str):
    """The text the annotators see, or None to skip the candidate."""
    if source in ("bash_failure", "bash_output"):
        return shape_output(text)
    if HARNESS.match(text):
        return None
    if source == "commit_subject":
        return text
    return shorten(text, MAX_PROSE)


def norm_key(text: str) -> str:
    return re.sub(r"\W+", " ", re.sub(r"\d+", "0", text.lower()))[:160]


def build(cands: list, counter: Counter):
    pool = {s: {"trigger": [], "plain": []} for s in SOURCES}
    seen, skipped = set(), Counter()
    for source, sid, ts, raw in sorted(cands, key=lambda c: sha(c[1])):
        text = shape(source, raw)
        if text is None:
            skipped["harness_text"] += 1
            continue
        if len(text.strip()) < MIN_CHARS:
            skipped["short"] += 1
            continue
        if PRIVATE.search(raw):
            skipped["private"] += 1
            continue
        key = norm_key(text)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        text = redact(text, counter)
        half = "trigger" if triggers(text) else "plain"
        pool[source][half].append({"source_id": sid, "source": source, "created_at": ts, "text": text,
                                   "trigger": half == "trigger"})

    chosen = []
    for source in SOURCES:
        quota = QUOTAS[source]
        trig, plain = pool[source]["trigger"], pool[source]["plain"]
        n_trig = min(len(trig), quota - min(len(plain), quota // 2))
        n_plain = min(len(plain), quota - n_trig)
        if n_trig + n_plain < quota:
            print(f"note: {source} has {n_trig + n_plain} of {quota}", file=sys.stderr)
        chosen += trig[:n_trig] + plain[:n_plain]

    chosen.sort(key=lambda c: sha(c["source_id"]))
    items = []
    for n, c in enumerate(chosen, 1):
        item_id = f"u3-{n:03d}"
        items.append({
            "id": item_id,
            "source_id": c["source_id"],
            "source": c["source"],
            "split": split_of(item_id),
            "regex_trigger": c["trigger"],
            "text": c["text"],
        })
    return items, pool, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="Axil database to read commits from (read-only)")
    ap.add_argument("--axil", default="axil", help="axil binary (default: on PATH)")
    ap.add_argument("--transcripts", default=str(TRANSCRIPTS), help="Claude Code project transcript dir")
    ap.add_argument("--pool", help="rebuild from a saved candidate pool instead of transcripts + DB")
    ap.add_argument("--save-pool", help="also write the raw candidate pool here (keep it out of git)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.pool:
        cands = [tuple(json.loads(line)) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    else:
        cands = read_transcripts(Path(args.transcripts)) + read_commits(args)
    if args.save_pool:
        Path(args.save_pool).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_pool, "w") as fh:
            for c in cands:
                fh.write(json.dumps(list(c), ensure_ascii=False) + "\n")

    redactions = Counter()
    items, pool, skipped = build(cands, redactions)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(items)} texts to {out}")
    print("candidates per source:", dict(Counter(c[0] for c in cands)))
    print("pool per source (trigger/plain):",
          {s: (len(v["trigger"]), len(v["plain"])) for s, v in pool.items()})
    print("chosen per source:", dict(Counter(i["source"] for i in items)))
    print("chosen with a regex trigger:", sum(i["regex_trigger"] for i in items))
    print("split:", dict(Counter(i["split"] for i in items)))
    print("skipped:", dict(skipped))
    print("redactions:", dict(redactions) or "none")


if __name__ == "__main__":
    main()
