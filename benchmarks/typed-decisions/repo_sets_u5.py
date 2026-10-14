#!/usr/bin/env python3
"""Harvest the U5 (entity extraction) texts from this repo's own Axil memory.

Writes sets/u5_texts.jsonl: about 160 unlabeled texts, shortened from real
records in the dogfood `.axil`. Gold entity spans are added later by the
annotators, following validation/u5-guidelines.md.

Sources and how each record becomes one text:
  decisions            summary + reason
  errors               error + "Root cause:" root_cause + "Fix:" fix
  context              summary (auto-generated fallback captures skipped)
  rules                rule
  _checkpoint_records  the markdown "Resume Here" block `axil boot` renders
  commits              subject + body; used only for the markdown stratum,
                       since no other table stores bulleted text
Internal tables (_idx_*, _dep_*, _sessions, ...) are never read; _sessions
rows hold no summary text, so the checkpoints are the session summaries.
Superseded records, near-duplicate texts and records that mention the closed
commercial product (Atlas) are skipped.

Each text is cut to at most MAX_CHARS at a sentence (or bullet) boundary,
falling back to a clause boundary only when the first sentence alone is too
long. Multi-field records give each field a share of the room so the fix or
reason is not always cut. Hard-wrapped commit lines are rejoined (a line
break inside a sentence would otherwise split a name) and commit trailers
dropped. Email addresses and token-shaped strings become [REDACTED]; home
directories (/Users/<name>/) become ~/.

Every text gets exactly one stratum, in this priority order:
  markdown    two or more bullet/heading lines, or **bold** markup
  junk_trap   prose that a pattern extractor turns into junk: an English word
              after a declaration keyword ("a trait with"), keyword-as-word
              prose ("fn", "impl", "struct"), slash-separated numbers read as
              a path ("3.1/3.2"), plural acronyms the CamelCase rule mints
              ("IDs", "PRs", "GiB"), or apostrophes that pair up into a fake
              quoted string ("Axil's ... doesn't")
  code_heavy  six or more distinct code-shaped tokens (inline code, paths,
              `a::b`, snake_case, _tables, --flags, calls, env vars, #[attrs])
  prose       no code-shaped token and at most one CamelCase name
  mixed       everything else
Within a stratum, the scarce tables (rules, checkpoints) go first, then
records that mention Python, TypeScript/JS, Go or Java (this DB is mostly
Rust), then the rest in sha256 order. A stratum with fewer candidates than
its quota leaves room that junk_trap, code_heavy and mixed take in turn.

Sampling is deterministic: candidates are ordered by sha256 of the Axil
record id, and only records created before CUTOFF are read, so a rerun on a
grown DB picks the same pool. Neutral ids (u5-001 ...) follow sha256 order of
the source id across strata; `split` is 40% train / 60% test from sha256 of
the neutral id, the rule the U1/U2 sets use.

Usage:
  python repo_sets_u5.py --db <path/to/memory.axil>    # runs `axil export`
  python repo_sets_u5.py --export saved-export.jsonl   # rebuild from a saved export
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sets" / "u5_texts.jsonl"

# Records created after this are ignored, so the pool is fixed.
CUTOFF = "2026-10-14T05:00:00+00:00"
MAX_CHARS = 600
TABLES = ["decisions", "errors", "context", "rules", "_checkpoint_records", "commits"]

TARGET = 160
QUOTAS = {"markdown": 24, "junk_trap": 30, "code_heavy": 42, "prose": 28, "mixed": 36}
STRATUM_ORDER = ["markdown", "junk_trap", "code_heavy", "prose", "mixed"]
# Strata that absorb, one record each in turn, the room a short stratum leaves.
SPILL_ORDER = ["junk_trap", "code_heavy", "mixed"]
# Taken first within their stratum: the tables with only a handful of records,
# then records that mention a language other than Rust (rare in this DB).
SCARCE_TABLES = {"rules", "_checkpoint_records"}
RARE_LANGS = ["python", "ts", "go", "java"]
# Records about the closed commercial product stay out of this public set.
PRIVATE = re.compile(r"\batlas\b|axil-atlas|enterprise-design|\bpricing\b", re.I)
HOME_DIR = re.compile(r"/Users/[^/\s]+/")


def split_of(item_id: str) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 4 else "test"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── Reading the DB ──────────────────────────────────────────────────


def load_export(args) -> list:
    if args.export:
        lines = Path(args.export).read_text().splitlines()
    else:
        db = args.db or os.environ.get("AXIL_DB")
        if not db:
            sys.exit("pass --db <memory.axil> or --export <file>")
        cmd = [args.axil, "export", "--db", db, "--tables", ",".join(TABLES), "--quiet"]
        lines = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [r for r in rows if r.get("kind") == "record"]


# ── Text shaping ────────────────────────────────────────────────────

ABBREVIATIONS = {"e.g.", "i.e.", "vs.", "etc.", "cf.", "approx.", "no.", "ca."}
BULLET_START = re.compile(r"[ \t]*(?:[-*+] |\d+[.)] |#{1,6} |```)")
TRAILER = re.compile(r"^(?:Co-Authored-By|Signed-off-by|Reviewed-by):|^🤖 Generated with", re.I)


def boundaries(text: str) -> list:
    """Offsets right after a sentence end or before a new line item."""
    cuts = []
    for m in re.finditer(r"[.!?][\"')\]]*(?=\s|$)", text):
        words = text[: m.end()].split()
        if words and words[-1].lstrip("([{\"'").lower() in ABBREVIATIONS:
            continue
        rest = text[m.end():].lstrip()
        if rest and rest[0].islower():
            continue
        cuts.append(m.end())
    for m in re.finditer(r"\n", text):
        nxt = text[m.end():]
        if nxt.startswith("\n") or BULLET_START.match(nxt) or text[: m.start()].endswith("\n"):
            cuts.append(m.start())
    return sorted(set(cuts))


def drop_dangling_header(text: str) -> str:
    """Drop trailing list headers ("- **Next:**") whose items were cut."""
    lines = text.rstrip().split("\n")
    while len(lines) > 1 and re.search(r":(?:\*\*)?\s*$", lines[-1]) and BULLET_START.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def shorten(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    fits = [c for c in boundaries(text) if c <= limit]
    best = max(fits) if fits else 0
    if best >= limit * 0.4:
        return drop_dangling_header(text[:best])
    # The first sentence alone is longer than the room: cut at a clause.
    window = text[:limit]
    floor = max(best, int(limit * 0.4))
    for sep in ("; ", " — ", " – ", ": ", ", ", " ("):
        i = window.rfind(sep)
        if i > floor:
            cut = window[:i]
            # Do not leave a parenthesis open: cut before it instead.
            j = cut.rfind(" (")
            if cut.count("(") > cut.count(")") and j > floor:
                cut = cut[:j]
            return cut.rstrip(" ,;:—–(")
    i = window.rfind(" ")
    return window[: i if i > 0 else limit].rstrip(" ,;:—–(")


def unwrap(text: str) -> str:
    """Rejoin hard-wrapped lines; keep bullets, headings, fences, blank lines."""
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (
            out
            and stripped
            and out[-1].strip()
            and not BULLET_START.match(line)
            and not out[-1].lstrip().startswith("```")
        ):
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(line.rstrip())
    joined = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def end_sentence(s: str) -> str:
    s = s.rstrip()
    return s if not s or s[-1] in ".!?" else s + "."


EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
TOKEN = re.compile(
    r"\b(?:ghp|gho|ghu|ghs|github_pat|glpat|xox[abps])_?[A-Za-z0-9_]{16,}"
    r"|\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)
SECRET_KV = re.compile(r"(?i)\b((?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[=:]\s*)[^\s,;]{8,}")


def redact(text: str, counter: Counter) -> str:
    def sub(pattern, repl, s, kind):
        new, n = pattern.subn(repl, s)
        counter[kind] += n
        return new

    text = sub(EMAIL, "[REDACTED]", text, "email")
    text = sub(TOKEN, "[REDACTED]", text, "token")
    text = sub(SECRET_KV, r"\1[REDACTED]", text, "secret_kv")
    text = sub(HOME_DIR, "~/", text, "home_dir")
    return text


def render_checkpoint(d: dict) -> str:
    """The "Resume Here" block, as the checkpoint extension renders it."""
    out = ["## Resume Here"]
    if d.get("summary"):
        out.append(f"- {d['summary']}")
    if d.get("goal"):
        out.append(f"- **Goal:** {d['goal']}")
    if d.get("state"):
        out.append(f"- **State:** {d['state']}")
    if d.get("next_steps"):
        out.append("- **Next:**")
        out += [f"  - {s}" for s in d["next_steps"]]
    if d.get("open_questions"):
        out.append("- **Open:**")
        out += [f"  - {q}" for q in d["open_questions"]]
    if d.get("references"):
        out.append("- **Refs:**")
        for r in d["references"]:
            note = f" — {r['note']}" if r.get("note") else ""
            out.append(f"  - [{r.get('kind')}] {r.get('ref')}{note}")
    return "\n".join(out)


def compose(rec: dict):
    """One record's text, already cut to MAX_CHARS; None to skip the record."""
    t, d = rec["table"], rec["data"]
    s = lambda k: (d.get(k) or "").strip() if isinstance(d.get(k), str) else ""
    if t == "decisions":
        head = end_sentence(shorten(s("summary"), 300))
        room = MAX_CHARS - len(head) - 1
        reason = shorten(s("reason"), room) if room >= 80 and s("reason") else ""
        return (head + " " + reason).strip()
    if t == "errors":
        err = end_sentence(shorten(s("error"), 260))
        room = MAX_CHARS - len(err) - len(" Root cause: ") - len(" Fix: ")
        rc = end_sentence(shorten(s("root_cause"), room // 2)) if s("root_cause") else ""
        fix = shorten(s("fix"), room - len(rc)) if s("fix") else ""
        text = err
        if rc:
            text += " Root cause: " + rc
        if fix:
            text += " Fix: " + fix
        return text
    if t == "context":
        if d.get("type") == "fallback_capture" or d.get("_origin") == "fallback_capture":
            return None
        return shorten(s("summary"), MAX_CHARS)
    if t == "rules":
        return shorten(s("rule"), MAX_CHARS)
    if t == "_checkpoint_records":
        return shorten(render_checkpoint(d), MAX_CHARS)
    if t == "commits":
        body = s("content") or (s("subject") + "\n\n" + s("body"))
        lines = [ln for ln in body.split("\n") if not TRAILER.match(ln.strip())]
        return shorten(unwrap("\n".join(lines)), MAX_CHARS)
    return None


# ── Features ────────────────────────────────────────────────────────

INLINE_CODE = re.compile(r"`([^`\n]+)`")
CODE_TOKENS = [
    re.compile(r"(?<![\w/.~-])(?:~|\.{1,2})?/?(?:[\w.@-]+/)+[\w.@-]*[\w/]"),  # paths
    re.compile(r"(?<![\w.-])[\w-]+\.(?:rs|py|pyi|ts|tsx|js|mjs|cjs|go|java|kt|toml|json|jsonl|ya?ml|md|lock|sh|onnx|scip|axil|wasm|wit)\b"),
    re.compile(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+"),  # qualified paths
    re.compile(r"(?<![\w])_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),  # snake_case
    re.compile(r"(?<![\w])_[a-z]+\b"),  # _entities, _sessions
    re.compile(r"(?<![\w-])--[a-z][\w-]*"),  # CLI flags
    re.compile(r"\b[A-Za-z_][\w.]*\((?=[\w&\"')])"),  # calls
    re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"),  # env vars / constants
    re.compile(r"#\[[^\]\n]+\]"),  # Rust attributes
]
# Slash lists of plain lowercase words ("and/or", "local/offline") are prose.
PLAIN_SLASH = re.compile(r"^[a-z-]+(?:/[a-z-]+)+$")
# CamelCase with an inner capital: a type name (GraphEngine) or a product
# name (LongMemEval). Prose may hold one; more reads as code.
CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")


def code_tokens(text: str) -> set:
    found = {m.group(1) for m in INLINE_CODE.finditer(text)}
    for pat in CODE_TOKENS:
        for m in pat.finditer(text):
            tok = m.group(0)
            if PLAIN_SLASH.match(tok) or re.fullmatch(r"[\d./]+", tok):
                continue
            found.add(tok)
    return found


def is_markdown(text: str) -> bool:
    items = sum(1 for ln in text.split("\n") if BULLET_START.match(ln))
    return items >= 2 or "**" in text


# The declaration patterns as they read all text before they were limited to
# code: an English word after the keyword became a code symbol.
OLD_DECLARATIONS = re.compile(
    r"\b(?:fn|struct|impl|trait|enum|def|class|function|func)\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)"
)
KEYWORD_AS_WORD = re.compile(r"\b(?:fn|impl|struct|trait|enum|def|func)\b")
# Read as a file path because the last part has a "." (4/12.9, 3.1/3.2).
SLASH_NUMBERS = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?:/\d+(?:\.\d+)*)*/\d+\.\d+\b")
# Plural or mixed-case acronyms the CamelCase rule mints as code (IDs -> i_ds).
ACRONYM_CAMEL = re.compile(r"\b(?:[A-Z]{2,}s|[KMGT]iB|macOS|iOS)\b")
# Two apostrophes on one line with 3+ words between them read as a quote.
APOSTROPHE_PAIR = re.compile(r"[A-Za-z]'[^'\n]*?\s[^'\n]*?\s[^'\n]*?'")


def junk_signals(text: str) -> list:
    prose = INLINE_CODE.sub(" ", text)
    signals = []
    for m in OLD_DECLARATIONS.finditer(prose):
        if re.fullmatch(r"[a-z]+", m.group(1)):
            signals.append("keyword_then_word")
            break
    if "keyword_then_word" not in signals and KEYWORD_AS_WORD.search(prose):
        signals.append("keyword_as_word")
    if SLASH_NUMBERS.search(text):
        signals.append("slash_numbers")
    if APOSTROPHE_PAIR.search(text):
        signals.append("apostrophe_pair")
    if ACRONYM_CAMEL.search(prose):
        signals.append("acronym_camel")
    return signals


LANGS = {
    "python": re.compile(r"\.pyi?\b|\bpython3?\b|\bpip\b|pyproject|uv\.lock|poetry\.lock|Pipfile|site-packages|dist-info|scip-python|__init__|\bpytest\b|\bnumpy\b|\bdef \w+\(", re.I),
    "ts": re.compile(r"\.(?:ts|tsx|js|mjs|cjs)\b|\bnpm\b|package(?:-lock)?\.json|node_modules|\bpnpm\b|\byarn\b|scip-typescript|\bTypeScript\b|\bJavaScript\b|\bReact\b|\bVite\b|\bnpx\b"),
    "go": re.compile(r"\.go\b|\bgo\.(?:mod|sum)\b|scip-go|\bgolang\b|\bGo module|\bfunc \w+|\bgo (?:build|test|mod|get)\b"),
    "java": re.compile(r"\.java\b|pom\.xml|\.m2\b|scip-java|\bMaven\b|\bGradle\b|\bKotlin\b"),
    "rust": re.compile(r"\.rs\b|::|\bcargo\b|Cargo\.(?:toml|lock)|#\[|\bimpl\b|\bfn\b|\bcrates?\b|rustc|clippy"),
}


def langs_of(text: str) -> list:
    return [name for name, pat in LANGS.items() if pat.search(text)]


def stratum_of(text: str, table: str) -> str:
    if is_markdown(text):
        return "markdown"
    if table == "commits":
        return ""  # commits only feed the markdown stratum
    if junk_signals(text):
        return "junk_trap"
    n = len(code_tokens(text))
    if n >= 6:
        return "code_heavy"
    if n == 0 and len(set(CAMEL.findall(text))) <= 1:
        return "prose"
    return "mixed"


# ── Sampling ────────────────────────────────────────────────────────


def norm_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower())[:200]


def build(records: list, counter: Counter):
    pool = {s: [] for s in STRATUM_ORDER}
    seen = set()
    skipped = Counter()
    for rec in sorted(records, key=lambda r: sha(r["id"])):
        if rec["table"] not in TABLES:
            continue
        if (rec.get("created_at") or "") >= CUTOFF:
            skipped["after_cutoff"] += 1
            continue
        if rec["data"].get("_superseded"):
            skipped["superseded"] += 1
            continue
        text = compose(rec)
        if not text or len(text) < 40:
            skipped["empty_or_short"] += 1
            continue
        if PRIVATE.search(text):
            skipped["private"] += 1
            continue
        key = norm_key(text)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        stratum = stratum_of(text, rec["table"])
        if not stratum:
            skipped["commit_not_markdown"] += 1
            continue
        pool[stratum].append({
            "source_id": rec["id"],
            "source_table": rec["table"],
            "stratum": stratum,
            "text": redact(text, counter),
            "langs": langs_of(text),
        })

    def priority(c):
        rare = any(lang in c["langs"] for lang in RARE_LANGS)
        return (c["source_table"] not in SCARCE_TABLES, not rare, sha(c["source_id"]))

    chosen, rest = [], {}
    for stratum in STRATUM_ORDER:
        ranked = sorted(pool[stratum], key=priority)
        quota = QUOTAS[stratum]
        if len(ranked) < quota:
            print(f"note: {stratum} has {len(ranked)} of {quota}; the rest spills over", file=sys.stderr)
        chosen += ranked[:quota]
        rest[stratum] = ranked[quota:]
    while len(chosen) < TARGET and any(rest[s] for s in SPILL_ORDER):
        for stratum in SPILL_ORDER:
            if rest[stratum] and len(chosen) < TARGET:
                chosen.append(rest[stratum].pop(0))

    chosen.sort(key=lambda c: sha(c["source_id"]))
    items = []
    for n, c in enumerate(chosen, 1):
        item_id = f"u5-{n:03d}"
        items.append({
            "id": item_id,
            "source_id": c["source_id"],
            "source_table": c["source_table"],
            "stratum": c["stratum"],
            "split": split_of(item_id),
            "text": c["text"],
        })
    return items, pool, skipped, {c["source_id"]: c["langs"] for c in chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="Axil database to export from (read-only)")
    ap.add_argument("--export", help="a saved `axil export` JSONL to read instead of the DB")
    ap.add_argument("--axil", default="axil", help="axil binary (default: on PATH)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    redactions = Counter()
    items, pool, skipped, langs = build(load_export(args), redactions)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(items)} texts to {out}")
    print("pool per stratum:", {s: len(v) for s, v in pool.items()})
    print("chosen per stratum:", dict(Counter(i["stratum"] for i in items)))
    print("chosen per table:", dict(Counter(i["source_table"] for i in items)))
    print("split:", dict(Counter(i["split"] for i in items)))
    print("languages mentioned:", dict(Counter(l for ls in langs.values() for l in ls)))
    print("skipped:", dict(skipped))
    print("redactions:", dict(redactions) or "none")


if __name__ == "__main__":
    main()
