#!/usr/bin/env python3
"""Harvest U2 (error resolved?) training pairs from repos held out of the test sets.

The fine-tune spike needs thousands of labeled pairs; the U2 test sets
(sets/u2_github.jsonl, sets/u2_repo.jsonl) hold about two hundred. This builds
training pairs with exactly the construction and cleaning rules of
build_u2_github.py, so train and test come from one distribution:

  fixed (1)                 bug issue + the merged PR that closed it
  mentioned_not_fix (0)     + a merged same-repo PR that cross-references it
  downstream_workaround (0) + a merged PR in another repo that references it
  other_fix (0)             + the closing PR of a different bug in the same
                              repo, picked for maximum word overlap

Leakage guards:
  - no repo of the test sets is harvested (asserted);
  - a pair is dropped if its error or later text, normalized, appears anywhere
    in the U2 sets, so a cross-repo PR title cannot carry a test row in;
  - the dev split is chosen per repo, never per pair.

Raw GraphQL pages are cached under data/github-train/ and the pairs are
written to data/u2_train/pairs.jsonl (both gitignored).

Requires an authenticated `gh` CLI.  Usage: python harvest_u2_train.py [--max-pages N]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import build_u2_github as base

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "github-train"
OUT = HERE / "data" / "u2_train" / "pairs.jsonl"
SETS = [HERE / "sets" / "u2_github.jsonl", HERE / "sets" / "u2_repo.jsonl"]

TEST_REPOS = {f"{o}/{n}".lower() for o, n in base.REPOS} | {"fc4b/axildb"}

# Active projects that close bugs with merged PRs ("Fixes #N"). Projects that
# land work by commit (node, golang/go) or through bots are left out: their
# issues rarely carry a merged closing PR.
REPOS = [
    # Rust
    ("rust-lang", "rust-analyzer"), ("rust-lang", "cargo"), ("rust-lang", "rust-clippy"),
    ("rust-lang", "rustfmt"), ("rust-lang", "regex"), ("rust-lang", "rustup"),
    ("serde-rs", "serde"), ("hyperium", "hyper"), ("seanmonstar", "reqwest"),
    ("tokio-rs", "axum"), ("tokio-rs", "tracing"), ("clap-rs", "clap"),
    ("sharkdp", "bat"), ("sharkdp", "fd"), ("alacritty", "alacritty"),
    ("helix-editor", "helix"), ("nushell", "nushell"), ("starship", "starship"),
    ("zellij-org", "zellij"), ("bevyengine", "bevy"), ("tauri-apps", "tauri"),
    ("denoland", "deno"), ("astral-sh", "ruff"), ("pola-rs", "polars"),
    ("apache", "arrow-rs"), ("apache", "datafusion"), ("PyO3", "pyo3"),
    ("rustls", "rustls"), ("launchbadge", "sqlx"), ("diesel-rs", "diesel"),
    ("meilisearch", "meilisearch"), ("qdrant", "qdrant"), ("huggingface", "tokenizers"),
    ("bytecodealliance", "wasmtime"), ("gfx-rs", "wgpu"), ("rayon-rs", "rayon"),
    # Python
    ("pallets", "flask"), ("pallets", "click"), ("psf", "black"),
    ("pytest-dev", "pytest"), ("python-poetry", "poetry"), ("pypa", "pip"),
    ("pydantic", "pydantic"), ("encode", "httpx"), ("python", "mypy"),
    ("scrapy", "scrapy"), ("pandas-dev", "pandas"), ("matplotlib", "matplotlib"),
    # TypeScript / JavaScript
    ("vitejs", "vite"), ("vuejs", "core"), ("sveltejs", "svelte"),
    ("prettier", "prettier"), ("microsoft", "TypeScript"), ("biomejs", "biome"),
    ("oven-sh", "bun"), ("withastro", "astro"),
    # Go
    ("gohugoio", "hugo"), ("junegunn", "fzf"), ("cli", "cli"),
    ("prometheus", "prometheus"), ("caddyserver", "caddy"), ("spf13", "cobra"),
]
POSITIVES_PER_REPO = 150
DEV_SHARE = 10  # percent of repos


def gh_graphql(owner, name, cursor):
    args = ["gh", "api", "graphql", "-F", f"owner={owner}", "-F", f"name={name}", "-f", f"query={base.QUERY}"]
    if cursor:
        args += ["-F", f"cursor={cursor}"]
    for attempt in range(5):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        err = proc.stderr.strip()
        # Secondary rate limits and transient 5xx: back off and retry.
        if any(s in err.lower() for s in ("rate limit", "abuse", "502", "503", "504", "timeout")):
            time.sleep(30 * (attempt + 1))
            continue
        raise RuntimeError(f"{owner}/{name}: {err[:300]}")
    raise RuntimeError(f"{owner}/{name}: gave up after retries")


def fetch(owner, name, max_pages):
    pages, cursor = [], None
    for k in range(max_pages):
        path = CACHE / f"{owner}_{name}_p{k}.json"
        if path.exists():
            page = json.loads(path.read_text())
        else:
            page = gh_graphql(owner, name, cursor)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(page))
        repo = (page.get("data") or {}).get("repository")
        if not repo:
            raise RuntimeError(f"{owner}/{name}: no repository in response")
        issues = repo["issues"]
        pages.append(issues["nodes"])
        if not issues["pageInfo"]["hasNextPage"]:
            break
        cursor = issues["pageInfo"]["endCursor"]
    return [n for p in pages for n in p]


def norm(text):
    return " ".join(text.lower().split())


def split_of(repo):
    h = int(hashlib.sha256(repo.lower().encode()).hexdigest()[:8], 16)
    return "dev" if h % 100 < DEV_SHARE else "train"


def pairs_for(owner, name, issues, seen_texts):
    """The build_u2_github.py construction, minus its 20-per-repo cap."""
    repo = f"{owner}/{name}"
    cands = []
    for issue in issues:
        closers = [p for p in issue["closedByPullRequestsReferences"]["nodes"] if p.get("merged")]
        if not closers or not base.is_bug(issue):
            continue
        fix = closers[0]
        err, later = base.clean(issue["title"]), base.clean(fix["title"])
        if len(err) < 12 or len(later) < 8:
            continue
        closer_ids = {p["number"] for p in closers}
        xrefs = []
        for node in issue["timelineItems"]["nodes"]:
            src = (node or {}).get("source") or {}
            same_repo = (src.get("repository") or {}).get("nameWithOwner", "").lower() == repo.lower()
            if (src.get("number") and src.get("merged") and not (same_repo and src["number"] in closer_ids)
                    and not base.SKIP_XREF.search(src["title"])
                    and not base.VERSION_ONLY.match(base.clean(src["title"]))
                    and base.jaccard(src["title"], fix["title"]) < 0.5):
                xrefs.append((src, same_repo))
        cands.append((issue, fix, err, later, xrefs))
    cands.sort(key=lambda c: hashlib.sha256(f"{repo}#{c[0]['number']}".encode()).hexdigest())
    cands = cands[:POSITIVES_PER_REPO]
    fixes = [(c[0]["number"], c[1], c[3]) for c in cands]

    rows, dropped = [], 0

    def add(kind, issue, pr, err, later, label):
        nonlocal dropped
        if norm(err) in seen_texts or norm(later) in seen_texts:
            dropped += 1
            return
        rows.append({"id": f"ght-{owner}-{name}-{issue['number']}-{kind}", "error": err, "later": later,
                     "label": label, "kind": kind, "repo": repo,
                     "source": f"github:{repo}#{issue['number']}->#{pr['number']}", "split": split_of(repo)})

    for issue, fix, err, later, xrefs in cands:
        add("fixed", issue, fix, err, later, 1)
        if xrefs:
            x, same_repo = xrefs[0]
            add("mentioned_not_fix" if same_repo else "downstream_workaround", issue, x, err,
                base.clean(x["title"]), 0)
        others = [(base.jaccard(err, t), n, pr, t) for n, pr, t in fixes if n != issue["number"]]
        others = [o for o in others if o[0] > 0]
        if others:
            _, _, pr, t = max(others, key=lambda o: (o[0], -o[1]))
            add("other_fix", issue, pr, err, t, 0)
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=25, help="40 closed issues per page")
    args = ap.parse_args()

    harvested = {f"{o}/{n}".lower() for o, n in REPOS}
    assert not harvested & TEST_REPOS, f"test repos in the harvest list: {harvested & TEST_REPOS}"

    seen_texts = set()
    for path in SETS:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            seen_texts.add(norm(row["error"]))
            seen_texts.add(norm(row["later"]))

    rows, failed, total_dropped = [], [], 0
    for owner, name in REPOS:
        try:
            issues = fetch(owner, name, args.max_pages)
        except RuntimeError as e:
            failed.append(str(e))
            print(f"  skip {owner}/{name}: {e}", file=sys.stderr, flush=True)
            continue
        repo_rows, dropped = pairs_for(owner, name, issues, seen_texts)
        total_dropped += dropped
        rows.extend(repo_rows)
        pos = sum(r["label"] for r in repo_rows)
        print(f"  {owner}/{name:<22} issues {len(issues):>5}  pairs {len(repo_rows):>4} ({pos} fixed)"
              f"  {split_of(f'{owner}/{name}')}", flush=True)

    rows.sort(key=lambda r: r["id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = {}
    for r in rows:
        key = (r["split"], r["kind"])
        by[key] = by.get(key, 0) + 1
    print(f"wrote {len(rows)} pairs ({sum(r['label'] for r in rows)} positive) to {OUT.relative_to(HERE)}")
    print(f"dropped {total_dropped} pairs sharing text with the U2 sets; {len(failed)} repos failed")
    for k in sorted(by):
        print(f"  {k[0]:<6} {k[1]:<22} {by[k]}")


if __name__ == "__main__":
    main()
