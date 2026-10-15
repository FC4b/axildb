#!/usr/bin/env python3
"""Harvest U2 (error resolved?) training pairs from repos held out of the test sets.

The fine-tune spike needs thousands of labeled pairs; the U2 test sets
(sets/u2_github.jsonl, sets/u2_repo.jsonl) hold about two hundred. This builds
training pairs with the construction and cleaning rules of build_u2_github.py:

  fixed (1)                 bug issue + the merged PR that closed it
  mentioned_not_fix (0)     + a merged same-repo PR that cross-references it
  downstream_workaround (0) + a merged PR in another repo that references it
  other_fix (0)             + the closing PR of a different bug in the same
                              repo, picked for maximum word overlap

It differs from the test construction in three ways, all deliberate:
  - up to 150 bugs per repo instead of 20, so other_fix picks its overlap
    winner from more candidates and the negatives come out somewhat harder;
  - other_fix never uses a PR that also closed the issue itself (one PR often
    closes several issues; the test set had such rows removed by hand);
  - an (error, later) text pair that ends up with both labels is dropped.

Leakage guards:
  - no repo of the test sets is harvested, and no PR from a test repo is used
    as a cross-reference (both asserted or filtered);
  - a pair is dropped if its error or later text, normalized, appears anywhere
    in the U2 sets;
  - the dev split is chosen per repo, never per pair.

Raw GraphQL pages are cached under data/github-train/<query hash>/; pairs go
to data/u2_train/pairs.jsonl and the run's parameters, per-repo page counts
and the pairs' sha256 to data/u2_train/harvest-meta.json (all gitignored).
Any repo that fails stops the run before anything is written, unless
--allow-partial is given.

Requires an authenticated `gh` CLI.  Usage: python harvest_u2_train.py [--max-pages 12]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import build_u2_github as base

HERE = Path(__file__).resolve().parent
QUERY_HASH = hashlib.sha256(base.QUERY.encode()).hexdigest()[:12]
CACHE = HERE / "data" / "github-train" / QUERY_HASH
OUT = HERE / "data" / "u2_train" / "pairs.jsonl"
META = HERE / "data" / "u2_train" / "harvest-meta.json"
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
RETRIES = 6


def gh_graphql(owner, name, cursor):
    args = ["gh", "api", "graphql", "-F", f"owner={owner}", "-F", f"name={name}", "-f", f"query={base.QUERY}"]
    if cursor:
        args += ["-F", f"cursor={cursor}"]
    err = ""
    for attempt in range(RETRIES):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        err = proc.stderr.strip()
        if "could not resolve to a repository" in err.lower():
            break
        # Rate limits, resets and 5xx are transient; the primary GraphQL limit
        # resets hourly, so the last waits are long.
        time.sleep(min(60 * 2 ** attempt, 1200))
    raise RuntimeError(f"{owner}/{name}: {err[:300]}")


def valid_page(page):
    issues = ((page.get("data") or {}).get("repository") or {}).get("issues")
    return issues if isinstance(issues, dict) and "nodes" in issues and "pageInfo" in issues else None


def fetch(owner, name, max_pages):
    pages, cursor = [], None
    for k in range(max_pages):
        path = CACHE / f"{owner}_{name}_p{k}.json"
        issues = None
        if path.exists():
            try:
                issues = valid_page(json.loads(path.read_text()))
            except json.JSONDecodeError:
                issues = None
        if issues is None:
            page = gh_graphql(owner, name, cursor)
            issues = valid_page(page)
            if issues is None:
                raise RuntimeError(f"{owner}/{name}: no issues in response page {k}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(page))
            os.replace(tmp, path)
        pages.append(issues["nodes"])
        if not issues["pageInfo"]["hasNextPage"]:
            break
        cursor = issues["pageInfo"]["endCursor"]
    return [n for p in pages for n in p], len(pages)


def norm(text):
    return " ".join(text.lower().split())


def split_of(repo):
    h = int(hashlib.sha256(repo.lower().encode()).hexdigest()[:8], 16)
    return "dev" if h % 100 < DEV_SHARE else "train"


def pairs_for(owner, name, issues, seen_texts, stats):
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
            src_repo = (src.get("repository") or {}).get("nameWithOwner", "").lower()
            same_repo = src_repo == repo.lower()
            if src_repo in TEST_REPOS:
                stats["xref_from_test_repo"] += 1
                continue
            if (src.get("number") and src.get("merged") and not (same_repo and src["number"] in closer_ids)
                    and not base.SKIP_XREF.search(src["title"])
                    and not base.VERSION_ONLY.match(base.clean(src["title"]))
                    and base.jaccard(src["title"], fix["title"]) < 0.5):
                xrefs.append((src, same_repo))
        cands.append((issue, fix, err, later, xrefs, closer_ids))
    cands.sort(key=lambda c: hashlib.sha256(f"{repo}#{c[0]['number']}".encode()).hexdigest())
    cands = cands[:POSITIVES_PER_REPO]
    fixes = [(c[0]["number"], c[1], c[3]) for c in cands]

    rows = []

    def add(kind, issue, pr, err, later, label):
        if norm(err) in seen_texts or norm(later) in seen_texts:
            stats["shares_text_with_sets"] += 1
            return
        rows.append({"id": f"ght-{owner}-{name}-{issue['number']}-{kind}", "error": err, "later": later,
                     "label": label, "kind": kind, "repo": repo,
                     "source": f"github:{repo}#{issue['number']}->#{pr['number']}", "split": split_of(repo)})

    for issue, fix, err, later, xrefs, closer_ids in cands:
        add("fixed", issue, fix, err, later, 1)
        if xrefs:
            x, same_repo = xrefs[0]
            add("mentioned_not_fix" if same_repo else "downstream_workaround", issue, x, err,
                base.clean(x["title"]), 0)
        # A PR that also closed this issue is this issue's fix, never "a fix
        # for something else", even when it is filed under another issue.
        others = [(base.jaccard(err, t), n, pr, t) for n, pr, t in fixes
                  if n != issue["number"] and pr["number"] not in closer_ids and norm(t) != norm(later)]
        others = [o for o in others if o[0] > 0]
        if others:
            _, _, pr, t = max(others, key=lambda o: (o[0], -o[1]))
            add("other_fix", issue, pr, err, t, 0)
    return rows


def drop_conflicts(rows, stats):
    """Drop every (error, later) text pair that carries both labels."""
    labels = {}
    for r in rows:
        labels.setdefault((norm(r["error"]), norm(r["later"])), set()).add(r["label"])
    conflicted = {k for k, v in labels.items() if len(v) > 1}
    kept = [r for r in rows if (norm(r["error"]), norm(r["later"])) not in conflicted]
    stats["conflicting_label_pairs_dropped"] = len(rows) - len(kept)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=12, help="40 closed issues per page")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the pairs even if some repos failed")
    args = ap.parse_args()

    harvested = {f"{o}/{n}".lower() for o, n in REPOS}
    assert not harvested & TEST_REPOS, f"test repos in the harvest list: {harvested & TEST_REPOS}"

    seen_texts = set()
    for path in SETS:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            seen_texts.add(norm(row["error"]))
            seen_texts.add(norm(row["later"]))

    stats = {"xref_from_test_repo": 0, "shares_text_with_sets": 0}
    rows, failed, per_repo = [], [], {}
    for owner, name in REPOS:
        try:
            issues, n_pages = fetch(owner, name, args.max_pages)
        except RuntimeError as e:
            failed.append(str(e))
            print(f"  FAILED {owner}/{name}: {e}", file=sys.stderr, flush=True)
            continue
        repo_rows = pairs_for(owner, name, issues, seen_texts, stats)
        rows.extend(repo_rows)
        per_repo[f"{owner}/{name}"] = {"pages": n_pages, "issues": len(issues), "pairs": len(repo_rows),
                                        "split": split_of(f"{owner}/{name}")}
        print(f"  {owner}/{name:<22} pages {n_pages:>2}  issues {len(issues):>5}  pairs {len(repo_rows):>4}"
              f"  {split_of(f'{owner}/{name}')}", flush=True)

    if failed and not args.allow_partial:
        sys.exit(f"{len(failed)} repos failed; nothing written (rerun, or pass --allow-partial)")

    rows = drop_conflicts(rows, stats)
    rows.sort(key=lambda r: r["id"])
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body)
    by = {}
    for r in rows:
        key = f"{r['split']}/{r['kind']}"
        by[key] = by.get(key, 0) + 1
    META.write_text(json.dumps({
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args), "query_sha256_12": QUERY_HASH, "positives_per_repo": POSITIVES_PER_REPO,
        "dev_share_percent_of_repos": DEV_SHARE, "pairs": len(rows),
        "positives": sum(r["label"] for r in rows), "by_split_kind": dict(sorted(by.items())),
        "dropped": stats, "failed_repos": failed, "per_repo": per_repo,
        "pairs_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }, indent=2))
    print(f"wrote {len(rows)} pairs ({sum(r['label'] for r in rows)} positive) to {OUT.relative_to(HERE)}")
    print(f"dropped: {stats}; {len(failed)} repos failed")
    for k in sorted(by):
        print(f"  {k:<30} {by[k]}")


if __name__ == "__main__":
    main()
