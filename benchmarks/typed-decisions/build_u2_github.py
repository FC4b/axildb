#!/usr/bin/env python3
"""Build U2 (error resolved?) pairs from real GitHub bug issues and pull requests.

Labels come from GitHub's own links, not from an annotator:
  fixed (1)            the bug issue + the merged PR that closed it
                       (`closedByPullRequestsReferences`, i.e. "Fixes #N")
  mentioned_not_fix (0) the issue + a merged PR in the same repo that
                       cross-references it but did not close it
  downstream_workaround (0) the issue + a merged PR in *another* repo that
                       references it (pins, workarounds, tests downstream)
  other_fix (0)        the issue + the closing PR of a *different* bug in the
                       same repo, picked for maximum word overlap with the
                       issue title (a fix for something else in the same area)

Only titles are used, with issue numbers, repo refs, URLs and closing keywords
stripped so no model can match on "#1234". Raw GraphQL pages are cached under
data/github/ (gitignored), so re-running reproduces sets/u2_github.jsonl.

Requires an authenticated `gh` CLI.  Usage: python build_u2_github.py
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "github"
OUT = HERE / "sets" / "u2_github.jsonl"
# Pairs whose link-derived label the text does not support: both blind
# annotators disagreed with it (see validation/). Excluded on every rebuild.
DROPS = HERE / "validation" / "u2_github_drops.json"

REPOS = [
    ("cberner", "redb"),
    ("quickwit-oss", "tantivy"),
    ("pykeio", "ort"),
    ("BurntSushi", "ripgrep"),
    ("astral-sh", "uv"),
    ("tokio-rs", "tokio"),
]
MAX_PAGES = 8
POSITIVES_PER_REPO = 20

QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issues(states:CLOSED, first:40, after:$cursor, orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title
        labels(first:15) { nodes { name } }
        closedByPullRequestsReferences(first:5, includeClosedPrs:false) { nodes { number title merged } }
        timelineItems(first:25, itemTypes:[CROSS_REFERENCED_EVENT]) {
          nodes { ... on CrossReferencedEvent { source { ... on PullRequest { number title merged repository { nameWithOwner } } } } }
        }
      }
    }
  }
}
"""

NOT_A_BUG = re.compile(r"feature|enhancement|question|discussion|documentation|\bdocs?\b|proposal|idea", re.I)
BUG_LABEL = re.compile(r"bug|crash|panic|regression|defect", re.I)
BUGGY_TITLE = re.compile(
    r"panic|crash|fail|error|wrong|incorrect|broken|\bbug\b|doesn'?t|does not|can'?t|cannot|not work|"
    r"hang|leak|overflow|regress|abort|missing|invalid|unexpected|mismatch|segfault|deadlock", re.I)
SKIP_XREF = re.compile(r"backport|cherry|release|bump|\bsync\b.*upstream|merge branch|changelog", re.I)
# Release PRs ("ripgrep 15.0.0", "v0.9.2") cross-reference every issue in their changelog.
VERSION_ONLY = re.compile(r"^(?:[\w.-]+\s+)?v?\d+\.\d+(?:\.\d+)?(?:[-+.\w]*)?$")
STOP = set("a an the of to in on for and or is are be with when from by at as it its this that not no "
           "fix fixes fixed use using should does do can into".split())


def fetch(owner, name):
    pages, cursor = [], None
    for k in range(MAX_PAGES):
        path = CACHE / f"{owner}_{name}_p{k}.json"
        if path.exists():
            page = json.loads(path.read_text())
        else:
            args = ["gh", "api", "graphql", "-F", f"owner={owner}", "-F", f"name={name}", "-f", f"query={QUERY}"]
            if cursor:
                args += ["-F", f"cursor={cursor}"]
            page = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(page))
        issues = page["data"]["repository"]["issues"]
        pages.append(issues["nodes"])
        if not issues["pageInfo"]["hasNextPage"]:
            break
        cursor = issues["pageInfo"]["endCursor"]
    return [n for p in pages for n in p]


def clean(title):
    t = re.sub(r"https?://\S+", " ", title)
    t = re.sub(r"\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s*:?\s*(?:[\w.-]+/[\w.-]+)?#\d+", " ", t, flags=re.I)
    t = re.sub(r"(?:[\w.-]+/[\w.-]+)?#\d+|\bGH-\d+\b", " ", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", " ", t)
    return " ".join(t.split()).strip(" -:,.")


def words(text):
    return {w for w in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def is_bug(issue):
    labels = [l["name"] for l in issue["labels"]["nodes"]]
    if any(NOT_A_BUG.search(l) for l in labels):
        return False
    if labels and any(BUG_LABEL.search(l) for l in labels):
        return True
    return bool(BUGGY_TITLE.search(issue["title"]))


def split_of(pair_id):
    h = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 4 else "test"


def main():
    rows = []
    for owner, name in REPOS:
        repo = f"{owner}/{name}"
        cands = []
        for issue in fetch(owner, name):
            closers = [p for p in issue["closedByPullRequestsReferences"]["nodes"] if p.get("merged")]
            if not closers or not is_bug(issue):
                continue
            fix = closers[0]
            err, later = clean(issue["title"]), clean(fix["title"])
            if len(err) < 12 or len(later) < 8:
                continue
            closer_ids = {p["number"] for p in closers}
            xrefs = []
            for node in issue["timelineItems"]["nodes"]:
                src = (node or {}).get("source") or {}
                same_repo = (src.get("repository") or {}).get("nameWithOwner", "").lower() == repo.lower()
                if (src.get("number") and src.get("merged") and not (same_repo and src["number"] in closer_ids)
                        and not SKIP_XREF.search(src["title"]) and not VERSION_ONLY.match(clean(src["title"]))
                        and jaccard(src["title"], fix["title"]) < 0.5):
                    xrefs.append((src, same_repo))
            cands.append((issue, fix, err, later, xrefs))
        cands.sort(key=lambda c: hashlib.sha256(f"{repo}#{c[0]['number']}".encode()).hexdigest())
        cands = cands[:POSITIVES_PER_REPO]
        fixes = [(c[0]["number"], c[1], c[3]) for c in cands]

        def add(kind, issue, pr, err, later, label):
            pid = f"gh-{name}-{issue['number']}-{kind}"
            rows.append({"id": pid, "error": err, "later": later, "label": label, "kind": kind,
                         "source": f"github:{repo}#{issue['number']}->#{pr['number']}", "split": split_of(pid)})

        for issue, fix, err, later, xrefs in cands:
            add("fixed", issue, fix, err, later, 1)
            if xrefs:
                x, same_repo = xrefs[0]
                add("mentioned_not_fix" if same_repo else "downstream_workaround", issue, x, err, clean(x["title"]), 0)
            others = [(jaccard(err, t), n, pr, t) for n, pr, t in fixes if n != issue["number"]]
            others = [o for o in others if o[0] > 0]
            if others:
                _, _, pr, t = max(others, key=lambda o: (o[0], -o[1]))
                add("other_fix", issue, pr, err, t, 0)

    if DROPS.exists():
        dropped = {d["id"] for d in json.loads(DROPS.read_text())}
        rows = [r for r in rows if r["id"] not in dropped]
    rows.sort(key=lambda r: r["id"])
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = {}
    for r in rows:
        key = (r["source"].split("#")[0].removeprefix("github:"), r["kind"])
        by[key] = by.get(key, 0) + 1
    print(f"wrote {len(rows)} pairs ({sum(r['label'] for r in rows)} positive) to {OUT.relative_to(HERE)}")
    for k in sorted(by):
        print(f"  {k[0]:<22} {k[1]:<18} {by[k]}")


if __name__ == "__main__":
    main()
