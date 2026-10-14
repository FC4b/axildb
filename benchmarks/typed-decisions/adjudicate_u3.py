#!/usr/bin/env python3
"""Adjudicate the U3 capture labels and write the gold set.

Inputs (all under validation/u3/):
  claude.jsonl   Claude's labels, written with the guidelines
  glm.jsonl      GLM, blind (ids mapped back via glm-map.json)
  gptoss.jsonl   gpt-oss-120b, blind (ids mapped back via gptoss-map.json)
  gemini.jsonl   Gemini, blind, 39 items only (its quota ran out); reported for
                 agreement, not a vote

Each row is {"id", "class", "importance", "rule"}; importance is null exactly
when class is none (validation/u3-guidelines.md).

Gold rule:
  * class: the majority of the three voters (Claude, GLM, gpt-oss); a 1/1/1
    split is broken by a named guideline rule (CLASS_TIEBREAKS); a majority is
    overridden only by a named guideline rule (OVERRIDES), and both are logged;
  * importance (gold class not none): the votes of every voter who gave the
    item any class but none; the majority bucket; low against high (or one of
    each) is medium, the ordinal median; low against medium or medium against
    high, with no majority, is broken by a named guideline rule
    (IMPORTANCE_TIEBREAKS), logged;
  * an item is dropped only if it is too ambiguous to label (DROPS).

Writes sets/u3_repo.jsonl (the u3_texts fields plus "class" and "importance")
and validation/<DATE>-u3-labels.json (annotators, agreement as accuracy and
Cohen's kappa for class and importance, overall and per source, overrides,
tie-breaks, drops and every disputed item).
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "validation" / "u3"
TEXTS = HERE / "sets" / "u3_texts.jsonl"
OUT_SET = HERE / "sets" / "u3_repo.jsonl"
DATE = "2026-10-14"
OUT_JSON = HERE / "validation" / f"{DATE}-u3-labels.json"

CLASSES = ["decision", "error", "context", "rule", "none"]
BUCKETS = ["low", "medium", "high"]
SOURCES = ["bash_failure", "bash_output", "agent_text", "agent_brief", "commit_subject"]
VOTERS = ["claude", "glm", "gptoss"]
ANNOTATORS = {
    "claude": {
        "model": "claude-opus-5-5 (Claude Code)",
        "blind": False,
        "note": "Written alongside the guidelines by the session that built the set, reading "
                "every item; the reference labels, as for U1/U2/U5.",
    },
    "glm": {
        "model": "zai-coding-plan/glm-5.3 via OpenCode",
        "blind": True,
        "note": "Neutral ids (b-NNN) in shuffled order, batches of 20, guidelines plus item text "
                "only (relabel_u3.py); map in glm-map.json, raw output in glm-raw/.",
    },
    "gptoss": {
        "model": "gpt-oss-120b-medium via agy (Antigravity CLI)",
        "blind": True,
        "note": "Same protocol, its own shuffled order; map in gptoss-map.json, raw output in "
                "gptoss-raw/. Stands in for Gemini, whose quota ran out after 39 items.",
    },
    "gemini": {
        "model": "gemini-3.8-flash-high via agy (Antigravity CLI)",
        "blind": True,
        "vote": False,
        "note": "Same protocol, its own shuffled order. Every call from batch 03 on returned "
                "RESOURCE_EXHAUSTED (individual quota, resets in about 162 h), so only batches 01 "
                "and 02 were labeled (39 items; batch 01 hit the 300 s print timeout with 19 of 20 "
                "answers). Reported for agreement on those items only; not a vote.",
    },
}

# A majority overridden by a named guideline rule.
OVERRIDES = [
    {"id": "u3-115", "from": "context", "to": "none", "rule": "N2 (Normal output)",
     "reason": "A failed shell call whose output is a stored memory record printed as JSON "
               "(`_importance`, `summary`, `type: howto`) and no failure line. N2 lists JSON rows "
               "and text printed for reading as normal output; storing it again would duplicate the "
               "record it prints. The same rule decides the printed commit messages in u3-157 and "
               "u3-172."},
]

# Gold class when the three voters all differ.
CLASS_TIEBREAKS = {
    "u3-015": ("none", "N2 (Normal output)",
               "`axil boot` output, exit 0: the open errors it lists are stored records printed for "
               "reading, not a failure of this command (votes none / error / context)."),
    "u3-113": ("error", "E2 + 3.6 precedence",
               "A brief listing review findings to fix: item 1 is a spec gap in the code (higher-priority "
               "rows dropped while lower ones show) and item 2 an observed defect (no Decisions section "
               "at the hook budget). E2 (a defect that exists) fits; R1 does not, since the \"must\" "
               "constrains the code, not later work; error outranks rule in 3.6 (votes error / rule / "
               "none)."),
    "u3-157": ("none", "N2 + N4",
               "A commit message printed by git for reading (N2) followed by zsh's `=== not found` "
               "(N4); nothing in the project failed (votes none / error / decision)."),
    "u3-172": ("none", "N2 + N4",
               "`git show` output of a commit (N2) followed by zsh's `===== not found` (N4), as in "
               "u3-157 (votes none / context / error)."),
}

# Gold importance when the bucket votes tie between adjacent buckets.
IMPORTANCE_TIEBREAKS = {
    "u3-045": ("medium", "4 (medium otherwise)",
               "A full test pass verifying one branch increment before its merge: a finding about one "
               "piece of work; no H test holds (it is not a project-wide verdict or baseline) and no L "
               "test (it is not transient noise). Votes high / medium."),
    "u3-171": ("medium", "4 (medium: a scoped design choice)",
               "Routing the code-graph hint through extension_blocks is a scoped design choice; L2/L3 do "
               "not clearly hold. Votes low / medium."),
    "u3-196": ("low", "L3",
               "A point-in-time health snapshot (build sha, record counts, an empty queue): a narrow fact "
               "that goes stale with the next session. Votes medium / low."),
}

# Items too ambiguous to label.
DROPS = []


def rnd(x, k=3):
    return None if x is None else round(x, k)


def load(name):
    path = V / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            assert r["id"] not in rows, f"duplicate id {r['id']} in {path.name}"
            assert r["class"] in CLASSES, (name, r)
            assert (r["importance"] is None) == (r["class"] == "none"), (name, r)
            assert r["importance"] is None or r["importance"] in BUCKETS, (name, r)
            rows[r["id"]] = r
    return rows


def kappa(pairs, labels, weights=None):
    """Cohen's kappa; `weights="linear"` gives the linear-weighted kappa for ordinal labels."""
    n = len(pairs)
    if not n:
        return None
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    w = [[(abs(i - j) / (k - 1) if weights == "linear" else float(i != j)) for j in range(k)]
         for i in range(k)]
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    observed = sum(w[idx[a]][idx[b]] for a, b in pairs) / n
    expected = sum(w[idx[a]][idx[b]] * ca[a] * cb[b] for a in labels for b in labels) / (n * n)
    return rnd(1 - observed / expected) if expected else None


def agreement(a, b, ids):
    """Class and importance agreement between two annotators over `ids`."""
    ids = [i for i in ids if i in a and i in b]
    cls = [(a[i]["class"], b[i]["class"]) for i in ids]
    imp = [(a[i]["importance"], b[i]["importance"]) for i in ids
           if a[i]["importance"] and b[i]["importance"]]
    capture = [(a[i]["class"] != "none", b[i]["class"] != "none") for i in ids]
    return {
        "items": len(ids),
        "class": {"agree": sum(x == y for x, y in cls), "accuracy": rnd(sum(x == y for x, y in cls) / len(cls)) if cls else None,
                  "kappa": kappa(cls, CLASSES)},
        "capture_vs_none": {"agree": sum(x == y for x, y in capture),
                            "kappa": kappa([(str(x), str(y)) for x, y in capture], ["True", "False"])},
        "importance_where_both_capture": {
            "items": len(imp), "agree": sum(x == y for x, y in imp),
            "accuracy": rnd(sum(x == y for x, y in imp) / len(imp)) if imp else None,
            "kappa": kappa(imp, BUCKETS), "linear_weighted_kappa": kappa(imp, BUCKETS, "linear")},
    }


def gold_importance(item_id, votes):
    """Majority bucket; low/high is medium; an adjacent tie needs a logged guideline rule."""
    c = Counter(votes)
    top = c.most_common()
    if top[0][1] > len(votes) / 2:
        return top[0][0], None
    if set(votes) >= {"low", "high"}:
        return "medium", "ordinal median (low against high)"
    assert item_id in IMPORTANCE_TIEBREAKS, f"unresolved importance tie {item_id} {votes}"
    bucket, rule, reason = IMPORTANCE_TIEBREAKS[item_id]
    assert bucket in votes
    return bucket, f"{rule}: {reason}"


def main():
    texts = {}
    for line in TEXTS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            texts[r["id"]] = r
    ids = sorted(texts)
    labels = {name: load(name) for name in ANNOTATORS}
    for name in VOTERS:
        assert set(labels[name]) == set(texts), f"{name} does not cover every item"

    by_source = {s: [i for i in ids if texts[i]["source"] == s] for s in SOURCES}
    pairwise = {}
    for x in range(len(VOTERS)):
        for y in range(x + 1, len(VOTERS)):
            a, b = VOTERS[x], VOTERS[y]
            pairwise[f"{a}_vs_{b}"] = {
                "overall": agreement(labels[a], labels[b], ids),
                "per_source": {s: agreement(labels[a], labels[b], sids) for s, sids in by_source.items()},
            }
    gemini_ids = sorted(labels["gemini"])
    gemini = {f"gemini_vs_{v}": agreement(labels["gemini"], labels[v], gemini_ids) for v in VOTERS}

    overrides = {o["id"]: o for o in OVERRIDES}
    drop_ids = {d["id"] for d in DROPS}
    used_overrides, used_ties, used_imp = set(), set(), set()
    gold, disputed, splits = {}, [], Counter()
    for i in ids:
        if i in drop_ids:
            continue
        votes = {v: labels[v][i]["class"] for v in VOTERS}
        c = Counter(votes.values()).most_common()
        splits["unanimous" if c[0][1] == 3 else "2-1" if c[0][1] == 2 else "1-1-1"] += 1
        note = None
        if c[0][1] >= 2:
            cls = c[0][0]
        else:
            assert i in CLASS_TIEBREAKS, f"unresolved class tie {i} {votes}"
            cls, rule, reason = CLASS_TIEBREAKS[i]
            note = f"class tie broken by {rule}: {reason}"
            used_ties.add(i)
        if i in overrides:
            o = overrides[i]
            assert o["from"] == cls, (i, o, cls)
            cls = o["to"]
            note = f"majority {o['from']} overridden by {o['rule']}: {o['reason']}"
            used_overrides.add(i)
        imp, imp_note = None, None
        if cls != "none":
            imp_votes = [labels[v][i]["importance"] for v in VOTERS if labels[v][i]["importance"]]
            if not imp_votes:
                raise AssertionError(f"{i}: gold {cls} but no importance vote")
            imp, imp_note = gold_importance(i, imp_votes)
            if i in IMPORTANCE_TIEBREAKS:
                used_imp.add(i)
        gold[i] = {"class": cls, "importance": imp}
        imp_votes = {v: labels[v][i]["importance"] for v in VOTERS}
        if c[0][1] < 3 or len({x for x in imp_votes.values() if x}) > 1 or note:
            disputed.append({
                "id": i, "source": texts[i]["source"], "split": texts[i]["split"],
                "class_votes": votes, "importance_votes": imp_votes,
                "rules_cited": {v: labels[v][i].get("rule") for v in VOTERS},
                "gold": gold[i],
                **({"note": note} if note else {}),
                **({"importance_note": imp_note} if imp_note else {}),
                **({"gemini": {k: labels["gemini"][i][k] for k in ("class", "importance")}}
                   if i in labels["gemini"] else {}),
                "text": texts[i]["text"][:300],
            })
    assert used_overrides == set(overrides), set(overrides) - used_overrides
    assert used_ties == set(CLASS_TIEBREAKS), set(CLASS_TIEBREAKS) - used_ties
    assert used_imp == set(IMPORTANCE_TIEBREAKS), set(IMPORTANCE_TIEBREAKS) - used_imp

    kept = [i for i in ids if i not in drop_ids]
    with OUT_SET.open("w") as fh:
        for i in kept:
            fh.write(json.dumps({**texts[i], **gold[i]}, ensure_ascii=False) + "\n")

    gold_labels = {i: {**gold[i], "rule": ""} for i in kept}
    vs_gold = {v: agreement(labels[v], gold_labels, kept) for v in ANNOTATORS}
    by_split = {s: Counter(gold[i]["class"] for i in kept if texts[i]["split"] == s) for s in ("train", "test")}
    summary = {
        "date": DATE,
        "set": "benchmarks/typed-decisions/sets/u3_repo.jsonl",
        "texts": "benchmarks/typed-decisions/sets/u3_texts.jsonl",
        "guidelines": "benchmarks/typed-decisions/validation/u3-guidelines.md",
        "scripts": ["benchmarks/typed-decisions/repo_sets_u3.py", "benchmarks/typed-decisions/relabel_u3.py",
                    "benchmarks/typed-decisions/adjudicate_u3.py"],
        "items": len(kept),
        "split": dict(Counter(texts[i]["split"] for i in kept)),
        "annotators": {
            name: {**cfg, "file": f"validation/u3/{name}.jsonl", "items": len(labels[name]),
                   "by_class": dict(Counter(r["class"] for r in labels[name].values())),
                   "by_importance": dict(Counter(r["importance"] for r in labels[name].values() if r["importance"]))}
            for name, cfg in ANNOTATORS.items()
        },
        "gold_rule": "class = majority of Claude, GLM and gpt-oss; importance = majority bucket of the "
                     "voters who gave a class other than none, low against high = medium; ties and "
                     "overrides only by a named guideline rule, all logged below",
        "codex_relabel": "pending: the Codex blind relabel used for U1/U2 was not run because the user "
                         "paused Codex",
        "gemini_relabel": "partial: 39 of 200 items before the Antigravity quota ran out (resets in "
                          "about 162 h); gpt-oss-120b is the second external voter instead",
        "class_splits": dict(splits),
        "gold": {
            "by_class": dict(Counter(g["class"] for g in gold.values())),
            "by_importance": dict(Counter(g["importance"] for g in gold.values() if g["importance"])),
            "by_split_and_class": {s: dict(c) for s, c in by_split.items()},
            "by_source_and_class": {s: dict(Counter(gold[i]["class"] for i in sids if i in gold))
                                    for s, sids in by_source.items()},
        },
        "agreement_metric": "class: raw agreement and Cohen's kappa over the five classes; "
                            "capture_vs_none: kappa over the binary store / do not store decision; "
                            "importance: agreement, kappa and linear-weighted kappa over the items both "
                            "annotators labeled with a class other than none",
        "pairwise": pairwise,
        "gemini_partial": gemini,
        "annotator_vs_gold": vs_gold,
        "annotator_vs_gold_note": "Claude's labels are one of the three votes, so its agreement with "
                                  "the gold is not independent evidence.",
        "overrides": OVERRIDES,
        "class_tiebreaks": [{"id": k, "class": v[0], "rule": v[1], "reason": v[2]} for k, v in CLASS_TIEBREAKS.items()],
        "importance_tiebreaks": [{"id": k, "importance": v[0], "rule": v[1], "reason": v[2]}
                                 for k, v in IMPORTANCE_TIEBREAKS.items()],
        "drops": DROPS,
        "disputed_items": disputed,
        "caveat": "The annotators are models, not people; a human spot-check of the disputed items is "
                  "still recommended before publishing anything.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")

    print(f"items {len(kept)}  gold {summary['gold']['by_class']}  importance {summary['gold']['by_importance']}")
    print(f"class splits {dict(splits)}")
    for k, v in pairwise.items():
        o = v["overall"]
        print(f"{k:20} class {o['class']['agree']}/{o['items']} kappa {o['class']['kappa']}  capture kappa "
              f"{o['capture_vs_none']['kappa']}  importance {o['importance_where_both_capture']['agree']}/"
              f"{o['importance_where_both_capture']['items']} kappa {o['importance_where_both_capture']['kappa']}"
              f" wkappa {o['importance_where_both_capture']['linear_weighted_kappa']}")
    for k, v in gemini.items():
        print(f"{k:20} class {v['class']['agree']}/{v['items']} kappa {v['class']['kappa']}")
    print(f"disputed {len(disputed)}  overrides {len(OVERRIDES)}  class ties {len(CLASS_TIEBREAKS)}  "
          f"importance ties {len(IMPORTANCE_TIEBREAKS)}")


if __name__ == "__main__":
    main()
