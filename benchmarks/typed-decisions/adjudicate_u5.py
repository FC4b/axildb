#!/usr/bin/env python3
"""Adjudicate the U5 entity labels and write the gold set.

Inputs (all under validation/u5/):
  claude-shard{0,1,2}.jsonl   Claude's labels, written with the guidelines
  glm.jsonl                   GLM, blind (ids mapped back via glm-map.json)
  gemini.jsonl                Gemini, blind (ids mapped back via gemini-map.json)

Surface forms are compared after `norm`: case-insensitive, with surrounding
backticks and quotes and trailing sentence punctuation stripped. Each
annotator contributes one form per item at most (the first occurrence keeps
its type, per guidelines section 4 "Choosing between types").

Gold rule:
  * a form is gold if at least 2 of the 3 annotators listed it;
  * the majority is overridden only by an explicit guideline rule, listed in
    OVERRIDES with the rule it applies;
  * the gold type is the majority type; a tie is broken by the guidelines
    (TYPE_TIEBREAKS, each with its reason);
  * an item is dropped only if it is too ambiguous to label (DROPS).

Writes sets/u5_repo.jsonl (the u5_texts fields plus "entities") and
validation/<DATE>-u5-labels.json (annotators, agreement, overrides, ties,
drops and every disputed item).
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "validation" / "u5"
TEXTS = HERE / "sets" / "u5_texts.jsonl"
OUT_SET = HERE / "sets" / "u5_repo.jsonl"
DATE = "2026-10-14"
OUT_JSON = HERE / "validation" / f"{DATE}-u5-labels.json"

TYPES = ["file", "code", "project", "concept"]
STRATA = ["code_heavy", "mixed", "junk_trap", "markdown", "prose"]
ANNOTATORS = {
    "claude": {
        "model": "claude-opus-5-5 (Claude Code)",
        "files": [f"claude-shard{k}.jsonl" for k in range(3)],
        "blind": False,
        "note": "Written alongside the guidelines by the session that built the set, in three "
                "shards; the reference labels, as for U1/U2.",
    },
    "glm": {
        "model": "zai-coding-plan/glm-5.3 via OpenCode",
        "files": ["glm.jsonl"],
        "blind": True,
        "note": "Neutral ids (b-NNN) in shuffled order, batches of 20, guidelines plus item text "
                "only; map in glm-map.json, raw output in glm-raw/.",
    },
    "gemini": {
        "model": "gemini-3.8-flash-high via agy (Antigravity CLI)",
        "files": ["gemini.jsonl"],
        "blind": True,
        "note": "Neutral ids (b-NNN) in its own shuffled order, batches of 20, guidelines plus "
                "item text only; map in gemini-map.json, raw output in gemini-raw/ (the first "
                "attempts that hit the 300 s print timeout are kept in gemini-raw/timeout300/).",
    },
}

# Majority overridden by an explicit guideline rule. Keys are normalized forms.
OVERRIDES = [
    {"id": "u5-009", "form": "store", "action": "add", "text": "Store", "type": "code",
     "rule": "guidelines 6 (Capitalized words) + 3.3",
     "reason": "\"creates a Store<PluginState>\": Store is capitalized mid-sentence where English "
               "would not capitalize it, so it is a code type name; Store<PluginState> is a type "
               "expression whose names are Store and PluginState. Claude listed Store, GLM listed "
               "the whole expression, Gemini only PluginState."},
    {"id": "u5-076", "form": "recall", "action": "add", "text": "recall", "type": "code",
     "rule": "guidelines 6 (Commands)",
     "reason": "`recall --recall-format context-block ...` is a backticked command line; its "
               "command word counts even when it is a plain word (cf. worked example 9, heal)."},
    {"id": "u5-101", "form": "exit", "action": "add", "text": "exit", "type": "code",
     "rule": "guidelines 3.4 + 6 (Plain words)",
     "reason": "\"reproduce e.exit()\": a method called on a local variable keeps the method "
               "(rule 3.4, e.exit_code() gives exit_code), and a plain word joined to code syntax "
               "is code-marked (render() example). The same item's render, to_string, use_stderr "
               "and exit_code are gold; exit is not in the 5.7 generic-name list."},
    {"id": "u5-153", "form": "jev", "action": "add", "text": "Jev", "type": "project",
     "rule": "guidelines 3.6 (Derived words)",
     "reason": "\"the Jev-style question framing\": label the name inside a hyphenated adjective "
               "(SQLite-style gives SQLite). All three annotators labeled Jev in the same "
               "construction in u5-137."},
    {"id": "u5-158", "form": "jev", "action": "add", "text": "Jev", "type": "project",
     "rule": "guidelines 3.6 (Derived words)",
     "reason": "\"The Jev-style framing\": same rule and same case as u5-153."},
    {"id": "u5-067", "form": "heal", "action": "remove",
     "rule": "guidelines 6 (Plain words, Commands)",
     "reason": "\"heal reports ...\" and \"So heal --reindex returns ...\": heal is one of the "
               "guidelines' named plain-word examples, labeled only when code-marked (a backtick "
               "span by itself, or joined to code syntax without a space). Neither occurrence is "
               "marked, and neither starts a command line (no program, no backticks)."},
    {"id": "u5-099", "form": "heal", "action": "remove",
     "rule": "guidelines 6 (Plain words, Commands)",
     "reason": "\"incl. heal via open_with_all_detected\" and \"so heal --reindex --dry-run "
               "reported\": unmarked plain word outside a command line, as in u5-067. The "
               "command `axil heal` in the same item stays gold."},
]

# Gold type when the annotators who listed a form split evenly (or 1/1/1).
TYPE_TIEBREAKS = {
    ("u5-011", "rg"): ("project", "\"(rg/grep/find/fd/ls/tree)\" names the tools in prose: "
                                  "\"named in prose, the same tool is project\" (guidelines 4)."),
    ("u5-011", "grep"): ("project", "Same list as rg."),
    ("u5-011", "fd"): ("project", "Same list as rg."),
    ("u5-011", "ls"): ("project", "Same list as rg."),
    ("u5-034", "bindgen"): ("code", "\"via bindgen\" refers to the wasmtime::component::bindgen! "
                                    "macro the body names; another project's identifier is code "
                                    "(guidelines 4), as all voters typed it in u5-009 and u5-088."),
    ("u5-115", "macos-13"): ("code", "\"the retired macos-13 runner\" is a CI runner label, a "
                                     "config identifier like the target triples and feature "
                                     "names the guidelines list under code, not the operating "
                                     "system (macOS)."),
}

# Items too ambiguous to label. None were found.
DROPS = []

QUOTES = "`\"'“”‘’"


def norm(s: str) -> str:
    """Comparison form: case-folded, outer quotes/backticks and trailing punctuation removed."""
    prev = None
    s = s.strip()
    while s != prev:
        prev = s
        s = s.strip(QUOTES).strip()
        s = s.rstrip(".,;:!?")
    return s.lower()


def load_annotator(files):
    rows = {}
    for f in files:
        for line in (V / f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                assert r["id"] not in rows, f"duplicate id {r['id']} in {f}"
                rows[r["id"]] = r
    return rows


def form_map(entities):
    """norm -> {"text", "type"}; the first occurrence of a form wins."""
    out = {}
    for e in entities:
        n = norm(e["text"])
        if n and n not in out:
            out[n] = {"text": e["text"], "type": e["type"]}
    return out


def prf(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else None
    r = tp / n_ref if n_ref else None
    f = 2 * tp / (n_pred + n_ref) if (n_pred + n_ref) else None
    return {"precision": rnd(p), "recall": rnd(r), "f1": rnd(f), "tp": tp, "pred": n_pred, "ref": n_ref}


def rnd(x, k=3):
    return None if x is None else round(x, k)


def kappa(pairs):
    """Cohen's kappa over type labels on matched mentions."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in TYPES) / (n * n)
    return rnd((po - pe) / (1 - pe)) if pe < 1 else None


def compare(pred, ref, ids, texts):
    """Micro P/R/F1 of `pred` against `ref` (both id -> form_map) over `ids`."""
    def block(subset_ids, type_filter=None, typed=False):
        tp = n_p = n_r = 0
        for i in subset_ids:
            p, r = pred[i], ref[i]
            if type_filter:
                p = {k: v for k, v in p.items() if v["type"] == type_filter}
                r = {k: v for k, v in r.items() if v["type"] == type_filter}
            n_p += len(p)
            n_r += len(r)
            if typed or type_filter:
                tp += sum(1 for k in p if k in r and p[k]["type"] == r[k]["type"])
            else:
                tp += sum(1 for k in p if k in r)
        return prf(tp, n_p, n_r)

    matched = [(pred[i][k]["type"], ref[i][k]["type"]) for i in ids for k in pred[i] if k in ref[i]]
    return {
        "span": block(ids),
        "span_and_type": block(ids, typed=True),
        "per_type": {t: block(ids, type_filter=t) for t in TYPES},
        "per_stratum": {
            s: {"span": block([i for i in ids if texts[i]["stratum"] == s]),
                "span_and_type": block([i for i in ids if texts[i]["stratum"] == s], typed=True)}
            for s in STRATA
        },
        "type_agreement_on_matched": {
            "matched": len(matched),
            "same_type": sum(a == b for a, b in matched),
            "rate": rnd(sum(a == b for a, b in matched) / len(matched)) if matched else None,
            "kappa": kappa(matched),
        },
    }


def main():
    texts = {}
    for line in TEXTS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            texts[r["id"]] = r
    ids = sorted(texts)

    raw = {a: load_annotator(cfg["files"]) for a, cfg in ANNOTATORS.items()}
    for a, rows in raw.items():
        assert set(rows) == set(texts), f"{a} does not cover every item"
        for i, r in rows.items():
            for e in r["entities"]:
                assert e["text"] in texts[i]["text"], (a, i, e)
                assert e["type"] in TYPES, (a, i, e)
    forms = {a: {i: form_map(raw[a][i]["entities"]) for i in ids} for a in ANNOTATORS}
    names = list(ANNOTATORS)

    # ── Pairwise agreement ──────────────────────────────────────────
    pairwise = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            pairwise[f"{a}_vs_{b}"] = compare(forms[a], forms[b], ids, texts)

    # ── Votes and gold ──────────────────────────────────────────────
    overrides = {(o["id"], o["form"]): o for o in OVERRIDES}
    used_overrides, used_ties = set(), set()
    drop_ids = {d["id"] for d in DROPS}
    gold, disputed, vote_hist = {}, [], Counter()
    for i in ids:
        if i in drop_ids:
            continue
        all_forms = set().union(*(forms[a][i].keys() for a in names))
        g, item_disputes = {}, []
        for f in sorted(all_forms, key=lambda k: (texts[i]["text"].lower().find(k), k)):
            voters = [a for a in names if f in forms[a][i]]
            vote_hist[len(voters)] += 1
            types = {a: forms[a][i][f]["type"] for a in voters}
            tc = Counter(types.values()).most_common()
            in_gold = len(voters) >= 2
            gtype, tie = tc[0][0], len(tc) > 1 and tc[0][1] == tc[1][1]
            if in_gold and tie:
                key = (i, f)
                assert key in TYPE_TIEBREAKS, f"unresolved type tie {key} {types}"
                gtype = TYPE_TIEBREAKS[key][0]
                used_ties.add(key)
            texts_seen = Counter(forms[a][i][f]["text"] for a in voters)
            gtext = max(texts_seen, key=lambda t: (texts_seen[t], -texts[i]["text"].find(t)))
            ov = overrides.get((i, f))
            if ov:
                used_overrides.add((i, f))
                if ov["action"] == "remove":
                    assert in_gold, f"override removes a non-gold form {i} {f}"
                    in_gold = False
                else:
                    assert not in_gold, f"override adds a gold form {i} {f}"
                    in_gold, gtype, gtext = True, ov["type"], ov["text"]
            if in_gold:
                assert gtext in texts[i]["text"]
                g[f] = {"text": gtext, "type": gtype}
            if len(voters) < 3 or len(tc) > 1:
                item_disputes.append({
                    "form": f, "voters": voters, "types": types,
                    "gold": in_gold, "gold_type": gtype if in_gold else None,
                    **({"override": ov["action"]} if ov else {}),
                })
        # Unlisted forms an override adds.
        for (oi, of), ov in overrides.items():
            if oi == i and of not in all_forms:
                raise AssertionError(f"override for a form nobody listed: {oi} {of}")
        gold[i] = g
        if item_disputes:
            disputed.append({
                "id": i, "stratum": texts[i]["stratum"], "split": texts[i]["split"],
                "disputed_forms": item_disputes,
                "sets": {a: [[v["text"], v["type"]] for v in forms[a][i].values()] for a in names},
            })
    assert used_overrides == set(overrides), set(overrides) - used_overrides
    assert used_ties == set(TYPE_TIEBREAKS), set(TYPE_TIEBREAKS) - used_ties

    kept = [i for i in ids if i not in drop_ids]
    vs_gold = {a: compare(forms[a], gold, kept, texts) for a in names}

    # ── Write the set ───────────────────────────────────────────────
    with OUT_SET.open("w") as fh:
        for i in kept:
            row = dict(texts[i])
            ents = sorted(gold[i].values(), key=lambda e: (texts[i]["text"].find(e["text"]), e["text"]))
            row["entities"] = [{"text": e["text"], "type": e["type"]} for e in ents]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    gold_types = Counter(e["type"] for g in gold.values() for e in g.values())
    split_counts = Counter(texts[i]["split"] for i in kept)
    summary = {
        "date": DATE,
        "set": "benchmarks/typed-decisions/sets/u5_repo.jsonl",
        "texts": "benchmarks/typed-decisions/sets/u5_texts.jsonl",
        "guidelines": "benchmarks/typed-decisions/validation/u5-guidelines.md",
        "script": "benchmarks/typed-decisions/adjudicate_u5.py",
        "items": len(kept),
        "split": dict(split_counts),
        "annotators": {
            a: {**{k: v for k, v in cfg.items() if k != "files"},
                "files": [f"validation/u5/{f}" for f in cfg["files"]],
                "items": len(raw[a]),
                "mentions_listed": sum(len(r["entities"]) for r in raw[a].values()),
                "forms_after_norm": sum(len(forms[a][i]) for i in ids),
                "by_type": dict(Counter(v["type"] for i in ids for v in forms[a][i].values()))}
            for a, cfg in ANNOTATORS.items()
        },
        "normalization": "case-insensitive; surrounding backticks and quotes and trailing .,;:!? "
                         "stripped; one form per annotator per item (first occurrence keeps its "
                         "type)",
        "gold_rule": "a form is gold if at least 2 of the 3 annotators listed it; gold type is the "
                     "majority type; majority overridden only by an explicit guideline rule",
        "codex_relabel": "pending: the Codex blind relabel used for U1/U2 was not run because the "
                         "user paused Codex; the external blind annotators here are GLM and Gemini",
        "votes_per_form": {str(k): vote_hist[k] for k in sorted(vote_hist)},
        "gold": {
            "mentions": sum(len(g) for g in gold.values()),
            "by_type": dict(gold_types),
            "items_with_no_entities": sum(1 for g in gold.values() if not g),
            "per_split": {s: sum(len(gold[i]) for i in kept if texts[i]["split"] == s) for s in ("train", "test")},
        },
        "agreement_metric": "mention-level micro precision/recall/F1 of the first annotator against "
                            "the second as reference (F1 is symmetric); span = normalized form "
                            "match, span_and_type = form and type match; per_type counts only "
                            "mentions both typed as that type; type agreement is over form-matched "
                            "mentions",
        "pairwise": pairwise,
        "annotator_vs_gold": vs_gold,
        "annotator_vs_gold_note": "Claude's labels are one of the three votes, so its agreement "
                                  "with the gold is not independent evidence.",
        "overrides": OVERRIDES,
        "type_tiebreaks": [{"id": k[0], "form": k[1], "type": v[0], "reason": v[1]}
                           for k, v in TYPE_TIEBREAKS.items()],
        "drops": DROPS,
        "drops_note": "No item was too ambiguous to label.",
        "disputed_items": disputed,
        "caveat": "The annotators are models, not people; a human spot-check of the disputed "
                  "items is still recommended before publishing anything.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")

    print(f"items {len(kept)}  gold mentions {summary['gold']['mentions']} {dict(gold_types)}")
    print(f"votes per form {summary['votes_per_form']}")
    for k, v in pairwise.items():
        print(f"{k:22} span F1 {v['span']['f1']}  typed F1 {v['span_and_type']['f1']}  "
              f"type agree {v['type_agreement_on_matched']['rate']} (kappa {v['type_agreement_on_matched']['kappa']})")
    for a, v in vs_gold.items():
        print(f"{a:8} vs gold  P {v['span']['precision']} R {v['span']['recall']} F1 {v['span']['f1']}")
    print(f"overrides {len(OVERRIDES)}  type ties {len(TYPE_TIEBREAKS)}  drops {len(DROPS)}  "
          f"disputed items {len(disputed)}")


if __name__ == "__main__":
    main()
