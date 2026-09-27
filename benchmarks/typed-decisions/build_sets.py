#!/usr/bin/env python3
"""Build the U1 supersession pairs derived from LongMemEval (oracle split).

Positives come from `knowledge-update` questions: the earliest and latest
evidence user turns are the OLD and NEW statement of one changing fact.

Negatives are the pairs a naive detector confuses with an update:
  * both_hold  - two evidence turns of a `multi-session` question: separate
                 facts the answer aggregates, so neither replaces the other.
  * same_topic - the OLD evidence turn against a non-evidence user turn from
                 the NEW session: same conversation, nothing changed.

Deterministic: no randomness beyond a fixed seed, so re-running on the same
dataset bytes reproduces `sets/u1_lme.jsonl` exactly.

Usage:
  curl -L -o data/longmemeval_oracle.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
  python build_sets.py
"""

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "data" / "longmemeval_oracle.json"
OUT = HERE / "sets" / "u1_lme.jsonl"

# Long chatty turns are capped so every candidate sees the same text, and so
# 512-token models are not silently judged on a truncated tail.
MAX_CHARS = 1200
SEED = 13

# Pairs whose dataset label is right but whose clipped text does not show it.
# Removed after blind annotation (validation/2026-09-27-labels.json).
EXCLUDE = {
    "lme-ku-dfde3500": "tutor change Juan -> Maria is implicit; both annotators judged the texts compatible",
}


def clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def split_of(pair_id: str) -> str:
    """40% train (calibration fitting) / 60% test, stable per pair id."""
    h = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 4 else "test"


def evidence_turns(question):
    """Evidence user turns per session, ordered by session date."""
    sessions = sorted(
        zip(question["haystack_dates"], question["haystack_sessions"]),
        key=lambda pair: pair[0],
    )
    out = []
    for date, session in sessions:
        ev = [t for t in session if t.get("has_answer") and t["role"] == "user"]
        other = [t for t in session if not t.get("has_answer") and t["role"] == "user"]
        out.append((date, ev, other))
    return out


def main():
    data = json.loads(ORACLE.read_text())
    rng = random.Random(SEED)
    rows = []

    def add(pair_id, old, new, label, kind, qid):
        rows.append({
            "id": pair_id,
            "old": clip(old),
            "new": clip(new),
            "label": label,
            "kind": kind,
            "source": f"longmemeval_oracle:{qid}",
            "split": split_of(pair_id),
        })

    for q in data:
        qid = q["question_id"]
        sessions = [s for s in evidence_turns(q) if s[1]]
        if q["question_type"] == "knowledge-update" and len(sessions) >= 2:
            (_, old_ev, _), (_, new_ev, new_other) = sessions[0], sessions[-1]
            add(f"lme-ku-{qid}", old_ev[0]["content"], new_ev[-1]["content"], 1, "update", qid)
            if new_other:
                pick = rng.choice(new_other)
                add(f"lme-st-{qid}", old_ev[0]["content"], pick["content"], 0, "same_topic", qid)
        elif q["question_type"] == "multi-session" and len(sessions) >= 2:
            (_, a_ev, _), (_, b_ev, _) = sessions[0], sessions[1]
            add(f"lme-bh-{qid}", a_ev[0]["content"], b_ev[0]["content"], 0, "both_hold", qid)

    # Keep the classes comparable in size: all updates, and the same number of
    # each negative kind (multi-session has more candidates than we need).
    n_pos = sum(r["label"] for r in rows)
    both = [r for r in rows if r["kind"] == "both_hold"]
    rng.shuffle(both)
    keep_both = {r["id"] for r in both[:n_pos]}
    rows = [r for r in rows if r["kind"] != "both_hold" or r["id"] in keep_both]
    rows = [r for r in rows if r["id"] not in EXCLUDE]
    rows.sort(key=lambda r: r["id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = {}
    for r in rows:
        kinds[(r["kind"], r["split"])] = kinds.get((r["kind"], r["split"]), 0) + 1
    print(f"wrote {len(rows)} pairs to {OUT.relative_to(HERE)}")
    for k in sorted(kinds):
        print(f"  {k[0]:<11} {k[1]:<5} {kinds[k]}")


if __name__ == "__main__":
    main()
