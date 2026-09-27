# Typed-decisions bake-off (Phase 29)

For each judgement call Axil makes about its own memories, this harness asks
one question: does a small model make the call better than the rule Axil uses
today?

| Decision | Question | What Axil does today |
|---|---|---|
| **U1** | Does NEW replace, update or contradict OLD? | `check_conflict` (bge-small cosine ≥ 0.92, then entity + negation rules) |
| **U2** | Does LATER show that ERROR is actually fixed? | Nothing: errors have no resolved status |

There are three arms:
- the **production heuristic** (`heuristic/`, a Rust crate that calls the real `axil_core::check_conflict`);
- **small NLI cross-encoders** via ONNX Runtime (`run_bakeoff.py`);
- **Jev-style typed-decision models:**
  - Laya (`run_laya.py`).
  - Fastino GLiNER2.5-Decide, 340M and 1B (`run_gliner_decide.py`).
  - TypeSafe Jev itself, via its hosted API (`run_jev.py`). This one is opt-in: it needs `TYPESAFE_API_KEY` and `--confirm-external`, because it uploads every pair to TypeSafe.

The plan and the go/no-go rules are in `tasks/phase-29-typed-decisions-laya.md`. That file is local-only.

## Sets (`sets/`, tracked)

| File | Pairs | Source | Labels |
|---|---|---|---|
| `u1_lme.jsonl` | 209 | LongMemEval oracle split (MIT, see `NOTICE-longmemeval.md`), rebuilt by `build_sets.py` | Mechanical, from the dataset's evidence flags: 69 knowledge-update pairs (positives); 70 multi-session "both still hold" pairs and 70 "same conversation, nothing changed" pairs (negatives). One pair is excluded; see `EXCLUDE` |
| `u1_repo.jsonl` | 40 | This repo's own `.axil` decisions / errors / context, shortened (`repo_sets.py`) | Written by Claude, then blind-validated (see below) |
| `u2_repo.jsonl` | 31 | Same (`repo_sets.py`) | Same. The negatives are deliberate traps: still broken, only planned, merged but not published, a fix for something else |
| `u2_github.jsonl` | 170 | Bug issues + PR titles from redb, tantivy, ort, ripgrep, uv and tokio (`build_u2_github.py`; needs `gh`) | From GitHub links: **fixed** = the merged PR that closed the issue (94). The 76 negatives are three kinds: **other_fix** = the closing PR of a different bug in the same repo, picked for word overlap; **mentioned_not_fix** = a merged same-repo PR that references the issue but didn't close it; **downstream_workaround** = a merged PR in another repo that references it. Issue numbers, URLs and closing keywords are stripped from the titles. 9 pairs were dropped after validation (`validation/u2_github_drops.json`) |

### Label validation (`validation/2026-09-27-labels.json`)

Two other models relabeled the pairs **blind**: all 72 repo pairs, plus 30 LongMemEval pairs (10 per kind). They saw neutral IDs in shuffled order, with no labels, kinds or sources:
- Codex `gpt-6-sol` (xhigh reasoning, read-only sandbox);
- GLM `glm-5.3-flash` via OpenCode.

| Set | Claude vs Codex | Claude vs GLM | Codex vs GLM |
|---|---|---|---|
| U1 repo (40) | 40/40 | 40/40 | 40/40 |
| U2 repo (32) | 30/32, κ 0.88 | 29/32, κ 0.81 | 29/32, κ 0.81 |
| U1 LongMemEval sample (30) | 28/30, κ 0.84 | 29/30, κ 0.92 | 29/30, κ 0.92 |

How disagreements were resolved:
- **Majority vote**, overridden only by evidence in the dataset.
- **Two pairs dropped**, where both external annotators disagreed and the text alone supports their reading:
  - `e0-09`: "README hero rewrite" read as fixing the unsourced headline.
  - `lme-ku-dfde3500`: the tutor change is only implicit in the clipped text.
- **Four split votes kept with the majority label.**

The same two models also re-annotated **all 179 GitHub pairs** blind, before any drops (`validation/2026-09-27-u2-github-labels.json`):
- Codex agreed with the link label on 162/179 (κ 0.81).
- GLM agreed on 162/179 (κ 0.81).
- The two agreed with each other on 163/179 (κ 0.82).

Pairs where **both** rejected the link label were dropped: 5 "fixed" pairs whose titles don't show the connection, and 4 "other_fix" negatives that were really near-duplicate fixes. Split votes keep the link label.

Remaining caveat: the annotators are models, not people.

Each pair's `split` is 40% `train` / 60% `test`, derived from sha256 of its id. The train split is used only to fit the decision threshold and the Platt calibration; every reported metric is on test.

## Run

```bash
# Python env (any venv; the scripts need onnxruntime, tokenizers, huggingface_hub, numpy; laya pulls torch)
pip install -r requirements.txt

# Rebuild the LongMemEval-derived pairs (optional; the output is committed)
curl -L -o data/longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
python build_sets.py && python repo_sets.py

# NLI arm (downloads ONNX models from HF on first run)
python run_bakeoff.py --threads 4

# Jev-style arm
python run_laya.py                              # English root checkpoint
python run_laya.py --subfolder multilingual
python run_laya.py --subfolder typed-decisions
python run_gliner_decide.py                                   # fastino/GLiNER2.5-Decide
python run_gliner_decide.py --model fastino/GLiNER2.5-Decide-1B
TYPESAFE_API_KEY=... python run_jev.py --confirm-external     # hosted Jev; sends data to TypeSafe

# Production-heuristic arm
cat sets/u1_lme.jsonl sets/u1_repo.jsonl > out/u1_all.jsonl
cargo run --release --manifest-path heuristic/Cargo.toml -- out/u1_all.jsonl > out/heuristic-u1.jsonl
```

Raw outputs go to `out/`, which is gitignored. The committed summary is `benchmarks/results/typed-decisions-bakeoff-<date>.json`.

## Metrics

- **AUROC** on test: threshold-free ranking quality.
- **TPR / FPR at the fitted threshold:** the highest-recall threshold whose *train* FPR is ≤ 5%, applied to test. False positives are the costly error: a false supersede hides a correct memory, and a false "fixed" hides an open error.
- **ECE** before and after Platt scaling fit on train.
- **Per-kind accuracy:** shows which kinds of negative fool a model.
- **CPU latency:** p50/p95 per pair, batch 1, model already loaded, Python-side. Load time is reported separately.

## Numbers policy

U2 now has 201 validated pairs. Still, none of these numbers go into README, docs or CLI output: they measure a candidate for a feature that isn't built yet. Once a winner is wired into Axil, its end-to-end numbers get their own committed result. The labels are model-validated (blind agreement between models); a human spot-check of the disputed items listed in the validation files is still recommended before publishing anything.
