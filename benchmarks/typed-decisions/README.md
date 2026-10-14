# Typed-decisions bake-off (Phase 29)

For each judgement call Axil makes about its own memories, this harness asks
one question: does a small model make the call better than the rule Axil uses
today?

| Decision | Question | What Axil does today |
|---|---|---|
| **U1** | Does NEW replace, update or contradict OLD? | `check_conflict` (bge-small cosine ≥ 0.92, then entity + negation rules) |
| **U2** | Does LATER show that ERROR is actually fixed? | Nothing: errors have no resolved status |
| **U5** | Which strings in a stored note are entities worth linking? | `extract_entities` (backticks, paths, CamelCase/snake_case, quoted strings, code-symbol patterns), run by `auto_link` on every insert |

For U1 and U2 there are three arms:
- the **production heuristic** (`heuristic/`, a Rust crate that calls the real `axil_core::check_conflict`);
- **small NLI cross-encoders** via ONNX Runtime (`run_bakeoff.py`);
- **Jev-style typed-decision models:**
  - Laya (`run_laya.py`).
  - Fastino GLiNER2.5-Decide, 340M and 1B (`run_gliner_decide.py`).
  - TypeSafe Jev itself, via its hosted API (`run_jev.py`). This one is opt-in: it needs `TYPESAFE_API_KEY` and `--confirm-external`, because it uploads every pair to TypeSafe.

U5 has two (`run_u5_bakeoff.py`):
- the **production extractor**, through the same Rust crate (`heuristic/ --entities`);
- **zero-shot GLiNER span models** on CPU: Fastino GLiNER2.5-small (74M) and -base (194M), and `urchade/gliner_small-v2.1`, each pinned by revision.

The plan and the go/no-go rules are in `tasks/phase-29-typed-decisions-laya.md`. That file is local-only.

## Sets (`sets/`, tracked)

| File | Pairs | Source | Labels |
|---|---|---|---|
| `u1_lme.jsonl` | 209 | LongMemEval oracle split (MIT, see `NOTICE-longmemeval.md`), rebuilt by `build_sets.py` | Mechanical, from the dataset's evidence flags: 69 knowledge-update pairs (positives); 70 multi-session "both still hold" pairs and 70 "same conversation, nothing changed" pairs (negatives). One pair is excluded; see `EXCLUDE` |
| `u1_repo.jsonl` | 40 | This repo's own `.axil` decisions / errors / context, shortened (`repo_sets.py`) | Written by Claude, then blind-validated (see below) |
| `u2_repo.jsonl` | 31 | Same (`repo_sets.py`) | Same. The negatives are deliberate traps: still broken, only planned, merged but not published, a fix for something else |
| `u2_github.jsonl` | 170 | Bug issues + PR titles from redb, tantivy, ort, ripgrep, uv and tokio (`build_u2_github.py`; needs `gh`) | From GitHub links: **fixed** = the merged PR that closed the issue (94). The 76 negatives are three kinds: **other_fix** = the closing PR of a different bug in the same repo, picked for word overlap; **mentioned_not_fix** = a merged same-repo PR that references the issue but didn't close it; **downstream_workaround** = a merged PR in another repo that references it. Issue numbers, URLs and closing keywords are stripped from the titles. 9 pairs were dropped after validation (`validation/u2_github_drops.json`) |
| `u5_repo.jsonl` | 160 texts, 1,138 entity mentions | This repo's own `.axil` decisions, errors, context, rules, checkpoints and commits, shortened (`repo_sets_u5.py`, which writes the unlabeled `u5_texts.jsonl`). Five strata: code_heavy 44, mixed 38, junk_trap 33 (prose that pattern extractors turn into junk), markdown 24, prose 21 | Entity spans typed `file` / `code` / `project` / `concept` per `validation/u5-guidelines.md`. Written by Claude, then blind-labeled by two other models; gold is the 2-of-3 majority (`adjudicate_u5.py`, see below) |

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

**U5 entities** (`validation/2026-10-14-u5-labels.json`). Claude labeled all 160 texts from the guidelines (1,150 mentions). Two models then labeled them **blind**, from the guidelines and the item text only, under neutral ids in shuffled order:
- GLM `zai-coding-plan/glm-5.3` via OpenCode (1,179 mentions);
- Gemini `gemini-3.8-flash-high` via `agy` (1,093 mentions).

The Codex relabel used for U1/U2 is still pending: Codex was paused by the user. Mentions are compared as normalized surface forms (case-insensitive; outer backticks, quotes and trailing punctuation stripped), micro-averaged:

| Pair | Span F1 | Span + type F1 | Type agreement on matched spans | Lowest stratum (span F1) | `concept` F1 |
|---|---|---|---|---|---|
| Claude vs GLM | 0.951 | 0.941 | 1,093/1,105, κ 0.98 | prose 0.896 | 0.753 |
| Claude vs Gemini | 0.950 | 0.944 | 1,057/1,063, κ 0.99 | markdown 0.916 | 0.870 |
| GLM vs Gemini | 0.944 | 0.936 | 1,061/1,069, κ 0.99 | markdown 0.922 | 0.721 |

Gold is every form at least two of the three listed (1,051 unanimous, 84 two-vote; 93 single-vote forms left out), typed by majority. The majority was overridden 7 times, each by a named guideline rule (5 additions, 2 removals of the unmarked plain word `heal`), and 6 even type splits were broken by the guidelines; every case is logged with its reason. No item was dropped. The result: 1,138 mentions (code 663, project 266, file 173, concept 36), 690 of them in the 97 test texts. `concept` is the least reliable type, as the agreement shows.

Remaining caveat: the annotators are models, not people.

Each pair's `split` is 40% `train` / 60% `test`, derived from sha256 of its id. The train split is used only to fit the decision threshold and the Platt calibration; every reported metric is on test. U5 texts use the same rule (63 train, 97 test); there, train only picks each model's label phrasing and span threshold.

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

# U1 calibration for the Rust runtime (crates/extensions/axil-decide).
# Use onnxruntime 1.22.0, the version ort 2.0.0-rc.10 bundles: the int8
# model's logits move by up to 1.3 between onnxruntime versions.
pip install onnxruntime==1.22.0 tokenizers==0.21.4 huggingface_hub numpy
python calibrate_u1.py
```

`calibrate_u1.py` fits one temperature on the one-vs-rest contradiction logit
and a threshold whose validation FPR has a 95% upper bound of 5%, both on the
`train` split, and reports on `test`. It writes
`benchmarks/results/typed-decisions-u1-calibration-<date>.json`, the runtime
file `crates/extensions/axil-decide/calibration/u1-nli-deberta-v3-xsmall-qint8.json`
and the Rust parity fixture `crates/extensions/axil-decide/tests/fixtures/u1_parity.json`.
The model and tokenizer are pinned by revision and sha256.

```bash
# U5 entity bake-off. The GLiNER arms also need gliner, sentencepiece and psutil
# (requirements.txt); the control arm needs the heuristic binary.
cargo build --release --manifest-path heuristic/Cargo.toml
python adjudicate_u5.py                # optional: rebuild sets/u5_repo.jsonl and the validation JSON
python run_u5_bakeoff.py --arm all     # every arm in its own process, then the summary
python run_u5_bakeoff.py --summarize   # re-score the saved arm outputs only
```

`run_u5_bakeoff.py` runs each GLiNER arm once per label phrasing (two at most)
over every text at a low threshold, then sweeps the threshold offline. Phrasing
and threshold are picked on `train`: the best train F1 whose train precision is
at least the production extractor's. The chosen setting then runs for real on
`test` (batch 1, 4 threads), which gives the reported predictions and latency.
Each arm runs in its own process, so peak RSS is per arm. The 1-minute load
average is recorded next to every latency, because other jobs share the
machine. It writes `benchmarks/results/typed-decisions-u5-bakeoff-<date>.json`;
`--write-junk-list` regenerates `u5_junk_words.txt`.

Raw outputs go to `out/`, which is gitignored. The committed summary is `benchmarks/results/typed-decisions-bakeoff-<date>.json`.

## Metrics

- **AUROC** on test: threshold-free ranking quality.
- **TPR / FPR at the fitted threshold:** the highest-recall threshold whose *train* FPR is ≤ 5%, applied to test. False positives are the costly error: a false supersede hides a correct memory, and a false "fixed" hides an open error.
- **ECE** before and after Platt scaling fit on train.
- **Per-kind accuracy:** shows which kinds of negative fool a model.
- **CPU latency:** p50/p95 per pair, batch 1, model already loaded, Python-side. Load time is reported separately.

For U5:
- **Mention-level micro precision / recall / F1** on test, over normalized surface forms (`adjudicate_u5.norm`), one form per text. Span = the form matches; span + type = the type matches too. Reported per type and per stratum. Precision plays the false-positive role: a wrong entity becomes a graph node that links unrelated notes.
- **Junk rate:** the share of predicted mentions that are wrong *and* are markdown syntax or made only of words in `u5_junk_words.txt` (stopwords, generic nouns, keywords, dictionary words).
- **Entities per text**, against the gold mean.
- **CPU latency** p50/p95 per text, batch 1, model loaded, with the load average. Load time, peak RSS and size on disk are reported per arm.

## Numbers policy

U2 now has 201 validated pairs. Still, none of these numbers go into README, docs or CLI output: they measure a candidate for a feature that isn't built yet. The same holds for the U5 bake-off results; the U5 latencies were also measured on a shared, heavily loaded machine, so read them with the load average recorded next to each. Once a winner is wired into Axil, its end-to-end numbers get their own committed result. The labels are model-validated (blind agreement between models); a human spot-check of the disputed items listed in the validation files is still recommended before publishing anything.
