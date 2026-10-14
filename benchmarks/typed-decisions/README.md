# Typed-decisions bake-off (Phase 29)

For each judgement call Axil makes about its own memories, this harness asks
one question: does a small model make the call better than the rule Axil uses
today?

| Decision | Question | What Axil does today |
|---|---|---|
| **U1** | Does NEW replace, update or contradict OLD? | `check_conflict` (bge-small cosine ≥ 0.92, then entity + negation rules) |
| **U2** | Does LATER show that ERROR is actually fixed? | Nothing: errors have no resolved status |
| **U5** | Which strings in a stored note are entities worth linking? | `extract_entities` (backticks, paths, CamelCase/snake_case, quoted strings, code-symbol patterns), run by `auto_link` on every insert |
| **U3** | Should text from an agent's actions become a memory record, in which table (decision / error / context / rule) and at what importance? | `auto_capture::analyze` (substring patterns for errors and decisions, each with a confidence; the brain hook stores captures at 0.8 or more from a failed shell command's output), then `compute_importance` on insert |

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

U3 has four kinds (`run_u3_bakeoff.py`):
- the **production classifier**, through the same Rust crate (`heuristic/ --capture`), at the hook's `--min-confidence 0.8` (the control) and the CLI default 0.7;
- a **logistic-regression head on the bge-small vector** Axil already computes (`heuristic/ --embed`), trained on the live DB's own records with their table as a weak label, and a variant that also trains on the U3 train split;
- **zero-shot GLiClass** (Knowledgator `gliclass-edge-v3.0`, `gliclass-base-v3.0`);
- **GLiNER2.5 classification** (Fastino `gliner2.5-small-v1`, `gliner2.5-base-v1`), each pinned by revision.

The plan and the go/no-go rules are in `tasks/phase-29-typed-decisions-laya.md`. That file is local-only.

## Sets (`sets/`, tracked)

| File | Pairs | Source | Labels |
|---|---|---|---|
| `u1_lme.jsonl` | 209 | LongMemEval oracle split (MIT, see `NOTICE-longmemeval.md`), rebuilt by `build_sets.py` | Mechanical, from the dataset's evidence flags: 69 knowledge-update pairs (positives); 70 multi-session "both still hold" pairs and 70 "same conversation, nothing changed" pairs (negatives). One pair is excluded; see `EXCLUDE` |
| `u1_repo.jsonl` | 40 | This repo's own `.axil` decisions / errors / context, shortened (`repo_sets.py`) | Written by Claude, then blind-validated (see below) |
| `u2_repo.jsonl` | 31 | Same (`repo_sets.py`) | Same. The negatives are deliberate traps: still broken, only planned, merged but not published, a fix for something else |
| `u2_github.jsonl` | 170 | Bug issues + PR titles from redb, tantivy, ort, ripgrep, uv and tokio (`build_u2_github.py`; needs `gh`) | From GitHub links: **fixed** = the merged PR that closed the issue (94). The 76 negatives are three kinds: **other_fix** = the closing PR of a different bug in the same repo, picked for word overlap; **mentioned_not_fix** = a merged same-repo PR that references the issue but didn't close it; **downstream_workaround** = a merged PR in another repo that references it. Issue numbers, URLs and closing keywords are stripped from the titles. 9 pairs were dropped after validation (`validation/u2_github_drops.json`) |
| `u5_repo.jsonl` | 160 texts, 1,138 entity mentions | This repo's own `.axil` decisions, errors, context, rules, checkpoints and commits, shortened (`repo_sets_u5.py`, which writes the unlabeled `u5_texts.jsonl`). Five strata: code_heavy 44, mixed 38, junk_trap 33 (prose that pattern extractors turn into junk), markdown 24, prose 21 | Entity spans typed `file` / `code` / `project` / `concept` per `validation/u5-guidelines.md`. Written by Claude, then blind-labeled by two other models; gold is the 2-of-3 majority (`adjudicate_u5.py`, see below) |
| `u3_repo.jsonl` | 200 texts | What auto-capture is shown, rebuilt from this repo's Claude Code session transcripts and its `.axil` (`repo_sets_u3.py`, which writes the unlabeled `u3_texts.jsonl`). Five sources: bash_failure 56 (the output of a shell command that exited non-zero, exactly what the brain hook pipes to `axil auto-capture`), bash_output 30 (successful output and status lines), agent_text 56 (the agent's own paragraphs), agent_brief 22 (paragraphs of briefs written for subagents), commit_subject 36. Half of each source's quota went to texts containing a production pattern where there were enough (72 in all; agent_brief had 1 and commit_subject none, so the rest of their quota is plain text) | One class (decision / error / context / rule / none) and, unless none, an importance (low / medium / high) per `validation/u3-guidelines.md`. Written by Claude, then blind-labeled by two other models; gold is the 2-of-3 majority (`adjudicate_u3.py`, see below): none 143, error 35, context 11, decision 6, rule 5 |

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

**U3 capture classes** (`validation/2026-10-14-u3-labels.json`). Claude labeled all 200 texts from the guidelines. Two models then labeled them **blind** (`relabel_u3.py`): neutral ids in their own shuffled order, batches of 20, the guidelines and the item text only, a strict JSON answer parsed defensively, every attempt kept under `validation/u3/`:
- GLM `zai-coding-plan/glm-5.3` via OpenCode;
- `gpt-oss-120b-medium` via `agy`. It stands in for Gemini `gemini-3.8-flash-high`, whose Antigravity quota ran out after 39 items (every later call returned `RESOURCE_EXHAUSTED`, resetting in about 162 hours); those 39 Gemini labels are reported for agreement, not counted as a vote.

The Codex relabel is pending here too (Codex is paused).

| Pair | Class agreement | Class κ | Store-or-not κ | Importance (both stored) | Importance κ (linear-weighted) | Lowest source (class κ) |
|---|---|---|---|---|---|---|
| Claude vs GLM | 182/200 | 0.81 | 0.80 | 42/53 | 0.67 (0.73) | bash_failure 0.69 |
| Claude vs gpt-oss | 163/200 | 0.63 | 0.66 | 31/48 | 0.44 (0.47) | bash_output 0.24 |
| GLM vs gpt-oss | 159/200 | 0.61 | 0.66 | 34/54 | 0.40 (0.47) | bash_output 0.32 |
| Gemini vs Claude / GLM / gpt-oss (39 items) | 36 / 35 / 34 | 0.83 / 0.80 / 0.74 | 0.82 / 0.77 / 0.71 | 6/10, 9/11, 8/10 | | |

gpt-oss labels more texts `context` (28 against Claude's 9 and GLM's 14), most often for shell output that prints source code, a stored record or a document (none by rule N2), and for status reports. Gold is the 2-of-3 majority: 154 unanimous, 42 split 2-1, and 4 three-way splits broken by a named guideline rule. One majority was overridden by rule N2 (a stored record printed as JSON is ordinary output, and so are the printed commit messages that decided two of the three-way splits). Importance takes the majority bucket of the voters who stored the item, low against high is medium, and 3 adjacent ties were broken by a named rule. Every case is logged with its reason, along with all 63 disputed items; none was dropped. Importance is the least reliable label (κ 0.40–0.67).

Each pair's `split` is 40% `train` / 60% `test`, derived from sha256 of its id. The train split is used only to fit the decision threshold and the Platt calibration; every reported metric is on test. U5 texts use the same rule (63 train, 97 test); there, train only picks each model's label phrasing and span threshold. U3 texts too (75 train, 125 test): train picks each model's setting, and the LR variant and the importance head also learn from it. By chance only 1 of the 11 context items fell in train.

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

```bash
# U3 capture-classification set and bake-off. The model arms also need
# scikit-learn and gliclass (requirements.txt); the control and the LR heads
# need the heuristic binary; --prepare reads the live DB with `axil export` only.
python repo_sets_u3.py --db .axil/memory.axil --save-pool out/u3_pool.jsonl   # optional: rebuild sets/u3_texts.jsonl
python relabel_u3.py --annotator glm --lanes 2       # optional: blind relabels (raw output kept)
python relabel_u3.py --annotator gptoss --lanes 2
python adjudicate_u3.py                              # optional: rebuild sets/u3_repo.jsonl and the validation JSON
cargo build --release --manifest-path heuristic/Cargo.toml
python run_u3_bakeoff.py --prepare --db .axil/memory.axil   # weak-label records + bge-small vectors
python run_u3_bakeoff.py --arm all --parallel 2      # every arm in its own process, then the summary
python run_u3_bakeoff.py --summarize                 # re-score the saved arm outputs only
```

`repo_sets_u3.py` reads the Claude Code transcripts under
`~/.claude/projects/<this repo>/` and the DB's `commits`, both up to a fixed
cutoff, so a rerun picks the same pool; Claude Code deletes old transcripts,
so `--save-pool` keeps the candidates and `--pool` rebuilds from them.
`run_u3_bakeoff.py` trains the LR heads in Python on vectors the Rust embedder
wrote, so they see exactly the vector Axil stores. Every model setting (label
phrasing and none threshold; C, class weighting and threshold) is chosen on
`train`: the best train macro-F1 whose train false-capture rate is at most the
control's. Zero-shot scores are computed once per phrasing and swept offline;
the chosen setting then runs for real on `test` (batch 1, 4 threads). Each arm
runs in its own process under `/usr/bin/time -l`, at most two at once, with
the load average next to every latency. It writes
`benchmarks/results/typed-decisions-u3-bakeoff-<date>.json` and, for the head
the classifier step would load, `typed-decisions-u3-heads-<date>.json`
(weights, bias, decision rule and parity fixtures).

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

For U3:
- **Accuracy and macro-F1** over the five classes on test, and **per-class precision / recall**.
- **False-capture rate:** gold `none` stored as anything else, over gold `none`. It is the governing error: a junk error is shown as open at the start of every session until someone resolves it. The verdict compares models with the control at an equal or lower rate.
- **Wrong-table rate:** stored in the wrong table, over items that should be stored and were. **Miss rate:** gold not-none predicted none.
- **Importance-bucket accuracy** over gold not-none test items, next to the accuracy of always answering train's most common bucket.
- **CPU latency** p50/p95 per text with the model loaded, the load average, load time, peak RSS and peak memory footprint (`/usr/bin/time -l`), and size on disk. For the heads, latency includes the bge-small embed of the text.

## Numbers policy

U2 now has 201 validated pairs. Still, none of these numbers go into README, docs or CLI output: they measure a candidate for a feature that isn't built yet. The same holds for the U5 and U3 bake-off results; their latencies were also measured on a shared, heavily loaded machine, so read them with the load average recorded next to each. Once a winner is wired into Axil, its end-to-end numbers get their own committed result. The labels are model-validated (blind agreement between models); a human spot-check of the disputed items listed in the validation files is still recommended before publishing anything.
