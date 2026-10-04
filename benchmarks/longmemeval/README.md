# LongMemEval — Axil retrieval benchmark

This crate is Axil's LongMemEval retrieval harness. It is **workspace-excluded**
(its own Cargo workspace) so the main `cargo build` from the repo root stays
fast — build it explicitly via `--manifest-path`.

Upstream dataset: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
Paper: LongMemEval (ICLR 2025), 500 questions across 5 abilities.

## Bootstrap

The dataset isn't checked into git (~265 MB). Pull the cleaned split once:

```bash
mkdir -p benchmarks/longmemeval/data
# Pick one — the `s` split is the CI gate target.
curl -L -o benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
# Optional: the `m` split for the larger ablation runs.
curl -L -o benchmarks/longmemeval/data/longmemeval_m_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
```

## Run the gate

```bash
# Full gate: 500 questions, recall-qtc, bge-small — the configuration the
# committed baseline was recorded at (slow)
scripts/longmemeval-gate.sh

# Seed a fresh baseline (overwrites benchmarks/longmemeval/baseline.jsonl)
scripts/longmemeval-gate.sh --save

# The PR smoke CI runs: 20 questions over `recall`, 10% tolerance
scripts/longmemeval-gate.sh --questions 20 --strategy recall --tolerance 0.1

# Switch strategies (a new configuration needs its own `--save` first)
scripts/longmemeval-gate.sh --strategy vector --questions 30

# With reranker (needs `--features rerank` on the bench + ONNX model present)
scripts/longmemeval-gate.sh --rerank --questions 30
```

`--variant s` is the default; `m` and `oracle` are also supported. The `m`
split (about 480 sessions per question, 2.6 GB) is the retrieval setting the
paper used; a question takes roughly ten times as long as on `s`. The gate
skips loudly when the requested split's file is missing.

Each configuration (split, strategy, top-k, question count, rerank) has its own
baseline: `baseline.jsonl` for the default 500-question `recall-qtc` run, and
`baselines/<variant>-<strategy>-k<top_k>-n<questions>.json` for the rest. A
comparison against a baseline recorded under another configuration fails
instead of passing on a meaningless delta.

The gate exit codes:

| Code | Meaning |
|------|---------|
| 0    | Pass (or dataset missing → skip) |
| 1    | Bench binary failed / usage error |
| 2    | No baseline on disk; re-run with `--save` |
| 3    | `overall.avg_recall` regressed beyond `--tolerance` (default 2 %) |

## Direct bench invocation

```bash
cargo run --release --manifest-path benchmarks/longmemeval/Cargo.toml -- \
  --variant s --limit 20 --strategy recall-qtc --top-k 5 --model bge-small
```

Output is a single JSON `BenchmarkReport` on stdout (progress on stderr). For
this command it looks like the committed
[`baselines/s-recall-qtc-k5-n20.json`](baselines/s-recall-qtc-k5-n20.json)
(values rounded):

```json
{
  "benchmark": "LongMemEval",
  "variant": "s",
  "strategy": "recall-qtc",
  "rerank": "off",
  "top_k": 5,
  "total_questions": 20,
  "overall": { "hit_rate": 0.90, "avg_recall": 0.90, "avg_precision": 0.18,
               "recall_all": 0.90, "ndcg_at_10": 0.857, "turn_recall_compact": 0.0,
               "turn_recall_full": 0.90, "tokens_compact": 250, "tokens_full": 15515, … },
  "by_category": { "single-session-user": { ... } },
  "misses": [ ... ]
}
```

## Published numbers

The Recall-QTC row is the headline baseline: recall@5 **0.935**, which backs the
README's **93.5 %** figure. It is committed at `benchmarks/results/qtc-500.json`
and copied verbatim to `benchmarks/longmemeval/baseline.jsonl`, the gate
baseline, so the gate compares a QTC run against it when the dataset is
present. Both 2026-06-27 rows come from one re-run of the suite (`795f1e9`):
bge-small, top-k 5, ONNX Runtime CUDA on an RTX 3080. The older rows are kept
for reference and history.

| Run | Strategy | Questions | recall@5 | hit rate | precision@5 | Source | Notes |
|-----|----------|-----------|----------|----------|-------------|--------|-------|
| 2026-06-27 | **recall-qtc** | **500 (full `s`)** | **0.935** | 96.4 % | 31.7 % | [`results/qtc-500.json`](../results/qtc-500.json) | **Gate baseline** (= `baseline.jsonl`). Backs the README 93.5 %. |
| 2026-06-27 | recall (fusion) | 500 (full `s`) | 0.915 | 96.2 % | 30.4 % | [`results/fusion-500.json`](../results/fusion-500.json) | Backs the README 91.5 %. |
| initial release (`f363ffa`) | vector | 500 (full `s`) | 0.936 | 96.4 % | 32.0 % | [`results/baseline-500.json`](../results/baseline-500.json) | Reference. bge-small. |
| initial release (`f363ffa`) | recall-qtc | 500 (full `s`) | 0.945 | 97.2 % | 31.9 % | `git show f363ffa:benchmarks/results/qtc-500.json` | **Historical, superseded** by the 2026-06-27 re-run, which re-based the published figure to 93.5 %. |
| 2026-09-28 | recall | 20 | 0.900 | 90.0 % | 18.0 % | [`baselines/s-recall-k5-n20.json`](baselines/s-recall-k5-n20.json) | PR smoke baseline (see CI integration). |

To refresh a row, re-run its configuration; `--save` after a clean pass
promotes the result into the baseline for that configuration.

## CI integration

`.github/workflows/longmemeval.yml` runs `scripts/longmemeval-gate.sh`:

- **On PRs** that touch the recall core (`crates/axil-core/src/{query,scoring,brain,db}.rs`),
  `crates/engines/axil-vector/`, the gate script or the workflow: a 20-question
  `recall` smoke against `baselines/s-recall-k5-n20.json`, with 10 % tolerance.
- **Nightly and on manual dispatch:** the full gate at its defaults (500
  questions, `recall-qtc`, 2 % tolerance) against `baseline.jsonl`.

The dataset is never downloaded from a public URL in CI. The job restores it
from an Actions cache or from an optional `LONGMEMEVAL_DATASET_URL` repo
secret; when neither has it, the gate **skips loudly** (exit 0 plus a
`::warning`), so a green check does not mean recall was verified. The workflow
also caches the bge-small model and the bench's `target/`.

## Beyond session hit rate

Session hit rate on `s` is close to saturated, so it can't show most changes.
Every report also carries:

| Metric | Meaning |
|---|---|
| `recall_all` | Share of questions with *every* answer session in the top-k |
| `ndcg_at_10` | Ranking quality over the top 10 sessions (binary relevance) |
| `turn_recall_compact` | Share of answer turns the agent would actually read in the CLI's default compact output (the first 200 characters of each hit) |
| `turn_recall_full` | The same, reading each hit's full text |
| `tokens_compact`, `tokens_full` | Tokens (bytes / 4) of the top-k hits in each form |
| `evidence_per_1k_tokens_*` | Answer turns found per 1,000 tokens shown |

`ndcg_at_10` is gated with `avg_recall`; the rest is printed. The gap between
`turn_recall_compact` and `turn_recall_full` is the cost of showing a hit's
opening lines instead of the part that matched: on the first 20 `s` questions
(`recall-qtc`, top-5;
[`baselines/s-recall-qtc-k5-n20.json`](baselines/s-recall-qtc-k5-n20.json))
the right session was found for 90% of them, yet the compact output contained
the answer turn for none, while the full text cost about 15,500 tokens per
question.

