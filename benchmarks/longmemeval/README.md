# LongMemEval — Axil retrieval benchmark

This crate is the Phase 15 P0.1 benchmark harness. It is **workspace-excluded**
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
# 20-question smoke run, vector strategy, bge-small embedder
scripts/longmemeval-gate.sh

# Seed a fresh baseline (overwrites benchmarks/longmemeval/baseline.jsonl)
scripts/longmemeval-gate.sh --save

# Full 500-question run (slow; needs a beefy machine — see R1 in task spec)
scripts/longmemeval-gate.sh --questions 0 --save

# Switch strategies
scripts/longmemeval-gate.sh --strategy recall-qtc --questions 30

# With reranker (needs `--features rerank` on the bench + ONNX model present)
scripts/longmemeval-gate.sh --rerank --questions 30
```

`--variant s` is the default; `m` and `oracle` are also supported.

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
  --variant s --limit 20 --strategy vector --top-k 5 --model bge-small
```

Output is a single JSON `BenchmarkReport` on stdout (progress on stderr):

```json
{
  "benchmark": "LongMemEval",
  "variant": "s",
  "strategy": "vector",
  "rerank": "off",
  "top_k": 5,
  "total_questions": 20,
  "overall": { "hit_rate": 0.90, "avg_recall": 0.88, "avg_precision": 0.176 },
  "by_category": { "single-session-user": { ... } },
  "misses": [ ... ]
}
```

## Published numbers

The 2026-05-17 row is the headline baseline — the first full 500-question
run, committed at `benchmarks/results/baseline-500.json` and gated in CI via
`benchmarks/longmemeval/baseline.jsonl`. The smaller rows are Phase 15 P0
CPU verification runs.

| Run | Strategy | Questions | recall@5 | recall@1 | Notes |
|-----|----------|-----------|----------|----------|-------|
| 2026-05-17 | vector | **500 (full `s`)** | **0.936** | — | **Baseline.** bge-small, RTX 3080. hit_rate 96.4 %, precision@5 32.0 %. 15m19s wall. Clears the Phase 13 historical 94.4 %. |
| 2026-05-16 | vector | 50 (single-session-user) | 0.880 | 0.760 | bge-small, P0 CPU verify |
| 2026-05-16 | vector | 10 | 0.800 | — | gte-modernbert-base (P0.2 verify; ties bge-small, ~2× slower on CPU) |
| 2026-05-16 | recall + rerank | 30 | — | 0.900 | answerai-colbert via P0.3 (+12.5 % vs no rerank) |

Re-running any row reproduces it; `--save` after a clean pass promotes it
back into `baseline.jsonl`.

## CI integration

`.github/workflows/longmemeval.yml` runs `scripts/longmemeval-gate.sh` on PRs
that touch `crates/axil-{core,vector,fts,graph,indexer,memory}`. The gate
**skips silently** on a fresh checkout when `data/longmemeval_s_cleaned.json`
isn't present — useful for forks that don't want to pay the dataset download.
The reference workflow caches both the dataset and the bench's `target/` to
keep wall time tractable.
