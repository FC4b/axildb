# Axil vs. redis/agent-memory-server — LongMemEval retrieval

A reproducible head-to-head comparing **Axil** against
**[redis/agent-memory-server](https://github.com/redis/agent-memory-server)**
("AMS" — Redis's OSS self-hosted memory server, Apache 2.0) on the exact metric
the in-tree [`benchmarks/longmemeval`](../longmemeval/) harness uses:

> **retrieval recall@5** — for each question, is every answer-bearing session in
> the top-5 retrieved sessions? Averaged over the 500 LongMemEval-S questions.

Both sides run on the **same dataset bytes**, at the **same session-level
granularity**, scored by the **same recall@5 definition**, in a **no-generative-LLM
condition**. This harness builds the AMS side; the Axil side is the existing
`benchmarks/longmemeval` crate.

> **This harness is workspace-neutral.** It is Python-only (no Cargo crate), so
> it is not a workspace member and needs no entry in the root `Cargo.toml`
> `exclude` list. `out/` and `data/` here are already gitignored by the repo's
> `benchmarks/**` rules.

---

## ⛔ NO RESULTS YET — nothing here is citable

**This harness has produced no committed numbers. Zero.** No figure from an AMS
run may appear in the README, docs, CLI/MCP output, marketing, or anywhere else
until:

1. a full run's results JSON is **committed to `benchmarks/results/`** (e.g.
   `benchmarks/results/ams-longmemeval-s-<mode>-500.json`), **and**
2. that commit records the **environment**: AMS image tag, `search_mode`,
   `recency_boost`, extraction on/off, embedding model, dataset variant + size,
   date, and hardware.

This is repo policy, not a suggestion — see
[Numbers integrity](../../docs/src/advanced/context-economics.md#numbers-integrity-policy)
and `CLAUDE.md` ("Numbers integrity"). Until a committed baseline exists,
**there is no Axil-vs-AMS number.** The driver prints this reminder after every
run, and writes to gitignored `out/` precisely so a raw run cannot be mistaken
for a published result.

Any PR that quotes an Axil-vs-AMS recall figure without a matching committed
`benchmarks/results/ams-*.json` is quoting a fabricated number.

---

## Prerequisites

| Need | Why |
|------|-----|
| Docker + Docker Compose v2 | Runs Redis Stack + the AMS API + its task-worker |
| Python 3.12 | The AMS driver (`run_ams.py`) |
| `pip install -r requirements.txt` | Just `requests` (pinned) |
| The LongMemEval-S dataset | Same file the Axil harness reads |
| An embedding-provider key (`OPENAI_API_KEY`) | **AMS embeds at ingest + query — required in every mode.** See [Fairness & asymmetries](#fairness--asymmetries) |
| The Axil side (for the other half) | Build `benchmarks/longmemeval` per its own README |

### Dataset placement

The driver defaults to the **same path the Axil harness uses**, so both sides
read identical bytes:

```bash
mkdir -p benchmarks/longmemeval/data
curl -L -o benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

If the file is missing, the driver **skips loudly** (prints a `::warning::`
banner explaining where to get it and exits 0) — the same dataset-gated,
skip-loud pattern the in-tree harnesses use. A green run with no dataset
verified nothing.

---

## Run it

### 1. Bring up the AMS stack

```bash
cd benchmarks/agent-memory-compare
cp .env.example .env          # then put your OPENAI_API_KEY in .env
docker compose up -d          # pulls redis-stack + agent-memory-server, starts api (:8000)
docker compose logs -f api    # wait for it to report healthy
```

Defaults (see `docker-compose.yml`): extraction flags **OFF**
(`ENABLE_DISCRETE_MEMORY_EXTRACTION`/`ENABLE_TOPIC_EXTRACTION`/`ENABLE_NER` =
`False`), `LONG_TERM_MEMORY=True`, `DISABLE_AUTH=true`, images pinned to
`redislabs/agent-memory-server:0.15.2` and `redis/redis-stack-server:7.4.0-v8`.

### 2. Run the AMS side (this harness)

```bash
# From the repo root. Default: variant s, all 500 questions, semantic search,
# extraction off, recency off, top-k 5.
python benchmarks/agent-memory-compare/run_ams.py

# Smaller smoke run:
python benchmarks/agent-memory-compare/run_ams.py --variant s --limit 20 --verbose

# Key-free lexical condition (different from Axil's semantic recall — see below):
python benchmarks/agent-memory-compare/run_ams.py --search-mode keyword

# Recency-blended condition (approximates Axil's recall-qtc, not the vector run):
python benchmarks/agent-memory-compare/run_ams.py --recency-boost
```

Output: a `BenchmarkReport`-shaped JSON written to gitignored
`benchmarks/agent-memory-compare/out/ams-<variant>-<mode>-<N>.json`, with the
**same top-level keys as `benchmarks/results/qtc-500.json`** (`benchmark`,
`variant`, `strategy`, `top_k`, `total_questions`, `overall`, `by_category`,
`misses`) plus a `run_meta` block recording the exact conditions. That parity
lets you diff the two reports directly.

Config knobs (`--help` for all): `--server-url`, `--top-k`, `--search-mode
{semantic,keyword,hybrid}`, `--hybrid-alpha`, `--recency-boost`, `--extraction
{off,on}`, `--distance-threshold`, `--index-timeout`, `--limit`, `--no-cleanup`,
`--out`.

### 3. Run the Axil side (existing harness)

The Axil recall@5 numbers come from the in-tree
[`benchmarks/longmemeval`](../longmemeval/README.md) crate. Quoting its README
verbatim — the **vector** strategy is the like-for-like analog to AMS
`--search-mode semantic`:

```bash
cargo run --release --manifest-path benchmarks/longmemeval/Cargo.toml -- \
  --variant s --limit 20 --strategy vector --top-k 5 --model bge-small
```

The headline Axil baseline is the **recall-qtc** strategy (recency-blended),
committed at `benchmarks/results/qtc-500.json` (recall@5 = 0.935 — the
README's 93.5% Recall-QTC figure). Its recency-aware analog on the AMS side is
`--recency-boost`. Run whichever pair you intend to compare — and label the
report with which strategy pair it is.

---

## Granularity & fairness decisions

These choices make the two sides measure the same thing. Each mirrors a
concrete detail of `benchmarks/longmemeval/src/main.rs`.

- **Ingest granularity — one memory per session.** The Axil harness inserts one
  record per haystack session (`session_text` = each turn rendered `role:
  content`, newline-joined). This driver creates **one AMS long-term memory per
  session**, with byte-identical text rendering. Same unit, same text.
- **`session_id` tag + `session_<index>` scheme.** Each memory carries
  `session_id = "session_<0-based-index>"`, the same tag the Axil harness scores
  against. We read it back from search results to map a hit to a session.
- **Answer-bearing sessions from `has_answer`, not `answer_session_ids`.** Both
  sides define the ground truth as "sessions containing a turn flagged
  `has_answer: true`", tagged `session_<index>`. The dataset's
  `answer_session_ids` (a different naming scheme) is deliberately unused — on
  both sides — so scoring is identical. (Verified: the driver's
  `answer_session_tags` reproduces the Axil harness's answer indices.)
- **Over-fetch then collapse.** The Axil harness fetches `top_k*8` (min 40) then
  collapses to unique sessions; this driver fetches `min(top_k*8, 100)` (AMS's
  `limit` max is 100) and collapses the same way.
- **`deduplicate: false` on ingest.** AMS's create endpoint defaults to
  deduplicating; we disable it to preserve the 1:1 session→memory mapping (dedup
  could collapse near-identical sessions and corrupt session-level scoring).
- **Recency off by default.** `recency_boost=false` gives pure similarity/lexical
  ranking — the analog to Axil's `similar_to`/`vector` strategy. `--recency-boost`
  is the analog to `recall-qtc`.
- **Extraction off by default.** No generative LLM is called on ingest — the
  no-LLM retrieval condition matching Axil's Path 0. `--extraction on` (with the
  server's extraction flags set True) measures a **different, LLM-assisted**
  condition and must be labeled as such.

### Fairness & asymmetries

- **Same dataset, same metric, same granularity, same ground-truth definition.**
  Enforced by the decisions above.
- **Embedding models DIFFER — an inherent asymmetry.** Axil embeds in-process
  with **bge-small** (384-dim, no key, no network). AMS embeds with its
  configured provider — default **OpenAI `text-embedding-3-small`** (1536-dim),
  a **hosted** model requiring `OPENAI_API_KEY` and network calls. There is no
  way to make these identical; the comparison is "each system's default local /
  configured embedding," not "same embedder." State this next to any number.
- **"No LLM" means no *generative* LLM.** With extraction off, neither system
  calls a generative model. But AMS **always** embeds memories at ingest and
  embeds the query for semantic/hybrid search, so it needs an embedding provider
  even in the no-LLM condition. Axil needs neither (bge-small is bundled,
  in-process). Do not describe the AMS side as "no external dependency."
- **`--search-mode keyword` is a different condition, not a cheaper semantic
  run.** Keyword mode is Redis full-text (BM25STD) over the stored text — lexical,
  not vector. It does not embed the query, but AMS still embeds each memory at
  ingest (the vector index is always populated). Compare keyword-vs-keyword or
  semantic-vs-semantic; don't cross them.

### Why the task-worker is required

AMS's `POST /v1/long-term-memory/` **enqueues** indexing as a background task
(Docket, `use_docket=True` by default) and returns an ack immediately — the
memories are not searchable until the `task-worker` service drains the queue.
The compose file includes `task-worker` for this reason, and the driver **polls
until the expected count is indexed** (`--index-timeout`, default 120s) before
searching. Without the worker, ingests are accepted but never become
searchable, and recall would read as ~0.

---

## Verified sources (no invented endpoints)

Every endpoint and image below was verified against the AMS docs and source
(commit on `main` at authoring time; release `server/v0.15.2`). If AMS changes
these, update the driver and re-verify.

| Thing | Value | Source |
|-------|-------|--------|
| Create (bulk) long-term memories | `POST /v1/long-term-memory/` — body `{"memories": [{id, text, session_id, namespace, memory_type, ...}], "deduplicate": bool}` | `agent_memory_server/api.py` (`create_long_term_memory`), `models.py` (`CreateMemoryRecordRequest`, `MemoryRecord`) · docs: <https://redis.github.io/agent-memory-server/long-term-memory/> |
| Search long-term memories | `POST /v1/long-term-memory/search` — filters are **top-level** fields (`namespace`, `session_id`, ... as `{"eq": ...}`/`{"any": [...]}` tag filters), plus `text`, `search_mode`, `hybrid_alpha`, `limit` (≤100), `recency_boost`, `distance_threshold` | `models.py` (`SearchRequest`, `TagFilter`), `api.py` (`search_long_term_memory`) · docs: <https://redis.github.io/agent-memory-server/api/> |
| Search result shape | `{"memories": [{id, text, session_id, namespace, memory_type, dist, ...}], "total": int, "next_offset": int?}` — `dist` = distance (lower = closer) | `models.py` (`MemoryRecordResult(dist: float)`, `MemoryRecordResults`) |
| Delete by IDs | `DELETE /v1/long-term-memory?memory_ids=a&memory_ids=b` (repeated query param) | `api.py` (`delete_long_term_memory`) |
| Health | `GET /v1/health` → `{"now": int}` | AMS compose healthcheck (`/v1/health`) |
| API image | `redislabs/agent-memory-server:0.15.2` (Docker **Hub**, not ghcr.io) | AMS `docker-compose.yml`; Docker Hub tags |
| Redis image | `redis/redis-stack-server:7.4.0-v8` | AMS `docker-compose.yml`; Docker Hub tags |
| Env: `DISABLE_AUTH`, `LONG_TERM_MEMORY`, `ENABLE_*_EXTRACTION`, `ENABLE_NER`, `EMBEDDING_MODEL`, `GENERATION_MODEL` | defaults per `config.py`; overridden in our compose | `agent_memory_server/config.py` |

Docs index: <https://redis.github.io/agent-memory-server/> ·
Quick start: <https://redis.github.io/agent-memory-server/quick-start/>

**Could not fully verify from docs (flagged):**

- Whether AMS can **skip embedding at ingest** so that a `keyword`-only run needs
  no embedding provider at all. The RedisVL index always has a vector field, so
  the driver/README assume ingest embeds in every mode. If you can configure a
  no-embed ingest, this assumption (and the "embedding key required in every
  mode" note) can be relaxed — verify before relying on it.
- Whether AMS supports a **local/offline embedding model** (avoiding a hosted
  key). `config.py` model configs are all `ModelProvider.OPENAI` (plus an AWS
  Bedrock variant image). A LiteLLM-compatible local endpoint via
  `EMBEDDING_MODEL` may be possible but was not verified here.

---

## What a future runner must do (step by step)

1. **Get the dataset** into `benchmarks/longmemeval/data/longmemeval_s_cleaned.json`
   (see [Dataset placement](#dataset-placement)). Without it, both sides skip loud.
2. **Provide an embedding key.** Put `OPENAI_API_KEY=...` in
   `benchmarks/agent-memory-compare/.env` (copy from `.env.example`). Required —
   AMS embeds at ingest.
3. **Start the AMS stack:** `cd benchmarks/agent-memory-compare && docker compose up -d`,
   then `docker compose logs -f api` until healthy.
4. **Install driver deps:** `pip install -r benchmarks/agent-memory-compare/requirements.txt`.
5. **Run the AMS side:** `python benchmarks/agent-memory-compare/run_ams.py`
   (add `--limit 20` for a smoke run first). Note the `out/…json` path it prints.
6. **Run the Axil side** at the matching condition (vector for AMS-semantic;
   recall-qtc for AMS `--recency-boost`) via the
   [`benchmarks/longmemeval`](../longmemeval/README.md) command above.
7. **Compare** the two reports' `overall.avg_recall` (recall@5) and
   `by_category`. They share a JSON shape, so a plain diff works.
8. **To publish anything:** copy the AMS `out/…json` to
   `benchmarks/results/ams-longmemeval-s-<mode>-<N>.json`, commit it **with**
   full environment details (image tags, embedding model, search mode, recency,
   extraction, hardware, date), and only then may the number be cited — always
   next to the embedding-asymmetry caveat.
9. **Tear down:** `docker compose down -v` (the `-v` wipes the Redis volume).
