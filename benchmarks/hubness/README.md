# Hubness of fallback-embedded records

`searchable_text` (`crates/axil-core/src/util.rs`) gives a record's text from
the first of its key fields (`SEARCHABLE_TEXT_KEYS`: `full_text`, `content`,
`text`, `description`, `message`, `summary`, `fact`, `error`, `statement`).
A record with none of them falls back to every string value joined, ids,
hashes and timestamps included, and that join is what gets embedded. The
worry: such a vector is a *hub*, close to everything, taking top-k slots from
unrelated queries. This measures it before anything changes what is embedded.

```bash
python3 benchmarks/hubness/run.py \
    --snapshot benchmarks/dogfood-recall/data/snap-prefix \
    --out benchmarks/results/hubness-<date>-<snapshot>.json
```

It builds `crates/engines/axil-vector/examples/hubness.rs` (or takes
`--example-bin`), then works on two copies of the snapshot, so the snapshot is
never written. A run needs the dogfood snapshot, since the questions' expected
ids exist only in this repo's memory.

## Method

Every record falls in one class:

| Class | Meaning |
|---|---|
| `keyed` | `searchable_text` read a key field |
| `fallback` | it joined every string value |
| `internal` | an `_`-prefixed table an extension embeds with its own text (code proxies, file summaries) |

**Vector view** (the example). Exact cosine search over the default vector
index with the embedder the CLI opens (`axil.toml`'s model, else bge-small).
A record is scored by its closest vector, so a recall chunk counts for its
source record. Records created after the dogfood cutoff are left out. Two
query sets:

- the 57 dogfood questions, embedded as queries (`embed_query`, as recall
  does);
- every record's own stored vector, the record itself excluded.

N_k is how many queries have a record in their top k (k = 10). A hub is a
record with N_k above the population mean plus two standard deviations. The
report gives N_k skewness, each class's share of top-k slots against its share
of the population (`slot_lift`, 1.0 = no over-representation), hubs per class,
and for records-as-queries the share of a class's slots that come from other
tables (a hub that matches everything draws from every table). It runs over
three populations:

- `memory`: non-`_` tables only;
- `index`: every vector, which is what recall's vector search draws from;
- `what_if`: `memory` with every keyed commit re-embedded from its data minus
  all key fields, which leaves sha, author, date, subject and body joined: the
  shape commits had before the hook stored `content` and `summary`. A real
  database holds few fallback records, so this pushes a whole table through
  the fallback to see whether it makes hubs. The paired comparison
  (`questions_not_about_table`) runs the questions whose answers are not
  commits before and after the rewrite.

It also re-embeds each memory record's current `searchable_text` and checks
the stored vector against it (`stored_vector_matches_text`), which confirms
the class is what the vector was actually built from.

**Recall view.** The real `axil recall --recall-format full`, the dogfood
eval's ranked view (fetch 25, drop hits after the cutoff, keep the top 10):
how often a fallback record takes a slot for a question whose expected answers
are all keyed records.

## Result: not confirmed (2026-09-30)

Snapshot `benchmarks/dogfood-recall/data/snap-prefix` (the healed dogfood
copy of 2026-09-28), bge-small, k = 10, dogfood cutoff. Every number below is
from `benchmarks/results/hubness-2026-09-30-snap-prefix.json`; the recall view
ran `axil` at `65d45ff`.

- **Few fallback records.** 3 of the 219 memory records in the window: one
  commit in the old hook shape (sha, author, date, subject, body; superseded
  by a keyed copy) and both rules. A rule keeps its text in `rule`, which is
  not a key field, so rules are embedded only through the fallback. The
  stored vectors match the text the class implies: all 229 keyed ones and 2
  of the 3 fallback ones at cosine >= 0.99, the third at 0.98.
- **Not hubs as neighbours.** Records as queries, memory population: the
  fallback records have N_10 of 6, 2 and 3, against a mean of 10.0 and a hub
  threshold of 27.2. They take 0.5% of the slots at 1.4% of the population.
  None is a hub in the index population either.
- **Not over-represented for questions.** Vector top 10 for the 57 questions,
  memory population: fallback records take 1.2% of the slots at 1.4% of the
  population (slot lift 0.90). Of the 55 questions whose answers are all
  keyed, 5 had a fallback record in the top 10, one slot each, always the
  legacy commit: a long commit covering boot, recall and rerank fixes. Its
  question N_10 of 5 is under the hub threshold of 7.6; keyed commits reach
  15. In the index population that commit's N_10 of 3 does pass the hub
  threshold, which is low there (2.25) because the questions rarely reach the
  code proxies and file summaries that make up 84% of it; keyed memory records
  are over-represented about as much (slot lift 3.46, against 4.11 for the
  three fallback records).
- **Real recall.** `axil recall`: fallback records took 6 of the 570 top-10
  slots (1.1%). 5 of the 55 keyed-answer questions had one in the top 10, one
  slot each, and it ranked above the first expected answer in 2.
- **What if every commit had no key field.** All 95 keyed commits re-embedded
  from sha, author, date, subject and body joined (mean cosine 0.96 to their
  stored vectors). For the 52 questions not about commits, commit slots in the
  top 10 fell from 222 to 189 (fewer in 28 questions, more in 7, the same in
  17; two-sided sign test p = 0.0005), and vector-only hit@1 on those
  questions rose from 0.635 to 0.692 (MRR@10 0.768 to 0.805). As
  records-as-queries the rewritten commits took more slots (1,104 to 1,234,
  from mean N_10 and table size), but slots from other tables fell (516 to
  488): the shared sha/author/date text pulls commits toward each other, not
  toward everything. N_k skewness fell too (1.56 to 1.30 records-as-queries,
  2.08 to 1.68 questions).

So the join does not make hubs here, and nothing changes what is embedded. A
gate that stopped embedding records without a key field would also take the
vectors from both rules; the rule questions need them (q28's rule ranks first
in both views).

Two things the run turned up that are not hubness:

- 870 of the 2,314 vectors in this snapshot's index belong to no record
  (`vectors.without_a_record`). `heal --reindex`, which the dogfood eval runs,
  does not remove them (`heal --orphans` does). Recall's vector search still
  returns them and drops them when it resolves candidates, so they take slots
  from its candidate pool.
- Rules are embedded from `rule` joined with `reason` or `_seed`. Adding
  `rule` to the key fields would embed the rule alone; that changes recall and
  is a separate decision.
