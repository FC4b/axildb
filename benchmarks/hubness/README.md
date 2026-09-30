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
`--example-bin`), hashes the snapshot's `memory.axil` before anything opens a
database (opening one rewrites the file, so a hash of an opened copy would not
name the snapshot), then works on two copies. The snapshot itself is only
read, and its hash is checked again at the end. A run needs the dogfood
snapshot, since the questions' expected ids exist only in this repo's memory.

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
report gives N_k skewness, each class's and table's top-k slots against its
share of the population (`slot_lift`, 1.0 = no over-representation), hubs,
and for records-as-queries the slots that come from other tables' queries (a
hub that matches everything draws from every table). Four populations:

- `memory`: non-`_` tables only;
- `index`: every vector, which is what recall's vector search draws from;
- `what_if_prose`: `memory` with every keyed commit re-embedded from its data
  minus all key fields, which leaves sha, author, date, subject and body
  joined: the shape commits had before the hook stored `content` and
  `summary`;
- `what_if_text_poor`: the same with every string of more than one word
  removed as well, which leaves sha, author and date: a fallback record with
  almost no prose, whose join is ids, hashes and timestamps.

A real database holds few fallback records, so the what-ifs push a whole
table through the fallback. Each rewritten record gets the vectors insert
would build for its new text: its own and, when the text is longer than one
recall chunk (`RECALL_CHUNK_MAX_BYTES`), one per chunk. Each what-if reports
the table's own numbers before and after (`table_before_after`) and a paired
comparison over the questions whose answers are not commits
(`questions_not_about_table`), with a two-sided exact sign test over the
questions whose commit slots changed.

It also re-embeds each memory record's current `searchable_text` and checks
the stored vector against it (`stored_vector_matches_text`), which confirms
the class is what the vector was actually built from.

**Recall view.** The real `axil recall --recall-format full`, the dogfood
eval's ranked view (fetch 25, drop hits after the cutoff, keep the top 10):
how often a fallback record takes a slot for a question whose expected answers
are all keyed records.

## Result: not confirmed (2026-09-30)

Snapshot `benchmarks/dogfood-recall/data/snap-prefix` (the healed dogfood
copy of 2026-09-28; the result's `memory_axil_sha256` is that file's hash),
bge-small, k = 10, dogfood cutoff. Every number below is in
`benchmarks/results/hubness-2026-09-30-snap-prefix.json`, or computed from it
where the text says how; the recall view ran `axil` at `65d45ff`.

- **Few fallback records.** 3 of the 219 memory records in the window: one
  commit in the old hook shape (sha, author, date, subject, body; superseded
  by a keyed copy) and both rules. A rule keeps its text in `rule`, which is
  not a key field, so rules are embedded only through the fallback. The
  stored vectors match the text the class implies: all 229 keyed ones and 2
  of the 3 fallback ones at cosine >= 0.99, the third at 0.98.
- **Not hubs as neighbours.** Records as queries, memory population: the
  fallback records have N_10 of 6, 2 and 3, against a mean of 10.0 and a hub
  threshold of 27.2, so 0 of the 3 are hubs, against 10 of the 216 keyed
  records. They take 0.5% of the slots at 1.4% of the population. None is a
  hub in the index population either.
- **Not over-represented for questions.** Vector top 10 for the 57 questions,
  memory population: fallback records take 1.2% of the slots at 1.4% of the
  population (slot lift 0.90), and 0 of the 3 are hubs, against 11 of the 216
  keyed records. Of the 55 questions whose answers are all keyed, 5 had a
  fallback record in the top 10, one slot each, always the legacy commit: a
  long commit covering boot, recall and rerank fixes. Its question N_10 of 5
  is under the hub threshold of 7.6; keyed commits reach 15. In the index
  population that commit's N_10 of 3 does pass the hub threshold, which is
  low there (2.25) because the questions rarely reach the code proxies and
  file summaries that make up 84% of it; keyed memory records are
  over-represented about as much (slot lift 3.46, against 4.11 for the three
  fallback records).
- **Real recall.** `axil recall`: fallback records took 6 top-10 slots, a
  share of 0.0105. 5 of the 55 keyed-answer questions had one in the top 10,
  one slot each, and it ranked above the first expected answer in 2.
- **Every commit through the fallback, prose kept.** All 95 keyed commits
  re-embedded from sha, author, date, subject and body joined (mean cosine
  0.96 to their stored vectors; their chunk vectors went from 22 to 30). For
  the 52 questions not about commits, commit slots in the top 10 went from 222
  to 217: fewer in 20 questions, more in 17, the same in 15 (sign test
  p = 0.74), so no crowding. Vector-only hit@1 on those questions went from
  0.635 to 0.692 (MRR@10 0.768 to 0.799). N_k skewness fell a little
  (records-as-queries 1.56 to 1.51, questions 2.08 to 1.88).
  **Hubs, under the same rule (above each population's mean + 2 std):** as
  records-as-queries, 14 of the 95 rewritten commits are hubs, against 8 of
  the same commits before (the `commits` table: 8 of 96 before, 14 of 96
  after; the one real fallback commit is a hub in neither). As question
  results, commit hubs went the other way, 6 to 3. The rewritten commits took
  more records-as-queries slots (1,104 to 1,272), but the slots they took
  from other tables' queries barely moved (516 to 524); the rest, 588 to 748
  (slots minus `slots_from_other_tables`), came from commits' own queries. So
  the fallback makes more commits hubs among commits, most likely through the
  sha, author and date text they all share (the text-poor what-if below, which
  keeps only that text, draws every slot from commits), and not across
  tables.
- **Every commit through the fallback, ids and timestamps only.** The same 95
  commits re-embedded from sha, author and date alone (71 bytes each; mean
  cosine 0.54 to their stored vectors). Toward the questions they are the
  opposite of hubs: none reaches any question's top 10 (`what_if_fallback`
  question slots: 0; 0 of 95 hubs, against 6 commit hubs before). As
  records-as-queries 8 of the 95 are hubs (8 of 96 commits before), and all
  950 of their slots come from each other (0 from other tables). For the 52
  questions not about commits, commit slots fell from 222 to 9, the 9 left
  being the one real old-shape commit, which is not rewritten and is now the
  only commit with prose (it passes the questions' hub threshold in this
  view, N_10 11 against 9.56). Such a record is lost to vector search rather
  than crowding it; FTS still indexes its strings.

So on this snapshot the fallback does not make hubs that reach across
tables, and nothing changes what is embedded:

- in the memory population the three real fallback records are hubs in
  neither query set and take less than their share of slots (in the index
  population the legacy commit passes the low question threshold, as 41 of
  the 216 keyed records do);
- all 95 commits through the fallback with their prose leave the other
  questions' results where they were (commit slots 222 to 217, p = 0.74) and
  take no more slots from other tables' records; they do make more commits
  hubs among commits (8 to 14 of 95);
- with only ids and a timestamp left, the commits drop out of every
  question's top 10 instead of crowding it.

A gate that stopped embedding records without a key field would also take the
vectors from both rules; the rule questions need them (q28's rule ranks first
in both views).

The evidence is narrower than "the join never makes hubs". The real fallback
records are 2 rules (298 and 450 bytes) and 1 old-shape commit (1,587 bytes),
and the prose what-if's joins run from 162 to 2,368 bytes (mean 1,071). In all
of them the join is mostly natural-language text (subject and body, rule and
reason) with a sha, an author and a timestamp added, and the prose what-if's
vectors stay close to the keyed ones (mean cosine 0.96). The
text-poor what-if is one synthetic shape: 95 records of exactly 71 bytes. Not
covered:

- fallback records that mix a little prose with many ids, paths, hashes or
  enum values (small structured context rows, say), the shape most likely to
  make a hub; this snapshot has none and neither what-if builds one;
- a few text-poor records among many prose records: in the text-poor what-if
  the 95 look alike and fill each other's top 10, which a lone one could not;
- other tables' shapes, other models than bge-small, and larger or older
  databases with many fallback records;
- the what-ifs through real `axil recall`, where FTS and the other signals
  are fused in: they were measured in the vector view only;
- with 3 real fallback records the real-data figures have little statistical
  power; the what-ifs carry most of the weight.

Two things the run turned up that are not hubness:

- 870 of the 2,314 vectors in this snapshot's index belong to no record
  (`vectors.without_a_record`). `heal --reindex`, which the dogfood eval runs,
  does not remove them (`heal --orphans` does). Recall's vector search still
  returns them and drops them when it resolves candidates, so they take slots
  from its candidate pool.
- Rules are embedded from `rule` joined with `reason` or `_seed`. Adding
  `rule` to the key fields would embed the rule alone; that changes recall and
  is a separate decision.
