//! Hubness in the default vector space: do records embedded through
//! `searchable_text`'s fallback (no key text field, so every string value
//! joined, ids and timestamps included) turn up in top-k lists far more often
//! than records embedded from a key field?
//!
//! Every vector in the index is credited to the record that owns it (a recall
//! chunk belongs to its source record), and owners fall into three classes:
//! `keyed` (text from a [`SEARCHABLE_TEXT_KEYS`] field), `fallback` (the
//! join), and `internal` (an `_`-prefixed table an extension embeds with its
//! own text). Exact cosine search then runs for two query sets, the questions
//! (embedded as queries, the way recall embeds them) and every owner's own
//! vector (itself excluded), and counts N_k: how many queries have an owner in
//! their top k. A hub is an owner whose N_k is above the population's mean
//! plus two standard deviations.
//!
//! Populations:
//! - `memory`: non-`_` tables only, the records the questions are about.
//! - `index`: every vector, which is what recall's vector search draws from.
//! - `what_if_prose`: `memory`, with each keyed record of one table
//!   re-embedded as if it had none of the key fields (default `commits`,
//!   which gives the shape commits had before the hook stored `content` and
//!   `summary`: sha, author, date, subject and body joined).
//! - `what_if_text_poor`: the same, with every multi-word string removed as
//!   well, so the join is only ids, hashes, timestamps and one-word tags: the
//!   case where a fallback record has almost no prose.
//!
//! A real database may hold only a handful of fallback records; the what-ifs
//! put a whole table through the fallback to see whether that makes hubs.
//! Each rewritten record gets the vectors insert would give its new text,
//! its own and, past one recall chunk, one per chunk.
//!
//! ```text
//! cargo run --release -p axil-vector --features embed --example hubness -- \
//!     --db <copy>/memory.axil \
//!     --questions benchmarks/dogfood-recall/questions.jsonl \
//!     [--cutoff 2026-09-28T02:00:00Z] [--k 10] [--what-if commits|none]
//! ```
//!
//! Point it at a copy: opening a database can migrate it. It prints one JSON
//! report on stdout; `benchmarks/hubness/run.py` wraps it.
//!
//! [`SEARCHABLE_TEXT_KEYS`]: axil_core::util::SEARCHABLE_TEXT_KEYS

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use axil_core::{util, Axil, RecordId};
use axil_vector::models::EmbeddingModel;
use axil_vector::AxilBuilderVectorExt;
use chrono::{DateTime, Utc};
use serde_json::{json, Value};

const CHUNKS_TABLE: &str = "_recall_chunks";

/// A stored vector counts as embedded from the record's current text when a
/// fresh embedding of that text is at least this close to it.
const SAME_TEXT_COSINE: f32 = 0.99;

/// How many of the highest-N_k owners each report lists.
const TOP_HUBS: usize = 15;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Keyed,
    Fallback,
    Internal,
    /// A keyed record re-embedded through the fallback (the what-if views).
    WhatIf,
}

impl Class {
    const ALL: [Class; 4] = [
        Class::Keyed,
        Class::Fallback,
        Class::Internal,
        Class::WhatIf,
    ];

    fn name(self) -> &'static str {
        match self {
            Class::Keyed => "keyed",
            Class::Fallback => "fallback",
            Class::Internal => "internal",
            Class::WhatIf => "what_if_fallback",
        }
    }

    /// Embedded from the join of every string value.
    fn is_fallback(self) -> bool {
        matches!(self, Class::Fallback | Class::WhatIf)
    }
}

struct Owner {
    id: RecordId,
    table: String,
    class: Class,
    /// The field `searchable_text` read, `None` for a fallback record.
    key: Option<&'static str>,
    /// Names of the record's string fields, the ones the fallback joins.
    string_keys: Vec<String>,
    created_at: DateTime<Utc>,
    in_window: bool,
    /// The record's data, kept for memory records only.
    data: Value,
    /// `searchable_text` of the data, memory records only.
    text: String,
}

/// One population: the vectors it can match and the class of each owner.
struct Space {
    /// `(owner, unit vector)` for every vector in the population.
    points: Vec<(usize, Vec<f32>)>,
    /// Each owner's own record-level vector, the one it queries with.
    own: Vec<Option<usize>>,
    class: Vec<Class>,
    member: Vec<bool>,
}

struct Question {
    id: String,
    kind: String,
    expect: Vec<Vec<String>>,
    vector: Vec<f32>,
}

struct Args {
    db: PathBuf,
    questions: PathBuf,
    cutoff: Option<DateTime<Utc>>,
    k: usize,
    what_if: Option<String>,
}

fn parse_args() -> Args {
    let mut db = None;
    let mut questions = None;
    let mut cutoff = None;
    let mut k = 10usize;
    let mut what_if = Some("commits".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--db" => db = Some(PathBuf::from(value())),
            "--questions" => questions = Some(PathBuf::from(value())),
            "--cutoff" => {
                let raw = value();
                let at = DateTime::parse_from_rfc3339(&raw)
                    .unwrap_or_else(|e| panic!("--cutoff {raw}: {e}"));
                cutoff = Some(at.with_timezone(&Utc));
            }
            "--k" => k = value().parse().expect("--k takes a number"),
            "--what-if" => what_if = Some(value()).filter(|t| t != "none"),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        db: db.expect("--db <path> is required"),
        questions: questions.expect("--questions <jsonl> is required"),
        cutoff,
        k,
        what_if,
    }
}

/// The class `searchable_text` puts a record in, and the key it read.
fn classify(table: &str, data: &Value) -> (Class, Option<&'static str>) {
    if table.starts_with('_') {
        return (Class::Internal, None);
    }
    match data {
        Value::String(_) => (Class::Keyed, Some("(plain string)")),
        Value::Object(map) => util::SEARCHABLE_TEXT_KEYS
            .iter()
            .copied()
            .find(|k| map.get(*k).is_some_and(Value::is_string))
            .map_or((Class::Fallback, None), |k| (Class::Keyed, Some(k))),
        _ => (Class::Fallback, None),
    }
}

fn string_keys(data: &Value) -> Vec<String> {
    data.as_object()
        .map(|m| {
            m.iter()
                .filter(|(_, v)| v.is_string())
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact search: the `k` member owners closest to `query`, best first, each
/// scored by its closest vector.
fn top_k(
    query: &[f32],
    space: &Space,
    exclude: Option<usize>,
    k: usize,
    scores: &mut [f32],
) -> Vec<usize> {
    scores.fill(f32::NEG_INFINITY);
    for (owner, v) in &space.points {
        if space.member[*owner] {
            let s = dot(query, v);
            if s > scores[*owner] {
                scores[*owner] = s;
            }
        }
    }
    let mut ranked: Vec<usize> = (0..scores.len())
        .filter(|&o| scores[o].is_finite() && Some(o) != exclude)
        .collect();
    let by_score = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
    if ranked.len() > k {
        ranked.select_nth_unstable_by(k, by_score);
        ranked.truncate(k);
    }
    ranked.sort_by(by_score);
    ranked
}

fn mean_std_skew(xs: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let skew = if std > 0.0 {
        xs.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };
    (mean, std, skew)
}

fn median(mut xs: Vec<u32>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_unstable();
    let m = xs.len() / 2;
    if xs.len().is_multiple_of(2) {
        (xs[m - 1] + xs[m]) as f64 / 2.0
    } else {
        xs[m] as f64
    }
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// `x` to `digits` significant digits, so a tiny p-value keeps its size.
fn round_significant(x: f64, digits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(digits - 1 - x.abs().log10().floor() as i32);
    (x * scale).round() / scale
}

/// The N_k distribution over a population, overall and per class.
fn nk_report(nk: &[u32], owners: &[Owner], space: &Space, queries: usize, k: usize) -> Value {
    let members: Vec<usize> = (0..owners.len()).filter(|&o| space.member[o]).collect();
    let values: Vec<f64> = members.iter().map(|&o| nk[o] as f64).collect();
    let (mean, std, skew) = mean_std_skew(&values);
    let hub_threshold = mean + 2.0 * std;
    let total_slots: u64 = members.iter().map(|&o| nk[o] as u64).sum();

    let mut classes = serde_json::Map::new();
    for class in Class::ALL {
        let of: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&o| space.class[o] == class)
            .collect();
        if of.is_empty() {
            continue;
        }
        let slots: u64 = of.iter().map(|&o| nk[o] as u64).sum();
        let share_pop = of.len() as f64 / members.len() as f64;
        let share_slots = if total_slots > 0 {
            slots as f64 / total_slots as f64
        } else {
            0.0
        };
        let hubs = of.iter().filter(|&&o| nk[o] as f64 > hub_threshold).count();
        let zero = of.iter().filter(|&&o| nk[o] == 0).count();
        classes.insert(
            class.name().to_string(),
            json!({
                "owners": of.len(),
                "share_of_population": round(share_pop),
                "slots": slots,
                "share_of_slots": round(share_slots),
                "slot_lift": round(if share_pop > 0.0 { share_slots / share_pop } else { 0.0 }),
                "mean_nk": round(slots as f64 / of.len() as f64),
                "median_nk": median(of.iter().map(|&o| nk[o]).collect()),
                "max_nk": of.iter().map(|&o| nk[o]).max().unwrap_or(0),
                "hubs": hubs,
                "hub_rate": round(hubs as f64 / of.len() as f64),
                "zero_nk": zero,
                "zero_nk_rate": round(zero as f64 / of.len() as f64),
            }),
        );
    }

    let mut tables: BTreeMap<&str, (usize, u64, usize, u32)> = BTreeMap::new();
    for &o in &members {
        let row = tables.entry(owners[o].table.as_str()).or_default();
        row.0 += 1;
        row.1 += nk[o] as u64;
        row.2 += usize::from(nk[o] as f64 > hub_threshold);
        row.3 = row.3.max(nk[o]);
    }
    let by_table: serde_json::Map<String, Value> = tables
        .into_iter()
        .map(|(table, (n, slots, hubs, max_nk))| {
            let share_slots = if total_slots > 0 {
                slots as f64 / total_slots as f64
            } else {
                0.0
            };
            (
                table.to_string(),
                json!({
                    "owners": n,
                    "share_of_population": round(n as f64 / members.len() as f64),
                    "slots": slots,
                    "share_of_slots": round(share_slots),
                    "mean_nk": round(slots as f64 / n as f64),
                    "max_nk": max_nk,
                    "hubs": hubs,
                }),
            )
        })
        .collect();

    let describe = |o: usize| {
        let below = members.iter().filter(|&&m| nk[m] < nk[o]).count();
        json!({
            "id": owners[o].id.as_str(),
            "table": owners[o].table,
            "class": space.class[o].name(),
            "key": owners[o].key,
            "string_keys": owners[o].string_keys,
            "nk": nk[o],
            "percentile": round(below as f64 / members.len() as f64),
            "hub": nk[o] as f64 > hub_threshold,
        })
    };
    // Every real fallback record; the what-if class is summarized by class.
    let fallback: Vec<Value> = members
        .iter()
        .copied()
        .filter(|&o| space.class[o] == Class::Fallback)
        .map(describe)
        .collect();
    let mut by_nk = members.clone();
    by_nk.sort_by(|a, b| nk[*b].cmp(&nk[*a]).then(a.cmp(b)));
    let top: Vec<Value> = by_nk.iter().take(TOP_HUBS).map(|&o| describe(o)).collect();

    json!({
        "queries": queries,
        "k": k,
        "owners": members.len(),
        "nk_mean": round(mean),
        "nk_std": round(std),
        "nk_skewness": round(skew),
        "hub_threshold": round(hub_threshold),
        "by_class": classes,
        "by_table": by_table,
        "fallback_records": fallback,
        "top_hubs": top,
    })
}

/// Both query sets over one population, plus each question's top k.
fn run_space(
    space: &Space,
    owners: &[Owner],
    index: &HashMap<String, usize>,
    questions: &[Question],
    k: usize,
) -> (Value, Vec<Vec<usize>>) {
    let mut scores = vec![f32::NEG_INFINITY; owners.len()];

    let mut nk_q = vec![0u32; owners.len()];
    let mut per_question = Vec::new();
    let mut tops = Vec::new();
    let (mut keyed_q, mut with_fallback, mut fallback_slots, mut fallback_above) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut hit1, mut hitk, mut mrr) = (0.0f64, 0.0f64, 0.0f64);
    for q in questions {
        let top = top_k(&q.vector, space, None, k, &mut scores);
        top.iter().for_each(|&o| nk_q[o] += 1);
        let expected: Vec<&str> = q.expect.iter().flatten().map(String::as_str).collect();
        let all_keyed = expected.iter().all(|id| {
            index
                .get(*id)
                .is_some_and(|&o| space.class[o] == Class::Keyed)
        });
        let first_expected = top
            .iter()
            .position(|&o| expected.contains(&owners[o].id.as_str()))
            .map(|r| r + 1);
        let fallback_hits: Vec<(usize, usize)> = top
            .iter()
            .enumerate()
            .filter(|(_, &o)| space.class[o].is_fallback())
            .map(|(r, &o)| (r + 1, o))
            .collect();
        if let Some(r) = first_expected {
            hit1 += f64::from(u8::from(r == 1));
            hitk += 1.0;
            mrr += 1.0 / r as f64;
        }
        if all_keyed {
            keyed_q += 1;
            with_fallback += usize::from(!fallback_hits.is_empty());
            fallback_slots += fallback_hits.len();
            fallback_above += usize::from(
                fallback_hits
                    .first()
                    .is_some_and(|&(f, _)| first_expected.is_none_or(|e| f < e)),
            );
        }
        per_question.push(json!({
            "id": q.id,
            "kind": q.kind,
            "answers_all_keyed": all_keyed,
            "first_expected_rank": first_expected,
            "fallback_hits": fallback_hits
                .iter()
                .map(|&(rank, o)| json!({
                    "rank": rank,
                    "id": owners[o].id.as_str(),
                    "class": space.class[o].name(),
                }))
                .collect::<Vec<_>>(),
        }));
        tops.push(top);
    }
    let n = questions.len().max(1) as f64;
    let mut question_report = nk_report(&nk_q, owners, space, questions.len(), k);
    question_report["vector_only"] = json!({
        "hit@1": round(hit1 / n),
        format!("hit@{k}"): round(hitk / n),
        format!("mrr@{k}"): round(mrr / n),
    });
    question_report["keyed_answer_questions"] = json!({
        "questions": keyed_q,
        "with_fallback_in_top_k": with_fallback,
        "fallback_slots": fallback_slots,
        "fallback_above_first_expected": fallback_above,
    });
    question_report["per_question"] = Value::Array(per_question);

    // Records as queries: each member's own vector, itself excluded. A hub
    // that matches everything draws its slots from every table; one that
    // only clusters with its own kind draws them from its own table.
    let mut nk_r = vec![0u32; owners.len()];
    let mut cross = vec![0u32; owners.len()];
    let mut record_queries = 0usize;
    for o in (0..owners.len()).filter(|&o| space.member[o]) {
        let Some(p) = space.own[o] else { continue };
        record_queries += 1;
        for hit in top_k(&space.points[p].1, space, Some(o), k, &mut scores) {
            nk_r[hit] += 1;
            cross[hit] += u32::from(owners[hit].table != owners[o].table);
        }
    }
    let mut records_report = nk_report(&nk_r, owners, space, record_queries, k);
    // Adds the slots a group of owners took from queries of another table,
    // as a count and as a share of the group's slots.
    let add_cross = |row: &mut Value, of: &mut dyn Iterator<Item = usize>| {
        let (slots, other) = of.fold((0u64, 0u64), |(s, c), o| {
            (s + u64::from(nk_r[o]), c + u64::from(cross[o]))
        });
        if slots > 0 {
            row["slots_from_other_tables"] = json!(other);
            row["share_of_slots_from_other_tables"] = json!(round(other as f64 / slots as f64));
        }
    };
    for class in Class::ALL {
        if let Some(row) = records_report["by_class"].get_mut(class.name()) {
            let mut of = (0..owners.len()).filter(|&o| space.member[o] && space.class[o] == class);
            add_cross(row, &mut of);
        }
    }
    if let Some(tables) = records_report["by_table"].as_object_mut() {
        for (table, row) in tables.iter_mut() {
            let mut of =
                (0..owners.len()).filter(|&o| space.member[o] && owners[o].table == *table);
            add_cross(row, &mut of);
        }
    }

    let report = json!({
        "questions": question_report,
        "records_as_queries": records_report,
    });
    (report, tops)
}

/// Summed rank scores over a set of questions.
#[derive(Default)]
struct RankTotals {
    hit1: f64,
    hitk: f64,
    rr: f64,
}

impl RankTotals {
    fn add(&mut self, (hit1, hitk, rr): (f64, f64, f64)) {
        self.hit1 += hit1;
        self.hitk += hitk;
        self.rr += rr;
    }
}

/// The what-if's paired comparison over the questions not about its table.
#[derive(Default)]
struct Paired {
    questions: usize,
    slots_before: usize,
    slots_after: usize,
    fewer: usize,
    same: usize,
    more: usize,
    before: RankTotals,
    after: RankTotals,
}

/// Rank-based scores of one question's top k: (hit@1, hit@k, reciprocal rank).
fn rank_scores(top: &[usize], owners: &[Owner], expect: &[Vec<String>]) -> (f64, f64, f64) {
    let rank = top.iter().position(|&o| {
        expect
            .iter()
            .flatten()
            .any(|id| id == owners[o].id.as_str())
    });
    match rank {
        Some(r) => (f64::from(u8::from(r == 0)), 1.0, 1.0 / (r + 1) as f64),
        None => (0.0, 0.0, 0.0),
    }
}

/// How a what-if rewrites a keyed record's data before re-embedding it.
#[derive(Clone, Copy)]
enum Rewrite {
    /// Every key field removed, so the join keeps the record's other strings,
    /// prose included.
    Prose,
    /// Every key field and every string of more than one word removed, so the
    /// join holds only ids, hashes, timestamps and one-word tags.
    TextPoor,
}

impl Rewrite {
    const ALL: [Rewrite; 2] = [Rewrite::Prose, Rewrite::TextPoor];

    fn name(self) -> &'static str {
        match self {
            Rewrite::Prose => "prose",
            Rewrite::TextPoor => "text_poor",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Rewrite::Prose => "every key field removed; the other string values joined",
            Rewrite::TextPoor => {
                "every key field and every string value of more than one word removed; \
                 the remaining one-word values (ids, hashes, timestamps, tags) joined"
            }
        }
    }

    fn apply(self, data: &Value) -> Value {
        let mut data = data.clone();
        if let Some(map) = data.as_object_mut() {
            map.retain(|key, value| {
                let keyed = util::SEARCHABLE_TEXT_KEYS.contains(&key.as_str());
                let multi_word = value
                    .as_str()
                    .is_some_and(|s| s.split_whitespace().nth(1).is_some());
                let dropped = keyed || (matches!(self, Rewrite::TextPoor) && multi_word);
                !dropped
            });
        }
        data
    }
}

/// The population a what-if is compared against.
struct Before<'a> {
    space: &'a Space,
    report: &'a Value,
    tops: &'a [Vec<usize>],
}

/// Two-sided exact sign test: the chance of a split at least as uneven as
/// `a` against `b` if either direction were equally likely (ties dropped).
fn sign_test(a: usize, b: usize) -> f64 {
    let n = a + b;
    if n == 0 {
        return 1.0;
    }
    let mut term = 0.5f64.powi(n as i32);
    let mut tail = 0.0;
    for i in 0..=a.min(b) {
        tail += term;
        term *= (n - i) as f64 / (i + 1) as f64;
    }
    (2.0 * tail).min(1.0)
}

/// Re-embeds each keyed record of `table` from its data after `rewrite`,
/// which leaves `searchable_text` only the fallback, with the vectors insert
/// would give that text: one for the record and, when the text is longer
/// than one recall chunk, one per chunk. Runs both query sets over the
/// result and compares it with `before`: the table's own numbers, and the
/// questions not about the table, paired.
#[allow(clippy::too_many_arguments)]
fn what_if(
    rewrite: Rewrite,
    table: &str,
    db: &Axil,
    owners: &[Owner],
    index: &HashMap<String, usize>,
    questions: &[Question],
    k: usize,
    before: &Before,
) -> (Value, Value) {
    let old = before.space;
    let mut rewritten: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
    let mut not_embeddable = Vec::new();
    let mut cosines = Vec::new();
    let mut text_bytes = Vec::new();
    for (o, owner) in owners.iter().enumerate() {
        if owner.table != table || owner.class != Class::Keyed || !old.member[o] {
            continue;
        }
        let data = rewrite.apply(&owner.data);
        if classify(table, &data).0 != Class::Fallback {
            continue;
        }
        let text = util::searchable_text(&data);
        // Insert embeds only text longer than five bytes.
        if text.len() <= 5 {
            not_embeddable.push(o);
            continue;
        }
        let own = normalize(db.embed_passage(&text).expect("embed what-if text"));
        if let Some(p) = old.own[o] {
            cosines.push(f64::from(dot(&own, &old.points[p].1)));
        }
        text_bytes.push(text.len() as f64);
        let mut vectors = vec![own];
        let chunks = util::overlapping_chunks(
            &text,
            util::RECALL_CHUNK_MAX_BYTES,
            util::RECALL_CHUNK_OVERLAP_BYTES,
        );
        if chunks.len() > 1 {
            for chunk in &chunks {
                vectors.push(normalize(
                    db.embed_passage(chunk).expect("embed what-if chunk"),
                ));
            }
        }
        rewritten.insert(o, vectors);
    }

    // The rewritten records lose every vector they had (chunks included)
    // and get the ones built above; a record whose rewritten text insert
    // would not embed leaves the population.
    let mut member = old.member.clone();
    not_embeddable.iter().for_each(|&o| member[o] = false);
    let mut points = Vec::new();
    let mut own = vec![None; owners.len()];
    let mut chunks_before = 0usize;
    for (p, (o, v)) in old.points.iter().enumerate() {
        if rewritten.contains_key(o) || not_embeddable.contains(o) {
            chunks_before += usize::from(old.own[*o] != Some(p));
            continue;
        }
        if old.own[*o] == Some(p) {
            own[*o] = Some(points.len());
        }
        points.push((*o, v.clone()));
    }
    let mut class = old.class.clone();
    let mut chunks_after = 0usize;
    let rewritten_count = rewritten.len();
    for (o, vectors) in rewritten {
        own[o] = Some(points.len());
        chunks_after += vectors.len() - 1;
        points.extend(vectors.into_iter().map(|v| (o, v)));
        class[o] = Class::WhatIf;
    }
    let space = Space {
        points,
        own,
        class,
        member,
    };
    let (view, tops) = run_space(&space, owners, index, questions, k);

    // Paired: the questions not answered by the rewritten table, before and
    // after the rewrite. Does the table crowd them out?
    let in_table = |o: &usize| owners[*o].table == table;
    let mut paired = Paired::default();
    for (qi, q) in questions.iter().enumerate() {
        let about_table = q
            .expect
            .iter()
            .flatten()
            .any(|id| index.get(id).is_some_and(|&o| owners[o].table == table));
        if about_table {
            continue;
        }
        let slots_before = before.tops[qi].iter().filter(|o| in_table(o)).count();
        let slots_after = tops[qi].iter().filter(|o| in_table(o)).count();
        paired.questions += 1;
        paired.slots_before += slots_before;
        paired.slots_after += slots_after;
        paired.fewer += usize::from(slots_after < slots_before);
        paired.same += usize::from(slots_after == slots_before);
        paired.more += usize::from(slots_after > slots_before);
        paired
            .before
            .add(rank_scores(&before.tops[qi], owners, &q.expect));
        paired.after.add(rank_scores(&tops[qi], owners, &q.expect));
    }
    let n = paired.questions.max(1) as f64;
    let scores = |s: &RankTotals| {
        json!({
            "hit@1": round(s.hit1 / n),
            format!("hit@{k}"): round(s.hitk / n),
            format!("mrr@{k}"): round(s.rr / n),
        })
    };

    // The table's own numbers in both populations, from the two reports.
    let side = |report: &Value, set: &str| {
        json!({
            "nk_skewness": report[set]["nk_skewness"],
            "hub_threshold": report[set]["hub_threshold"],
            "table": report[set]["by_table"][table],
        })
    };
    let compare = |set: &str| {
        json!({
            "before": side(before.report, set),
            "after": side(&view, set),
        })
    };

    let (mean_cos, _, _) = mean_std_skew(&cosines);
    let (mean_bytes, _, _) = mean_std_skew(&text_bytes);
    let summary = json!({
        "rewrite": rewrite.describe(),
        "rewritten": rewritten_count,
        "not_embeddable": not_embeddable.len(),
        "text_bytes": {
            "mean": round(mean_bytes),
            "min": text_bytes.iter().copied().fold(f64::INFINITY, f64::min),
            "max": text_bytes.iter().copied().fold(0.0, f64::max),
        },
        "chunk_vectors_before": chunks_before,
        "chunk_vectors_after": chunks_after,
        "mean_cosine_to_stored_vector": round(mean_cos),
        "min_cosine_to_stored_vector": round(cosines.iter().copied().fold(f64::INFINITY, f64::min)),
        "table_before_after": {
            "questions": compare("questions"),
            "records_as_queries": compare("records_as_queries"),
        },
        "questions_not_about_table": {
            "questions": paired.questions,
            "table_slots_before": paired.slots_before,
            "table_slots_after": paired.slots_after,
            "questions_with_fewer_table_slots": paired.fewer,
            "questions_with_same_table_slots": paired.same,
            "questions_with_more_table_slots": paired.more,
            "sign_test": {
                "method": "two-sided exact binomial over the questions whose table slots changed, ties dropped",
                "p": round_significant(sign_test(paired.fewer, paired.more), 4),
            },
            "vector_only_before": scores(&paired.before),
            "vector_only_after": scores(&paired.after),
        },
    });
    (view, summary)
}

fn main() {
    let args = parse_args();
    let dir = args.db.parent().unwrap_or(std::path::Path::new("."));
    let config = axil_core::load_config_from(dir).unwrap_or_default();
    let model = config
        .database
        .embedding_model
        .as_deref()
        .and_then(EmbeddingModel::from_name)
        .unwrap_or(EmbeddingModel::BgeSmall);
    let db = Axil::open(&args.db)
        .with_embedder_model(model.clone())
        .and_then(|b| b.build())
        .expect("open database with embedder");

    // Every record that owns a vector, plus every memory record (so an
    // expected answer without a vector still has a class).
    let mut owners: Vec<Owner> = Vec::new();
    let mut points: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut own: Vec<Option<usize>> = Vec::new();
    let mut chunk_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    let mut record_vectors = 0usize;
    for table in db.tables().expect("list tables") {
        for record in db.list(&table).expect("list table") {
            let vector = db.get_vector(&record.id).expect("read vector");
            if table == CHUNKS_TABLE {
                if let (Some(v), Some(source)) = (
                    vector,
                    record.data.get("source_record").and_then(Value::as_str),
                ) {
                    chunk_vectors.push((source.to_string(), normalize(v)));
                }
                continue;
            }
            let (class, key) = classify(&table, &record.data);
            if class == Class::Internal && vector.is_none() {
                continue;
            }
            let o = owners.len();
            own.push(vector.map(|v| {
                record_vectors += 1;
                points.push((o, normalize(v)));
                points.len() - 1
            }));
            let memory = class != Class::Internal;
            owners.push(Owner {
                id: record.id.clone(),
                table: table.clone(),
                class,
                key,
                string_keys: string_keys(&record.data),
                created_at: record.created_at,
                in_window: args.cutoff.is_none_or(|c| record.created_at <= c),
                text: if memory {
                    util::searchable_text(&record.data)
                } else {
                    String::new()
                },
                data: if memory { record.data } else { Value::Null },
            });
        }
    }
    let index: HashMap<String, usize> = owners
        .iter()
        .enumerate()
        .map(|(i, o)| (o.id.as_str().to_string(), i))
        .collect();
    let mut orphan_chunks = 0usize;
    let chunk_count = chunk_vectors.len();
    for (source, v) in chunk_vectors {
        match index.get(&source) {
            Some(&o) => points.push((o, v)),
            None => orphan_chunks += 1,
        }
    }
    let mut has_points = vec![false; owners.len()];
    points.iter().for_each(|(o, _)| has_points[*o] = true);

    // Was each memory record's stored vector embedded from the text its
    // class implies? A record changed later without a re-embed would not be.
    let mut same_text: BTreeMap<&str, (usize, usize, f32)> = BTreeMap::new();
    for (o, owner) in owners.iter().enumerate() {
        let Some(p) = own[o].filter(|_| owner.class != Class::Internal) else {
            continue;
        };
        let fresh = normalize(db.embed_passage(&owner.text).expect("embed record text"));
        let cos = dot(&fresh, &points[p].1);
        let entry = same_text
            .entry(owner.class.name())
            .or_insert((0, 0, f32::INFINITY));
        entry.0 += 1;
        entry.1 += usize::from(cos >= SAME_TEXT_COSINE);
        entry.2 = entry.2.min(cos);
    }

    let questions: Vec<Question> = std::fs::read_to_string(&args.questions)
        .expect("read questions")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let q: Value = serde_json::from_str(l).expect("question json");
            let text = q["question"].as_str().expect("question text");
            Question {
                id: q["id"].as_str().unwrap_or_default().to_string(),
                kind: q["kind"].as_str().unwrap_or_default().to_string(),
                expect: serde_json::from_value(q["expect"].clone()).expect("expect groups"),
                vector: normalize(db.embed_query(text).expect("embed question")),
            }
        })
        .collect();

    let class: Vec<Class> = owners.iter().map(|o| o.class).collect();
    let member = |internal: bool| -> Vec<bool> {
        owners
            .iter()
            .enumerate()
            .map(|(o, w)| w.in_window && has_points[o] && (internal || w.class != Class::Internal))
            .collect()
    };
    let memory = Space {
        points,
        own,
        class,
        member: member(false),
    };
    let mut views = serde_json::Map::new();
    let (memory_report, memory_tops) = run_space(&memory, &owners, &index, &questions, args.k);
    views.insert("memory".into(), memory_report.clone());
    let memory = Space {
        member: member(true),
        ..memory
    };
    views.insert(
        "index".into(),
        run_space(&memory, &owners, &index, &questions, args.k).0,
    );
    let memory = Space {
        member: member(false),
        ..memory
    };

    let mut what_if_report = Value::Null;
    if let Some(table) = &args.what_if {
        let before = Before {
            space: &memory,
            report: &memory_report,
            tops: &memory_tops,
        };
        let mut summaries = serde_json::Map::new();
        summaries.insert("table".into(), json!(table));
        for rewrite in Rewrite::ALL {
            let (view, summary) = what_if(
                rewrite, table, &db, &owners, &index, &questions, args.k, &before,
            );
            views.insert(format!("what_if_{}", rewrite.name()), view);
            summaries.insert(rewrite.name().into(), summary);
        }
        what_if_report = Value::Object(summaries);
    }

    // Per-table counts, by class and key.
    let mut tables: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    for (o, w) in owners.iter().enumerate() {
        let row = tables.entry(w.table.as_str()).or_default();
        let label = match w.class {
            Class::Keyed => format!("key:{}", w.key.unwrap_or_default()),
            other => other.name().to_string(),
        };
        *row.entry(label).or_default() += 1;
        if memory.own[o].is_some() {
            *row.entry(format!("{}_with_vector", w.class.name()))
                .or_default() += 1;
        }
        if !w.in_window {
            *row.entry("after_cutoff".to_string()).or_default() += 1;
        }
    }
    let fallback_records: Vec<Value> = owners
        .iter()
        .enumerate()
        .filter(|(_, w)| w.class == Class::Fallback)
        .map(|(o, w)| {
            json!({
                "id": w.id.as_str(),
                "table": w.table,
                "string_keys": w.string_keys,
                "created_at": w.created_at.to_rfc3339(),
                "has_vector": memory.own[o].is_some(),
                "in_window": w.in_window,
                "text_bytes": w.text.len(),
            })
        })
        .collect();

    let in_index = db.vector_count();
    let report = json!({
        "benchmark": "hubness",
        "model": model.name(),
        "dimensions": model.dimensions(),
        "k": args.k,
        "cutoff": args.cutoff.map(|c| c.to_rfc3339()),
        "hub_rule": "an owner is a hub when its N_k exceeds the population mean + 2 std",
        "vectors": {
            "in_index": in_index,
            "record_vectors": record_vectors,
            "chunk_vectors": chunk_count,
            "orphan_chunk_vectors": orphan_chunks,
            "without_a_record": in_index.map(|n| n.saturating_sub(record_vectors + chunk_count)),
        },
        "records_by_table": tables,
        "stored_vector_matches_text": same_text
            .iter()
            .map(|(class, (n, same, min))| {
                (class.to_string(), json!({
                    "with_vector": n,
                    "cosine_at_least": SAME_TEXT_COSINE,
                    "matching": same,
                    "min_cosine": round(f64::from(*min)),
                }))
            })
            .collect::<serde_json::Map<_, _>>(),
        "fallback_records": fallback_records,
        "what_if": what_if_report,
        "views": views,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize")
    );
}
