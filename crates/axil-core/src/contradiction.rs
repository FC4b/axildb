//! Possible contradictions: a [`PairJudge`] reads a new record against the
//! older records it may have replaced, and the pairs it flags become a
//! review signal.
//!
//! Candidates come from [`Axil::supersede_candidates`] (same table, live,
//! similar enough to be about the same thing), and only those created before
//! the record are judged, as `(older, newer)`: the order the U1 judge was
//! calibrated in. Each judged record gets one row in
//! [`CONTRADICTION_REVIEW_TABLE`] with every pair's score, flagged or not,
//! and each flagged pair a [`POSSIBLY_CONTRADICTS`] edge (newer → older)
//! carrying the score, threshold, model and calibration.
//!
//! It is a suggestion and nothing more. A flag never supersedes, hides,
//! demotes or edits either record: deciding whether the newer one replaced
//! the older is left to whoever reviews the flag, with
//! [`Axil::mark_superseded`] if it did or
//! [`Axil::dismiss_possible_contradiction`] if both hold.
//!
//! Scoring loads a model, so it belongs in a background pass
//! ([`Axil::review_new_records`]), never on the write path: the judge is
//! passed in as a loader that only runs when a pass finds records to judge.

use std::collections::{BTreeSet, HashSet};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::db::{is_superseded_record, Axil};
use crate::decide::PairJudge;
use crate::error::{AxilError, Result};
use crate::plugin::Direction;
use crate::record::{Record, RecordId};
use crate::util::edge_types::POSSIBLY_CONTRADICTS;

/// Internal table holding one row per judged record: the judge, its
/// threshold, every pair it scored, and the pairs a reviewer dismissed.
pub const CONTRADICTION_REVIEW_TABLE: &str = "_contradiction_reviews";

/// Tables a review pass judges unless configured otherwise: the ones that
/// hold claims a later record can overturn. `errors` are closed by
/// `resolve`, not by a newer claim; sessions, commits and `_`-prefixed
/// tables are logs.
pub const DEFAULT_REVIEW_TABLES: &[&str] = &["decisions", "context", "rules"];

/// Settings for [`Axil::review_new_records`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewOptions {
    /// Tables whose new records are judged. `_`-prefixed tables are skipped.
    pub tables: Vec<String>,
    /// Most records judged in one pass, newest first; the rest wait for the
    /// next pass.
    pub max_records: usize,
    /// Only records created this recently are judged by a pass, so turning
    /// review on does not judge a whole history.
    /// [`Axil::review_possible_contradictions`] judges an older record on
    /// request.
    pub window: Duration,
    /// Older records judged against each new one, most similar first.
    pub candidates: usize,
}

impl Default for ReviewOptions {
    /// [`DEFAULT_REVIEW_TABLES`], 20 records a pass, a 7-day window and 5
    /// candidates (the top-5 candidate search measured in
    /// `benchmarks/results/step4-29.16-supersede-candidates-2026-09-28.json`).
    fn default() -> Self {
        Self {
            tables: DEFAULT_REVIEW_TABLES
                .iter()
                .map(|t| t.to_string())
                .collect(),
            max_records: 20,
            window: Duration::days(7),
            candidates: 5,
        }
    }
}

/// One older record judged against a newer one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgedPair {
    /// The older record (the judge's premise).
    pub older: RecordId,
    /// Cosine similarity that made it a candidate.
    pub similarity: f32,
    /// The judge's calibrated score.
    pub probability: f64,
    /// Whether the score reached the judge's threshold.
    pub flagged: bool,
}

/// What one [`Axil::review_new_records`] pass did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewReport {
    /// Records judged in this pass.
    pub records_reviewed: usize,
    /// `(older, newer)` pairs scored.
    pub pairs_judged: usize,
    /// Pairs at or above the threshold.
    pub flagged: usize,
    /// Records due for review that this pass left for the next one.
    pub pending: usize,
    /// The judge, when one was loaded.
    pub model: Option<String>,
    /// Time to load the judge; `None` when nothing was due, so none was loaded.
    pub load_ms: Option<u64>,
    /// Wall time of the whole pass.
    pub duration_ms: u64,
}

/// An open flag: a judged pair whose records both still exist, neither
/// superseded, that no reviewer has dismissed.
#[derive(Debug, Clone)]
pub struct PossibleContradiction {
    /// The newer record.
    pub newer: Record,
    /// The older record it may contradict or replace.
    pub older: Record,
    /// The judge's calibrated score.
    pub probability: f64,
    /// The threshold it reached.
    pub threshold: f64,
    /// Cosine similarity that made the older record a candidate.
    pub similarity: f32,
    /// The judge's model.
    pub model: String,
    /// The judge's calibration.
    pub calibration: String,
    /// When the pair was judged.
    pub judged_at: DateTime<Utc>,
}

impl Axil {
    /// Judge `record` against the older live records in its table that it
    /// may have replaced, without writing anything.
    ///
    /// Takes up to `candidates` of [`Axil::supersede_candidates`] created
    /// before `record`, most similar first, and scores each as
    /// `(older, newer)`, each record read as its headline
    /// ([`crate::boot::record_headline`]: the `rule` of a rule, otherwise
    /// its summary or main text). `record` need not be stored.
    pub fn judge_possible_contradictions(
        &self,
        record: &Record,
        judge: &dyn PairJudge,
        candidates: usize,
    ) -> Result<Vec<JudgedPair>> {
        let newer_text = crate::boot::record_headline(record);
        if newer_text.trim().is_empty() || candidates == 0 {
            return Ok(Vec::new());
        }
        // Over-fetch: a record judged late can have newer neighbours, which
        // are judged when their own turn comes, against it.
        let older: Vec<(Record, f32)> = self
            .supersede_candidates(record, candidates.saturating_mul(2))?
            .into_iter()
            .filter(|(c, _)| created_before(c, record))
            .take(candidates)
            .collect();
        if older.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = older
            .iter()
            .map(|(c, _)| crate::boot::record_headline(c))
            .collect();
        let pairs: Vec<(&str, &str)> = texts
            .iter()
            .map(|t| (t.as_str(), newer_text.as_str()))
            .collect();
        let verdicts = judge.judge(&pairs)?;
        if verdicts.len() != pairs.len() {
            return Err(AxilError::plugin(format!(
                "judge {} returned {} verdicts for {} pairs",
                judge.name(),
                verdicts.len(),
                pairs.len()
            )));
        }
        Ok(older
            .into_iter()
            .zip(verdicts)
            .map(|((c, similarity), v)| JudgedPair {
                older: c.id,
                similarity,
                probability: v.probability,
                flagged: v.flagged,
            })
            .collect())
    }

    /// Judge the stored record `id` (see
    /// [`Axil::judge_possible_contradictions`]) and record the outcome: its
    /// row in [`CONTRADICTION_REVIEW_TABLE`] with every pair's score, and a
    /// [`POSSIBLY_CONTRADICTS`] edge to each flagged older record when a
    /// graph is attached.
    ///
    /// Judging a record again replaces its pairs and keeps its dismissals;
    /// an edge the new judgement no longer supports is removed. Neither
    /// record is ever changed.
    pub fn review_possible_contradictions(
        &self,
        id: &RecordId,
        judge: &dyn PairJudge,
        candidates: usize,
    ) -> Result<Vec<JudgedPair>> {
        let record = self
            .get(id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {id}")))?;
        let pairs = self.judge_possible_contradictions(&record, judge, candidates)?;
        let existing = self.review_row(id)?;
        let dismissed = existing
            .as_ref()
            .map(|r| dismissed_set(&r.data))
            .unwrap_or_default();

        if self.has_graph_index() {
            let flagged: HashSet<&str> = pairs
                .iter()
                .filter(|p| p.flagged && !dismissed.contains(p.older.as_str()))
                .map(|p| p.older.as_str())
                .collect();
            let mut linked = HashSet::new();
            for edge in self.edges(id, Some(POSSIBLY_CONTRADICTS), Direction::Out)? {
                if flagged.contains(edge.to.as_str()) {
                    linked.insert(edge.to.as_str().to_string());
                } else {
                    self.unrelate(&edge.id)?;
                }
            }
            for pair in pairs.iter().filter(|p| {
                flagged.contains(p.older.as_str()) && !linked.contains(p.older.as_str())
            }) {
                self.relate(
                    id,
                    POSSIBLY_CONTRADICTS,
                    &pair.older,
                    Some(json!({
                        "p": pair.probability,
                        "threshold": judge.threshold(),
                        "similarity": pair.similarity,
                        "model": judge.name(),
                        "calibration": judge.calibration(),
                    })),
                )?;
            }
        }

        let row = json!({
            "record": id.as_str(),
            "table": record.table,
            "model": judge.name(),
            "calibration": judge.calibration(),
            "threshold": judge.threshold(),
            "pairs": pairs,
            "flagged": pairs.iter().filter(|p| p.flagged).count(),
            "dismissed": dismissed.iter().collect::<Vec<_>>(),
        });
        match existing {
            Some(r) => {
                self.update(&r.id, row)?;
            }
            None => {
                self.insert(CONTRADICTION_REVIEW_TABLE, row)?;
            }
        }
        Ok(pairs)
    }

    /// The review pass: judge the records in `opts.tables` created within
    /// `opts.window` that are live and not yet judged, newest first, at most
    /// `opts.max_records` of them (see
    /// [`Axil::review_possible_contradictions`]).
    ///
    /// `load_judge` runs only when at least one record is due, so a pass with
    /// nothing to do loads no model. Stops at the first error; the records
    /// already judged keep their rows.
    pub fn review_new_records<F>(&self, opts: &ReviewOptions, load_judge: F) -> Result<ReviewReport>
    where
        F: FnOnce() -> Result<Box<dyn PairJudge>>,
    {
        let started = std::time::Instant::now();
        let due = self.records_due_for_review(opts)?;
        let mut report = ReviewReport {
            pending: due.len(),
            ..ReviewReport::default()
        };
        if !due.is_empty() && opts.max_records > 0 {
            let loading = std::time::Instant::now();
            let judge = load_judge()?;
            report.load_ms = Some(loading.elapsed().as_millis() as u64);
            report.model = Some(judge.name().to_string());
            for record in due.iter().take(opts.max_records) {
                let pairs = self.review_possible_contradictions(
                    &record.id,
                    judge.as_ref(),
                    opts.candidates,
                )?;
                report.records_reviewed += 1;
                report.pending -= 1;
                report.pairs_judged += pairs.len();
                report.flagged += pairs.iter().filter(|p| p.flagged).count();
            }
        }
        report.duration_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Records a review pass would judge now, newest first (see
    /// [`Axil::review_new_records`]).
    pub fn records_due_for_review(&self, opts: &ReviewOptions) -> Result<Vec<Record>> {
        let reviewed: HashSet<String> = self
            .list(CONTRADICTION_REVIEW_TABLE)?
            .iter()
            .filter_map(|r| r.data.get("record").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        let since = Utc::now() - opts.window;
        let mut due = Vec::new();
        for table in opts.tables.iter().filter(|t| !t.starts_with('_')) {
            due.extend(self.list(table)?.into_iter().filter(|r| {
                r.created_at >= since
                    && !is_superseded_record(r)
                    && !reviewed.contains(r.id.as_str())
            }));
        }
        due.sort_by(|a, b| (b.created_at, &b.id).cmp(&(a.created_at, &a.id)));
        Ok(due)
    }

    /// Open possible contradictions, most recently judged first: flagged
    /// pairs whose records both still exist, neither superseded, that no
    /// reviewer has dismissed.
    pub fn possible_contradictions(&self) -> Result<Vec<PossibleContradiction>> {
        let mut rows = self.list(CONTRADICTION_REVIEW_TABLE)?;
        rows.retain(|r| r.data.get("flagged").and_then(Value::as_u64).unwrap_or(0) > 0);
        rows.sort_by(|a, b| (b.updated_at, &b.id).cmp(&(a.updated_at, &a.id)));
        let mut open = Vec::new();
        for row in rows {
            let Some(newer_id) = row_record_id(&row.data) else {
                continue;
            };
            let Some(newer) = self.get(&newer_id)? else {
                continue;
            };
            if is_superseded_record(&newer) {
                continue;
            }
            let dismissed = dismissed_set(&row.data);
            for pair in row_pairs(&row.data) {
                if !pair.flagged || dismissed.contains(pair.older.as_str()) {
                    continue;
                }
                let Some(older) = self.get(&pair.older)? else {
                    continue;
                };
                if is_superseded_record(&older) {
                    continue;
                }
                open.push(PossibleContradiction {
                    newer: newer.clone(),
                    older,
                    probability: pair.probability,
                    threshold: row
                        .data
                        .get("threshold")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    similarity: pair.similarity,
                    model: row_str(&row.data, "model"),
                    calibration: row_str(&row.data, "calibration"),
                    judged_at: row.updated_at,
                });
            }
        }
        Ok(open)
    }

    /// Mark a flagged pair reviewed with both records holding: the pair
    /// leaves [`Axil::possible_contradictions`] and its
    /// [`POSSIBLY_CONTRADICTS`] edge is removed, while the review row keeps
    /// the judged score. Returns false when `newer` has no open flagged pair
    /// with `older`.
    pub fn dismiss_possible_contradiction(
        &self,
        newer: &RecordId,
        older: &RecordId,
    ) -> Result<bool> {
        let Some(row) = self.review_row(newer)? else {
            return Ok(false);
        };
        let mut dismissed = dismissed_set(&row.data);
        let flagged = row_pairs(&row.data)
            .iter()
            .any(|p| p.flagged && p.older == *older);
        if !flagged || !dismissed.insert(older.as_str().to_string()) {
            return Ok(false);
        }
        if self.has_graph_index() {
            for edge in self.edges(newer, Some(POSSIBLY_CONTRADICTS), Direction::Out)? {
                if edge.to == *older {
                    self.unrelate(&edge.id)?;
                }
            }
        }
        let mut data = row.data.clone();
        data["dismissed"] = json!(dismissed.into_iter().collect::<Vec<_>>());
        self.update(&row.id, data)?;
        Ok(true)
    }

    /// Every review row, most recently judged first: each judged record with
    /// the score of every pair, flagged or not.
    pub fn contradiction_reviews(&self) -> Result<Vec<Record>> {
        let mut rows = self.list(CONTRADICTION_REVIEW_TABLE)?;
        rows.sort_by(|a, b| (b.updated_at, &b.id).cmp(&(a.updated_at, &a.id)));
        Ok(rows)
    }

    /// The review row of the record `id`, if it was judged.
    fn review_row(&self, id: &RecordId) -> Result<Option<Record>> {
        Ok(self
            .list(CONTRADICTION_REVIEW_TABLE)?
            .into_iter()
            .find(|r| r.data.get("record").and_then(Value::as_str) == Some(id.as_str())))
    }
}

/// Whether `a` was created before `b` (ties broken by id, so exactly one of
/// any two records is the older).
fn created_before(a: &Record, b: &Record) -> bool {
    (a.created_at, &a.id) < (b.created_at, &b.id)
}

fn row_record_id(data: &Value) -> Option<RecordId> {
    data.get("record")
        .and_then(Value::as_str)
        .map(|s| RecordId(s.to_string()))
}

fn row_pairs(data: &Value) -> Vec<JudgedPair> {
    data.get("pairs")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn row_str(data: &Value, key: &str) -> String {
    data.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Older-record ids a reviewer dismissed, sorted so a rewritten row is stable.
fn dismissed_set(data: &Value) -> BTreeSet<String> {
    data.get("dismissed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: i64, id: &str) -> Record {
        let mut r = Record::new("decisions", json!({}));
        r.created_at = DateTime::from_timestamp(secs, 0).unwrap();
        r.id = RecordId(id.to_string());
        r
    }

    #[test]
    fn exactly_one_of_two_records_is_the_older() {
        let a = at(100, "01A");
        let b = at(200, "01B");
        assert!(created_before(&a, &b) && !created_before(&b, &a));
        // Same instant: the id decides, the same way both times.
        let c = at(100, "01C");
        assert!(created_before(&a, &c) && !created_before(&c, &a));
        assert!(!created_before(&a, &a));
    }

    #[test]
    fn review_rows_parse_their_pairs_and_dismissals() {
        let data = json!({
            "record": "01NEW",
            "pairs": [
                {"older": "01OLD", "similarity": 0.8, "probability": 0.9, "flagged": true},
                {"older": "01OTHER", "similarity": 0.7, "probability": 0.1, "flagged": false},
            ],
            "dismissed": ["01OTHER"],
        });
        assert_eq!(row_record_id(&data), Some(RecordId("01NEW".into())));
        let pairs = row_pairs(&data);
        assert_eq!(pairs.len(), 2);
        assert!(pairs[0].flagged && !pairs[1].flagged);
        assert_eq!(
            dismissed_set(&data).into_iter().collect::<Vec<_>>(),
            ["01OTHER"]
        );
        // A row written before any dismissal, or a malformed one, has none.
        assert!(dismissed_set(&json!({})).is_empty());
        assert!(row_pairs(&json!({"pairs": "junk"})).is_empty());
    }

    #[test]
    fn default_tables_hold_overturnable_claims_only() {
        let opts = ReviewOptions::default();
        assert_eq!(opts.tables, ["decisions", "context", "rules"]);
        assert!(opts.tables.iter().all(|t| !t.starts_with('_')));
        assert!(opts.max_records > 0 && opts.candidates > 0);
    }
}
