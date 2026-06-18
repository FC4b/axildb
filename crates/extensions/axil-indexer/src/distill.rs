//! Failure → corrective-rule write-back loop (the `axil rule distill` command).
//!
//! The rest of the rules pipeline is one-directional: [`auto_extract_rules`]
//! reads conventions *from* `CLAUDE.md` *into* the `_rules` store. This module
//! closes the loop in the other direction — it distills recurring failures
//! (the `errors` table, written by `axil store errors` and the auto-capture
//! hook) back *out* into corrective directives, persists them where `axil boot`
//! already surfaces them, and writes them into `CLAUDE.md` inside an idempotent
//! marker block.
//!
//! The synthesis is deliberately template-based (Path 0 — no LLM): a failure
//! seen `N ≥ min_evidence` times with a recorded fix becomes the directive
//! *"Last N times you hit X, the fix was Y."* Failures are grouped with the
//! shared [`axil_core::simhash`] near-duplicate primitive so case / whitespace /
//! line-number variants of the same failure collapse into one group.
//!
//! [`auto_extract_rules`]: crate::rules::auto_extract_rules

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

use axil_core::{simhash, Axil, Result};

/// Start marker for the managed CLAUDE.md correction block.
pub const LEARNED_BLOCK_BEGIN: &str = "<!-- axil:learned:start -->";
/// End marker for the managed CLAUDE.md correction block.
pub const LEARNED_BLOCK_END: &str = "<!-- axil:learned:end -->";

/// `source` tag stamped on persisted learned rules so they can be found and
/// replaced idempotently without touching user- or seed-set rules.
pub const LEARNED_RULE_SOURCE: &str = "learned";

/// Table `axil boot`'s Constraints / pinned-rules block reads from. Distinct
/// from [`crate::rules::TABLE_RULES`] (`_rules`, the key/value convention
/// store): the boot path surfaces pinned records in the plain `rules` table.
const TABLE_BOOT_RULES: &str = "rules";

/// Minimum number of occurrences before a failure earns a directive.
pub const DEFAULT_MIN_EVIDENCE: usize = 2;

/// Cap on how many directives are emitted (keeps the CLAUDE.md block small).
pub const DEFAULT_MAX_DIRECTIVES: usize = 10;

/// SimHash Hamming threshold for grouping failures. Looser than recall's tight
/// collapse threshold (3) because here false-grouping is low-cost — it merely
/// merges two corrections — while we want near-identical failure restatements
/// (the auto-capture hook re-records the same error verbatim) to land together.
const GROUP_HAMMING_THRESHOLD: u32 = 8;

/// Half-life (days) for the recency weight in the impact score.
const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// Longest error excerpt kept in a directive (chars).
const MAX_ERROR_EXCERPT: usize = 100;
/// Longest fix excerpt kept in a directive (chars).
const MAX_FIX_EXCERPT: usize = 200;

/// A corrective directive distilled from a cluster of recurring failures.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedDirective {
    /// Stable hex signature of the failure cluster (used as the idempotent key).
    pub signature: String,
    /// The rendered directive line, e.g. *"Last 3 times you hit X, the fix was Y."*
    pub directive: String,
    /// How many failures backed this directive.
    pub frequency: usize,
    /// Most recent occurrence in the cluster.
    pub last_seen: DateTime<Utc>,
    /// frequency × recency-decay — drives ranking and the block cap.
    pub impact: f64,
}

/// A single failure occurrence pulled from the `errors` table.
struct Failure {
    error: String,
    fix: Option<String>,
    created_at: DateTime<Utc>,
    fingerprint: u64,
}

/// Read the `errors` table, group recurring failures, and synthesize ranked
/// corrective directives. Pure read — does not write anything.
///
/// A cluster qualifies when it has at least `min_evidence` occurrences *and* at
/// least one recorded fix (the directive needs a "the fix was Y"). Results are
/// sorted by descending impact (frequency × recency) and capped at
/// `max_directives`.
pub fn distill_directives(
    db: &Axil,
    min_evidence: usize,
    max_directives: usize,
) -> Result<Vec<LearnedDirective>> {
    let records = db.list("errors").unwrap_or_default();

    let failures: Vec<Failure> = records
        .iter()
        .filter_map(|r| {
            let error = r
                .data
                .get("error")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())?;
            let fix = r
                .data
                .get("fix")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            Some(Failure {
                error: error.to_string(),
                fix,
                created_at: r.created_at,
                fingerprint: simhash::simhash(&canonicalize(error)),
            })
        })
        .collect();

    // Greedy single-link clustering by SimHash Hamming distance. Each failure
    // joins the first existing cluster within threshold of its seed, else
    // starts a new one.
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for (i, f) in failures.iter().enumerate() {
        let mut placed = false;
        for cluster in &mut clusters {
            let seed = &failures[cluster[0]];
            if simhash::hamming(seed.fingerprint, f.fingerprint) <= GROUP_HAMMING_THRESHOLD {
                cluster.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push(vec![i]);
        }
    }

    let now = Utc::now();
    let mut directives: Vec<LearnedDirective> = Vec::new();
    for cluster in &clusters {
        if cluster.len() < min_evidence.max(1) {
            continue;
        }
        let members: Vec<&Failure> = cluster.iter().map(|&i| &failures[i]).collect();

        // Most recent member supplies the representative error text and fix.
        let latest = members
            .iter()
            .max_by_key(|f| f.created_at)
            .expect("non-empty cluster");
        let last_seen = latest.created_at;

        // Prefer the most recent recorded fix; skip the cluster if none exists.
        let fix = members
            .iter()
            .filter(|f| f.fix.is_some())
            .max_by_key(|f| f.created_at)
            .and_then(|f| f.fix.clone());
        let Some(fix) = fix else {
            continue;
        };

        let frequency = members.len();
        let age_days = (now - last_seen).num_seconds().max(0) as f64 / 86_400.0;
        let recency = 0.5_f64.powf(age_days / RECENCY_HALF_LIFE_DAYS);
        let impact = frequency as f64 * recency;

        directives.push(LearnedDirective {
            signature: format!("{:016x}", latest.fingerprint),
            directive: synthesize(frequency, &latest.error, &fix),
            frequency,
            last_seen,
            impact,
        });
    }

    // Rank by impact (frequency × recency), then frequency, then recency as
    // tie-breakers so the order is deterministic for a given input.
    directives.sort_by(|a, b| {
        b.impact
            .partial_cmp(&a.impact)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.frequency.cmp(&a.frequency))
            .then(b.last_seen.cmp(&a.last_seen))
    });
    directives.truncate(max_directives);
    Ok(directives)
}

/// Build the directive sentence: *"Last N times you hit X, the fix was Y."*
fn synthesize(frequency: usize, error: &str, fix: &str) -> String {
    format!(
        "Last {frequency} times you hit \"{}\", the fix was: {}",
        excerpt(error, MAX_ERROR_EXCERPT),
        excerpt(fix, MAX_FIX_EXCERPT),
    )
}

/// Canonicalize failure text for grouping: lowercase, drop digits (so line
/// numbers / counts don't split a cluster) and other noise, collapse
/// whitespace. The result is only ever fed to SimHash, never displayed.
fn canonicalize(text: &str) -> String {
    let lowered: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_digit() {
                ' '
            } else if c.is_alphanumeric() || c.is_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    simhash::normalize(&lowered)
}

/// Truncate `text` to at most `max` chars, appending `…` when cut. Newlines are
/// collapsed to spaces so a directive stays on one line.
fn excerpt(text: &str, max: usize) -> String {
    let oneline = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if oneline.chars().count() <= max {
        return oneline;
    }
    let truncated: String = oneline.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", truncated.trim_end())
}

/// Render the marker-delimited CLAUDE.md block for `directives`. Returns `None`
/// when there are no directives (the caller removes any stale block).
pub fn render_block(directives: &[LearnedDirective]) -> Option<String> {
    if directives.is_empty() {
        return None;
    }
    let mut body = String::new();
    body.push_str(LEARNED_BLOCK_BEGIN);
    body.push('\n');
    body.push_str("## Learned Corrections (auto-distilled from recurring failures)\n\n");
    body.push_str(
        "_Maintained by `axil rule distill`. Everything between the markers is regenerated on \
         each run — edit elsewhere. Remove the markers to disable._\n\n",
    );
    for d in directives {
        body.push_str("- ");
        body.push_str(&d.directive);
        body.push('\n');
    }
    body.push_str(LEARNED_BLOCK_END);
    body.push('\n');
    Some(body)
}

/// Splice `block` into `existing` CLAUDE.md content, replacing any prior managed
/// block in place and leaving all human-edited content untouched.
///
/// - `Some(block)` replaces an existing marker block, or appends one when none
///   is present.
/// - `None` removes an existing marker block (used when no directives remain).
///
/// Pure string transform so the marker logic is unit-testable without I/O.
pub fn apply_block(existing: &str, block: Option<&str>) -> String {
    let prior = find_block(existing);

    // No managed block present and nothing to add → leave the file untouched.
    if prior.is_none() && block.is_none() {
        return existing.to_string();
    }

    // Split the file into the human content before and after any prior block.
    let (before, after) = match prior {
        Some((start, end)) => (&existing[..start], &existing[end..]),
        None => (existing, ""),
    };
    let before = before.trim_end();
    let after = after.trim();

    let mut out = String::new();
    if !before.is_empty() {
        out.push_str(before);
    }
    if let Some(block) = block {
        if !before.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(block.trim());
    }
    if !after.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(after);
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Locate the byte range `[start, end)` of an existing managed block, markers
/// included. Returns `None` when no complete block is present.
fn find_block(text: &str) -> Option<(usize, usize)> {
    let start = text.find(LEARNED_BLOCK_BEGIN)?;
    let end_rel = text[start..].find(LEARNED_BLOCK_END)?;
    let mut end = start + end_rel + LEARNED_BLOCK_END.len();
    // Swallow a single trailing newline so repeated runs don't accrete blanks.
    if text[end..].starts_with('\n') {
        end += 1;
    }
    Some((start, end))
}

/// Write the directive block into the CLAUDE.md at `path`, replacing any prior
/// managed block in place. Returns `true` when the file changed.
///
/// When `directives` is empty an existing block is removed; if the file does
/// not exist it is only created when there is a block to write.
pub fn write_claude_md(
    path: &std::path::Path,
    directives: &[LearnedDirective],
) -> Result<bool> {
    let block = render_block(directives);
    let existing = std::fs::read_to_string(path).unwrap_or_default();

    // Avoid creating an empty file just to hold nothing.
    if existing.is_empty() && block.is_none() {
        return Ok(false);
    }

    let next = apply_block(&existing, block.as_deref());
    if next == existing {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                axil_core::AxilError::plugin(format!("create {}: {e}", parent.display()))
            })?;
        }
    }
    std::fs::write(path, next)
        .map_err(|e| axil_core::AxilError::plugin(format!("write {}: {e}", path.display())))?;
    Ok(true)
}

/// Persist directives as pinned records in the boot-surfaced `rules` table so
/// `axil boot` echoes them even when no CLAUDE.md edit happened. Idempotent:
/// records are keyed by signature and upserted; learned rules whose signature
/// is no longer present are removed. Returns the number of active learned rules.
pub fn persist_rules(db: &Axil, directives: &[LearnedDirective]) -> Result<usize> {
    let existing = db.list(TABLE_BOOT_RULES).unwrap_or_default();

    // Index prior learned rules by signature.
    let mut by_sig: std::collections::HashMap<String, axil_core::Record> =
        std::collections::HashMap::new();
    let mut stale: Vec<axil_core::RecordId> = Vec::new();
    let wanted: std::collections::HashSet<&str> =
        directives.iter().map(|d| d.signature.as_str()).collect();
    for r in existing {
        let is_learned =
            r.data.get("source").and_then(|v| v.as_str()) == Some(LEARNED_RULE_SOURCE);
        if !is_learned {
            continue;
        }
        match r.data.get("_learned_sig").and_then(|v| v.as_str()) {
            Some(sig) if wanted.contains(sig) => {
                by_sig.insert(sig.to_string(), r);
            }
            _ => stale.push(r.id),
        }
    }

    for id in stale {
        let _ = db.delete(&id);
    }

    for d in directives {
        let data = json!({
            "rule": d.directive,
            "source": LEARNED_RULE_SOURCE,
            "_learned_sig": d.signature,
            "frequency": d.frequency,
            "last_seen": d.last_seen.to_rfc3339(),
            "_importance": 0.95,
            "_importance_pinned": true,
        });
        match by_sig.get(&d.signature) {
            Some(existing) => {
                let _ = db.update(&existing.id, data);
            }
            None => {
                db.insert(TABLE_BOOT_RULES, data)?;
            }
        }
    }

    Ok(directives.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    fn store_error(db: &Axil, error: &str, fix: &str) {
        let mut data = json!({ "error": error });
        if !fix.is_empty() {
            data["fix"] = json!(fix);
        }
        db.insert("errors", data).unwrap();
    }

    #[test]
    fn distill_requires_min_evidence() {
        let (db, _dir) = temp_db();
        store_error(&db, "connection pool exhausted under load", "raised pool size to 20");

        // Single occurrence — below the default ≥2 threshold.
        let directives = distill_directives(&db, DEFAULT_MIN_EVIDENCE, DEFAULT_MAX_DIRECTIVES).unwrap();
        assert!(directives.is_empty(), "one occurrence should not earn a directive");
    }

    #[test]
    fn distill_groups_recurring_failures() {
        let (db, _dir) = temp_db();
        // Same failure three times, with line-number / case noise that
        // canonicalization + SimHash should fold into one cluster.
        store_error(&db, "Connection pool exhausted at worker.rs:42", "raise pool size");
        store_error(&db, "connection pool exhausted at worker.rs:88", "raise pool size to 20");
        store_error(&db, "CONNECTION POOL EXHAUSTED at worker.rs:5", "raise pool size to 30");

        let directives =
            distill_directives(&db, DEFAULT_MIN_EVIDENCE, DEFAULT_MAX_DIRECTIVES).unwrap();
        assert_eq!(directives.len(), 1, "three variants collapse to one directive");
        let d = &directives[0];
        assert_eq!(d.frequency, 3);
        assert!(d.directive.starts_with("Last 3 times you hit"));
        // Most recent fix wins.
        assert!(d.directive.contains("raise pool size to 30"), "got: {}", d.directive);
    }

    #[test]
    fn distill_skips_clusters_without_a_fix() {
        let (db, _dir) = temp_db();
        store_error(&db, "flaky network test timeout", "");
        store_error(&db, "flaky network test timeout again", "");

        let directives =
            distill_directives(&db, DEFAULT_MIN_EVIDENCE, DEFAULT_MAX_DIRECTIVES).unwrap();
        assert!(directives.is_empty(), "no recorded fix → no directive");
    }

    #[test]
    fn distill_respects_max_cap() {
        let (db, _dir) = temp_db();
        // Five lexically distinct failures (no digits — canonicalize strips
        // those), each seen twice so each earns its own directive.
        let kinds = [
            "alpha database connection pool exhausted timeout",
            "beta filesystem permission denied while writing output",
            "gamma json parser hit an unexpected closing token",
            "delta network socket connection reset by remote peer",
            "epsilon heap memory allocation failed out of space",
        ];
        for err in kinds {
            store_error(&db, err, "some fix");
            store_error(&db, err, "some fix");
        }
        let all = distill_directives(&db, DEFAULT_MIN_EVIDENCE, DEFAULT_MAX_DIRECTIVES).unwrap();
        assert_eq!(all.len(), 5, "five distinct clusters");
        let capped = distill_directives(&db, DEFAULT_MIN_EVIDENCE, 3).unwrap();
        assert_eq!(capped.len(), 3, "capped at max_directives");
    }

    #[test]
    fn apply_block_appends_when_absent() {
        let block = "<!-- axil:learned:start -->\nX\n<!-- axil:learned:end -->";
        let out = apply_block("# Title\n\nbody", Some(block));
        assert!(out.starts_with("# Title\n\nbody"));
        assert!(out.contains(LEARNED_BLOCK_BEGIN));
        assert!(out.trim_end().ends_with(LEARNED_BLOCK_END));
    }

    #[test]
    fn apply_block_replaces_in_place_idempotently() {
        let human = "# Title\n\nhuman content\n";
        let v1 = apply_block(
            human,
            Some("<!-- axil:learned:start -->\nv1\n<!-- axil:learned:end -->"),
        );
        let v2 = apply_block(
            &v1,
            Some("<!-- axil:learned:start -->\nv2\n<!-- axil:learned:end -->"),
        );
        // Human content survives; only the managed block changed; no accretion.
        assert!(v2.contains("human content"));
        assert!(v2.contains("v2"));
        assert!(!v2.contains("v1"));
        assert_eq!(v2.matches(LEARNED_BLOCK_BEGIN).count(), 1);

        // Re-applying the same block is a no-op.
        let v3 = apply_block(
            &v2,
            Some("<!-- axil:learned:start -->\nv2\n<!-- axil:learned:end -->"),
        );
        assert_eq!(v2, v3);
    }

    #[test]
    fn apply_block_removes_stale_block() {
        let with_block =
            "# Title\n\nhuman\n\n<!-- axil:learned:start -->\nX\n<!-- axil:learned:end -->\n";
        let out = apply_block(with_block, None);
        assert!(!out.contains(LEARNED_BLOCK_BEGIN));
        assert!(out.contains("human"));
    }

    #[test]
    fn write_claude_md_round_trips_and_is_idempotent() {
        let (db, dir) = temp_db();
        let path = dir.path().join("CLAUDE.md");
        std::fs::write(&path, "# Project\n\nhand-written notes\n").unwrap();

        let directives = vec![LearnedDirective {
            signature: "deadbeefdeadbeef".into(),
            directive: "Last 2 times you hit \"X\", the fix was: Y".into(),
            frequency: 2,
            last_seen: Utc::now(),
            impact: 2.0,
        }];

        assert!(write_claude_md(&path, &directives).unwrap(), "first write changes file");
        assert!(
            !write_claude_md(&path, &directives).unwrap(),
            "identical second write is a no-op"
        );

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("hand-written notes"));
        assert!(content.contains("the fix was: Y"));
        assert_eq!(content.matches(LEARNED_BLOCK_BEGIN).count(), 1);

        // Empty directive set removes the block.
        assert!(write_claude_md(&path, &[]).unwrap(), "removal changes file");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(!content.contains(LEARNED_BLOCK_BEGIN));
        assert!(content.contains("hand-written notes"));
    }

    #[test]
    fn write_claude_md_skips_empty_nonexistent_file() {
        let (_db, dir) = temp_db();
        let path = dir.path().join("CLAUDE.md");
        assert!(!write_claude_md(&path, &[]).unwrap(), "no directives, no file → no-op");
        assert!(!path.exists(), "should not create an empty file");
    }

    #[test]
    fn persist_rules_upserts_and_prunes() {
        let (db, _dir) = temp_db();

        let d1 = LearnedDirective {
            signature: "1111111111111111".into(),
            directive: "Last 2 times you hit \"A\", the fix was: a".into(),
            frequency: 2,
            last_seen: Utc::now(),
            impact: 2.0,
        };
        let d2 = LearnedDirective {
            signature: "2222222222222222".into(),
            directive: "Last 3 times you hit \"B\", the fix was: b".into(),
            frequency: 3,
            last_seen: Utc::now(),
            impact: 3.0,
        };

        assert_eq!(persist_rules(&db, &[d1.clone(), d2.clone()]).unwrap(), 2);
        let learned: Vec<_> = db
            .list(TABLE_BOOT_RULES)
            .unwrap()
            .into_iter()
            .filter(|r| r.data.get("source").and_then(|v| v.as_str()) == Some(LEARNED_RULE_SOURCE))
            .collect();
        assert_eq!(learned.len(), 2);
        assert!(learned
            .iter()
            .all(|r| r.data.get("_importance_pinned").and_then(|v| v.as_bool()) == Some(true)));

        // Re-running with only d1 prunes d2 and does not duplicate d1.
        assert_eq!(persist_rules(&db, &[d1.clone()]).unwrap(), 1);
        let learned: Vec<_> = db
            .list(TABLE_BOOT_RULES)
            .unwrap()
            .into_iter()
            .filter(|r| r.data.get("source").and_then(|v| v.as_str()) == Some(LEARNED_RULE_SOURCE))
            .collect();
        assert_eq!(learned.len(), 1);
        assert_eq!(
            learned[0].data.get("_learned_sig").and_then(|v| v.as_str()),
            Some("1111111111111111")
        );
    }

    #[test]
    fn persist_rules_leaves_other_rules_untouched() {
        let (db, _dir) = temp_db();
        db.insert("rules", json!({ "rule": "human rule", "source": "user" }))
            .unwrap();

        let d = LearnedDirective {
            signature: "abcabcabcabcabca".into(),
            directive: "Last 2 times you hit \"X\", the fix was: Y".into(),
            frequency: 2,
            last_seen: Utc::now(),
            impact: 2.0,
        };
        persist_rules(&db, &[d]).unwrap();

        // Pruning to empty must not touch the user rule.
        persist_rules(&db, &[]).unwrap();
        let rules = db.list("rules").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].data.get("source").and_then(|v| v.as_str()), Some("user"));
    }
}
