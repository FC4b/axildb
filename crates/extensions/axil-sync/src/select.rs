//! Distillate selection: turn a local `Axil`'s promoted memory into a batch of
//! protocol [`Op`]s to push to Atlas.
//!
//! What crosses the boundary is the **distillate**, never the raw episodic
//! trail. Selection reuses Axil's own signals: it reads the promoted
//! convention tables and each record's decayed importance, computed at
//! selection time from `_importance`, the record's age and the table's
//! configured half-life — the same function recall and tiering use. The embedding,
//! when requested, is computed locally by the caller's `Axil` — the server
//! never embeds.
//!
//! Follow-ups (out of this first cut): a `_sync_*` cursor table so selection is
//! incremental instead of a full scan; `canonical.publish` ops (today
//! `canonical_id` rides on the upsert payload when present); graph-edge ops.

use axil_atlas_proto::{Lww, Op, OpKind, Tier};
use axil_core::{Axil, DecayConfig, Record};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::SyncError;

/// Convention tables whose contents are promoted knowledge — the distillate.
/// Mirrors the tables the brain/boot pipeline treats as durable knowledge.
pub const DEFAULT_PROMOTED_TABLES: &[&str] =
    &["decisions", "rules", "errors", "patterns", "_beliefs"];

/// Controls what [`select_distillate`] emits.
#[derive(Debug, Clone)]
pub struct SelectOpts {
    /// The workspace member id this device syncs as.
    pub member: String,
    /// Tables to scan for promoted records.
    pub tables: Vec<String>,
    /// Drop records whose effective importance is below this. Defaults to
    /// `axil_core`'s archive threshold (0.1), i.e. "everything not archived".
    pub min_importance: f32,
    /// Embed each record's content locally (requires the `Axil` to have an
    /// embedder). The server never embeds.
    pub embed: bool,
    /// Safety cap on records read per table.
    pub max_per_table: usize,
}

impl Default for SelectOpts {
    fn default() -> Self {
        Self {
            member: "local".into(),
            tables: DEFAULT_PROMOTED_TABLES.iter().map(|s| s.to_string()).collect(),
            min_importance: axil_core::importance::ARCHIVE_THRESHOLD,
            embed: true,
            max_per_table: 100_000,
        }
    }
}

/// Scan the promoted tables of `db` and emit one `record.upsert` [`Op`] per
/// selected record.
///
/// Per-table half-lives come from the `axil.toml` nearest the database — the
/// same resolution `axil decay` and tiering use, so `[decay] tables.<t>` and
/// `[lifecycle.tables.<t>] decay = false` apply here too.
pub fn select_distillate(db: &Axil, opts: &SelectOpts) -> Result<Vec<Op>, SyncError> {
    let decay = db
        .path()
        .parent()
        .and_then(|dir| axil_core::load_config_from(dir).ok())
        .map(|c| c.decay)
        .unwrap_or_default();
    select_distillate_with_decay(db, opts, &decay)
}

/// [`select_distillate`] with an explicit per-table half-life configuration.
pub fn select_distillate_with_decay(
    db: &Axil,
    opts: &SelectOpts,
    decay: &DecayConfig,
) -> Result<Vec<Op>, SyncError> {
    let now_secs = unix_now_secs();
    let mut ops = Vec::new();
    for table in &opts.tables {
        // Never sync the private episodic trail, even if it was passed in.
        if is_episodic(table) {
            continue;
        }
        let half_life = decay.half_life_for(table);
        let recs = db.query().table(table).limit(opts.max_per_table).exec()?;
        for rec in recs {
            let importance = record_importance(&rec, half_life, now_secs);
            if importance < opts.min_importance {
                continue;
            }
            ops.push(record_to_op(
                db,
                &rec,
                &opts.member,
                opts.embed,
                importance,
            )?);
        }
    }
    Ok(ops)
}

/// Build a `record.upsert` op from a promoted record.
fn record_to_op(
    db: &Axil,
    rec: &Record,
    member: &str,
    embed: bool,
    importance: f32,
) -> Result<Op, SyncError> {
    // Prefer an explicit `content` field; fall back to the whole record body.
    // Strip Axil-internal (`_`-prefixed) fields first: the body carries volatile
    // local state — `_importance`, `_effective_importance`, `_access_count` —
    // that decay/access sweeps rewrite in place (`axil_core::importance`), so
    // hashing them would churn `op_id` on every sweep and leak that state across
    // the sync boundary. Both the content hash and the pushed payload use the
    // stripped body.
    let content = strip_internal_fields(
        &rec.data
            .get("content")
            .cloned()
            .unwrap_or_else(|| rec.data.clone()),
    );
    let content_hash = content_hash(&content)?;

    // op_id is stable while the content is unchanged (re-syncs dedup) and
    // changes when the content changes (a genuinely new op, never a reused id
    // with a different hash — which the server would reject).
    let op_id = format!(
        "{}~{}",
        rec.id.as_str(),
        &content_hash[..16.min(content_hash.len())]
    );

    let canonical_id = rec
        .data
        .get("canonical_id")
        .and_then(Value::as_str)
        .map(str::to_string);

    let embedding = if embed && db.has_embedder() {
        let text = render_text(&content);
        if text.is_empty() {
            None
        } else {
            Some(db.embed_query(&text)?)
        }
    } else {
        None
    };

    let payload = serde_json::json!({
        "source_record_id": rec.id.as_str(),
        "kind": rec.table,
        "canonical_id": canonical_id,
        "content": content,
        "importance": importance,
        "model_id": Value::Null,
    });

    Ok(Op {
        op_id,
        kind: OpKind::RecordUpsert,
        table: rec.table.clone(),
        member: member.to_string(),
        tier: Tier::Distillate,
        content_hash,
        lww: Some(Lww {
            updated_at: rec.updated_at.timestamp_millis(),
            origin: member.to_string(),
        }),
        payload,
        embedding,
    })
}

/// Effective importance of `rec` now.
///
/// Decay is lazy: nothing rewrites `_effective_importance` on a schedule (only
/// a manual `axil decay` does), so the stored field can be arbitrarily stale.
/// Compute it instead from `_importance`, the record's age and the table's
/// half-life with core's own decay function. A record that was never scored
/// (no `_importance`, as in `_`-prefixed tables) has no decay signal: it keeps
/// a stored `_effective_importance` if one exists, else 1.0 — always synced.
fn record_importance(rec: &Record, half_life: f64, now_secs: i64) -> f32 {
    let data = &rec.data;
    if data.get("_importance").is_none() && !axil_core::importance::is_pinned(data) {
        return data
            .get("_effective_importance")
            .and_then(Value::as_f64)
            .map(|f| f as f32)
            .unwrap_or(1.0);
    }
    let age_days = (now_secs - rec.created_at.timestamp()).max(0) as f64 / 86_400.0;
    axil_core::importance::effective_importance(data, age_days, half_life)
}

/// Seconds since the Unix epoch (0 if the clock is before it).
fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Drop Axil-internal (`_`-prefixed) top-level fields from a content value.
///
/// Mirrors the canonicalization in `axil_core::portable` (source of truth for
/// the field set), which drops the same `_`-prefixed top-level keys before its
/// export/import content hash — those fields are Axil-internal and drift on
/// their own (importance decays, access counts climb, tiers change), so keeping
/// them would make the "same content" hash differently over time and across
/// machines. Non-object values pass through unchanged.
fn strip_internal_fields(content: &Value) -> Value {
    match content {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| !k.starts_with('_'))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Stable SHA-256 hex of a JSON value.
fn content_hash(v: &Value) -> Result<String, SyncError> {
    let bytes = serde_json::to_vec(v)?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(hex(h.finalize().as_slice()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// A best-effort plain-text rendering of a content value, for local embedding.
fn render_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Object(m) => m
            .values()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Tables whose contents are the private episodic trail — never synced.
fn is_episodic(table: &str) -> bool {
    let t = table.to_ascii_lowercase();
    t.contains("session")
        || t.contains("episode")
        || t.contains("file_touch")
        || t.contains("edit")
        || t == "_working"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn db(tmp: &TempDir) -> Axil {
        Axil::open(tmp.path().join("m.axil")).build().unwrap()
    }

    fn opts(tables: &[&str]) -> SelectOpts {
        SelectOpts {
            member: "mem_a".into(),
            tables: tables.iter().map(|s| s.to_string()).collect(),
            min_importance: 0.0,
            embed: false,
            max_per_table: 1000,
        }
    }

    #[test]
    fn selects_promoted_records_as_upsert_ops() {
        let tmp = TempDir::new().unwrap();
        let db = db(&tmp);
        db.insert("decisions", json!({"summary": "chose A over B"})).unwrap();
        db.insert("decisions", json!({"summary": "chose C over D"})).unwrap();

        let ops = select_distillate(&db, &opts(&["decisions"])).unwrap();
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().all(|o| o.kind == OpKind::RecordUpsert));
        assert!(ops.iter().all(|o| o.tier == Tier::Distillate));
        assert!(ops.iter().all(|o| o.embedding.is_none()));
    }

    #[test]
    fn never_syncs_episodic_tables() {
        let tmp = TempDir::new().unwrap();
        let db = db(&tmp);
        db.insert("_sessions", json!({"summary": "worked on auth"})).unwrap();
        // Even if the caller passes an episodic table, it contributes nothing.
        let ops = select_distillate(&db, &opts(&["_sessions", "sessions", "file_touch"])).unwrap();
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn op_id_is_stable_until_content_changes() {
        let tmp = TempDir::new().unwrap();
        let db = db(&tmp);
        let rec = db.insert("decisions", json!({"summary": "v1"})).unwrap();

        let first = select_distillate(&db, &opts(&["decisions"])).unwrap();
        let second = select_distillate(&db, &opts(&["decisions"])).unwrap();
        assert_eq!(first[0].op_id, second[0].op_id, "unchanged record → stable op_id");

        db.update(&rec.id, json!({"summary": "v2"})).unwrap();
        let third = select_distillate(&db, &opts(&["decisions"])).unwrap();
        assert_ne!(first[0].op_id, third[0].op_id, "changed content → new op_id");
        assert_ne!(first[0].content_hash, third[0].content_hash);
    }

    #[test]
    fn op_id_stable_across_volatile_field_drift() {
        // A decay/access sweep rewrites `_effective_importance`/`_access_count`
        // in a record's body. That internal churn must not change the op_id
        // (which would re-push the record as "new"), and those volatile fields
        // must never ride inside the pushed payload's content.
        let tmp = TempDir::new().unwrap();
        let db = db(&tmp);
        let rec = db
            .insert("decisions", json!({"summary": "chose A", "reason": "faster"}))
            .unwrap();

        let before = record_to_op(&db, &rec, "mem_a", false, 1.0).unwrap();

        // Simulate a sweep mutating the internal fields in place.
        let mut mutated = rec.clone();
        let obj = mutated.data.as_object_mut().unwrap();
        obj.insert("_effective_importance".into(), json!(0.137));
        obj.insert("_access_count".into(), json!(9));
        obj.insert("_importance".into(), json!(0.42));
        let after = record_to_op(&db, &mutated, "mem_a", false, 1.0).unwrap();

        assert_eq!(
            before.op_id, after.op_id,
            "volatile internal-field drift must not churn the op_id"
        );
        assert_eq!(before.content_hash, after.content_hash);

        // The pushed payload's content excludes the volatile internal fields.
        let content = &after.payload["content"];
        assert!(content.get("_effective_importance").is_none());
        assert!(content.get("_access_count").is_none());
        assert!(content.get("_importance").is_none());
        assert_eq!(content["summary"], "chose A");
        assert_eq!(content["reason"], "faster");
    }

    #[test]
    fn min_importance_filters_records() {
        let now = unix_now_secs();
        let half_life = axil_core::importance::DEFAULT_HALF_LIFE_DAYS;
        let rec = |data| Record::new("decisions", data);
        // A fresh scored record: effective == base.
        let fresh = record_importance(&rec(json!({"_importance": 0.7})), half_life, now);
        assert!((fresh - 0.7).abs() < 1e-3, "{fresh}");
        // Unscored records have no decay signal and are always synced.
        assert_eq!(record_importance(&rec(json!({})), half_life, now), 1.0);
        assert_eq!(
            record_importance(&rec(json!({"_effective_importance": 0.3})), half_life, now),
            0.3
        );
    }

    /// A record inserted long ago with high base importance and a stale stored
    /// `_effective_importance`.
    fn old_decision(db: &Axil) -> Record {
        let mut rec = Record::new(
            "decisions",
            json!({"summary": "ancient choice", "_importance": 0.9, "_effective_importance": 0.9}),
        );
        rec.created_at = "2020-01-01T00:00:00Z".parse().unwrap();
        rec.updated_at = rec.created_at;
        db.insert_preserving(rec).unwrap()
    }

    #[test]
    fn selection_decays_importance_instead_of_trusting_stored_values() {
        // Years old at a 90-day half-life: far below the archive threshold, so
        // the default options must not push it — even though its stored
        // `_importance`/`_effective_importance` still say 0.9.
        let tmp = TempDir::new().unwrap();
        let db = db(&tmp);
        old_decision(&db);
        db.insert(
            "decisions",
            json!({"summary": "fresh choice", "_importance": 0.9}),
        )
        .unwrap();

        let ops = select_distillate(
            &db,
            &SelectOpts {
                embed: false,
                ..SelectOpts::default()
            },
        )
        .unwrap();
        assert_eq!(ops.len(), 1, "only the fresh decision survives decay");
        assert_eq!(ops[0].payload["content"]["summary"], "fresh choice");
    }

    #[test]
    fn selection_honors_the_configured_half_life() {
        // `[lifecycle.tables.decisions] decay = false` next to the database
        // stops decay, exactly as it does for recall and tiering.
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("axil.toml"),
            "[lifecycle.tables.decisions]\ndecay = false\n",
        )
        .unwrap();
        let db = db(&tmp);
        old_decision(&db);

        let ops = select_distillate(
            &db,
            &SelectOpts {
                embed: false,
                ..SelectOpts::default()
            },
        )
        .unwrap();
        assert_eq!(ops.len(), 1);
        let importance = ops[0].payload["importance"].as_f64().unwrap();
        assert!((importance - 0.9).abs() < 1e-3, "{importance}");
    }

    #[test]
    fn is_episodic_matches_the_private_trail() {
        assert!(is_episodic("_sessions"));
        assert!(is_episodic("sessions"));
        assert!(is_episodic("file_touch"));
        assert!(is_episodic("_working"));
        assert!(!is_episodic("decisions"));
        assert!(!is_episodic("_beliefs"));
    }
}
