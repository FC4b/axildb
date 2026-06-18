//! Reverse index over `data.code_refs[]` arrays.
//!
//! Memory records (decisions, errors, sessions, …) can carry a
//! `code_refs` array of pointers back to code proxies. This index stores
//! one row per `(anchor_key, record_id)` pair so recall can resolve all
//! memories attached to a given proxy with a single `db.list` call
//! instead of walking every non-internal table.
//!
//! `Axil::insert` and `Axil::update` automatically sync this index after
//! the storage write commits, so any caller (CLI, MCP server, embedded
//! library) maintains it transparently. Callers that bypass the index
//! still get the old O(records × refs) fallback walk in
//! `axil_indexer::recall::related_memories_for_proxies`.
//!
//! Anchor keys are flat strings derived per code_ref entry:
//!
//! | code_ref shape                  | emitted key                 |
//! |---------------------------------|-----------------------------|
//! | `{ proxy_id }`                  | `proxy:<id>`                |
//! | `{ canonical_id }`              | `canonical:<id>`            |
//! | `{ path, symbol }`              | `path_symbol:<p>::<s>`      |
//! | `{ path }` (no symbol)          | `path:<p>`                  |
//!
//! Multiple keys may be emitted per code_ref. The index dedupes nothing —
//! if two pointers in the same memory share a key, two rows land. That
//! keeps the writer trivial and the reader's `HashSet`-of-record-ids
//! collapses duplicates.

use serde_json::{json, Value};

use crate::{Axil, Record, RecordId, Result};

/// Internal table name. Hidden from search indexes via the `_` prefix.
pub const TABLE_CODE_REFS_INDEX: &str = "_idx_code_refs";

/// Build the flat anchor keys that a single `code_refs[]` entry
/// generates. The same set is consulted at recall time via
/// `proxy_match_keys` so the writer and reader can never drift on the
/// key format.
pub fn anchor_keys(code_ref: &Value) -> Vec<String> {
    proxy_match_keys(
        code_ref.get("proxy_id").and_then(|v| v.as_str()),
        code_ref.get("canonical_id").and_then(|v| v.as_str()),
        code_ref.get("path").and_then(|v| v.as_str()),
        code_ref.get("symbol").and_then(|v| v.as_str()),
    )
}

/// Build the same anchor keys from primitive fields. Used by recall to
/// turn a proxy hit into the set of `_idx_code_refs.key` values it
/// would have produced if a memory pointed at it.
pub fn proxy_match_keys(
    proxy_id: Option<&str>,
    canonical_id: Option<&str>,
    path: Option<&str>,
    symbol: Option<&str>,
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    if let Some(s) = proxy_id {
        keys.push(format!("proxy:{s}"));
    }
    if let Some(s) = canonical_id {
        keys.push(format!("canonical:{s}"));
    }
    match (path, symbol) {
        (Some(p), Some(s)) => keys.push(format!("path_symbol:{p}::{s}")),
        (Some(p), None) => keys.push(format!("path:{p}")),
        _ => {}
    }
    keys
}

/// Whether the calling write is creating a new record or modifying one.
/// First-time inserts can never have stale index rows, so they skip the
/// drop pass and avoid an O(N_index) scan on every memory write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SyncMode {
    Insert,
    Update,
}

/// Metadata field on a parent memory record listing the `_idx_code_refs`
/// row IDs it owns. Lets `sync_for_record` find prior rows in O(refs)
/// instead of scanning the full `_idx_code_refs` table on every update.
const META_INDEX_IDS: &str = "_code_ref_index_ids";

/// Sync the reverse index for `record`. Called from `Axil::insert` and
/// `Axil::update` after the storage write commits.
///
/// Best-effort: failures don't roll back the underlying record. The
/// `Insert` path skips listing `_idx_code_refs` when the new record
/// carries no `code_refs` — i.e. the common case for tables that never
/// carry pointers — keeping the per-write overhead at zero.
///
/// Once anchor rows are emitted, the parent's `_code_ref_index_ids`
/// metadata is back-written via `storage_update_raw` so the next
/// `sync_for_record` (e.g. on update) can drop existing rows by id —
/// no full-table scan, no list-and-filter.
pub(crate) fn sync_for_record(db: &Axil, record: &Record, mode: SyncMode) -> Result<()> {
    if record.table.starts_with('_') {
        return Ok(());
    }
    let refs = record.data.get("code_refs").and_then(|v| v.as_array());

    if mode == SyncMode::Update {
        drop_existing_for_parent(db, record)?;
    }

    let Some(refs) = refs else {
        clear_index_ids_metadata(db, record);
        return Ok(());
    };
    if refs.is_empty() {
        clear_index_ids_metadata(db, record);
        return Ok(());
    }

    let target_id = record.id.to_string();
    let now = chrono::Utc::now();
    let mut batch: Vec<(Value, chrono::DateTime<chrono::Utc>)> = Vec::new();
    for r in refs {
        for key in anchor_keys(r) {
            batch.push((
                json!({
                    "key": key,
                    "record_id": target_id,
                    "src_table": record.table,
                }),
                now,
            ));
        }
    }
    let emitted_ids: Vec<String> = if batch.is_empty() {
        Vec::new()
    } else {
        db.insert_batch_raw(TABLE_CODE_REFS_INDEX, batch)?
            .into_iter()
            .map(|r| r.id.to_string())
            .collect()
    };

    // Stash the index row IDs on the parent so a future update or delete
    // can drop them without scanning `_idx_code_refs`. Uses
    // `storage_update_raw` to bypass the hook chain — calling
    // `Axil::update` here would recurse.
    if !emitted_ids.is_empty() {
        let mut new_data = record.data.clone();
        if let Some(obj) = new_data.as_object_mut() {
            obj.insert(META_INDEX_IDS.to_string(), json!(emitted_ids));
            let _ = db.storage_update_raw(&record.id, new_data);
        }
    }

    Ok(())
}

/// Drop existing reverse-index rows for a parent memory.
///
/// Fast path: read `_code_ref_index_ids` from the parent's metadata and
/// delete each row by id. Falls back to the old full-table scan when
/// the parent lacks the metadata (e.g. a record written before the
/// metadata trick existed, or one not produced by this codepath).
fn drop_existing_for_parent(db: &Axil, record: &Record) -> Result<()> {
    if let Some(arr) = record.data.get(META_INDEX_IDS).and_then(|v| v.as_array()) {
        for v in arr {
            if let Some(rid_str) = v.as_str() {
                if let Ok(rid) = RecordId::from_string(rid_str) {
                    let _ = db.delete(&rid);
                }
            }
        }
        return Ok(());
    }
    drop_for_record(db, &record.id)
}

/// Remove the `_code_ref_index_ids` metadata when a record's `code_refs`
/// is cleared. Keeps the parent's data consistent.
fn clear_index_ids_metadata(db: &Axil, record: &Record) {
    if !record.data.get(META_INDEX_IDS).is_some() {
        return;
    }
    let mut new_data = record.data.clone();
    if let Some(obj) = new_data.as_object_mut() {
        obj.remove(META_INDEX_IDS);
        let _ = db.storage_update_raw(&record.id, new_data);
    }
}

/// Drop every reverse-index row pointing at `record_id`. Used by the
/// delete path and as the fallback in `drop_existing_for_parent` when a
/// memory was written before the metadata trick existed.
pub(crate) fn drop_for_record(db: &Axil, record_id: &RecordId) -> Result<()> {
    let target_id = record_id.to_string();
    let existing = match db.list(TABLE_CODE_REFS_INDEX) {
        Ok(rows) => rows,
        Err(_) => return Ok(()),
    };
    for row in existing {
        if row.data.get("record_id").and_then(|v| v.as_str()) == Some(target_id.as_str()) {
            let _ = db.delete(&row.id);
        }
    }
    Ok(())
}
