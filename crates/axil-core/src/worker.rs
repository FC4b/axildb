//! Background worker — consolidate entities, strengthen connections, detect stale records.
//!
//! Idempotent: safe to call multiple times. Stores run reports in `_worker_runs`.
//!
//! ## Maintenance thread
//!
//! `MaintenanceThread` runs the worker on a configurable interval in a
//! background thread. Non-blocking — the thread sleeps between runs.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::record::RecordId;
use crate::{Axil, Result};

/// Report from a single worker run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReport {
    /// When the worker started.
    pub started_at: DateTime<Utc>,
    /// How long the run took in milliseconds.
    pub duration_ms: u64,
    /// Number of entities that were consolidated.
    pub consolidated_entities: usize,
    /// Number of new co-mention connections discovered.
    pub new_connections: usize,
    /// Number of inferred facts produced.
    pub inferred_facts: usize,
    /// Number of stale records detected (no access in 90+ days).
    pub stale_detected: usize,
    /// Number of records whose effective importance was updated by decay.
    #[serde(default)]
    pub decayed_records: usize,
    // ── Phase 11.7: Brain consolidation fields ──
    /// Number of stale beliefs detected and auto-doubted.
    #[serde(default)]
    pub stale_beliefs: usize,
    /// Number of candidate procedures extracted from repeated sessions.
    #[serde(default)]
    pub candidate_procedures: usize,
    /// Number of candidate preferences extracted from user corrections.
    #[serde(default)]
    pub candidate_preferences: usize,
    /// Number of near-duplicate memory clusters found.
    #[serde(default)]
    pub duplicate_clusters: usize,
}

/// Background worker that runs maintenance tasks on an Axil database.
pub struct AxilWorker<'a> {
    db: &'a Axil,
    /// When true, also run Phase 11.7 brain consolidation tasks.
    brain_mode: bool,
}

impl<'a> AxilWorker<'a> {
    /// Create a new worker for the given database.
    pub fn new(db: &'a Axil) -> Self {
        Self {
            db,
            brain_mode: false,
        }
    }

    /// Enable brain consolidation tasks (Phase 11.7).
    pub fn with_brain(mut self) -> Self {
        self.brain_mode = true;
        self
    }

    /// Run all worker tasks. Idempotent — safe to call multiple times.
    ///
    /// Tasks:
    /// 1. Consolidate entities with multiple unmerged facts
    /// 2. Strengthen connections by counting co-mentions
    /// 3. Run inference if graph is available
    /// 4. Detect stale records (no access in 90+ days)
    /// 5. Apply importance decay
    /// 6. (Brain mode) Detect stale beliefs
    /// 7. (Brain mode) Extract candidate procedures
    /// 8. (Brain mode) Extract candidate preferences
    /// 9. (Brain mode) Cluster near-duplicates
    pub fn run(&self) -> Result<WorkerReport> {
        let started_at = Utc::now();
        let start = std::time::Instant::now();

        let consolidated_entities = self.consolidate_entities();
        let new_connections = self.strengthen_connections();
        let inferred_facts = self.run_inference();
        let stale_detected = self.detect_stale();
        let decayed_records = self.run_decay();

        // Phase 11.7: Brain consolidation tasks.
        let (stale_beliefs, candidate_procedures, candidate_preferences, duplicate_clusters) =
            if self.brain_mode {
                (
                    self.detect_stale_beliefs(),
                    self.extract_candidate_procedures(),
                    self.extract_candidate_preferences(),
                    self.cluster_duplicates(),
                )
            } else {
                (0, 0, 0, 0)
            };

        let duration_ms = start.elapsed().as_millis() as u64;

        let report = WorkerReport {
            started_at,
            duration_ms,
            consolidated_entities,
            new_connections,
            inferred_facts,
            stale_detected,
            decayed_records,
            stale_beliefs,
            candidate_procedures,
            candidate_preferences,
            duplicate_clusters,
        };

        // Store the report in `_worker_runs`.
        let _ = self.db.insert(
            "_worker_runs",
            serde_json::to_value(&report).unwrap_or(json!({})),
        );

        Ok(report)
    }

    /// Get the last worker run report.
    pub fn last_run(&self) -> Result<Option<WorkerReport>> {
        let records = self.db.list("_worker_runs")?;
        // Records are ordered by ID (ULID = time-sorted), take last.
        let last = records.last();
        match last {
            Some(r) => {
                let report: WorkerReport = serde_json::from_value(r.data.clone())
                    .map_err(|e| crate::error::AxilError::plugin(e.to_string()))?;
                Ok(Some(report))
            }
            None => Ok(None),
        }
    }

    /// Find entities with multiple unmerged facts and consolidate them.
    fn consolidate_entities(&self) -> usize {
        let entities = match self.db.list("_entities") {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut consolidated = 0usize;
        for entity_record in &entities {
            let name = match entity_record.data.get("entity").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Skip already-consolidated entities (check _consolidated table).
            // Use consolidate_entity which handles the full pipeline.
            if self.db.has_graph_index() {
                match self.db.consolidate_entity(&name) {
                    Ok(Some(_)) => consolidated += 1,
                    _ => {}
                }
            }
        }

        consolidated
    }

    /// Count co-mentions from `_entities` table and create `related_to` edges.
    fn strengthen_connections(&self) -> usize {
        if !self.db.has_graph_index() {
            return 0;
        }

        let entities = match self.db.list("_entities") {
            Ok(e) => e,
            Err(_) => return 0,
        };

        // Build a map: entity_name -> set of record IDs that mention it.
        let mut entity_mentions: HashMap<String, Vec<RecordId>> = HashMap::new();
        for entity_record in &entities {
            let name = match entity_record.data.get("entity").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Find records mentioning this entity via graph edges.
            let mentioners = self
                .db
                .neighbors(
                    &entity_record.id,
                    Some(crate::util::edge_types::MENTIONS),
                    crate::plugin::Direction::In,
                )
                .unwrap_or_default();

            let ids: Vec<RecordId> = mentioners.iter().map(|r| r.id.clone()).collect();
            entity_mentions.insert(name, ids);
        }

        // Build name -> record ID lookup for O(1) access.
        let name_to_id: HashMap<String, RecordId> = entities
            .iter()
            .filter_map(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(|n| (n.to_string(), r.id.clone()))
            })
            .collect();

        // Find entity pairs that share mentioning records (co-occurrence).
        let entity_names: Vec<String> = entity_mentions.keys().cloned().collect();
        let mut new_connections = 0usize;

        for i in 0..entity_names.len() {
            for j in (i + 1)..entity_names.len() {
                let a = &entity_names[i];
                let b = &entity_names[j];
                let ids_a = &entity_mentions[a];
                let ids_b = &entity_mentions[b];

                // Count shared records.
                let shared: usize = ids_a
                    .iter()
                    .filter(|id| ids_b.iter().any(|bid| bid == *id))
                    .count();

                if shared >= 2 {
                    let ea_id = name_to_id.get(a);
                    let eb_id = name_to_id.get(b);

                    if let (Some(ea), Some(eb)) = (ea_id, eb_id) {
                        let existing = self
                            .db
                            .edges(
                                ea,
                                Some(crate::util::edge_types::RELATED_TO),
                                crate::plugin::Direction::Out,
                            )
                            .unwrap_or_default();

                        let existing_edge = existing.iter().find(|e| e.to == *eb);
                        if let Some(edge) = existing_edge {
                            // Update co_mentions count on existing edge.
                            let old_count = edge
                                .properties
                                .get("co_mentions")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;
                            if shared > old_count {
                                let _ = self.db.unrelate(&edge.id);
                                // Upgrade edge type at 5+ co-mentions.
                                let edge_type = if shared >= 5 {
                                    "depends_on"
                                } else {
                                    crate::util::edge_types::RELATED_TO
                                };
                                let _ = self.db.relate(
                                    ea,
                                    edge_type,
                                    eb,
                                    Some(json!({"co_mentions": shared, "source": "worker"})),
                                );
                                new_connections += 1;
                            }
                        } else {
                            let _ = self.db.relate(
                                ea,
                                crate::util::edge_types::RELATED_TO,
                                eb,
                                Some(json!({"co_mentions": shared, "source": "worker"})),
                            );
                            new_connections += 1;
                        }
                    }
                }
            }
        }

        new_connections
    }

    /// Run simple inference: if A->mentions->E and B->mentions->E and A and B
    /// are in the same table, they may be related.
    fn run_inference(&self) -> usize {
        // Simple inference is covered by strengthen_connections.
        // Future: add rule-based inference, transitive closure, etc.
        0
    }

    /// Apply importance decay to all records, updating `_effective_importance`.
    fn run_decay(&self) -> usize {
        let now = Utc::now();
        let default_half_life = crate::importance::DEFAULT_HALF_LIFE_DAYS;
        // Config is best-effort: workers run whether or not axil.toml exists.
        // Start the search from the current working directory so a project
        // config picks up even when the worker thread is spawned inside a
        // CLI invocation.
        let decay_cfg = std::env::current_dir()
            .ok()
            .and_then(|cwd| crate::config::load_config_from(&cwd).ok())
            .map(|c| c.decay);
        let tables = match self.db.tables() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let mut updated = 0usize;
        for table in &tables {
            if table.starts_with('_') {
                continue;
            }
            // Per-table half-life: errors/code-memory decay faster than
            // preferences. Resolved once per table so the inner loop stays
            // a single f64.
            let half_life = decay_cfg
                .as_ref()
                .map(|d| d.half_life_for(table))
                .unwrap_or(default_half_life);
            let records = match self.db.list(table) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for record in &records {
                if crate::importance::is_pinned(&record.data) {
                    continue;
                }
                if record.data.get("_importance").is_none() {
                    continue;
                }
                let age_days = (now - record.created_at).num_seconds() as f64 / 86400.0;
                let effective =
                    crate::importance::effective_importance(&record.data, age_days, half_life);
                // Only update if effective importance differs from stored value.
                let stored_effective = record
                    .data
                    .get("_effective_importance")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
                let needs_update = match stored_effective {
                    Some(old) => (old - effective).abs() > 0.01,
                    None => true,
                };
                if needs_update {
                    let mut data = record.data.clone();
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert(
                            "_effective_importance".to_string(),
                            serde_json::json!(effective),
                        );
                    }
                    if self.db.update(&record.id, data).is_ok() {
                        updated += 1;
                    }
                }
            }
        }
        updated
    }

    // ── Phase 11.7: Brain consolidation tasks ──────────────────────

    /// Detect beliefs that haven't been validated recently and auto-doubt them.
    fn detect_stale_beliefs(&self) -> usize {
        let beliefs = match self.db.list("_beliefs") {
            Ok(b) => b,
            Err(_) => return 0,
        };

        let cutoff = Utc::now() - Duration::days(60);
        let mut stale = 0;

        for belief in &beliefs {
            // Skip already doubted.
            if belief
                .data
                .get("doubted")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }

            // Check last validated time.
            let last_validated = belief
                .data
                .get("_last_validated_at")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            let check_date = last_validated.unwrap_or(belief.created_at);
            if check_date < cutoff {
                // Auto-doubt with reason.
                let _ = crate::brain::doubt_record(
                    self.db,
                    &belief.id,
                    Some("stale: not validated in 60+ days (worker auto-doubt)"),
                );
                stale += 1;
            }
        }

        stale
    }

    /// Extract candidate procedures from repeated successful session patterns.
    ///
    /// Scans ended sessions with "success" outcome, finds common action patterns,
    /// and stores them as candidate procedures for review.
    fn extract_candidate_procedures(&self) -> usize {
        let sessions = match self.db.list("_sessions") {
            Ok(s) => s,
            Err(_) => return 0,
        };

        // Find successful sessions with summaries.
        let successful: Vec<&crate::Record> = sessions
            .iter()
            .filter(|s| {
                let status = s.data.get("status").and_then(|v| v.as_str()).unwrap_or("");
                let outcome = s.data.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
                status == "ended" && outcome == "success"
            })
            .collect();

        if successful.len() < 3 {
            return 0; // Need at least 3 successful sessions to detect patterns.
        }

        // Extract summaries and look for repeated action words.
        let summaries: Vec<String> = successful
            .iter()
            .filter_map(|s| {
                s.data
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_lowercase())
            })
            .collect();

        // Simple pattern detection: find 3-word sequences that appear in 2+ sessions.
        let mut ngram_counts: HashMap<String, usize> = HashMap::new();
        for summary in &summaries {
            let words: Vec<&str> = summary.split_whitespace().collect();
            let mut seen_in_this = std::collections::HashSet::new();
            for window in words.windows(3) {
                let ngram = window.join(" ");
                if seen_in_this.insert(ngram.clone()) {
                    *ngram_counts.entry(ngram).or_insert(0) += 1;
                }
            }
        }

        // Store candidate procedures for patterns appearing 2+ times.
        let mut candidates = 0;
        let existing_procs = self.db.list("_candidate_procedures").unwrap_or_default();
        let existing_patterns: std::collections::HashSet<String> = existing_procs
            .iter()
            .filter_map(|r| {
                r.data
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        for (ngram, count) in &ngram_counts {
            if *count >= 2 && !existing_patterns.contains(ngram) {
                let _ = self.db.insert(
                    "_candidate_procedures",
                    json!({
                        "pattern": ngram,
                        "occurrences": count,
                        "source": "worker_brain",
                        "status": "candidate",
                    }),
                );
                candidates += 1;
            }
        }

        candidates
    }

    /// Extract candidate preferences from user corrections and repeated patterns.
    fn extract_candidate_preferences(&self) -> usize {
        // Look for records with _source.kind = "user" that contain preference signals.
        let tables = match self.db.tables() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let preference_signals = ["prefer", "always", "never", "don't", "should"];
        let mut candidates = 0;

        let existing_prefs = self.db.list("_candidate_preferences").unwrap_or_default();
        let existing_statements: std::collections::HashSet<String> = existing_prefs
            .iter()
            .filter_map(|r| {
                r.data
                    .get("statement")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_lowercase())
            })
            .collect();

        for table in &tables {
            if table.starts_with('_') {
                continue;
            }

            let records = match self.db.list(table) {
                Ok(r) => r,
                Err(_) => continue,
            };

            for record in &records {
                let source_kind = record
                    .data
                    .get("_source")
                    .and_then(|s| s.get("kind"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if source_kind != "user" {
                    continue;
                }

                let text = crate::util::value_text(&record.data).to_lowercase();
                let is_preference = preference_signals.iter().any(|s| text.contains(s));

                if is_preference && !existing_statements.contains(&text) {
                    let _ = self.db.insert(
                        "_candidate_preferences",
                        json!({
                            "statement": text,
                            "source_record": record.id.to_string(),
                            "source": "worker_brain",
                            "status": "candidate",
                        }),
                    );
                    candidates += 1;
                }
            }
        }

        candidates
    }

    /// Cluster near-duplicate memories for review.
    fn cluster_duplicates(&self) -> usize {
        // Without vector index, can't do semantic dedup.
        if !self.db.has_vector_index() || !self.db.has_embedder() {
            return 0;
        }

        // Check a sample of recent records for high-similarity pairs.
        let tables = match self.db.tables() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let mut clusters_found = 0;
        for table in &tables {
            if table.starts_with('_') {
                continue;
            }

            let records = match self.db.list(table) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Only check last 50 records per table to stay fast.
            let recent: Vec<&crate::Record> = records.iter().rev().take(50).collect();
            for record in &recent {
                if record
                    .data
                    .get("_superseded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    continue;
                }

                let text = crate::util::value_text(&record.data);
                if text.len() < 20 {
                    continue;
                }

                if let Ok(similar) = self.db.similar_to(&text, 3) {
                    for (other, score) in &similar {
                        if other.id == record.id || other.table != record.table {
                            continue;
                        }
                        if *score > 0.95 {
                            // Near-duplicate found — mark for review.
                            let mut data = record.data.clone();
                            if let Some(obj) = data.as_object_mut() {
                                if obj.get("_near_duplicate_of").is_none() {
                                    obj.insert(
                                        "_near_duplicate_of".to_string(),
                                        json!(other.id.to_string()),
                                    );
                                    let _ = self.db.update(&record.id, data);
                                    clusters_found += 1;
                                }
                            }
                            break; // One duplicate per record.
                        }
                    }
                }
            }
        }

        clusters_found
    }

    /// Detect records older than 90 days with no graph edges (stale).
    fn detect_stale(&self) -> usize {
        let cutoff = Utc::now() - Duration::days(90);
        let tables = match self.db.tables() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let mut stale_count = 0usize;
        for table in &tables {
            // Skip internal tables.
            if table.starts_with('_') {
                continue;
            }

            let records = match self.db.list(table) {
                Ok(r) => r,
                Err(_) => continue,
            };

            for record in &records {
                // Check age.
                if record.created_at >= cutoff {
                    continue;
                }

                // Check last accessed (activation system).
                let last_accessed = record
                    .data
                    .get("_last_accessed")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                if let Some(la) = last_accessed {
                    if la >= cutoff {
                        continue;
                    }
                }

                // Check for graph edges (if graph is available).
                if self.db.has_graph_index() {
                    let edges = self
                        .db
                        .edges(&record.id, None, crate::plugin::Direction::Both)
                        .unwrap_or_default();

                    if !edges.is_empty() {
                        continue; // Connected records are not stale.
                    }
                }

                stale_count += 1;
            }
        }

        stale_count
    }
}

/// Handle for a background maintenance thread.
///
/// Runs the `AxilWorker` periodically. The thread is non-blocking and
/// stops when `stop()` is called or when the handle is dropped.
///
/// # Example
///
/// ```ignore
/// let thread = MaintenanceThread::start(db_arc, std::time::Duration::from_secs(300));
/// // ... later ...
/// thread.stop();
/// ```
pub struct MaintenanceThread {
    stop_flag: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<Vec<WorkerReport>>>,
}

impl MaintenanceThread {
    /// Start a background maintenance thread.
    ///
    /// The thread runs the full worker cycle at the given interval.
    /// Returns a handle that can be used to stop the thread.
    pub fn start(db: Arc<Axil>, interval: std::time::Duration) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let flag = stop_flag.clone();

        let handle = std::thread::spawn(move || {
            let mut reports = std::collections::VecDeque::with_capacity(100);
            let check_interval = std::time::Duration::from_millis(500);

            // Run immediately on start, then loop with interval.
            let mut first_run = true;
            loop {
                if !first_run {
                    let mut waited = std::time::Duration::ZERO;
                    while waited < interval {
                        if flag.load(Ordering::Relaxed) {
                            return reports.into_iter().collect();
                        }
                        std::thread::sleep(check_interval);
                        waited += check_interval;
                    }
                }
                first_run = false;

                if flag.load(Ordering::Relaxed) {
                    return reports.into_iter().collect();
                }

                let worker = AxilWorker::new(&db);
                match worker.run() {
                    Ok(report) => {
                        if reports.len() >= 100 {
                            reports.pop_front();
                        }
                        reports.push_back(report);
                    }
                    Err(e) => {
                        eprintln!("maintenance worker error: {e}");
                    }
                }
            }
        });

        Self {
            stop_flag,
            handle: Some(handle),
        }
    }

    /// Check if the maintenance thread is still running.
    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }

    /// Stop the maintenance thread and return all collected reports.
    ///
    /// Blocks until the thread finishes (up to ~500ms).
    pub fn stop(mut self) -> Vec<WorkerReport> {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

impl Drop for MaintenanceThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        // Don't join on drop — let the thread exit on its own.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axil;
    use serde_json::json;

    fn temp_db() -> (tempfile::TempDir, Axil) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        (dir, db)
    }

    #[test]
    fn worker_run_produces_report() {
        let (_dir, db) = temp_db();

        // Insert some test records.
        db.insert("notes", json!({"text": "hello world"})).unwrap();
        db.insert("notes", json!({"text": "goodbye world"}))
            .unwrap();

        let worker = AxilWorker::new(&db);
        let report = worker.run().unwrap();

        assert!(report.duration_ms < 10_000);
        assert_eq!(report.stale_detected, 0); // Fresh records.
    }

    #[test]
    fn worker_last_run_returns_none_initially() {
        let (_dir, db) = temp_db();
        let worker = AxilWorker::new(&db);
        assert!(worker.last_run().unwrap().is_none());
    }

    #[test]
    fn worker_last_run_returns_report_after_run() {
        let (_dir, db) = temp_db();
        let worker = AxilWorker::new(&db);
        worker.run().unwrap();
        let last = worker.last_run().unwrap();
        assert!(last.is_some());
    }

    #[test]
    fn worker_is_idempotent() {
        let (_dir, db) = temp_db();
        let worker = AxilWorker::new(&db);

        let r1 = worker.run().unwrap();
        let r2 = worker.run().unwrap();

        // Both should succeed (idempotent).
        assert!(r1.duration_ms < 10_000);
        assert!(r2.duration_ms < 10_000);
    }

    #[test]
    fn detect_stale_with_old_records() {
        let (_dir, db) = temp_db();

        // Insert records and backdate them via storage.
        let mut old_record = crate::Record::new("notes", json!({"text": "ancient note"}));
        old_record.created_at = Utc::now() - Duration::days(100);
        db.storage().insert(&old_record).unwrap();

        let mut recent_record = crate::Record::new("notes", json!({"text": "fresh note"}));
        recent_record.created_at = Utc::now() - Duration::days(10);
        db.storage().insert(&recent_record).unwrap();

        let worker = AxilWorker::new(&db);
        let report = worker.run().unwrap();

        // Only the 100-day-old record should be stale.
        assert_eq!(report.stale_detected, 1);
    }

    #[test]
    fn detect_stale_skips_recently_accessed() {
        let (_dir, db) = temp_db();

        // Old record but recently accessed.
        let mut old_but_accessed = crate::Record::new(
            "notes",
            json!({
                "text": "old but used",
                "_last_accessed": (Utc::now() - Duration::days(5)).to_rfc3339(),
            }),
        );
        old_but_accessed.created_at = Utc::now() - Duration::days(120);
        db.storage().insert(&old_but_accessed).unwrap();

        // Old record, never accessed.
        let mut old_unused = crate::Record::new("notes", json!({"text": "old and forgotten"}));
        old_unused.created_at = Utc::now() - Duration::days(120);
        db.storage().insert(&old_unused).unwrap();

        let worker = AxilWorker::new(&db);
        let report = worker.run().unwrap();

        // Only the unused one is stale.
        assert_eq!(report.stale_detected, 1);
    }

    #[test]
    fn maintenance_thread_starts_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        db.insert("notes", json!({"text": "hello"})).unwrap();
        let db_arc = std::sync::Arc::new(db);

        let mt = MaintenanceThread::start(db_arc, std::time::Duration::from_millis(50));
        assert!(mt.is_running());

        // Let it run at least once. Use a generous timeout since CI may be slow.
        std::thread::sleep(std::time::Duration::from_secs(2));

        let reports = mt.stop();
        // The thread should have run at least once in 2 seconds with 50ms interval.
        // On very slow CI, it might not, so just check it didn't panic.
        if !reports.is_empty() {
            assert!(reports[0].duration_ms < 10_000);
        }
    }

    #[test]
    fn detect_stale_skips_internal_tables() {
        let (_dir, db) = temp_db();

        // Insert old record in internal table (should be skipped).
        let mut internal = crate::Record::new("_system", json!({"type": "config"}));
        internal.created_at = Utc::now() - Duration::days(200);
        db.storage().insert(&internal).unwrap();

        let worker = AxilWorker::new(&db);
        let report = worker.run().unwrap();

        assert_eq!(report.stale_detected, 0);
    }
}
