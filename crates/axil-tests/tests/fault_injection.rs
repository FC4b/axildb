//! Seeded fault-injection (deterministic-simulation-test style) proving Axil's
//! repair path recovers from torn-write **forward orphans** — the symmetric
//! failure to the reverse orphans covered in `self_healing.rs`.
//!
//! A forward orphan is an index entry (vector / FTS doc / graph edge) whose
//! backing core record no longer exists: the kind of inconsistency a crash
//! between the core-storage commit and the index fan-out can leave behind.
//!
//! These tests synthesize a genuine orphan with **zero production-code change**
//! by calling [`Axil::storage`]`().delete(id)`, which removes only the core
//! record — the cascade that normally cleans the index fan-out lives in
//! [`Axil::delete`], not in `Storage::delete`. We then assert the existing
//! repair path (`detect_problems` → `clean_orphaned_*` / `compact` /
//! `heal_all`) drives the database back to consistent, and that recovery is
//! idempotent.
//!
//! Determinism: an in-test xorshift PRNG (no `rand` dependency) drives the
//! fan-out workload; the seed is printed on failure so any flake reproduces.

use axil_core::{Axil, Engine, HealingConfig, RecordId, Severity, TextEmbedder, VectorIndex};
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

// ── Deterministic PRNG (xorshift64*) — no `rand` dependency ──────────────

/// Minimal deterministic PRNG so the fan-out workload is reproducible from a
/// single seed. xorshift64* — adequate for choosing operations and indices in
/// a test; not for cryptography.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift cannot escape.
        Self {
            state: seed | 1,
        }
    }

    /// Next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform-ish `usize` in `[0, n)`. `n` must be non-zero.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── Test fixtures ────────────────────────────────────────────────────────

/// Deterministic 4-dim mock embedder (no ONNX model needed). Mirrors the
/// `FrontWindowEmbedder` used in `self_healing.rs` / `intelligent_db.rs` so the
/// vector fan-out runs without downloading a model.
struct FrontWindowEmbedder;

impl TextEmbedder for FrontWindowEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let window = text.chars().take(100).collect::<String>().to_lowercase();
        Ok(vec![
            if window.contains("auth") { 1.0 } else { 0.0 },
            if window.contains("timeout") { 1.0 } else { 0.0 },
            if window.contains("pool") { 1.0 } else { 0.0 },
            1.0,
        ])
    }
}

/// A database wired with the full fan-out: vector + graph + FTS, plus a mock
/// embedder. Every Engine that can hold a forward orphan is present.
fn full_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let vector = axil_vector::VectorEngine::open(&path, 4).unwrap();
    let db = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FrontWindowEmbedder))
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

/// Synthesize a forward orphan: remove only the core record, leaving every
/// index entry (vector / FTS / edge) dangling. This is the genuine torn-write
/// state — `Storage::delete` does not run the cascade that `Axil::delete` does.
fn torn_delete(db: &Axil, id: &RecordId) {
    assert!(
        db.storage().delete(id).unwrap(),
        "torn_delete target must exist in core storage"
    );
}

/// Total forward orphans across every Engine, computed without mutating the DB:
/// the sum of the three `count_*`-equivalent scans the repair path uses. We
/// drive this to zero.
fn detect_problems_is_clean(db: &Axil) -> bool {
    let problems = db.detect_problems();
    // Forward orphans that `detect_problems` reports: edges and vectors whose
    // record is gone. A consistent DB has none of these.
    !problems
        .iter()
        .any(|p| matches!(p.detector.as_str(), "orphaned_edges" | "orphaned_vectors"))
}

// ── Per-class orphan tests ───────────────────────────────────────────────

#[test]
fn orphaned_vector_is_detected_and_healed() {
    let (db, _dir) = full_db();
    let rec = db
        .insert("sessions", json!({"summary": "auth timeout"}))
        .unwrap();
    db.embed_field(&rec.id, "summary").unwrap();

    // Tear: core record gone, vector entry remains.
    torn_delete(&db, &rec.id);
    assert!(
        db.detect_problems()
            .iter()
            .any(|p| p.detector == "orphaned_vectors" && p.auto_fixable),
        "the orphaned vector must be reported before it is cleaned"
    );

    // Detected: a clean call removes exactly the one orphan.
    let cleaned = db.clean_orphaned_vectors();
    assert_eq!(cleaned, 1, "the single orphaned vector must be cleaned");

    // Idempotent: a second pass finds nothing.
    assert_eq!(
        db.clean_orphaned_vectors(),
        0,
        "recovery must be idempotent"
    );
    assert!(detect_problems_is_clean(&db));
}

/// The `orphaned_vectors` check from `doctor`.
fn orphaned_vectors_check(db: &Axil) -> axil_core::CheckResult {
    db.doctor()
        .unwrap()
        .checks
        .into_iter()
        .find(|c| c.name == "orphaned_vectors")
        .expect("doctor must report orphaned_vectors when a vector index is attached")
}

#[test]
fn doctor_reports_orphaned_vectors_by_share_of_the_index() {
    let (db, _dir) = full_db();
    let ids: Vec<RecordId> = (0..20)
        .map(|i| {
            db.insert(
                "sessions",
                json!({"summary": format!("auth timeout pool record {i}")}),
            )
            .unwrap()
            .id
        })
        .collect();
    let clean = orphaned_vectors_check(&db);
    assert_eq!(clean.status, Severity::Ok, "{}", clean.detail);

    // One orphan in twenty vectors: a warning, with the targeted fix.
    torn_delete(&db, &ids[0]);
    let check = orphaned_vectors_check(&db);
    assert_eq!(check.status, Severity::Warning, "{}", check.detail);
    assert!(
        check.detail.starts_with("1 of 20 vectors"),
        "{}",
        check.detail
    );
    assert_eq!(check.fix.as_deref(), Some("axil heal --orphans"));

    // Four in twenty is past a tenth of the index: an error, in doctor and in
    // detect_problems alike.
    for id in &ids[1..4] {
        torn_delete(&db, id);
    }
    assert_eq!(orphaned_vectors_check(&db).status, Severity::Error);
    let problem = db
        .detect_problems()
        .into_iter()
        .find(|p| p.detector == "orphaned_vectors")
        .expect("detect_problems must report the orphans too");
    assert_eq!(problem.severity, Severity::Error, "{}", problem.message);

    assert_eq!(db.clean_orphaned_vectors(), 4);
    let healed = orphaned_vectors_check(&db);
    assert_eq!(healed.status, Severity::Ok, "{}", healed.detail);
    assert_eq!(healed.detail, "16 vectors, no orphans");
}

#[test]
fn orphaned_fts_is_detected_and_healed() {
    // FTS is the weakest link — `index_text` writes to a separate tantivy
    // directory, so a torn delete leaves a dangling doc until reconciled.
    let (db, _dir) = full_db();
    let rec = db
        .insert("sessions", json!({"summary": "auth timeout in the pool"}))
        .unwrap();
    db.index_text(&rec.id, "summary", "auth timeout in the pool")
        .unwrap();

    torn_delete(&db, &rec.id);

    let cleaned = db.clean_orphaned_fts();
    assert_eq!(cleaned, 1, "the single orphaned FTS doc must be cleaned");

    assert_eq!(db.clean_orphaned_fts(), 0, "recovery must be idempotent");

    // The deleted record must never surface in search after healing.
    let hits = db.search_text("auth timeout", 5).unwrap();
    assert!(
        hits.iter().all(|(r, _)| r.id != rec.id),
        "healed FTS orphan must not surface in search"
    );
    // `insert` also embedded the record, so the tear orphaned its vector too.
    assert_eq!(db.clean_orphaned_vectors(), 1);
    assert!(detect_problems_is_clean(&db));
}

#[test]
fn orphaned_edge_is_detected_and_healed() {
    let (db, _dir) = full_db();
    let from = db.insert("t", json!({"x": 1})).unwrap();
    let to = db.insert("t", json!({"x": 2})).unwrap();
    db.relate(&from.id, "links_to", &to.id, None).unwrap();

    // Tear the edge target — the edge now dangles.
    torn_delete(&db, &to.id);

    // `detect_problems` flags it as an auto-fixable `orphaned_edges` problem.
    let problems = db.detect_problems();
    let orphan = problems
        .iter()
        .find(|p| p.detector == "orphaned_edges")
        .expect("orphaned_edges must be detected");
    assert_eq!(orphan.severity, Severity::Warning);
    assert!(orphan.auto_fixable, "orphaned edges must be auto-fixable");

    let cleaned = db.clean_orphaned_edges();
    assert_eq!(cleaned, 1, "the single orphaned edge must be cleaned");

    assert_eq!(db.clean_orphaned_edges(), 0, "recovery must be idempotent");
    assert!(
        db.detect_problems()
            .iter()
            .all(|p| p.detector != "orphaned_edges"),
        "no orphaned edges may remain after healing"
    );
}

// ── Seeded fan-out workload ──────────────────────────────────────────────

/// Run a seeded ~200-op workload of inserts, embeds, FTS-index, relates, and
/// torn deletes, then assert `heal_all` + `compact` drive `detect_problems`
/// back to clean. Returns nothing; panics (with the seed) on any divergence.
fn run_fanout_workload(seed: u64) {
    let (db, _dir) = full_db();
    let cfg = HealingConfig::default();
    let mut rng = XorShift64::new(seed);

    // IDs of records we know to still be live (never torn). Used so relates
    // target real endpoints most of the time.
    let mut live: Vec<RecordId> = Vec::new();
    // Records we tore (core row removed) — must never surface in recall again.
    let mut torn_ids: Vec<RecordId> = Vec::new();

    const OPS: usize = 200;
    for op in 0..OPS {
        match rng.below(5) {
            // Insert + full fan-out (embed + FTS).
            0 | 1 => {
                let rec = db
                    .insert(
                        "sessions",
                        json!({"summary": format!("auth timeout pool record {op}")}),
                    )
                    .unwrap();
                db.embed_field(&rec.id, "summary").unwrap();
                db.index_text(&rec.id, "summary", "auth timeout pool")
                    .unwrap();
                live.push(rec.id);
            }
            // Relate two live records.
            2 => {
                if live.len() >= 2 {
                    let a = live[rng.below(live.len())].clone();
                    let b = live[rng.below(live.len())].clone();
                    if a != b {
                        db.relate(&a, "links_to", &b, None).unwrap();
                    }
                }
            }
            // Torn delete: remove a live record's core row, orphaning its
            // index fan-out and any edges that reference it.
            3 => {
                if !live.is_empty() {
                    let idx = rng.below(live.len());
                    let id = live.swap_remove(idx);
                    torn_delete(&db, &id);
                    torn_ids.push(id);
                }
            }
            // Clean delete (cascades) — exercises the well-behaved path too.
            _ => {
                if !live.is_empty() {
                    let idx = rng.below(live.len());
                    let id = live.swap_remove(idx);
                    db.delete(&id).unwrap();
                }
            }
        }
    }

    // The workload must have actually injected faults, or it proves nothing.
    assert!(
        !torn_ids.is_empty(),
        "seed {seed}: workload injected no torn deletes — not a fault test"
    );

    // Repair: heal_all runs compact (which cleans orphaned edges/vectors/FTS)
    // plus reembed; a follow-up compact mops up anything heal deferred.
    db.heal_all(&cfg, false).unwrap();
    db.compact().unwrap();

    assert!(
        detect_problems_is_clean(&db),
        "seed {seed}: detect_problems not clean after heal_all + compact: {:?}",
        db.detect_problems()
    );

    // No torn record may resurface in recall (vector or FTS) — the user-visible
    // symptom a forward orphan would otherwise cause.
    for id in &torn_ids {
        let vec_hits = db.similar_to("auth timeout pool", 50).unwrap();
        assert!(
            vec_hits.iter().all(|(r, _)| &r.id != id),
            "seed {seed}: torn record {id} resurfaced in vector recall"
        );
        let fts_hits = db.search_text("auth timeout pool", 50).unwrap();
        assert!(
            fts_hits.iter().all(|(r, _)| &r.id != id),
            "seed {seed}: torn record {id} resurfaced in FTS recall"
        );
    }

    // Vector and graph cleaners use the live index as source of truth, so they
    // are fully idempotent: a second pass finds zero orphans.
    assert_eq!(
        db.clean_orphaned_vectors(),
        0,
        "seed {seed}: orphaned vectors remained after repair"
    );
    assert_eq!(
        db.clean_orphaned_edges(),
        0,
        "seed {seed}: orphaned edges remained after repair"
    );

    // Idempotent: a second repair pass changes nothing and stays clean.
    db.heal_all(&cfg, false).unwrap();
    db.compact().unwrap();
    assert!(
        detect_problems_is_clean(&db),
        "seed {seed}: repair was not idempotent"
    );
}

#[test]
fn seeded_fanout_workload_heals_clean() {
    // A spread of seeds so the workload exercises different op interleavings;
    // every one must heal clean. The seed is in every failure message.
    for seed in [1u64, 7, 42, 1337, 0xDEAD_BEEF, 0x5151_5151] {
        run_fanout_workload(seed);
    }
}

#[test]
fn pinned_seed_regression() {
    // A pinned "bugbase" seed: this exact interleaving must always heal clean,
    // guarding against regressions in the repair path. If this fails, the seed
    // is reproducible — rerun `run_fanout_workload(0x0000_0000_0042_1A4E)`.
    run_fanout_workload(0x0000_0000_0042_1A4E);
}

// ── Deletes while the vector store is detached ───────────────────────────

/// Open `path` with its default vector store attached, as a healthy CLI open
/// does.
fn open_with_vectors(path: &std::path::Path) -> Axil {
    let vector = axil_vector::VectorEngine::open(path, 4).unwrap();
    Axil::open(path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FrontWindowEmbedder))
        .build()
        .unwrap()
}

/// Open `path` without a vector index while its `.vec` store is on disk: what
/// the CLI does when the store fails to open or `[engines] disabled` lists it.
fn open_detached(path: &std::path::Path) -> Axil {
    assert!(axil_vector::vector_db_path(path).exists());
    Axil::open(path).build().unwrap()
}

#[test]
fn a_delete_while_the_vector_store_is_detached_leaves_no_orphan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let (gone, kept) = {
        let db = open_with_vectors(&path);
        let gone = db
            .insert("sessions", json!({"summary": "auth timeout in the pool"}))
            .unwrap()
            .id;
        let kept = db
            .insert("sessions", json!({"summary": "connection pool sizing"}))
            .unwrap()
            .id;
        assert!(db.get_vector(&gone).unwrap().is_some());
        (gone, kept)
    };

    assert!(open_detached(&path).delete(&gone).unwrap());

    let db = open_with_vectors(&path);
    assert_eq!(
        db.get_vector(&gone).unwrap(),
        None,
        "the deleted record's vector must go once the store is attached again"
    );
    assert!(
        db.get_vector(&kept).unwrap().is_some(),
        "other vectors stay"
    );
    assert!(detect_problems_is_clean(&db), "{:?}", db.detect_problems());
    assert_eq!(db.vector_count(), Some(1));
}

#[test]
fn a_record_reinserted_under_its_id_while_detached_keeps_its_vector() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let record = open_with_vectors(&path)
        .insert("sessions", json!({"summary": "auth timeout in the pool"}))
        .unwrap();

    {
        let db = open_detached(&path);
        assert!(db.delete(&record.id).unwrap());
        db.insert_preserving(record.clone()).unwrap();
    }

    // The note is cleared without taking a vector a live record owns.
    let db = open_with_vectors(&path);
    assert!(db.get_vector(&record.id).unwrap().is_some());
    drop(db);
    assert!(open_with_vectors(&path)
        .get_vector(&record.id)
        .unwrap()
        .is_some());
}

/// A vector index that, when asked to remove vectors, first lets a re-insert
/// land: the record goes back into storage and gets a new vector. That is
/// the interleaving a concurrent `insert_preserving` produces between an
/// orphan sweep's scan and its removal, made deterministic.
struct ReinsertDuringRemoval {
    inner: axil_vector::VectorEngine,
    db: std::sync::Arc<std::sync::OnceLock<std::sync::Weak<Axil>>>,
    record: std::sync::Arc<std::sync::Mutex<Option<axil_core::Record>>>,
}

impl ReinsertDuringRemoval {
    fn reinsert_first(&self) {
        if let Some(record) = self.record.lock().unwrap().take() {
            let db = self.db.get().and_then(std::sync::Weak::upgrade).unwrap();
            db.storage().insert(&record).unwrap();
            self.inner
                .add(record.id.clone(), &[0.0, 0.0, 0.0, 1.0])
                .unwrap();
        }
    }
}

impl Engine for ReinsertDuringRemoval {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> Vec<axil_core::Capability> {
        self.inner.capabilities()
    }
    fn on_record_insert(&self, record: &axil_core::Record) -> axil_core::Result<()> {
        self.inner.on_record_insert(record)
    }
    fn on_record_delete(&self, id: &RecordId) -> axil_core::Result<()> {
        self.inner.on_record_delete(id)
    }
}

impl VectorIndex for ReinsertDuringRemoval {
    fn add(&self, id: RecordId, vector: &[f32]) -> axil_core::Result<()> {
        self.inner.add(id, vector)
    }
    fn remove_batch(&self, ids: &[RecordId]) -> axil_core::Result<usize> {
        self.reinsert_first();
        self.inner.remove_batch(ids)
    }
    fn remove_batch_if(
        &self,
        ids: &[RecordId],
        doomed: &dyn Fn(&RecordId) -> bool,
    ) -> axil_core::Result<usize> {
        self.reinsert_first();
        self.inner.remove_batch_if(ids, doomed)
    }
    fn search(&self, query: &[f32], top_k: usize) -> axil_core::Result<Vec<(RecordId, f32)>> {
        self.inner.search(query, top_k)
    }
    fn count(&self) -> usize {
        self.inner.count()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    fn all_ids(&self) -> axil_core::Result<Vec<RecordId>> {
        self.inner.all_ids()
    }
    fn get_vector(&self, id: &RecordId) -> axil_core::Result<Option<Vec<f32>>> {
        self.inner.get_vector(id)
    }
}

#[test]
fn orphan_cleanup_keeps_the_vector_of_a_record_reinserted_meanwhile() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let slot = std::sync::Arc::new(std::sync::OnceLock::new());
    let reinsert = std::sync::Arc::new(std::sync::Mutex::new(None));
    let index = ReinsertDuringRemoval {
        inner: axil_vector::VectorEngine::open(&path, 4).unwrap(),
        db: slot.clone(),
        record: reinsert.clone(),
    };
    let db = std::sync::Arc::new(
        Axil::open(&path)
            .with_vector_index(Box::new(index))
            .with_embedder(Box::new(FrontWindowEmbedder))
            .build()
            .unwrap(),
    );
    slot.set(std::sync::Arc::downgrade(&db)).unwrap();

    let record = db
        .insert("sessions", json!({"summary": "auth timeout in the pool"}))
        .unwrap();
    torn_delete(&db, &record.id);
    // The sweep sees the record gone; by the time it removes vectors, the
    // record is back with a new one.
    *reinsert.lock().unwrap() = Some(record.clone());

    assert_eq!(db.clean_orphaned_vectors(), 0);
    assert_eq!(
        db.get_vector(&record.id).unwrap(),
        Some(vec![0.0, 0.0, 0.0, 1.0]),
        "the re-insert's vector must survive the sweep"
    );
    assert!(detect_problems_is_clean(&db), "{:?}", db.detect_problems());
}

/// The paths that delete on their own (compaction, recall-chunk cleanup, a
/// code re-index) all go through `Axil::delete`, so a detached store must not
/// leave an orphan behind any of them.
#[test]
fn cleanup_deletes_while_detached_leave_no_orphans() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let project = dir.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(
        project.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        project.join("src/lib.rs"),
        "/// Retry an auth call after a timeout.\npub fn retry_auth() {}\n\n\
         /// Size the connection pool.\npub fn pool_size() -> usize { 32 }\n",
    )
    .unwrap();

    let long_id = {
        let db = open_with_vectors(&path);
        let past = (chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        db.insert(
            "notes",
            json!({"summary": "auth timeout note", "valid_until": past, "_importance": 0.1}),
        )
        .unwrap();
        let long = db
            .insert(
                "notes",
                json!({"summary": "auth timeout in the pool, again. ".repeat(200)}),
            )
            .unwrap();
        axil_indexer::ProjectIndexer::new(&db, axil_indexer::IndexConfig::default())
            .index_full(&project)
            .unwrap();
        assert!(
            db.vector_count().unwrap() > 3,
            "the long note's chunks and the code index must hold vectors"
        );
        long.id
    };

    {
        let db = open_detached(&path);
        let report = db.compact().unwrap();
        assert_eq!(report.purged_expired, 1);
        assert!(db.delete(&long_id).unwrap());
        axil_indexer::ProjectIndexer::new(&db, axil_indexer::IndexConfig::default())
            .index_full(&project)
            .unwrap();
    }

    let db = open_with_vectors(&path);
    assert!(detect_problems_is_clean(&db), "{:?}", db.detect_problems());
    assert_eq!(db.clean_orphaned_vectors(), 0);
}
