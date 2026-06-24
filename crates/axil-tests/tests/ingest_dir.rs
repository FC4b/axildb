//! Integration tests for `axil ingest <dir>` — bulk filesystem ingest.
//!
//! Ports the shell e2e's ingest assertions (`tests/phase-12-e2e.sh`, step 2)
//! to a cargo-runnable test: ingest a temp corpus, assert the reported file
//! count, re-run and assert content-hash skip, and assert the `--stats`
//! dry-run plan counts. Each test uses a temp `.axil` DB so it is hermetic.
//!
//! These invoke the built `axil` binary (debug profile), matching the harness
//! in `agent_cli.rs`. The `ingest` subcommand is gated behind the binary's
//! `indexer` feature, which is on by default — so `cargo build -p axildb`
//! produces a binary that has it.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Path to the built CLI binary (debug profile).
fn axil_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates/
    path.pop(); // workspace root
    path.push("target/debug/axil");
    path.set_extension(std::env::consts::EXE_EXTENSION);
    assert!(
        path.exists(),
        "axil binary not found at {}. Run `cargo build -p axildb` first.",
        path.display()
    );
    path
}

/// Run axil with the given args, returning (stdout, stderr, exit_code).
fn run_axil(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(axil_bin())
        .args(args)
        .output()
        .expect("failed to execute axil binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

fn parse_json(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("failed to parse JSON: {e}\nstdout: {stdout}"))
}

/// Build a temp corpus of three small notes and return (tempdir, db_path, corpus_dir).
/// The DB lives under `<tmp>/.axil/memory.axil` so the sibling `ingest.state.json`
/// is created in a dir we own; the corpus is a separate sibling dir.
fn setup_corpus() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let axil_dir = dir.path().join(".axil");
    fs::create_dir_all(&axil_dir).unwrap();
    let db = axil_dir.join("memory.axil");

    let corpus = dir.path().join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("auth.md"),
        "# Auth refactor\n\nMoved AuthModule to OAuth2 using the standard JWT library.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("ingest.md"),
        "# Ingest pipeline\n\nBulk ingest walks a directory and chunks each file.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("notes.txt"),
        "Plain text note. Idempotent via FNV content hash.\n",
    )
    .unwrap();

    (dir, db, corpus)
}

#[test]
fn test_ingest_reports_file_count() {
    let (_dir, db, corpus) = setup_corpus();
    let db_str = db.to_str().unwrap();
    let corpus_str = corpus.to_str().unwrap();

    let (stdout, stderr, code) =
        run_axil(&["--db", db_str, "ingest", corpus_str, "--table", "notes"]);
    assert_eq!(code, 0, "ingest should succeed. stderr: {stderr}");

    let json = parse_json(&stdout);
    assert_eq!(
        json["files_ingested"], 3,
        "expected 3 files ingested, got report: {json}"
    );
    assert_eq!(json["files_skipped"], 0, "first run should skip nothing");
    assert!(
        json["chunks_written"].as_u64().unwrap() >= 3,
        "expected at least one chunk per file, got: {json}"
    );
    assert_eq!(json["table"], "notes");
}

#[test]
fn test_ingest_is_idempotent_via_content_hash() {
    let (_dir, db, corpus) = setup_corpus();
    let db_str = db.to_str().unwrap();
    let corpus_str = corpus.to_str().unwrap();

    // First ingest establishes state.
    let (stdout1, stderr1, code1) =
        run_axil(&["--db", db_str, "ingest", corpus_str, "--table", "notes"]);
    assert_eq!(code1, 0, "first ingest should succeed. stderr: {stderr1}");
    assert_eq!(parse_json(&stdout1)["files_ingested"], 3);

    // Second ingest with --resume must skip all three (unchanged content hash).
    let (stdout2, stderr2, code2) = run_axil(&[
        "--db", db_str, "ingest", corpus_str, "--table", "notes", "--resume",
    ]);
    assert_eq!(code2, 0, "resume ingest should succeed. stderr: {stderr2}");
    let json2 = parse_json(&stdout2);
    assert_eq!(
        json2["files_ingested"], 0,
        "re-ingest should write nothing, got: {json2}"
    );
    assert_eq!(
        json2["files_skipped"], 3,
        "re-ingest should skip all 3 unchanged files, got: {json2}"
    );
    assert_eq!(
        json2["chunks_written"], 0,
        "re-ingest should write no chunks, got: {json2}"
    );
}

#[test]
fn test_ingest_stats_dry_run_plan() {
    let (_dir, db, corpus) = setup_corpus();
    let db_str = db.to_str().unwrap();
    let corpus_str = corpus.to_str().unwrap();

    let (stdout, stderr, code) = run_axil(&[
        "--db", db_str, "ingest", corpus_str, "--table", "notes", "--stats",
    ]);
    assert_eq!(code, 0, "stats dry-run should succeed. stderr: {stderr}");

    let json = parse_json(&stdout);
    assert_eq!(json["dry_run"], true, "stats must flag a dry run");
    assert_eq!(
        json["total_files"], 3,
        "dry-run should count all 3 candidate files, got: {json}"
    );
    assert!(
        json["total_bytes"].as_u64().unwrap() > 0,
        "dry-run should report nonzero bytes, got: {json}"
    );
    // No prior state → nothing already indexed.
    assert_eq!(json["already_indexed"], 0);

    // A dry run must not write: a real ingest right after still reports 3 fresh files.
    let (stdout2, _e, c2) =
        run_axil(&["--db", db_str, "ingest", corpus_str, "--table", "notes"]);
    assert_eq!(c2, 0);
    assert_eq!(
        parse_json(&stdout2)["files_ingested"], 3,
        "stats must be a no-op; real ingest should still see 3 fresh files"
    );
}

#[test]
fn test_ingest_ext_filter_excludes_unlisted_extensions() {
    let (_dir, db, corpus) = setup_corpus();
    let db_str = db.to_str().unwrap();
    let corpus_str = corpus.to_str().unwrap();

    // Restrict to .md only → the .txt note is excluded, leaving 2 files.
    let (stdout, stderr, code) = run_axil(&[
        "--db", db_str, "ingest", corpus_str, "--table", "notes", "--ext", "md",
    ]);
    assert_eq!(code, 0, "ext-filtered ingest should succeed. stderr: {stderr}");
    assert_eq!(
        parse_json(&stdout)["files_ingested"], 2,
        "only the two .md files should be ingested with --ext md"
    );
}

#[test]
fn test_ingest_watch_flags_are_wired() {
    // We never run `--watch` (it blocks forever). Confirm the flags exist and are
    // documented in help — i.e. Part A's clap wiring is reachable from the binary.
    let (stdout, stderr, code) = run_axil(&["ingest", "--help"]);
    assert_eq!(code, 0, "ingest --help should succeed. stderr: {stderr}");
    let help = format!("{stdout}{stderr}");
    assert!(help.contains("--watch"), "ingest --help must list --watch");
    assert!(
        help.contains("--interval"),
        "ingest --help must list --interval"
    );
}
