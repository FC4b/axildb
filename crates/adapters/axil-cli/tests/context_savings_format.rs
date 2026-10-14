//! `axil context-savings --format` picks the report format (text, markdown,
//! json) even though the global `--format` (json, pretty, table) has the same
//! name, and the global flag still rejects the report formats everywhere
//! else.
//!
//! Drives the real binary: a debug build panics on a clap tree clash before
//! it parses anything, which a release build never shows.
#![cfg(feature = "indexer")]

use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;

const MARKDOWN_HEADER: &str =
    "| Task | Hits | Files | w/ Axil | no Axil | Saved |\n|---|---:|---:|---:|---:|---:|\n";
const TEXT_HEADLINE: &str = "Context savings — answering coding tasks with vs without Axil\n";

fn axil(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axil"))
        .current_dir(cwd)
        .env_remove("AXIL_DB")
        .args(args)
        .output()
        .expect("run axil")
}

/// stdout of a command that must succeed.
fn stdout_of(cwd: &Path, args: &[&str]) -> String {
    let out = axil(cwd, args);
    assert!(
        out.status.success(),
        "axil {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn context_savings_format_selects_the_report() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let proj = root.join("proj");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(
        proj.join("Cargo.toml"),
        "[package]\nname = \"f\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        proj.join("src/scoring.rs"),
        "//! Recall scoring helpers.\npub fn score() -> u32 { 0 }\n",
    )
    .unwrap();

    let db = root.join("t.axil");
    let db = db.to_str().unwrap();
    stdout_of(root, &["init", db]);
    stdout_of(root, &["--db", db, "index", proj.to_str().unwrap()]);

    let savings = |extra: &[&str]| {
        let mut args = vec!["--db", db];
        args.extend_from_slice(extra);
        stdout_of(root, &args)
    };
    let task = ["context-savings", "--task", "recall scoring"];
    let with = |flags: &[&str]| {
        let mut args = task.to_vec();
        args.extend_from_slice(flags);
        savings(&args)
    };

    let markdown = with(&["--format", "markdown"]);
    assert!(
        markdown.starts_with(MARKDOWN_HEADER),
        "--format markdown must print the markdown table, got:\n{markdown}"
    );
    assert!(markdown.contains("| recall scoring |"), "{markdown}");
    assert!(markdown.contains("| **Total** |"), "{markdown}");
    assert_eq!(with(&["--format", "md"]), markdown, "md is markdown");
    assert_eq!(with(&["--format=markdown"]), markdown);

    let json = with(&["--format", "json"]);
    assert_eq!(
        json.lines().count(),
        1,
        "the JSON report prints compact: {json}"
    );
    let report: Value = serde_json::from_str(&json).expect("--format json prints JSON");
    assert_eq!(report["tasks"][0]["task"], "recall scoring");

    let text = with(&[]);
    assert!(
        text.starts_with(TEXT_HEADLINE),
        "text is the default, got:\n{text}"
    );
    // A global `--format` before the subcommand leaves the text default alone.
    let mut before = vec!["--format", "json"];
    before.extend_from_slice(&task);
    assert_eq!(savings(&before), text);

    // The report formats are not values of the global flag.
    let out = axil(root, &["--db", db, "get", "x", "--format", "markdown"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("invalid value 'markdown' for '--format <FORMAT>'")
            && stderr.contains("[possible values: json, pretty, table]"),
        "{stderr}"
    );
}
