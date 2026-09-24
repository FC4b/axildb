//! A broken `axil.toml` must never be silently ignored: many open paths fall
//! back to defaults (`load_config(..).ok()`), so the handle builder reports
//! it itself, and `config show` explains why a malformed `[lifecycle]` entry
//! resolved to the protective policy.

use std::path::Path;
use std::process::{Command, Output};

fn axil(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axil"))
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("run axil")
}

#[test]
fn unparseable_config_is_reported_when_a_handle_opens() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.axil");
    let db_arg = db.to_str().unwrap();
    assert!(axil(dir.path(), &["init", db_arg]).status.success());

    std::fs::write(dir.path().join("axil.toml"), "[healing\nevent_log = true\n").unwrap();

    let out = axil(dir.path(), &["--db", db_arg, "list", "notes"]);
    assert!(
        out.status.success(),
        "a broken config must not stop the database from opening: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("invalid config") && stderr.contains("defaults apply"),
        "expected a config warning on stderr, got: {stderr}"
    );
}

#[test]
fn config_show_explains_a_malformed_lifecycle_entry() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("axil.toml"),
        "[lifecycle.tables.autopsies]\ncompact = \"Never\"\n",
    )
    .unwrap();

    let out = axil(dir.path(), &["config", "show"]);
    assert!(
        out.status.success(),
        "config show failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let shown: serde_json::Value = serde_json::from_slice(&out.stdout).expect("JSON output");
    let warnings = shown["lifecycle_warnings"]
        .as_array()
        .expect("lifecycle_warnings is listed");
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().unwrap_or_default().contains("autopsies")),
        "the warning names the malformed table: {warnings:?}"
    );
}
