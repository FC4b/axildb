//! Integration tests for : Skills, Config, and Report CLI commands.
//!
//! Tests config init/show/get/set, report generate/list/import,
//! and skill install/list/uninstall.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Path to the built CLI binary (debug profile).
fn axil_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates/
    path.pop(); // workspace root
    path.push("target/debug/axil");
    // On Windows the binary is `axil.exe`; EXE_EXTENSION is "" elsewhere.
    path.set_extension(std::env::consts::EXE_EXTENSION);
    assert!(
        path.exists(),
        "axil binary not found at {}. Run `cargo build -p axildb` first.",
        path.display()
    );
    path
}

/// Run axil in a specific working directory.
fn run_axil_in(cwd: &std::path::Path, args: &[&str]) -> (String, String, i32) {
    let output = Command::new(axil_bin())
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to execute axil binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Parse stdout as JSON Value.
fn parse_json(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("failed to parse JSON: {e}\nstdout: {stdout}"))
}

// ─── Config ─────────────────────────────────────────────────────────────────

#[test]
fn test_config_init_creates_file() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "init"]);
    assert_eq!(code, 0, "config init should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["created"], true);

    // Verify file exists
    assert!(dir.path().join("axil.toml").exists());
}

#[test]
fn test_config_init_fails_if_exists() {
    let dir = tempfile::TempDir::new().unwrap();

    // Create first
    let (_stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "init"]);
    assert_eq!(code, 0);

    // Second call should fail
    let (_stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "init"]);
    assert_eq!(code, 1, "config init should fail if axil.toml exists");
}

#[test]
fn test_config_show_defaults() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "show"]);
    assert_eq!(code, 0, "config show should succeed");

    let json = parse_json(&stdout);
    let config = &json["config"];
    assert_eq!(config["timeseries"]["full_retention_days"], 90);
    assert_eq!(config["debug"]["log_level"], "warn");
    assert_eq!(config["dev"]["reports_dir"], ".axil-reports");
}

#[test]
fn test_config_get_existing_key() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "get", "debug.log_level"]);
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["key"], "debug.log_level");
    assert_eq!(json["value"], "warn");
}

#[test]
fn test_config_get_nonexistent_key() {
    let dir = tempfile::TempDir::new().unwrap();

    let (_stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "get", "nonexistent.key"]);
    assert_eq!(code, 2, "should return NOT_FOUND for missing key");
}

#[test]
fn test_config_set_and_get() {
    let dir = tempfile::TempDir::new().unwrap();

    // Set a value
    let (stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["config", "set", "dev.source_repo", "../axildb"],
    );
    assert_eq!(code, 0);
    let json = parse_json(&stdout);
    assert_eq!(json["set"], true);

    // Get it back
    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "get", "dev.source_repo"]);
    assert_eq!(code, 0);
    let json = parse_json(&stdout);
    assert_eq!(json["value"], "../axildb");
}

#[test]
fn test_config_set_numeric() {
    let dir = tempfile::TempDir::new().unwrap();

    let (_stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["config", "set", "debug.slow_query_threshold_ms", "50"],
    );
    assert_eq!(code, 0);

    let (stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["config", "get", "debug.slow_query_threshold_ms"],
    );
    assert_eq!(code, 0);
    let json = parse_json(&stdout);
    assert_eq!(json["value"], "50");
}

#[test]
fn test_config_set_boolean() {
    let dir = tempfile::TempDir::new().unwrap();

    let (_stdout, _stderr, code) =
        run_axil_in(dir.path(), &["config", "set", "dev.auto_report", "true"]);
    assert_eq!(code, 0);

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "get", "dev.auto_report"]);
    assert_eq!(code, 0);
    let json = parse_json(&stdout);
    assert_eq!(json["value"], "true");
}

#[test]
fn test_config_show_reads_file() {
    let dir = tempfile::TempDir::new().unwrap();

    // Write a custom config
    std::fs::write(
        dir.path().join("axil.toml"),
        "[timeseries]\nfull_retention_days = 7\n",
    )
    .unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["config", "show"]);
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["config"]["timeseries"]["full_retention_days"], 7);
    // Other sections should still use defaults
    assert_eq!(json["config"]["debug"]["log_level"], "warn");
}

// ─── Report ─────────────────────────────────────────────────────────────────

#[test]
fn test_report_generate() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["report", "generate"]);
    assert_eq!(code, 0, "report generate should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["generated"], true);

    // Verify report file exists
    let reports_dir = dir.path().join(".axil-reports");
    assert!(reports_dir.exists(), "reports directory should be created");

    let entries: Vec<_> = std::fs::read_dir(&reports_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(entries.len(), 1, "should have one report");
}

#[test]
fn test_report_generate_with_db() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.axil");

    // Init a database first
    run_axil_in(dir.path(), &["init", db_path.to_str().unwrap()]);

    // Generate report with --db
    let (stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["--db", db_path.to_str().unwrap(), "report", "generate"],
    );
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["generated"], true);

    // Read the report and verify database info
    let report_path = json["path"].as_str().unwrap();
    let report_str = std::fs::read_to_string(report_path).unwrap();
    let report: Value = serde_json::from_str(&report_str).unwrap();

    assert_eq!(report["version"], "1.0");
    assert!(report["axil_version"].is_string());
    assert!(report["environment"]["os"].is_string());
    assert!(report["database"].is_object());
    assert!(
        report["database"] != Value::Null,
        "database info should be present"
    );
}

#[test]
fn test_report_list_empty() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["report", "list"]);
    assert_eq!(code, 0);

    let json: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[test]
fn test_report_list_after_generate() {
    let dir = tempfile::TempDir::new().unwrap();

    // Generate a report
    run_axil_in(dir.path(), &["report", "generate"]);

    // List reports
    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["report", "list"]);
    assert_eq!(code, 0);

    let json: Value = serde_json::from_str(stdout.trim()).unwrap();
    let reports = json.as_array().unwrap();
    assert_eq!(reports.len(), 1);
    assert!(reports[0]["filename"]
        .as_str()
        .unwrap()
        .starts_with("report-"));
    assert!(reports[0]["size_bytes"].as_u64().unwrap() > 0);
}

#[test]
fn test_report_import() {
    let source_dir = tempfile::TempDir::new().unwrap();
    let dest_dir = tempfile::TempDir::new().unwrap();

    // Generate a report in the source project
    run_axil_in(source_dir.path(), &["report", "generate"]);

    // Import it into the destination (Axil source repo)
    let (stdout, _stderr, code) = run_axil_in(
        dest_dir.path(),
        &[
            "report",
            "import",
            "--from",
            source_dir.path().to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "report import should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["imported"], true);

    // Verify report was copied
    let incoming = dest_dir.path().join("reports").join("incoming");
    assert!(incoming.exists());
    let entries: Vec<_> = std::fs::read_dir(&incoming)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(entries.len(), 1, "should have imported one report");
}

#[test]
fn test_report_import_direct_path() {
    let source_dir = tempfile::TempDir::new().unwrap();
    let dest_dir = tempfile::TempDir::new().unwrap();

    // Create a report file manually
    let report_path = source_dir.path().join("test-report.json");
    std::fs::write(&report_path, r#"{"version":"1.0","problems":[]}"#).unwrap();

    // Import by direct path
    let (stdout, _stderr, code) = run_axil_in(
        dest_dir.path(),
        &["report", "import", report_path.to_str().unwrap()],
    );
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["imported"], true);
}

#[test]
fn test_report_import_invalid_json() {
    let dir = tempfile::TempDir::new().unwrap();

    // Create an invalid JSON file
    let bad_path = dir.path().join("bad.json");
    std::fs::write(&bad_path, "not valid json").unwrap();

    let (_stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["report", "import", bad_path.to_str().unwrap()],
    );
    assert_eq!(code, 1, "should fail for invalid JSON");
}

// ─── Skill ──────────────────────────────────────────────────────────────────

#[test]
fn test_skill_list_shows_all() {
    let dir = tempfile::TempDir::new().unwrap();

    let (stdout, _stderr, code) = run_axil_in(dir.path(), &["skill", "list"]);
    assert_eq!(code, 0);

    let json: Value = serde_json::from_str(stdout.trim()).unwrap();
    let skills = json.as_array().unwrap();
    assert_eq!(skills.len(), 8, "should list 8 skills");

    let names: Vec<&str> = skills.iter().map(|s| s["name"].as_str().unwrap()).collect();
    for expected in [
        "memory", "report", "diagnose", "optimize", "autoagent", "learn", "retro", "brief",
    ] {
        assert!(names.contains(&expected), "skill list missing `{expected}`");
    }
}

#[test]
fn test_skill_install_and_uninstall() {
    // Use a custom HOME to avoid polluting the real home dir
    let fake_home = tempfile::TempDir::new().unwrap();
    let skills_dir = fake_home.path().join(".claude").join("skills");

    let output = Command::new(axil_bin())
        .env("HOME", fake_home.path())
        .args(["skill", "install"])
        .output()
        .expect("failed to execute axil");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    assert_eq!(code, 0, "skill install should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["installed"], 8);

    // Verify files exist
    assert!(skills_dir.join("axil.md").exists());
    assert!(skills_dir.join("axil-report.md").exists());
    assert!(skills_dir.join("axil-diagnose.md").exists());
    assert!(skills_dir.join("axil-optimize.md").exists());
    assert!(skills_dir.join("axil-autoagent.md").exists());

    // List should show installed
    let output = Command::new(axil_bin())
        .env("HOME", fake_home.path())
        .args(["skill", "list"])
        .output()
        .expect("failed to execute axil");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let json: Value = serde_json::from_str(stdout.trim()).unwrap();
    for skill in json.as_array().unwrap() {
        assert_eq!(skill["installed"], true, "all skills should be installed");
    }

    // Uninstall
    let output = Command::new(axil_bin())
        .env("HOME", fake_home.path())
        .args(["skill", "uninstall"])
        .output()
        .expect("failed to execute axil");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    assert_eq!(code, 0, "skill uninstall should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["removed"], 8);

    // Verify files removed
    assert!(!skills_dir.join("axil.md").exists());
    assert!(!skills_dir.join("axil-report.md").exists());
}

#[test]
fn test_skill_install_only_one() {
    let fake_home = tempfile::TempDir::new().unwrap();
    let skills_dir = fake_home.path().join(".claude").join("skills");

    let output = Command::new(axil_bin())
        .env("HOME", fake_home.path())
        .args(["skill", "install", "--only", "memory"])
        .output()
        .expect("failed to execute axil");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["installed"], 1);

    assert!(skills_dir.join("axil.md").exists());
    assert!(!skills_dir.join("axil-report.md").exists());
}

#[test]
fn test_skill_install_invalid_name() {
    let fake_home = tempfile::TempDir::new().unwrap();

    let output = Command::new(axil_bin())
        .env("HOME", fake_home.path())
        .args(["skill", "install", "--only", "nonexistent"])
        .output()
        .expect("failed to execute axil");
    let code = output.status.code().unwrap_or(-1);
    assert_eq!(code, 1, "should fail for invalid skill name");
}

// ─── Config Precedence ─────────────────────────────────────────────────────

#[test]
fn test_config_walks_up_directories() {
    let dir = tempfile::TempDir::new().unwrap();
    let subdir = dir.path().join("sub").join("deep");
    std::fs::create_dir_all(&subdir).unwrap();

    // Place config in parent
    std::fs::write(
        dir.path().join("axil.toml"),
        "[debug]\nlog_level = \"debug\"\n",
    )
    .unwrap();

    // Run from subdirectory — should find parent's config
    let (stdout, _stderr, code) = run_axil_in(&subdir, &["config", "get", "debug.log_level"]);
    assert_eq!(code, 0);

    let json = parse_json(&stdout);
    assert_eq!(json["value"], "debug");
}

// ─── Skill files valid markdown ─────────────────────────────────────────────

#[test]
fn test_skill_files_have_frontmatter() {
    let skills_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("skills");

    for name in &[
        "axil.md",
        "axil-report.md",
        "axil-diagnose.md",
        "axil-optimize.md",
        "generic-agent.md",
    ] {
        let path = skills_dir.join(name);
        assert!(path.exists(), "skill file should exist: {name}");

        let content = std::fs::read_to_string(&path).unwrap();
        // `lines()` strips a trailing `\r`, so this holds for both LF
        // and CRLF skill files (Windows checkouts use CRLF).
        assert!(
            content.lines().next() == Some("---"),
            "{name} should start with YAML frontmatter"
        );
        assert!(content.contains("name:"), "{name} should have a name field");
        assert!(
            content.contains("description:"),
            "{name} should have a description field"
        );
    }
}

// ─── Report JSON Schema ────────────────────────────────────────────────────

#[test]
fn test_report_json_schema() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.axil");

    // Init a database
    run_axil_in(dir.path(), &["init", db_path.to_str().unwrap()]);

    // Generate report
    let (stdout, _stderr, code) = run_axil_in(
        dir.path(),
        &["--db", db_path.to_str().unwrap(), "report", "generate"],
    );
    assert_eq!(code, 0);

    let gen = parse_json(&stdout);
    let report_path = gen["path"].as_str().unwrap();
    let report_str = std::fs::read_to_string(report_path).unwrap();
    let report: Value = serde_json::from_str(&report_str).unwrap();

    // Verify required fields
    assert_eq!(report["version"], "1.0");
    assert!(report["generated_at"].is_string());
    assert!(report["axil_version"].is_string());

    // Environment
    assert!(report["environment"]["os"].is_string());
    assert!(report["environment"]["arch"].is_string());
    assert!(report["environment"]["features"].is_array());

    // Database
    assert!(report["database"].is_object());
    assert!(report["database"]["record_count"].is_number());

    // Problems array exists
    assert!(report["problems"].is_array());
}
