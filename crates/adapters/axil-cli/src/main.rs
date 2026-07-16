use std::io::{self, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};

use axil_core::{Axil, Direction, Op, RecordId, SortDirection};

#[cfg(feature = "http")]
mod http_server;

mod features;
mod hook_brain;
mod install_wizard;
#[cfg(feature = "scip")]
mod scip_detect;
mod workspace;

#[cfg(feature = "vector")]
use axil_vector::AxilBuilderVectorExt;

#[cfg(feature = "graph")]
use axil_graph::AxilBuilderGraphExt;

#[cfg(feature = "fts")]
use axil_fts::AxilBuilderFtsExt;

#[cfg(feature = "timeseries")]
use axil_timeseries::AxilBuilderTimeSeriesExt;

#[cfg(feature = "wasm-host")]
mod wasm_plugins;

// ─── Exit codes ──────────────────────────────────────────────────────────────

const EXIT_OK: i32 = 0;
const EXIT_ERROR: i32 = 1;
const EXIT_NOT_FOUND: i32 = 2;
const EXIT_BENCH_REGRESSION: i32 = 3;

// ─── Progress reporting (indicatif on stderr) ────────────────────────────────

/// Indicatif-backed `IndexProgress` for `axil index` and the bootstrap path.
///
/// Renders a determinate bar to stderr while the indexer walks files, then
/// switches to a spinner for non-file phases (modules / symbols / proxies).
/// Only attached when stderr is a TTY and `--quiet` is off — see
/// `make_index_progress`.
#[cfg(feature = "indexer")]
struct IndicatifIndexProgress {
    bar: indicatif::ProgressBar,
}

#[cfg(feature = "indexer")]
impl IndicatifIndexProgress {
    fn new() -> Self {
        // Hidden until `start(total)` upgrades it to a bar with known length.
        // Drawing on stderr keeps stdout clean for JSON output.
        let bar = indicatif::ProgressBar::hidden();
        bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
        Self { bar }
    }
}

#[cfg(feature = "indexer")]
impl axil_indexer::IndexProgress for IndicatifIndexProgress {
    fn start(&self, total_files: usize) {
        self.bar.set_length(total_files as u64);
        self.bar.set_style(
            indicatif::ProgressStyle::with_template("  {bar:30.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        self.bar.set_message("indexing files...");
    }
    fn file_indexed(&self, _idx: usize, path: &str) {
        self.bar.set_message(path.to_string());
        self.bar.inc(1);
    }
    fn phase(&self, name: &str) {
        self.bar
            .set_style(indicatif::ProgressStyle::with_template("  {spinner:.cyan} {msg}").unwrap());
        self.bar.set_message(format!("{name}..."));
        self.bar
            .enable_steady_tick(std::time::Duration::from_millis(120));
    }
    fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

/// Build a progress reporter when stderr is a TTY and the user hasn't
/// passed `--quiet`. Falls back to `NoopProgress` otherwise so JSON
/// output stays clean in CI / pipelines.
#[cfg(feature = "indexer")]
fn make_index_progress(quiet: bool) -> Box<dyn axil_indexer::IndexProgress> {
    use std::io::IsTerminal;
    if quiet || !std::io::stderr().is_terminal() {
        Box::new(axil_indexer::NoopProgress)
    } else {
        Box::new(IndicatifIndexProgress::new())
    }
}

/// Run the SCIP half of `axil reindex` by re-execing `axil scip refresh`.
///
/// Re-execing (rather than calling the refresh logic inline) reuses scip
/// refresh's detached-spawn, lock-file (`.axil/scip-refresh.lock`), and
/// per-project staleness machinery instead of duplicating ~200 lines.
/// Background by default (`--in-background`, returns instantly after the
/// detached worker is spawned); `--wait` runs it to completion. `--full`
/// drops `--if-stale` so the refresh is unconditional. `root` pins the
/// SCIP scan to the same tree the proxy index used (via `--root`), so the
/// two layers can't refresh different directories when the DB lives outside
/// the scanned project. Returns a JSON status object embedded under `scip`
/// in the combined `reindex` output.
#[cfg(all(feature = "indexer", feature = "scip"))]
fn reindex_scip(db_path: &Path, root: &Path, full: bool, wait: bool, quiet: bool) -> Value {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return json!({ "status": "error", "reason": format!("current_exe: {e}") }),
    };
    let db = db_path.display().to_string();
    let root_str = root.display().to_string();
    // Pin scip to the same tree the proxy index scanned. Without this the
    // child infers its root from the DB location, which can differ from
    // `path` when `--db` points outside the project being indexed.
    let mut args: Vec<&str> = vec!["--db", &db, "scip", "refresh", "--root", &root_str];
    if !full {
        // --full means "refresh unconditionally"; otherwise only when stale.
        args.push("--if-stale");
    }
    if !wait {
        // Background by default so the command returns quickly.
        args.push("--in-background");
    }
    if quiet {
        args.push("--quiet");
    }
    match std::process::Command::new(&exe).args(&args).output() {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            match serde_json::from_str::<Value>(stdout.trim()) {
                // --wait: child ran to completion; surface its full report
                // (which already carries `ok`/`skipped`/`refreshed`).
                Ok(v) if wait => v,
                // Background: the child may have spawned a detached worker OR
                // legitimately skipped (fresh, indexer missing, already
                // running). Reflect which actually happened instead of always
                // reporting "spawned", so callers can tell if graph work began.
                Ok(v) => {
                    let skipped = v.get("skipped").and_then(Value::as_bool).unwrap_or(false);
                    let spawned = v.get("spawned").and_then(Value::as_bool).unwrap_or(false);
                    let status = if skipped {
                        "skipped"
                    } else if spawned {
                        "spawned"
                    } else {
                        "ok"
                    };
                    let reason = v.get("reason").cloned();
                    let mut wrapped = json!({ "status": status, "detail": v });
                    if let Some(r) = reason {
                        wrapped["reason"] = r;
                    }
                    wrapped
                }
                Err(_) => json!({
                    "status": if o.status.success() { "ok" } else { "error" },
                    "stderr": String::from_utf8_lossy(&o.stderr).trim(),
                }),
            }
        }
        Err(e) => json!({ "status": "error", "reason": format!("spawn scip refresh: {e}") }),
    }
}

/// SCIP feature compiled out — `reindex` still rebuilds the proxy index and
/// reports the skip so the combined JSON shape stays stable.
#[cfg(all(feature = "indexer", not(feature = "scip")))]
fn reindex_scip(_db_path: &Path, _full: bool, _wait: bool, _quiet: bool) -> Value {
    json!({ "status": "skipped", "reason": "scip_feature_disabled" })
}

// ─── Output formatting ──────────────────────────────────────────────────────

#[derive(Clone, ValueEnum, Default)]
enum OutputFormat {
    /// Compact JSON (default — for agents).
    #[default]
    Json,
    /// Pretty-printed JSON (for humans).
    Pretty,
    /// Human-readable table (for terminal use).
    Table,
}

/// Recall output format.
#[derive(Clone, ValueEnum, Default)]
enum RecallFormat {
    /// Full JSON (default).
    #[default]
    Full,
    /// Compact: id, score, summary only, nulls stripped.
    Compact,
    /// One line per result: score | summary | id.
    Oneline,
    /// XML context block: <context>...</context> with source citations (12.1, for UserPromptSubmit hook).
    ContextBlock,
}

/// Brief/retro output format (12.3).
#[derive(Clone, ValueEnum, Default)]
enum BriefFormat {
    #[default]
    Markdown,
    Json,
}

/// Rerank mode (12.4). Opt-in — default is `off` so the query engine stays fast.
#[derive(Clone, ValueEnum, Default, PartialEq, Eq)]
enum RerankMode {
    /// No reranking — use RRF fusion score only (default).
    #[default]
    Off,
    /// Cross-encoder ONNX model (requires `rerank` feature + local model).
    CrossEncoder,
    /// LLM-based reranking through the configured LlmProvider (requires `llm` config).
    Llm,
}

/// Boot context output format.
#[derive(Clone, ValueEnum, Default)]
enum BootFormat {
    /// Structured JSON (default).
    #[default]
    Json,
    /// Plain text narrative.
    Narrative,
    /// Compact JSON (minimal tokens).
    Compact,
}

/// Centralized output helper. All command output goes through here.
struct Output {
    format: OutputFormat,
    quiet: bool,
    jsonl: bool,
}

impl Output {
    /// Print a single JSON value to stdout.
    fn print(&self, v: &Value) {
        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string(v).expect("JSON serialization"))
            }
            OutputFormat::Pretty => println!(
                "{}",
                serde_json::to_string_pretty(v).expect("JSON serialization")
            ),
            OutputFormat::Table => print_table_object(v),
        }
    }

    /// Print an array of values — respects --jsonl for streaming output.
    fn print_array(&self, values: &[Value]) {
        if self.jsonl {
            for v in values {
                println!("{}", serde_json::to_string(v).expect("JSON serialization"));
            }
        } else {
            match self.format {
                OutputFormat::Table => print_table_rows(values),
                // Serialize the slice directly to avoid cloning into Value::Array.
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string(values).expect("JSON serialization")
                ),
                OutputFormat::Pretty => println!(
                    "{}",
                    serde_json::to_string_pretty(values).expect("JSON serialization")
                ),
            }
        }
    }

    /// Print a status message to stderr (suppressed by --quiet).
    fn status(&self, msg: &str) {
        if !self.quiet {
            eprintln!("{msg}");
        }
    }
}

/// Print a single JSON object as key: value lines.
fn print_table_object(v: &Value) {
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            println!("{:<20} {}", k, format_table_value(val));
        }
    } else {
        println!("{}", serde_json::to_string_pretty(v).unwrap());
    }
}

/// Print an array of JSON objects as an aligned table.
fn print_table_rows(values: &[Value]) {
    if values.is_empty() {
        println!("(no results)");
        return;
    }

    // Collect all unique column names in order of first appearance.
    let mut columns: Vec<String> = Vec::new();
    for v in values {
        if let Some(obj) = v.as_object() {
            for k in obj.keys() {
                if !columns.contains(k) {
                    columns.push(k.clone());
                }
            }
        }
    }

    // Compute column widths (header vs. data).
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for v in values {
        let mut row = Vec::new();
        for (i, col) in columns.iter().enumerate() {
            let cell = v.get(col).map(format_table_value).unwrap_or_default();
            widths[i] = widths[i].max(cell.len());
            row.push(cell);
        }
        rows.push(row);
    }

    // Cap column widths at 60 chars for readability.
    for w in &mut widths {
        if *w > 60 {
            *w = 60;
        }
    }

    // Print header.
    let header: Vec<String> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{:<width$}", c, width = widths[i]))
        .collect();
    println!("{}", header.join("  "));
    let separator: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    println!("{}", separator.join("  "));

    // Print rows.
    for row in &rows {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let w = widths[i];
                if cell.len() > w {
                    // Truncate at a char boundary to avoid panicking on multibyte UTF-8.
                    let truncated: String = cell.chars().take(w.saturating_sub(1)).collect();
                    format!("{truncated}…")
                } else {
                    format!("{:<width$}", cell, width = w)
                }
            })
            .collect();
        println!("{}", cells.join("  "));
    }
}

/// Format a JSON value for table display.
fn format_table_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => format!("[{} items]", a.len()),
        Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
    }
}

// ─── CLI definition ─────────────────────────────────────────────────────────

/// Shown at the bottom of `axil --help`. clap lists subcommands as a flat
/// alphabetical block; this groups the most-used ones by purpose so the
/// diagnostics / lifecycle / automation families are easy to tell apart.
const COMMANDS_OVERVIEW: &str = "\
Commands by purpose:

  Write & recall:
    store / checkpoint    save a record / a resume-able session checkpoint
    recall, code-search, fts, boot    read memory back

  Diagnostics (read-only — watch the DB, never change memories):
    doctor          quick \"is anything wrong right now?\" check
    health-report   scored health assessment + fix recommendations
    snapshot        record metrics for trend charts (NOT a data backup)
    trends          chart the metric history snapshot collects
    detect          deep / expensive problem scan

  Memory lifecycle (change or forget memories):
    compact         delete expired/superseded rows, reclaim space
    heal            rebuild indexes (--reindex) or downsample old records
    worker          decay, consolidation, inference (run by the Stop hook)

  Automation:
    maintain        run the diagnostics (snapshot, health-report) on a cadence

When to use which:
    something feels off             -> axil doctor
    want a scored checkup           -> axil health-report
    DB grew / slow, reclaim space   -> axil compact
    index returns wrong/empty hits  -> axil heal --reindex
    (recurring upkeep is automatic via the brain hook -> axil maintain)
";

#[derive(Parser)]
#[command(
    name = "axil",
    about = "One file. One binary. Built for agents.",
    version,
    after_help = COMMANDS_OVERVIEW
)]
struct Cli {
    /// Database path (or set AXIL_DB env var).
    #[arg(long, env = "AXIL_DB", global = true)]
    db: Option<PathBuf>,

    /// Output format: json (default) or pretty.
    #[arg(long, global = true, default_value = "json")]
    format: OutputFormat,

    /// Suppress non-essential output.
    #[arg(long, global = true)]
    quiet: bool,

    /// JSON Lines mode: one JSON object per line instead of array.
    #[arg(long, global = true)]
    jsonl: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, ValueEnum)]
enum CliDirection {
    Out,
    In,
    Both,
}

impl From<CliDirection> for Direction {
    fn from(d: CliDirection) -> Self {
        match d {
            CliDirection::Out => Direction::Out,
            CliDirection::In => Direction::In,
            CliDirection::Both => Direction::Both,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum CliSortDirection {
    Asc,
    Desc,
}

impl From<CliSortDirection> for SortDirection {
    fn from(d: CliSortDirection) -> Self {
        match d {
            CliSortDirection::Asc => SortDirection::Asc,
            CliSortDirection::Desc => SortDirection::Desc,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    // ── Database management (4b.6) ──────────────────────────────────
    /// Create a new .axil database with all features enabled.
    #[command(visible_alias = "create")]
    Init {
        /// Path for the new database (overrides --db).
        path: Option<PathBuf>,
        /// Vector dimensions (default: 384 for bge-small; use 768 for bge-base/nomic).
        #[cfg(feature = "vector")]
        #[arg(long, default_value = "384")]
        vector_dims: usize,
    },

    /// Install Axil agent memory in the current project.
    ///
    /// Creates a .axil/ directory with the memory database, optionally
    /// configures AI agent integration (Claude Code, generic agents).
    /// Run bare on a terminal for an interactive wizard that detects the
    /// project's agent tooling; any flag (or piped stdin) keeps the
    /// non-interactive behavior for scripts and CI.
    #[command(name = "install")]
    InstallProject {
        /// Also configure Claude Code integration (skills + hook).
        #[arg(long)]
        claude_code: bool,
        /// Configure Cursor integration (.cursor/rules).
        #[arg(long)]
        cursor: bool,
        /// Configure Windsurf integration (.windsurfrules).
        #[arg(long)]
        windsurf: bool,
        /// Deprecated no-op — Sourcegraph discontinued Cody in 2025; its
        /// successor Amp reads the AGENTS.md contract written by default.
        #[arg(long, hide = true)]
        cody: bool,
        /// Configure Aider integration (CONVENTIONS.md + .aider.conf.yml read list).
        #[arg(long)]
        aider: bool,
        /// Configure OpenAI Codex integration (.codex/hooks.json + project
        /// MCP + .agents/skills). AGENTS.md is written by default anyway.
        #[arg(long)]
        codex: bool,
        /// Configure GitHub Copilot CLI integration (.github/hooks/axil.json
        /// + ~/.copilot/mcp-config.json).
        #[arg(long)]
        copilot: bool,
        /// Configure Factory Droid integration (.factory/hooks.json +
        /// .factory/mcp.json).
        #[arg(long)]
        droid: bool,
        /// Configure Google Antigravity integration (.agents/ rules + skills
        /// + hooks + mcp_config.json).
        #[arg(long)]
        antigravity: bool,
        /// Configure Qwen Code integration (.qwen/settings.json hooks + MCP).
        #[arg(long)]
        qwen: bool,
        /// Configure OpenCode integration (.opencode/plugins/axil.ts + MCP).
        #[arg(long)]
        opencode: bool,
        /// Skip the AGENTS.md managed block. It is written by default because
        /// AGENTS.md is the cross-tool contract read by Codex, OpenCode,
        /// Qwen Code, Copilot, Droid, and most other agents.
        #[arg(long)]
        no_agents_md: bool,
        /// Install for all detected AI agents in the project.
        #[arg(long)]
        all: bool,
        /// Also print setup instructions for a generic AI agent.
        #[arg(long, value_name = "AGENT")]
        agent: Option<String>,
        /// Vector dimensions (default: 384 for bge-small; use 768 for bge-base/nomic).
        #[cfg(feature = "vector")]
        #[arg(long, default_value = "384")]
        vector_dims: usize,
        /// Print every file/entry that would be written without touching the filesystem (12.1).
        #[arg(long)]
        dry_run: bool,
        /// Remove Axil-owned hook entries from `.claude/settings.json` and delete hook scripts + skills.
        /// Preserves the database and non-Axil settings. Use before uninstalling the binary (12.1).
        #[arg(long)]
        uninstall: bool,
        /// One-shot setup: after the normal install, also run `axil index .` to build
        /// structural code proxies and kick off `axil scip refresh --in-background`
        /// to populate the SCIP cross-reference graph if a language indexer is on PATH.
        /// Skips gracefully when the corresponding features/indexers are unavailable.
        #[arg(long)]
        bootstrap: bool,
        /// Install skills into `<project>/.claude/skills/` instead of `~/.claude/skills/`.
        /// Use this for self-contained, repo-local installs that don't pollute the
        /// global skills directory and travel with the project (commit them or gitignore).
        #[arg(long)]
        local: bool,
    },

    /// Generate a daily brief: summary of recent memory, open threads, top-of-mind (12.3).
    ///
    /// Designed to replace manual standup / meeting prep. Synthesizes sessions,
    /// decisions, errors, and patterns from the given window into a 2-minute read.
    Brief {
        /// Time window (e.g. 24h, 3d, 7d). Default 24h.
        #[arg(long, default_value = "24h")]
        window: String,
        /// Only include records after this ISO 8601 date (overrides --window).
        #[arg(long)]
        after: Option<String>,
        /// Output format.
        #[arg(long, default_value = "markdown")]
        brief_format: BriefFormat,
        /// Token budget for the generated brief.
        #[arg(long)]
        budget: Option<usize>,
    },

    /// Run a retrospective: longer-horizon review, writes a durable report file (12.3).
    Retro {
        /// Time window (e.g. 7d, 30d, 90d). Default 30d.
        #[arg(long, default_value = "30d")]
        window: String,
        /// Output format.
        #[arg(long, default_value = "markdown")]
        brief_format: BriefFormat,
        /// Write the retro to .axil/reports/retro-YYYY-MM.md and store a context record.
        #[arg(long)]
        save: bool,
    },

    /// Manage scheduled axil tasks (daily briefs, weekly retros) via launchd / systemd / cron (12.3).
    Schedule {
        #[command(subcommand)]
        op: ScheduleOp,
    },

    /// Bulk-ingest markdown/text notes from a filesystem directory into memory (12.2).
    ///
    /// Walks `<DIR>`, chunks each matching file, and stores chunks as records.
    /// Embeddings, entity extraction, and importance scoring run automatically
    /// per the brain pipeline. Unchanged files are skipped via content-hash.
    /// State is checkpointed to `.axil/ingest.state.json` so interrupted runs
    /// resume cleanly with `--resume`.
    ///
    /// With `--watch`, the same incremental ingest re-runs every `--interval`
    /// seconds; because unchanged files are skipped via content-hash, each tick
    /// only ingests new/changed files. Runs until interrupted (Ctrl-C).
    #[cfg(feature = "indexer")]
    Ingest {
        /// Directory to scan.
        dir: PathBuf,
        /// Recurse into subdirectories (default: true).
        #[arg(long, default_value = "true")]
        recursive: bool,
        /// Comma-separated extensions to include (default: md,txt,org,markdown).
        #[arg(long = "ext", default_value = "md,txt,org,markdown")]
        ext: String,
        /// Path substrings to exclude. Repeatable. Match is case-sensitive `path.contains(pattern)`
        /// after stripping leading/trailing `*` and `/`, e.g. `--exclude drafts` skips any path
        /// containing "drafts". Not a full glob matcher.
        #[arg(long)]
        exclude: Vec<String>,
        /// Table name for ingested records (default: notes).
        #[arg(long, default_value = "notes")]
        table: String,
        /// Only print what would happen; do not write.
        #[arg(long)]
        stats: bool,
        /// Resume an interrupted run from `.axil/ingest.state.json`.
        #[arg(long)]
        resume: bool,
        /// Max chunk size in bytes (default: 2000).
        #[arg(long, default_value = "2000")]
        chunk_bytes: usize,
        /// Re-run the incremental ingest on an interval (Ctrl-C to stop).
        /// Each tick re-scans `<DIR>` and ingests only new/changed files
        /// (unchanged files are skipped via content-hash). Mutually implies
        /// `--resume` so prior state is reused across ticks.
        #[arg(long)]
        watch: bool,
        /// Seconds between watch ticks (default: 2). Only used with `--watch`.
        #[arg(long, default_value = "2")]
        interval: u64,
    },
    /// Ingest a SCIP code-index file into Axil's graph.
    ///
    /// Parses `<PATH>` (protobuf, typically `index.scip` produced by
    /// `scip-rust`, `scip-python`, `scip-typescript`, …), then emits
    /// symbol-level entities and edges: `defined_in`, `references`,
    /// `implements`, `type_of` (direct) + `calls`, `imports` (heuristic).
    /// Idempotent — safe to re-run on the same file.
    ///
    /// When `<PATH>` is omitted, looks for `.axil/index.scip` first
    /// (preferred — keeps SCIP alongside the DB), then the per-language
    /// `.axil/index-<lang>*.scip` set a polyglot `scip refresh` writes
    /// (ingests all of them), then `./index.scip`.
    #[cfg(feature = "scip")]
    #[command(name = "ingest-scip")]
    IngestScip {
        /// Path to the SCIP protobuf file. Defaults to `.axil/index.scip`,
        /// then `.axil/index-<lang>*.scip` (all), then `./index.scip`.
        path: Option<PathBuf>,
        /// Parse and count edges without writing.
        #[arg(long)]
        dry_run: bool,
        /// Watch the SCIP file and re-ingest on change, using a
        /// size/mtime stabilization gate to avoid reading a partial write.
        #[arg(long)]
        watch: bool,
    },

    /// Manage the SCIP code-graph index (`axil scip refresh` / `status`).
    ///
    /// `refresh` runs the appropriate language indexer (rust-analyzer,
    /// scip-typescript, scip-python, scip-go, scip-java) and immediately
    /// ingests the result — closing the discoverability gap doctor's
    /// `fix` field leaves open.
    #[cfg(feature = "scip")]
    #[command(subcommand)]
    Scip(ScipCommand),

    /// Manage built-in Extensions: list them and toggle them on or off in
    /// axil.toml — no rebuild required.
    ///
    /// A disabled Extension is skipped at registration: its CLI subcommands,
    /// MCP tools, and boot block all vanish until it is re-enabled.
    #[command(subcommand)]
    Extensions(ExtensionsCommand),

    /// Manage runtime WASM plugins in `.axil/plugins/` — install, list, or
    /// remove a `.wasm` component with no rebuild and no fork.
    #[cfg(feature = "wasm-host")]
    #[command(subcommand)]
    Ext(ExtCommand),

    /// Catch-all for a command no built-in subcommand claims: routed to
    /// whichever registered Extension owns it via generic Path-C dispatch
    /// (`dispatch_cli`). This is what lets a CLI-facing Extension need zero
    /// code in `axil-cli` — it declares a `CliSurface` + `handle_cli`,
    /// registers in the bundle, and its command works here automatically.
    #[command(external_subcommand)]
    External(Vec<String>),

    /// Dependency documentation memory.
    ///
    /// `deps list` resolves the project's dependencies to their exact
    /// lockfile versions — the foundation for version-pinned library
    /// docs in agent memory.
    #[cfg(feature = "deps")]
    #[command(subcommand)]
    Deps(DepsCommand),

    /// Query the dependency documentation memory.
    ///
    /// Returns version-pinned doc chunks for the project's
    /// dependencies. Run `axil deps sync` first to populate them.
    #[cfg(feature = "deps")]
    #[command(name = "dep-docs")]
    DepDocs {
        /// The library question to search for.
        query: String,
        /// Restrict results to a single dependency by name.
        #[arg(long)]
        dep: Option<String>,
        /// Maximum number of doc chunks to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Also return docs for superseded / removed versions, which
        /// are excluded by default.
        #[arg(long)]
        include_superseded: bool,
    },

    /// Write or read a structured session checkpoint so a fresh agent can
    /// pick up where the last session left off.
    ///
    /// Common shapes:
    ///   axil checkpoint '{"goal":"…","next_steps":["…"]}'
    ///   echo '{…}' | axil checkpoint -
    ///   axil checkpoint show
    ///
    /// Mid-session by default: writes a snapshot without ending the
    /// owning session. Pass `--final` to mark this as the session's
    /// final checkpoint.
    #[cfg(feature = "checkpoint")]
    Checkpoint {
        /// Inline JSON payload, `-` for stdin, or the literal `show`
        /// to print the current checkpoint (stored or derived).
        arg: Option<String>,
        /// Attach to a specific session id instead of the latest active.
        #[arg(long)]
        session: Option<String>,
        /// Stamp this checkpoint as the final one for its session.
        #[arg(long = "final")]
        is_final: bool,
    },

    /// Reuse a cached answer when a semantically similar question recurs,
    /// with code-aware invalidation.
    ///
    /// `axil cache put '{"question":"…","answer":"…"}'` stores a pair;
    /// `axil cache get "<question>"` returns a hit or an explained miss.
    #[cfg(feature = "cache")]
    #[command(subcommand)]
    Cache(CacheCommand),

    /// Resolve a display name to a canonical id via scoped aliases.
    /// Distinct from `entity-resolve` (which does fuzzy/strategy
    /// disambiguation against natural-language aliases) — this walks
    /// `_scip_aliases` narrowest-first.
    #[cfg(feature = "scip")]
    #[command(name = "entity-resolve-scoped")]
    EntityResolveScoped {
        /// Display name to resolve.
        name: String,
        /// Scopes to walk, narrowest-first. Repeatable.
        /// Example: `--scope file:src/auth.rs --scope lang:rust --scope global`.
        #[arg(long = "scope")]
        scopes: Vec<String>,
    },

    /// Merge two canonical entity ids (explicit, never auto).
    ///
    /// Moves aliases and graph edges from `<FROM>` onto `<TO>`, then
    /// tombstones the source entity row. SCIP ingest will not auto-merge
    /// on ambiguity — use this to make the call explicit.
    #[cfg(feature = "scip")]
    #[command(name = "entity-merge-canonical")]
    EntityMergeCanonical {
        /// Source canonical id (merged away).
        from: String,
        /// Target canonical id (keeper).
        to: String,
    },

    /// Update agent integration files (hooks, skills, instructions) to latest version.
    ///
    /// Use after upgrading axil to get the latest hook scripts, CLAUDE.md template,
    /// and skill files. Only updates agent files — does not modify the database.
    #[command(name = "sync", visible_alias = "refresh")]
    UpdateProject {
        /// Update Claude Code integration.
        #[arg(long)]
        claude_code: bool,
        /// Update Cursor integration.
        #[arg(long)]
        cursor: bool,
        /// Update Windsurf integration.
        #[arg(long)]
        windsurf: bool,
        /// Deprecated no-op — Cody was discontinued; see `axil install --help`.
        #[arg(long, hide = true)]
        cody: bool,
        /// Update Aider integration.
        #[arg(long)]
        aider: bool,
        /// Update Codex integration (hooks + MCP + skills).
        #[arg(long)]
        codex: bool,
        /// Update Copilot CLI integration (hooks + MCP).
        #[arg(long)]
        copilot: bool,
        /// Update Factory Droid integration (hooks + MCP).
        #[arg(long)]
        droid: bool,
        /// Update Google Antigravity integration (rules + skills + hooks + MCP).
        #[arg(long)]
        antigravity: bool,
        /// Update Qwen Code integration (hooks + MCP).
        #[arg(long)]
        qwen: bool,
        /// Update OpenCode integration (plugin + MCP).
        #[arg(long)]
        opencode: bool,
        /// Update all detected agent integrations.
        #[arg(long)]
        all: bool,
    },

    /// Show database statistics as JSON.
    Info,

    /// List tables with record counts.
    Tables,

    /// Show which optional components (Engines, Extensions, Adapters) this
    /// binary was compiled with. Features are compile-time, so changing them
    /// means a rebuild — `--wizard` composes that command interactively.
    Features {
        /// Interactive picker: toggle components, then emit (and optionally
        /// run) the matching `cargo install` command.
        #[arg(long)]
        wizard: bool,
    },

    // ── Diagnostics (5b) ───────────────────────────────────────────────
    /// Quick read-only health check — "is anything wrong right now?".
    /// For a scored report + fix recommendations use `health-report`.
    Doctor,

    /// Deep problem scan — the expensive detectors `doctor` skips
    /// (stale sessions, slow queries, storage growth, embedding drift).
    Detect,

    /// Show memory pressure status: tier distribution, archive candidates, and DB size.
    #[command(name = "memory-pressure")]
    MemoryPressure {
        /// Auto-archive records below the archive threshold.
        #[arg(long)]
        archive: bool,
    },

    /// Apply importance decay and show records below archive threshold.
    Decay {
        /// Show what would be affected without updating.
        #[arg(long)]
        dry_run: bool,
    },

    /// Show or set importance score for a record.
    Importance {
        /// Record ID.
        id: String,
        /// Pin importance to 1.0 (never decays).
        #[arg(long)]
        pin: bool,
        /// Unpin (recompute importance from content).
        #[arg(long)]
        unpin: bool,
    },

    /// Show comprehensive database statistics.
    Stats {
        /// Stats for a specific table only.
        #[arg(long)]
        table: Option<String>,
        /// Refresh every N seconds.
        #[arg(long)]
        watch: Option<u64>,
        /// Show activation-level distribution.
        #[arg(long)]
        activation: bool,
    },

    /// Run built-in micro-benchmarks.
    Bench {
        /// Save results for later comparison.
        #[arg(long)]
        save: bool,
        /// Compare with last saved benchmark.
        #[arg(long)]
        compare: bool,
    },

    /// View slow query log.
    SlowQueries {
        /// Maximum number of entries.
        #[arg(long)]
        limit: Option<usize>,
        /// Show entries after this date.
        #[arg(long)]
        after: Option<String>,
        /// Clear the slow query log.
        #[arg(long)]
        clear: bool,
    },

    /// Show query plan without executing.
    Explain {
        /// Table name.
        table: String,
        /// Filter: field=value, etc. Repeatable.
        #[arg(long = "where")]
        where_clauses: Vec<String>,
        /// Semantic search query.
        #[arg(long)]
        similar: Option<String>,
        /// Graph traversal path.
        #[arg(long, allow_hyphen_values = true)]
        traverse: Option<String>,
        /// Number of results.
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// View the audit trail of write operations.
    Log {
        /// Maximum number of entries.
        #[arg(long)]
        limit: Option<usize>,
        /// Show entries after this date.
        #[arg(long)]
        after: Option<String>,
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
        /// Filter by operation (insert, update, delete).
        #[arg(long)]
        op: Option<String>,
    },

    /// Memory lifecycle: hard-delete expired/superseded records and clean
    /// orphaned edges/vectors/FTS, reclaiming space. Does NOT downsample.
    Compact {
        /// Instead of compacting, delete the companion file left behind by a
        /// removed Engine: `vector`, `graph`, `timeseries`, or `fts`. The core
        /// `.axil` file is never touched. Use after rebuilding without an
        /// Engine's feature or disabling it in `[engines] disabled`.
        #[arg(long)]
        drop_engine: Option<String>,
    },

    /// Scored health assessment (0-100) + fix recommendations — the deeper
    /// sibling of `doctor`. `--save`/`--compare` track the score over time.
    HealthReport {
        /// Brief one-line summary only.
        #[arg(long)]
        brief: bool,
        /// Save this report to the database for trend tracking.
        #[arg(long)]
        save: bool,
        /// Compare with the last saved report.
        #[arg(long)]
        compare: bool,
    },

    /// Chart the metric history that `snapshot` records over time.
    Trends {
        /// Number of days to show trends for.
        #[arg(long, default_value = "30")]
        days: u64,
    },

    /// Record DB metrics (counts, latencies) for trend charts — NOT a data
    /// backup. Pairs with `trends`. (For a data copy, see `branch`.)
    #[command(visible_alias = "metrics-snapshot")]
    Snapshot,

    /// Automation: run the additive diagnostics — `snapshot` and
    /// `health-report --save` — only when their cadence (`[maintenance]`
    /// in axil.toml) has elapsed. Cheap when fresh; fired by the brain hook
    /// each session. Never runs destructive downsample/reindex.
    Maintain {
        /// Only run tasks whose cadence has elapsed, and only when
        /// `[maintenance] auto` is true. Without this flag, every
        /// eligible task runs now regardless of cadence.
        #[arg(long)]
        if_stale: bool,
        /// Spawn a detached child and return immediately. A lock at
        /// `.axil/maintain.lock` makes concurrent invocations no-op.
        #[arg(long)]
        in_background: bool,
        /// Print what would run without doing it.
        #[arg(long)]
        dry_run: bool,
    },

    // ── Agent-friendly CRUD (4b.2) ──────────────────────────────────
    /// Store a record. Use "-" as json_data to read from stdin.
    ///
    /// Categorize by FUNCTION, not TOPIC. The table is the record's kind —
    /// the question it answers / when you reach for it: `decisions`
    /// (a choice + rationale), `errors` (a failure + fix), `rules`
    /// (constraints to obey), `context` (durable how-it-works knowledge).
    /// TOPIC ("the auth feature", a module name) is already handled by
    /// embeddings + entities + `--scope` — do NOT encode it as a category.
    ///
    /// `context` records may carry a `type` facet to scope retrieval
    /// (filterable via `axil recall --type <t>`). Recommended values:
    /// architecture, gotcha, howto, reference. The vocabulary is a
    /// recommendation, not enforced — any string is accepted. `decisions`
    /// and `errors` don't need a `type`; their field shape already encodes
    /// their function.
    #[command(visible_alias = "insert")]
    Store {
        /// Table name.
        table: String,
        /// JSON data (or "-" to read from stdin).
        json_data: String,
        /// Auto-embed these fields after insert (comma-separated).
        #[cfg(feature = "embed")]
        #[arg(long)]
        embed: Option<String>,
        /// JSON array of entity names to store as metadata.
        #[arg(long)]
        entities: Option<String>,
        /// Use LLM-enhanced storage (entity extraction, categorization).
        #[arg(long)]
        llm: bool,
        /// Tag record with agent name (adds `_agent` field to data).
        #[arg(long)]
        agent: Option<String>,
        /// Attach a code reference. Accepts `proxy_id`, `path[:line]`,
        /// or a SCIP-style `canonical_id`. Repeat the flag to attach
        /// multiple references. The CLI resolves each ref against
        /// `_idx_code_proxies` and stores a normalized `code_refs`
        /// metadata array on the new record so future recalls for the
        /// same symbol/file surface this memory.
        #[arg(long = "code-ref", value_name = "REF")]
        code_ref: Vec<String>,
    },

    /// Batch insert records from a JSON array (or one JSON object per line from stdin).
    BatchInsert {
        /// Table name.
        table: String,
        /// JSON array of objects, or "-" to read JSON lines from stdin.
        json_data: String,
    },

    /// Get a record by ID.
    Get {
        /// Record ID.
        id: String,
    },

    /// Update a record's data. Use "-" as json_data to read from stdin.
    Update {
        /// Record ID.
        id: String,
        /// JSON data (or "-" to read from stdin).
        json_data: String,
    },

    /// Delete a record by ID.
    Delete {
        /// Record ID.
        id: String,
    },

    /// List records in a table with optional filters.
    List {
        /// Table name.
        table: String,
        /// Maximum number of results.
        #[arg(long)]
        limit: Option<usize>,
        /// Offset for pagination.
        #[arg(long)]
        offset: Option<usize>,
        /// Filter: field=value, field>value, field<value, etc. Repeatable.
        #[arg(long = "where")]
        where_clauses: Vec<String>,
        /// Filter by agent name (matches `_agent` field in record data).
        #[arg(long)]
        agent: Option<String>,
        /// Include archived records (excluded by default).
        #[arg(long)]
        include_archived: bool,
    },

    /// Export memory to mergeable JSONL (portable across machines / teammates).
    ///
    /// Records and the graph edges between them are written as one JSON object
    /// per line, prefixed by a header line. Embeddings are NOT exported — they
    /// are machine-local and rebuilt on import. Distinct from `branch`/`snapshot`
    /// (a binary whole-file clone you restore over a DB): an export is merged
    /// into another DB with `import`.
    Export {
        /// Write to this file instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only export these tables (comma-separated). Overrides the default
        /// user-tables filter; `_idx_*` index tables are always excluded.
        #[arg(long, value_delimiter = ',')]
        tables: Vec<String>,
        /// Only export records created at or after this ISO 8601 instant.
        #[arg(long)]
        since: Option<String>,
        /// Also export system tables (prefix `_`, except rebuilt `_idx_*`).
        #[arg(long)]
        include_system: bool,
    },

    /// Import memory from a JSONL file produced by `axil export`.
    ///
    /// Records are recreated through the normal insert path (so embeddings, FTS,
    /// and code_refs are rebuilt) with their original ids preserved. A record
    /// whose id already exists is OVERWRITTEN in place (reported as
    /// `overwritten`); pass --dedup to skip existing ids and same-content
    /// duplicates instead. Edges are recreated between resolvable endpoints,
    /// never duplicated on re-import.
    Import {
        /// Path to the JSONL export file (or "-" to read from stdin).
        file: String,
        /// Skip records whose id — or whose content — already exists.
        #[arg(long)]
        dedup: bool,
        /// Report what would be imported without writing anything.
        #[arg(long)]
        dry_run: bool,
    },

    // ── Agent recall & search (4b.3) ────────────────────────────────
    /// Recency-weighted semantic search (the primary agent command).
    #[cfg(feature = "embed")]
    Recall {
        /// Search query text.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Only include records after this date (ISO 8601).
        #[arg(long)]
        after: Option<String>,
        /// Only include records before this date (ISO 8601).
        #[arg(long)]
        before: Option<String>,
        /// Blend factor: alpha * similarity + (1-alpha) * recency. Default 0.7.
        #[arg(long, default_value = "0.7")]
        alpha: f32,
        /// Exclude results from files that changed since last index.
        #[arg(long)]
        fresh_only: bool,
        /// Show score breakdown per result (multi-signal scoring).
        #[arg(long)]
        explain: bool,
        /// Show which results have prior relevance feedback.
        #[arg(long)]
        feedback: bool,
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
        /// Filter by the record's `type` facet (matches `data.type`,
        /// case-insensitive exact). Records without a `type` field are
        /// excluded when this is set. See `axil store --help` for the
        /// recommended `context` vocabulary.
        #[arg(long = "type")]
        r#type: Option<String>,
        /// Token budget: truncate results to fit within this many tokens (1 token ≈ 4 bytes).
        #[arg(long)]
        budget: Option<usize>,
        /// Output format: compact (default — id+score+summary), full (whole
        /// record JSON), oneline, or context-block. Compact is the default so
        /// bare `axil recall` stays cheap; use `axil get <id>` (or
        /// `--recall-format full`) to expand a hit.
        #[arg(long, default_value = "compact")]
        recall_format: RecallFormat,
        /// Disable near-duplicate collapse. By default recall collapses
        /// near-identical hits so they don't each consume a top-k slot.
        #[arg(long)]
        no_dedup: bool,
        /// Disable completeness k-widening. By default, when the kept top-k
        /// compresses much better than the candidate pool (a diverse cluster
        /// was cut), recall widens k once and re-trims.
        #[arg(long)]
        no_widen: bool,
        /// Filter by agent name (matches `_agent` field in record data).
        #[arg(long)]
        agent: Option<String>,
        /// Minimum importance score (0.0–1.0). Filters out low-importance records.
        #[arg(long)]
        min_importance: Option<f32>,
        /// Filter by memory scope (session, agent, project, user, global). Comma-separated for multiple.
        #[arg(long)]
        scope: Option<String>,
        /// Minimum confidence score (0.0–1.0). Filters out low-confidence records.
        #[arg(long)]
        min_confidence: Option<f32>,
        /// Hard deadline in milliseconds. Returns partial results if retrieval exceeds this (12.1).
        /// Recommended for hook-injected recall so the agent never blocks on slow queries.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Reranking mode (12.4): off (default), cross-encoder, or llm.
        #[arg(long, default_value = "off")]
        rerank: RerankMode,
        /// Expand the query with graph neighbors + alias synonyms before retrieval (12.5).
        #[arg(long)]
        expand: bool,
        /// When --expand is set, include up to this many one-hop neighbors per entity
        /// (multiplied by 3 internally). Single-hop only — does NOT traverse deeper.
        #[arg(long, default_value = "1")]
        expand_neighbors: usize,
        /// Disable cascading fallback when primary recall returns empty.
        /// Cascade rungs (in order): filters_relaxed → expand → fts. Honors
        /// the same --timeout-ms deadline. The matched rung is logged to
        /// stderr as `[recall] cascade rung=<name>`.
        #[arg(long)]
        no_cascade: bool,
        /// Print recall profiling to stderr, including the query classification
        /// and whether an identifier FTS tilt was applied (e.g.
        /// `query_class=identifier:uuid (FTS tilt applied)`).
        #[arg(long)]
        profile: bool,
    },

    /// Search structural code proxies and return compact pointers.
    ///
    /// Uses fused vector + FTS retrieval over `_idx_code_proxies` and
    /// returns `path:line symbol — why` pointers (and JSON when `--json`
    /// is set). Output is much smaller than `axil recall` for coding
    /// queries because it skips raw source by default.
    #[cfg(feature = "indexer")]
    #[command(name = "code-search")]
    CodeSearch {
        /// Search query.
        query: String,
        #[arg(long, default_value = "5")]
        top_k: usize,
        #[arg(long)]
        json: bool,
        /// Also surface graph neighbors of matched proxies
        /// (callers/callees/refs/impls via SCIP edges). Each neighbor's
        /// `why` field shows whether it came from direct search or graph
        /// expansion.
        #[arg(long = "trace-graph")]
        trace_graph: bool,
    },

    /// Assemble a coding-task context block (code + memories + rules +
    /// recent changes) within a token budget.
    #[cfg(feature = "indexer")]
    #[command(name = "code-context")]
    CodeContext {
        /// Task description / question.
        #[arg(long)]
        task: String,
        /// Token budget for the assembled context. Omit to auto-size by
        /// indexed repo size (tiny→1500, large monorepo→4000, capped).
        #[arg(long)]
        budget: Option<usize>,
        /// Output: `compact` (lean pointer lines, default — much smaller; drops
        /// the JSON bundle's scores/ids/section bookkeeping) or `json` (full
        /// bundle with scores/ids/sections).
        #[arg(long = "context-format", default_value = "compact")]
        context_format: String,
    },

    /// Explain why a code proxy matched a recent query.
    #[cfg(feature = "indexer")]
    #[command(name = "explain-code-hit")]
    ExplainCodeHit {
        /// Proxy record id (`01K...`) or `proxy_id` data field.
        id: String,
        /// Optional original query used at recall time.
        #[arg(long)]
        query: Option<String>,
    },

    /// Run the code-recall benchmark and emit a comparison table.
    ///
    /// Pass `--cases <FILE>` to use a JSON eval fixture; otherwise the
    /// built-in axil-dogfood cases are used.
    #[cfg(feature = "indexer")]
    #[command(name = "code-recall-bench")]
    CodeRecallBench {
        /// JSON file with `EvalCase[]` (see `code_recall_eval` module).
        #[arg(long)]
        cases: Option<String>,
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Output format: `text` (default), `markdown`, or `json`.
        #[arg(long = "bench-format", default_value = "text")]
        bench_format: String,
        /// Persist the full JSON report to this path so future runs can
        /// diff quality over time. Always written regardless of
        /// `--bench-format`.
        #[arg(long = "save")]
        save: Option<String>,
        /// Compare current run against a saved JSON report and exit
        /// non-zero on regression: top-3 symbol hit rate decrease, or
        /// >10% context-token bloat for the structural-proxies strategy.
        #[arg(long = "regression-gate")]
        regression_gate: Option<String>,
    },

    /// Report how many context tokens Axil saves vs reading files directly.
    ///
    /// For each task, runs real recall and compares the compact context
    /// block Axil injects against the full source an unaided agent would
    /// read to reach the same answer. Pass `--task` (repeatable) for your
    /// own workload, `--tasks <FILE>` for a JSON `TaskSpec[]`, or neither
    /// to use the built-in dogfood tasks.
    #[cfg(feature = "indexer")]
    #[command(name = "context-savings")]
    ContextSavings {
        /// A task/question to measure. Repeat for multiple; overrides the
        /// built-in task set.
        #[arg(long = "task")]
        task: Vec<String>,
        /// JSON file with `TaskSpec[]` (`[{"task":"..."}]`).
        #[arg(long = "tasks")]
        tasks: Option<String>,
        /// Hits per task — more hits = more files an unaided agent reads.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Output format: `text` (default), `markdown`, or `json`.
        #[arg(long = "format", default_value = "text")]
        savings_format: String,
        /// Persist the full JSON report to this path (A/B baseline).
        #[arg(long = "save")]
        save: Option<String>,
    },

    /// Pure vector similarity search.
    #[cfg(feature = "embed")]
    Search {
        /// Search query text.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Full-text search.
    #[cfg(feature = "fts")]
    Fts {
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// Commit pending FTS writes and optimize the index.
    #[cfg(feature = "fts")]
    #[command(name = "fts-optimize")]
    FtsOptimize,

    // ── Agent graph commands (4b.4) ─────────────────────────────────
    /// Create a graph edge between two records.
    #[cfg(feature = "graph")]
    Link {
        /// Source record ID.
        from_id: String,
        /// Edge type label.
        edge_type: String,
        /// Target record ID.
        to_id: String,
        /// Optional JSON properties for the edge.
        #[arg(long)]
        props: Option<String>,
    },

    /// Remove a graph edge.
    #[cfg(feature = "graph")]
    Unlink {
        /// Edge ID to remove.
        edge_id: String,
    },

    /// Get neighbor records via graph edges.
    #[cfg(feature = "graph")]
    Neighbors {
        /// Record ID.
        id: String,
        /// Filter by edge type.
        #[arg(long = "type")]
        edge_type: Option<String>,
        /// Edge direction.
        #[arg(long, default_value = "out")]
        direction: CliDirection,
    },

    /// Traverse graph edges using path syntax (e.g. "->modified->file").
    #[cfg(feature = "graph")]
    Traverse {
        /// Starting record ID.
        id: String,
        /// Path expression (e.g. "->modified->file").
        #[arg(allow_hyphen_values = true)]
        path_expr: String,
    },

    // ── Agent session commands (4b.5) ───────────────────────────────
    /// Session management for agent workflows.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },

    // ── Enhanced query (4b.7) ───────────────────────────────────────
    /// Query records with filters, ordering, and limits.
    Query {
        /// Table name.
        table: String,
        /// Filter: field=value, field>value, etc. Repeatable.
        #[arg(long = "where")]
        where_clauses: Vec<String>,
        /// Sort field.
        #[arg(long)]
        order_by: Option<String>,
        /// Sort direction.
        #[arg(long, default_value = "asc")]
        direction: CliSortDirection,
        /// Maximum number of results.
        #[arg(long)]
        limit: Option<usize>,
        /// Offset for pagination.
        #[arg(long)]
        offset: Option<usize>,
        /// Show per-step timing breakdown alongside results.
        #[arg(long)]
        profile: bool,
    },

    // ── AxilQL query language (7d) ──────────────────────────────────
    /// Execute an AxilQL query (e.g. RECALL "auth error" TOP 5).
    #[cfg(feature = "ql")]
    Ql {
        /// AxilQL query string (or "-" to read from stdin, or omit for interactive REPL).
        query_str: Option<String>,
        /// Interactive REPL mode with history and line editing.
        #[arg(short, long)]
        interactive: bool,
        /// Show query plan without executing (EXPLAIN).
        #[arg(long)]
        explain: bool,
        /// Show per-step timing (PROFILE).
        #[arg(long)]
        profile: bool,
    },

    // ── Time-series commands ─────────────────────────────────────────
    /// Show recent records (created within a duration: 3d, 1h, 30m, 90s).
    #[cfg(feature = "timeseries")]
    Since {
        /// Duration string.
        duration: String,
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
    },

    /// Show a timeline of records (newest first).
    #[cfg(feature = "timeseries")]
    Timeline {
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
        /// Number of records.
        #[arg(long, default_value = "20")]
        limit: usize,
    },

    /// Show what changed (created + updated) within a duration.
    #[cfg(feature = "timeseries")]
    Diff {
        /// Duration string.
        #[arg(long)]
        since: String,
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
    },

    /// Show daily record counts.
    #[cfg(feature = "timeseries")]
    Activity {
        /// Number of days.
        #[arg(long, default_value = "7")]
        days: u64,
        /// Filter by table name.
        #[arg(long)]
        table: Option<String>,
    },

    /// Memory lifecycle / repair: rebuild drifted indexes (`--reindex`),
    /// compact (`--compact`), or clean orphans (`--orphans`). A bare `heal`
    /// also downsamples (purges records past the retention window). Run
    /// deliberately — check `axil doctor` first.
    Heal {
        /// Just compact (purge expired/superseded).
        #[arg(long)]
        compact: bool,
        /// Rebuild all indexes from records.
        #[arg(long)]
        reindex: bool,
        /// Clean orphaned edges/vectors/FTS.
        #[arg(long)]
        orphans: bool,
        /// Show what would be fixed without doing it.
        #[arg(long)]
        dry_run: bool,
    },

    /// End-of-session heal: replay session failures, run auto-fixes, log result.
    ///
    /// The Stop hook captures axil command failures and empty-result misses to
    /// a JSONL file during the session. This command reads that file, runs
    /// `detect_problems()`, fixes auto-fixable issues (compact / reindex /
    /// orphans), classifies misses (e.g. empty code-search → suggests reindex)
    /// and writes a `_heal_log` row so the next session sees what was fixed.
    SessionHeal {
        /// JSONL file of problems captured during the session (one event per line).
        #[arg(long)]
        problems_file: Option<PathBuf>,
        /// Session id to tag the _heal_log entry with.
        #[arg(long)]
        session: Option<String>,
        /// Don't perform fixes — just report what would happen.
        #[arg(long)]
        dry_run: bool,
        /// Print a one-line human-readable summary on stderr.
        #[arg(long)]
        quiet: bool,
    },

    // ── Embedding commands ──────────────────────────────────────────
    /// Embed a record field using a local model.
    #[cfg(feature = "embed")]
    Embed {
        /// Record ID.
        id: String,
        /// Field name to embed.
        field: String,
        /// Model name: bge-small, bge-base, nomic.
        #[arg(long, default_value = "bge-small")]
        model: String,
    },

    // ── Advanced vector commands ────────────────────────────────────
    /// Add a pre-computed vector for a record.
    #[cfg(feature = "vector")]
    AddVector {
        /// Record ID.
        id: String,
        /// Vector as JSON array of floats.
        vector: String,
        /// Vector dimensions (auto-detected if omitted).
        #[arg(long)]
        dimensions: Option<usize>,
    },

    /// Search for similar records using a raw vector.
    #[cfg(feature = "vector")]
    SearchVector {
        /// Query vector as JSON array of floats.
        vector: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Vector dimensions (auto-detected if omitted).
        #[arg(long)]
        dimensions: Option<usize>,
    },

    // ── Advanced graph commands ─────────────────────────────────────
    /// List graph edges for a record.
    #[cfg(feature = "graph")]
    Edges {
        /// Record ID.
        id: String,
        /// Edge direction.
        #[arg(long, default_value = "both")]
        direction: CliDirection,
    },

    // ── Advanced FTS commands ───────────────────────────────────────
    /// Index a record field for full-text search.
    #[cfg(feature = "fts")]
    IndexText {
        /// Record ID.
        id: String,
        /// Field name.
        field: String,
        /// Text content (reads from record field if omitted).
        text: Option<String>,
    },

    // ── Config commands (4c.6) ──────────────────────────────────────
    /// Configuration management.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    // ── Report commands (4c.3 / 4c.7) ────────────────────────────────
    /// Field report management.
    Report {
        #[command(subcommand)]
        command: ReportCommand,
    },

    // ── Skill commands (4c.8) ────────────────────────────────────────
    /// Skill installation and management.
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },

    // ── Hook runtime ─────────────────────────────────────────────────
    /// Agent lifecycle hook runtime (the Axil brain).
    ///
    /// Wired into the agent's hook config by `axil install`; the harness
    /// pipes each event's JSON to stdin and reads the response from stdout.
    /// Replaces the former bash hook scripts — no bash/jq needed.
    Hook {
        #[command(subcommand)]
        command: HookCommand,
    },

    // ── Model management ────────────────────────────────────────────
    /// Download an embedding model.
    #[cfg(feature = "vector")]
    ModelDownload {
        /// Model name: bge-small, bge-small-int8, bge-base, nomic, or bge-m3.
        #[arg(default_value = "bge-small")]
        model: String,
    },

    /// List downloaded embedding models.
    #[cfg(feature = "vector")]
    ModelList,

    /// Remove a downloaded embedding model.
    #[cfg(feature = "vector")]
    ModelRemove {
        /// Model name.
        model: String,
    },

    /// Re-embed all records with a different model.
    ///
    /// Deletes the existing vector store, creates a new one with the target model's
    /// dimensions, and embeds the specified field from every record that has it.
    /// This embeds ALL matching records, not just those that previously had vectors.
    #[cfg(feature = "embed")]
    Reembed {
        /// Model name: bge-small, bge-small-int8, bge-base, nomic, bge-m3,
        /// or a registered custom model.
        #[arg(long, default_value = "bge-small")]
        model: String,
        /// Record field to embed (must be a string field present on all records).
        #[arg(long)]
        field: String,
        /// Only re-embed records in this table (default: all tables).
        #[arg(long)]
        table: Option<String>,
    },

    /// Register a custom ONNX embedding model from a local path.
    #[cfg(feature = "embed")]
    ModelAdd {
        /// Name to register the model under.
        name: String,
        /// Path to the ONNX model file.
        path: PathBuf,
        /// Output vector dimensions.
        #[arg(long)]
        dimensions: usize,
        /// Pooling strategy: cls or mean.
        #[arg(long, default_value = "cls")]
        pooling: String,
        /// Maximum input sequence length.
        #[arg(long, default_value = "512")]
        max_seq_len: usize,
    },

    /// Benchmark embedding models: compare latency, dimensions, and throughput.
    #[cfg(feature = "embed")]
    ModelBench {
        /// Models to benchmark (comma-separated). Default: all available.
        #[arg(long)]
        models: Option<String>,
        /// Number of texts to embed per model.
        #[arg(long, default_value = "100")]
        count: usize,
    },

    // ── Project indexer (4d) ────────────────────────────────────────
    /// Scan a project directory and build a token-efficient knowledge base.
    #[cfg(feature = "indexer")]
    Index {
        /// Project directory to index (default: current directory).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Force full re-index (skip incremental detection).
        #[arg(long)]
        full: bool,
        /// Show what would be indexed without storing.
        #[arg(long)]
        dry_run: bool,
    },

    /// Refresh all code-knowledge in one call: the structural proxy
    /// index (`index`) plus the SCIP code-graph (`scip refresh`).
    ///
    /// The proxy index runs in the foreground (incremental, fast). The
    /// SCIP refresh is spawned in the background by default so the
    /// command returns quickly — pass `--wait` to block until the graph
    /// edges are rebuilt. This is the one-shot "make my code-knowledge
    /// current" command; `index` and `scip refresh` remain available for
    /// granular control.
    #[cfg(feature = "indexer")]
    Reindex {
        /// Project directory to index (default: current directory).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Force a full proxy re-index AND an unconditional SCIP refresh
        /// (skip incremental + staleness detection).
        #[arg(long)]
        full: bool,
        /// Skip the SCIP code-graph refresh; only rebuild the proxy index.
        #[arg(long)]
        no_scip: bool,
        /// Wait for the SCIP refresh to finish (foreground) instead of
        /// spawning it in the background. Slower, but graph queries are
        /// guaranteed current when the command returns.
        #[arg(long)]
        wait: bool,
    },

    /// Search the project index — compact, token-efficient results for agents.
    #[cfg(feature = "indexer")]
    #[command(name = "recall-index")]
    RecallIndex {
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Get project context for a new agent session (project overview + modules).
    #[cfg(feature = "indexer")]
    Context {
        /// Maximum tokens to return (overrides depth default).
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Focus on specific modules (comma-separated).
        #[arg(long)]
        focus: Option<String>,
        /// Show what changed since last session.
        #[arg(long)]
        diff: bool,
        /// Context depth: shallow (~500 tokens), medium (~2000), deep (~5000).
        #[arg(long, default_value = "medium")]
        depth: String,
        /// Task-focused context: combines vector + graph + rules + timeline for a task.
        #[arg(long)]
        task: Option<String>,
    },

    /// Show index statistics and token efficiency metrics.
    #[cfg(feature = "indexer")]
    #[command(name = "index-stats")]
    IndexStats,

    // ── Agent runtime (4e) ─────────────────────────────────────────
    /// Smart query — auto-routes to vector/graph/FTS/time/rules based on intent.
    #[cfg(feature = "indexer")]
    Ask {
        /// Natural language question.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Run all strategies in parallel with Reciprocal Rank Fusion.
        #[arg(long)]
        parallel: bool,
        /// Limit which strategies run (comma-separated: vector,fts,graph,time).
        #[arg(long)]
        strategy: Option<String>,
        /// Decompose the query into a multi-step plan and execute sequentially.
        #[arg(long)]
        plan: bool,
        /// Apply cross-encoder reranking to improve result ordering.
        #[arg(long)]
        rerank: bool,
    },

    /// Manage agent rules and conventions (key-value store).
    #[cfg(feature = "indexer")]
    #[command(visible_alias = "rules")]
    Rule {
        #[command(subcommand)]
        command: RuleCommand,
    },

    /// Show what's affected if a file changes (graph-powered impact analysis).
    #[cfg(all(feature = "indexer", feature = "graph"))]
    Impact {
        /// File path to analyze.
        path: String,
        /// Reverse: show what this file depends ON.
        #[arg(long)]
        reverse: bool,
    },

    /// Show why two files are connected (shortest graph path).
    #[cfg(all(feature = "indexer", feature = "graph"))]
    Why {
        /// First file path.
        path_a: String,
        /// Second file path.
        path_b: String,
    },

    /// Show agent usage analytics.
    #[cfg(feature = "indexer")]
    Analytics {
        /// Period in days (default: 7).
        #[arg(long, default_value = "7")]
        days: u64,
    },

    /// Pre-load context for an upcoming task or file.
    #[cfg(feature = "indexer")]
    Prefetch {
        /// Task intent (e.g. "fix auth timeout").
        intent: String,
        /// Maximum tokens to pre-load.
        #[arg(long, default_value = "2000")]
        max_tokens: usize,
        /// Prefetch context for opening a specific file (imports, dependents, recent changes).
        #[arg(long)]
        file: Option<String>,
    },

    // ── Agent memory commands ────────────────────────────
    /// Store a fact about an entity (semantic memory).
    #[cfg(feature = "memory")]
    Know {
        /// Entity name (e.g. "auth-module").
        entity: String,
        /// Fact about the entity.
        fact: String,
        /// Source session ID.
        #[arg(long)]
        source: Option<String>,
    },

    /// Query everything known about an entity.
    #[cfg(feature = "memory")]
    #[command(name = "know-about", visible_alias = "about")]
    KnowAbout {
        /// Entity name.
        entity: String,
    },

    /// List known entities and facts.
    #[cfg(feature = "memory")]
    #[command(name = "know-list")]
    KnowList {
        /// Filter by entity name.
        #[arg(long)]
        entity: Option<String>,
    },

    /// Store a learned procedure/pattern (procedural memory).
    #[cfg(feature = "memory")]
    Learn {
        /// Pattern name (e.g. "fix-timeout").
        pattern_name: String,
        /// Description of the approach.
        description: String,
    },

    /// Find relevant procedures for a task.
    #[cfg(feature = "memory")]
    How {
        /// Task description (e.g. "fix a database timeout").
        task: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// List episodes (past sessions with outcomes).
    #[cfg(feature = "memory")]
    Episodes {
        /// Filter by outcome: success, failure, partial.
        #[arg(long)]
        outcome: Option<String>,
        /// Maximum number of results.
        #[arg(long, default_value = "20")]
        limit: usize,
    },

    /// Find similar past episodes.
    #[cfg(feature = "memory")]
    #[command(name = "episodes-similar")]
    EpisodesSimilar {
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Search ALL memory types (cross-memory query).
    #[cfg(feature = "memory")]
    Remember {
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Maximum tokens in response.
        #[arg(long)]
        max_tokens: Option<usize>,
    },

    /// Show how knowledge about an entity evolved over time.
    #[cfg(feature = "memory")]
    History {
        /// Entity name.
        entity: String,
    },

    /// Set TTL on a record.
    #[cfg(feature = "memory")]
    #[command(name = "ttl-set")]
    TtlSet {
        /// Record ID.
        id: String,
        /// Duration string (e.g. "7d", "24h", "30m").
        duration: String,
    },

    /// Clear TTL from a record.
    #[cfg(feature = "memory")]
    #[command(name = "ttl-clear")]
    TtlClear {
        /// Record ID.
        id: String,
    },

    // ── Intelligent Database ─────────────────────────────
    /// Auto-link a record: extract entities and create graph edges.
    #[cfg(all(feature = "embed", feature = "graph"))]
    AutoLink {
        /// Record ID to auto-link.
        id: String,
        /// Similarity threshold for related_to edges (0.0–1.0).
        #[arg(long, default_value = "0.85")]
        threshold: f32,
    },

    /// Detect contradictions for a newly inserted record.
    #[cfg(all(feature = "embed", feature = "graph"))]
    DetectConflicts {
        /// Record ID to check.
        id: String,
    },

    /// Consolidate facts about an entity into a merged summary.
    #[cfg(all(feature = "embed", feature = "graph"))]
    Consolidate {
        /// Entity name to consolidate.
        entity: String,
    },

    /// Show fact evolution timeline for an entity.
    #[cfg(feature = "graph")]
    #[command(name = "entity-history")]
    EntityHistory {
        /// Entity name.
        entity: String,
    },

    /// Register an alias for an entity (e.g. "the VP of Engineering" → "Sarah").
    #[cfg(feature = "memory")]
    #[command(name = "entity-alias")]
    EntityAlias {
        /// Canonical entity name.
        entity: String,
        /// Alias to register.
        alias: String,
    },

    /// Resolve a name to its canonical entity (check if it's an alias).
    #[cfg(feature = "memory")]
    #[command(name = "entity-resolve")]
    EntityResolve {
        /// Name to resolve.
        name: String,
        /// Include fuzzy matches with confidence scores.
        #[arg(long)]
        fuzzy: bool,
        /// Disambiguation strategy: default, frequency, session, context.
        #[arg(long, default_value = "default")]
        strategy: String,
        /// Context terms for context-based disambiguation (comma-separated).
        #[arg(long)]
        context: Option<String>,
        /// Session ID for session-based disambiguation.
        #[arg(long, name = "session-id")]
        session_id: Option<String>,
    },

    /// Merge two entities: move facts from source to target, transfer aliases.
    #[cfg(feature = "memory")]
    #[command(name = "entity-merge")]
    EntityMerge {
        /// Target entity (keeps this name).
        target: String,
        /// Source entity (merged into target, becomes alias).
        source: String,
    },

    /// Warm up the database (rebuild indexes, prepare caches).
    WarmUp,

    // ── LLM provider commands ──────────────────────────────
    /// LLM provider management.
    Llm {
        #[command(subcommand)]
        command: LlmCommand,
    },

    // ── AI Agent Performance ──────────────────────────────
    /// Extract entities from text using pattern-based extraction (no LLM).
    ///
    /// Extracts file paths, CamelCase/snake_case identifiers, backtick code,
    /// and quoted strings. Used by hooks for auto-entity capture.
    #[command(name = "extract-entities")]
    ExtractEntities {
        /// Text to extract entities from (or "-" to read from stdin).
        text: String,
    },

    /// Session boot context — curated wake-up context for agent sessions.
    ///
    /// Combines recent sessions, decisions, errors, and architecture notes
    /// into a single context payload for session initialization.
    Boot {
        /// Maximum token budget for output (estimate: 1 token ≈ 4 bytes).
        #[arg(long)]
        budget: Option<usize>,
        /// Boot output format: json (default), narrative (plain text), compact.
        #[arg(long = "boot-format", visible_alias = "fmt", default_value = "json")]
        boot_format: BootFormat,
        /// Topic-focused boot: recall + filter context for a specific task.
        #[arg(long, name = "for")]
        topic: Option<String>,
        /// Context-aware: push memories related to these files (comma-separated).
        #[arg(long)]
        files: Option<String>,
        /// Context-aware: push memories related to these entities (comma-separated).
        #[arg(long)]
        entities: Option<String>,
        /// Context-aware: push memories related to this error text.
        #[arg(long)]
        error: Option<String>,
        /// Use the stable v1 boot schema (fixed section order + token budget
        /// discipline). Opt-in for one release; will become the default later.
        /// Emits JSON matching `axil_core::BootContext`.
        #[arg(long = "schema", value_parser = ["v1"])]
        schema: Option<String>,
    },

    /// Intent-native writes — higher-level than `axil store`. Auto-embed,
    /// auto-supersede, (agent_id, external_id) idempotency.
    ///
    /// Named `capture` to avoid colliding with the existing `remember`
    /// cross-memory search subcommand. Routes to `Axil::remember_*`.
    #[command(subcommand)]
    Capture(CaptureCmd),

    /// Set a user preference (overwrites by key, keeps `_previous_value`).
    #[command(name = "prefer")]
    Prefer {
        /// Preference key.
        key: String,
        /// Preference value (JSON — use a quoted string for text).
        value: String,
    },

    /// Mark a session as closed with an optional summary. Idempotent by id.
    #[command(name = "close-session")]
    CloseSession {
        /// Session id (used as the idempotency key).
        id: String,
        /// Optional session summary.
        #[arg(long)]
        summary: Option<String>,
    },

    /// List current beliefs — the agent's high-level understanding.
    Beliefs {
        /// Filter beliefs by topic.
        #[arg(long)]
        topic: Option<String>,
        /// Include doubted beliefs.
        #[arg(long)]
        all: bool,
        /// Auto-generate beliefs from high-importance facts.
        #[arg(long)]
        generate: bool,
    },

    /// Explicitly state a belief.
    Believe {
        /// The belief statement.
        statement: String,
    },

    /// Mark a belief as uncertain (doubted).
    Doubt {
        /// Belief record ID.
        id: String,
    },

    /// Analyze text and auto-capture errors/decisions/context.
    ///
    /// Reads text from stdin or argument, classifies it, and stores
    /// high-confidence captures automatically.
    #[command(name = "auto-capture")]
    AutoCapture {
        /// Text to analyze (or "-" to read from stdin).
        text: String,
        /// Only show what would be captured, don't store.
        #[arg(long)]
        dry_run: bool,
        /// Minimum confidence to auto-store (default: 0.7).
        #[arg(long, default_value = "0.7")]
        min_confidence: f32,
        /// Source label for the capture (e.g., "bash", "agent").
        #[arg(long, default_value = "auto")]
        source: String,
    },

    /// Axil Brain mode management.
    ///
    /// Enable/disable brain mode, check status, trigger reflection, debug memories, run evals.
    #[command(subcommand)]
    Brain(BrainCommand),

    /// Show Axil Brain banner with session stats.
    ///
    /// Style configurable via `[brain] banner` in axil.toml:
    /// compact (default), box, ascii, status, bold.
    #[command(name = "brain-banner")]
    BrainBanner {
        /// Override banner style (compact, box, ascii, status, bold).
        #[arg(long)]
        style: Option<String>,
    },

    /// Recall memories relevant to a code entity — walks the SCIP graph.
    ///
    /// Accepts either a display name (resolved via `_entity_aliases`) or a
    /// raw SCIP canonical id. Surfaces memories attached to the entity +
    /// its 1-hop neighbors on `calls`, `references`, `implements`, and
    /// `defined_in → file → mentions`.
    #[cfg(feature = "scip")]
    #[command(name = "recall-for-entity")]
    RecallForEntity {
        /// Entity display name or SCIP canonical id.
        entity: String,
        /// Maximum hop depth (default 1).
        #[arg(long, default_value = "1")]
        depth: usize,
        /// Comma-separated edge types to traverse (default: all of calls,
        /// references, implements, type_of, defined_in).
        #[arg(long)]
        edge_types: Option<String>,
        /// Scopes to resolve the display name through (repeatable).
        #[arg(long = "scope")]
        scopes: Vec<String>,
        /// Show which layer (_idx_files or _entities) and confidence each
        /// hop came from.
        #[arg(long)]
        trace_graph: bool,
        /// Max records to return.
        #[arg(long, default_value = "10")]
        top_k: usize,
    },

    /// Recall memories relevant to a file — for use by hooks and automation.
    ///
    /// Searches decisions, errors, and context for mentions of the file path
    /// or filename. Returns compact results suitable for context injection.
    #[command(name = "recall-for-file")]
    RecallForFile {
        /// File path to search for.
        file: String,
        /// Maximum number of results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    // ── Agent Brain ──────────────────────────────────────────
    /// Observe and remember — unified write entry point.
    ///
    /// Routes observations through the decision pipeline:
    /// classify → scope → resolve (dedup/supersede) → score → commit.
    /// Replaces ad-hoc store/know/learn for new code.
    ///
    /// Examples:
    ///   axil observe "User prefers reversible migrations"
    ///   axil observe --source user --scope user "Always ask before schema changes"
    ///   axil observe --kind tool-output --table errors "connection refused on port 5432"
    ///   echo '{"summary":"..."}' | axil observe --stdin
    Observe {
        /// Text to observe (omit if using --stdin or --file).
        text: Option<String>,
        /// Read observation from stdin.
        #[arg(long)]
        stdin: bool,
        /// Read observation from a file.
        #[arg(long)]
        file: Option<String>,
        /// Source kind: user, agent, tool-output, hook, file, inference, llm.
        #[arg(long, default_value = "agent")]
        source: String,
        /// Source reference (command, file path, hook name).
        #[arg(long)]
        source_ref: Option<String>,
        /// Memory scope: session, agent, project, user, global.
        #[arg(long)]
        scope: Option<String>,
        /// Memory type hint: working, semantic, episodic, procedural, preference, belief.
        #[arg(long)]
        kind: Option<String>,
        /// Target table override (bypass auto-classification).
        #[arg(long)]
        table: Option<String>,
        /// Classification hints (comma-separated).
        #[arg(long)]
        hints: Option<String>,
        /// Agent name (for multi-agent scoping).
        #[arg(long)]
        agent: Option<String>,
        /// Output format: json (default), quiet, verbose.
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Inspect the full provenance and metadata of a memory record.
    ///
    /// Shows source, scope, confidence, trust tier, supersede chain,
    /// contradiction links, and derivation history.
    #[command(name = "inspect-memory")]
    InspectMemory {
        /// Record ID to inspect.
        id: String,
    },

    /// Show the derivation chain for a memory (trace provenance).
    #[command(name = "trace-memory")]
    TraceMemory {
        /// Record ID to trace.
        id: String,
    },

    /// Show cross-project history for a record: blast-radius counters
    /// (how many times and from which callers it has been recalled) plus
    /// any bridges that reference it. Useful for contamination review.
    #[command(name = "trace-record")]
    TraceRecord {
        /// Record spec: `<member>:<record_id>`. Omit `<member>:` to
        /// target the current DB directly.
        target: String,
    },

    /// Mark a memory record as human-verified (trust tier → Observed).
    Verify {
        /// Record ID to verify.
        id: String,
    },

    /// Backfill provenance metadata on legacy records.
    ///
    /// Adds _source, _scope, _confidence, _verified fields to records
    /// that predate the brain pipeline. Safe to run multiple times.
    #[command(name = "migrate-provenance")]
    MigrateProvenance,

    /// Revise beliefs based on new evidence text.
    ///
    /// Checks all existing beliefs against the observation and automatically
    /// reinforces, supersedes, doubts, or creates competing hypotheses.
    #[command(name = "revise-beliefs")]
    ReviseBeliefs {
        /// New evidence text.
        text: String,
    },

    /// Agent self-memory management.
    #[command(subcommand, name = "self")]
    SelfMemory(SelfCommand),

    /// Project operating model.
    #[command(subcommand, name = "project-model")]
    ProjectModel(ProjectModelCommand),

    /// User contract rules.
    #[command(subcommand, name = "user-contract")]
    UserContract(UserContractCommand),

    /// Show belief history for a topic.
    ///
    /// Includes current and doubted/superseded beliefs to show evolution.
    #[command(name = "belief-history")]
    BeliefHistory {
        /// Topic to search for.
        topic: String,
    },

    /// Redact a field in a record (replace with [REDACTED]).
    Redact {
        /// Record ID.
        id: String,
        /// Field name to redact.
        #[arg(long)]
        field: String,
    },

    /// Set retention policy for a scope (days until auto-cleanup).
    #[command(name = "retention")]
    Retention {
        #[command(subcommand)]
        command: RetentionCommand,
    },

    /// Pin a record (prevent decay/archive/deletion).
    Pin {
        /// Record ID.
        id: String,
    },

    /// Unpin a record (re-enable decay/archive).
    Unpin {
        /// Record ID.
        id: String,
    },

    /// Show overall memory safety policy summary.
    #[command(name = "memory-policy")]
    MemoryPolicy,

    /// Run the brain eval suite.
    #[command(name = "brain-eval")]
    BrainEval,

    /// Explain why a record was remembered (stored) — .
    ///
    /// Shows source event, classifier decision, importance breakdown,
    /// scope assignment, related memories considered, resolution.
    #[command(name = "why-remembered")]
    WhyRemembered {
        /// Record ID.
        id: String,
    },

    /// Explain why a record was recalled for a query — .
    ///
    /// Shows score breakdown, scope filter, trust tier, and ranking position.
    #[command(name = "why-recalled")]
    WhyRecalled {
        /// Record ID.
        id: String,
        /// The query to check against.
        query: String,
    },

    /// Explain why a record was revised (superseded/doubted) — .
    ///
    /// Shows the evidence that caused the revision, confidence change, and related records.
    #[command(name = "why-revised")]
    WhyRevised {
        /// Record ID.
        id: String,
    },

    // ── Worker & Branching ──────────────────────────────────
    /// Run background worker tasks (consolidation, connection strengthening, stale detection).
    Worker {
        #[command(subcommand)]
        command: WorkerCommand,
    },

    /// Memory branching — create experimental copies of the database.
    Branch {
        #[command(subcommand)]
        command: BranchCommand,
    },

    // ── MCP server ─────────────────────────────────────────
    /// Start the MCP (Model Context Protocol) server over stdio,
    /// or register it in an agent's config (`axil mcp install <target>`).
    #[cfg(feature = "mcp")]
    Mcp {
        #[command(subcommand)]
        command: Option<McpCommand>,
        /// OpenTelemetry OTLP endpoint for tracing/metrics export.
        /// Example: http://localhost:4317
        #[arg(long)]
        otel_endpoint: Option<String>,
    },

    // ── HTTP API server ────────────────────────────────────
    /// Start the HTTP API server.
    #[cfg(feature = "http")]
    Serve {
        /// Host to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value = "3000")]
        port: u16,
    },

    // ── Active Memory ──────────────────────────────────────
    /// Reflect on memory — synthesize patterns and insights across all memory types.
    #[cfg(feature = "memory")]
    Reflect {
        /// Topic to reflect on (or omit for all).
        #[arg(long)]
        topic: Option<String>,
        /// Scope: all, recent, entity.
        #[arg(long, default_value = "all")]
        scope: String,
        /// Store insights as a new record in the given table (e.g. `--store insights`).
        #[arg(long)]
        store: Option<String>,
        /// Use LLM to enhance insight synthesis (requires LLM config in axil.toml).
        #[arg(long)]
        llm: bool,
    },

    /// Show discovered graph connections for an entity.
    #[cfg(all(feature = "memory", feature = "graph"))]
    Connections {
        /// Entity name.
        entity: String,
    },

    /// Show consolidated entity profile.
    #[cfg(feature = "memory")]
    Profile {
        /// Entity name.
        entity: String,
    },

    /// Detect recurring patterns across sessions.
    #[cfg(feature = "memory")]
    Patterns {
        /// Filter by pattern type: repeated_failure, hot_spot, knowledge_gap, workflow.
        #[arg(long, name = "type")]
        pattern_type: Option<String>,
        /// Dismiss a pattern by name.
        #[arg(long)]
        dismiss: Option<String>,
        /// Detect and store new patterns.
        #[arg(long)]
        detect: bool,
    },

    /// Run graph inference to derive new facts.
    #[cfg(feature = "graph")]
    Infer {
        /// Entity name (or omit for all entities).
        #[arg(long)]
        entity: Option<String>,
    },

    /// Show reasoning chain for an inferred fact.
    #[cfg(feature = "graph")]
    #[command(name = "why-fact")]
    WhyFact {
        /// Record ID of the inferred fact.
        id: String,
    },

    /// Confirm an inferred fact (set confidence to 1.0).
    #[cfg(feature = "graph")]
    #[command(name = "fact-confirm")]
    FactConfirm {
        /// Record ID of the inferred fact.
        id: String,
    },

    /// Reject an inferred fact (mark as invalid, won't be re-inferred).
    #[cfg(feature = "graph")]
    #[command(name = "fact-reject")]
    FactReject {
        /// Record ID of the inferred fact.
        id: String,
    },

    // ── Workspace / consent / bridges ────────────────────
    /// Manage multi-project workspaces.
    Workspace {
        #[command(subcommand)]
        op: WorkspaceOp,
    },

    /// Manage record-level consent scopes.
    Consent {
        #[command(subcommand)]
        op: ConsentOp,
    },

    /// Manage entity bridges across sibling DBs (Boundary Dialect).
    Bridge {
        #[command(subcommand)]
        op: BridgeOp,
    },

    /// Cross-project recall fan-out.
    ///
    /// Fans out the query to every named sibling DB, filters each
    /// sibling's results by its own `read_consent`, and merges with
    /// provenance tags. Use the top-level `axil recall` for same-DB recall.
    #[cfg(feature = "embed")]
    #[command(name = "recall-across")]
    RecallAcross {
        /// Search query text.
        query: String,
        /// Comma-separated member labels, or `*` for every member.
        #[arg(long, default_value = "*")]
        across: String,
        /// Number of merged results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
        /// Drop workspace-scoped records at remote siblings.
        #[arg(long)]
        strict_consent: bool,
        /// Print per-result provenance, bridge confidence, and warnings.
        #[arg(long)]
        trace: bool,
        /// Human-readable one-line-per-result output instead of JSON.
        /// Example: `[backend] [conf=0.88] "summary..."  (BR=3q)`.
        #[arg(long)]
        oneline: bool,
    },
}

/// `axil remember <kind>` — intent-native writes layered above `axil store`.
///
/// Each subcommand routes through `Axil::remember_*` so callers get
/// auto-embed, auto-supersede, metadata normalization, and idempotency
/// by `(agent_id, external_id)` or 5-minute content-hash window.
#[derive(Subcommand)]
enum CaptureCmd {
    /// Record an architectural / implementation decision.
    Decision {
        /// What was decided. Required.
        #[arg(long)]
        summary: String,
        /// Why this path was chosen (recommended).
        #[arg(long)]
        reason: Option<String>,
        /// Comma-separated file list.
        #[arg(long)]
        files: Option<String>,
        /// Agent identifier (part of the idempotency key).
        #[arg(long = "agent-id")]
        agent_id: Option<String>,
        /// Caller-supplied idempotency key (paired with `--agent-id`).
        #[arg(long = "external-id")]
        external_id: Option<String>,
        /// Bypass both idempotency paths for intentional rewrites.
        #[arg(long = "force-new")]
        force_new: bool,
    },
    /// Record an error, optionally with root cause and fix.
    Error {
        /// What went wrong. Required.
        #[arg(long)]
        error: String,
        /// Root cause analysis.
        #[arg(long = "root-cause")]
        root_cause: Option<String>,
        /// How it was fixed.
        #[arg(long)]
        fix: Option<String>,
        /// Comma-separated file list.
        #[arg(long)]
        files: Option<String>,
        #[arg(long = "agent-id")]
        agent_id: Option<String>,
        #[arg(long = "external-id")]
        external_id: Option<String>,
        #[arg(long = "force-new")]
        force_new: bool,
    },
}

#[cfg(feature = "indexer")]
#[derive(Subcommand)]
enum RuleCommand {
    /// Set a rule (key-value).
    Set {
        /// Rule key (e.g. "error_handling").
        key: String,
        /// Rule value (e.g. "Use thiserror in libs, anyhow in bins").
        value: String,
    },
    /// Get a rule by key.
    Get {
        /// Rule key.
        key: String,
    },
    /// List all rules.
    List,
    /// Delete a rule by key.
    Delete {
        /// Rule key.
        key: String,
    },
    /// Auto-extract rules from CLAUDE.md and similar files.
    Extract {
        /// Project directory (default: current directory).
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Distill recurring failures (the `errors` table) into corrective
    /// directives, write them into CLAUDE.md (idempotent marker block) and
    /// pin them so `axil boot` surfaces them. The write-back counterpart of
    /// `extract`. (`learn` is a hidden alias.)
    #[command(alias = "learn")]
    Distill {
        /// Target file for the managed correction block.
        #[arg(long, default_value = "CLAUDE.md")]
        file: PathBuf,
        /// Minimum occurrences before a failure earns a directive.
        #[arg(long, default_value_t = axil_indexer::distill::DEFAULT_MIN_EVIDENCE)]
        min_evidence: usize,
        /// Cap on how many directives are emitted.
        #[arg(long, default_value_t = axil_indexer::distill::DEFAULT_MAX_DIRECTIVES)]
        max: usize,
        /// Preview directives without writing CLAUDE.md or pinning rules.
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum LlmCommand {
    /// Test the LLM connection.
    Test,
    /// Show LLM usage for the current process (in-memory, not persisted across invocations).
    Usage,
    /// Show LLM configuration.
    Config,
}

#[derive(Subcommand)]
enum SessionCommand {
    /// Start a new agent session.
    Start {
        /// Session metadata as JSON.
        #[arg(long)]
        meta: Option<String>,
    },
    /// End an agent session.
    End {
        /// Session ID.
        session_id: String,
        /// Session summary text.
        #[arg(long)]
        summary: Option<String>,
    },
    /// Log a record to a session (store + auto-link via graph).
    Log {
        /// Session ID.
        session_id: String,
        /// Table name.
        table: String,
        /// JSON data (or "-" to read from stdin).
        json_data: String,
        /// Auto-embed these fields after insert (comma-separated).
        #[cfg(feature = "embed")]
        #[arg(long)]
        embed: Option<String>,
    },
    /// List sessions.
    List {
        /// Only show active (not ended) sessions.
        #[arg(long)]
        active: bool,
    },
    /// Show all records linked to a session.
    History {
        /// Session ID.
        session_id: String,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Create `axil.toml` with commented defaults.
    Init,
    /// Print resolved config (all sources merged).
    Show,
    /// Get a single config value (e.g. `axil config get dev.source_repo`).
    Get {
        /// Dotted key (e.g. "dev.source_repo", "debug.log_level").
        key: String,
    },
    /// Set a config value in `axil.toml`.
    Set {
        /// Dotted key (e.g. "dev.source_repo").
        key: String,
        /// Value to set.
        value: String,
    },
}

// ── Self memory, project model, user contract subcommands ──

#[derive(Subcommand)]
enum SelfCommand {
    /// Add a self-memory note.
    Note {
        /// The note text.
        text: String,
        /// Category (general, strength, weakness, pattern, gotcha).
        #[arg(long, default_value = "general")]
        category: String,
    },
    /// Show agent self-profile.
    Profile,
}

#[derive(Subcommand)]
enum ProjectModelCommand {
    /// Set a project model entry.
    Set {
        /// Key (e.g. "deployment", "testing", "branching").
        key: String,
        /// Value (the rule or norm).
        value: String,
    },
    /// Show the full project operating model.
    Show,
    /// Auto-generate project model from existing memories.
    Generate,
}

#[derive(Subcommand)]
enum UserContractCommand {
    /// Add a user contract rule.
    Add {
        /// The rule text.
        rule: String,
    },
    /// List all user contract rules.
    List,
}

// ── Brain mode subcommands ──

/// `axil scip` subcommands. Closes the loop between `axil doctor`
/// (which reports SCIP missing/stale) and `axil ingest-scip` (which
/// only consumes a pre-existing file): `refresh` actually runs the
/// indexer.
#[cfg(feature = "scip")]
#[derive(Subcommand)]
enum ScipCommand {
    /// Run the language indexer(s) for the current repo and ingest the output.
    ///
    /// Detects every (language, project dir) pair via marker files
    /// (Cargo.toml, package.json, pyproject.toml, go.mod, pom.xml,
    /// build.gradle) — including subfolder-only projects like
    /// `frontend/package.json` + `backend/pyproject.toml` — and runs
    /// each indexer from its own project dir. Single-project repos
    /// write `.axil/index.scip`; polyglot repos write one
    /// `.axil/index-<lang>[-<dir>].scip` per project. A missing indexer
    /// binary skips that project with an install hint (hard error only
    /// with an explicit `--language`).
    Refresh {
        /// Directory to scan for projects. Defaults to the repo root
        /// derived from the database location (`<db>/../..`). Set this
        /// when the database lives outside the project being indexed
        /// (e.g. `axil reindex <path>` propagates the indexed path here)
        /// so the SCIP scan and the proxy index cover the same tree.
        /// Output `.scip` files still land next to the database.
        #[arg(long)]
        root: Option<PathBuf>,
        /// Restrict the run to one language (all of its detected
        /// project dirs). One of: rust, python, typescript, go, java.
        #[arg(long)]
        language: Option<String>,
        /// Output path for the generated SCIP file. Only honored when a
        /// single project is targeted; polyglot runs derive per-project
        /// names from this flag's default location.
        #[arg(long, default_value = ".axil/index.scip")]
        output: PathBuf,
        /// Generate the SCIP file but don't ingest it into the DB.
        #[arg(long)]
        skip_ingest: bool,
        /// Pass through to ingest: parse + count edges without writing.
        #[arg(long)]
        dry_run: bool,
        /// Only refresh when the existing `.scip` file is missing or
        /// older than `--max-age-days`. Cheap (<50ms) when fresh —
        /// safe to call from the brain hook on every session start.
        #[arg(long)]
        if_stale: bool,
        /// Staleness threshold in days. Matches `axil doctor`'s
        /// internal 14-day warning threshold.
        #[arg(long, default_value = "14")]
        max_age_days: u64,
        /// Spawn the refresh as a detached child process and return
        /// immediately. Combine with `--if-stale` to make the brain
        /// hook non-blocking — the child writes a lock file at
        /// `.axil/scip-refresh.lock` so concurrent invocations no-op.
        #[arg(long)]
        in_background: bool,
    },
    /// Report SCIP setup status: detected (language, project dir) pairs,
    /// indexer presence on PATH, existing `.scip` files, and the
    /// suggested install/run command per language.
    Status,
}

/// `axil ext` subcommands — install and manage runtime WASM plugins.
#[cfg(feature = "wasm-host")]
#[derive(Subcommand)]
enum ExtCommand {
    /// List installed WASM plugins and whether each loads.
    List,
    /// Scaffold a new WASM plugin crate ready to `cargo component build`.
    /// Emits a detached (own `[workspace]`) cdylib crate with the bundled
    /// `axil:plugin` WIT, the `sdk::Plugin` authoring layer, and a `lib.rs`
    /// stub overriding one high-value hook. Needs no database.
    New {
        /// Plugin name (kebab-case, e.g. `my-plugin`). Becomes the crate name,
        /// the plugin id, and the table prefix `_<name>_`.
        name: String,
        /// Directory to create the crate in. Defaults to `./<name>`.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Comma-separated host capabilities the plugin will request
        /// (e.g. `recall,records.write`). Recorded in the generated README so
        /// the operator knows what to `axil ext grant` after install.
        #[arg(long)]
        caps: Option<String>,
    },
    /// Install a `.wasm` component plugin: validate it loads, then copy it into
    /// the plugins dir. No rebuild.
    Install {
        /// Path to the `.wasm` component file.
        path: PathBuf,
    },
    /// Remove an installed plugin by its id (deletes its `.wasm`).
    Remove {
        /// Plugin id (as shown by `axil ext list`).
        id: String,
    },
    /// Replace an installed plugin's `.wasm` in place, keeping its data and
    /// capability grants; rolls back if the new file fails to load.
    Upgrade {
        /// Plugin id of the installed plugin to replace (as shown by `axil ext list`).
        id: String,
        /// Path to the new `.wasm` component file.
        path: PathBuf,
    },
    /// Show one installed plugin's id, name, table prefixes, and file.
    Info {
        /// Plugin id.
        id: String,
    },
    /// Grant a capability to a plugin (edits `[plugins.<key>]` in axil.toml).
    /// Plugins are deny-by-default — they can't call back into Axil until
    /// granted. `<key>` is the `.wasm` filename stem (shown by `ext list`).
    Grant {
        /// Plugin key (the `.wasm` filename stem).
        key: String,
        /// Capability: records.read | records.write | recall | embed | graph | fts | config.read.
        capability: String,
    },
    /// Revoke a capability from a plugin.
    Revoke {
        /// Plugin key (the `.wasm` filename stem).
        key: String,
        /// Capability to revoke.
        capability: String,
    },
}

#[derive(Subcommand)]
enum ExtensionsCommand {
    /// List every compiled-in Extension and whether it is active or disabled.
    List,
    /// Re-enable a built-in Extension that was disabled in axil.toml.
    Enable {
        /// Extension id (e.g. `docs`, `checkpoint`).
        id: String,
    },
    /// Disable a built-in Extension for this project (edits axil.toml).
    Disable {
        /// Extension id (e.g. `docs`, `checkpoint`).
        id: String,
    },
}

#[derive(Subcommand)]
enum DepsCommand {
    /// List the project's dependencies resolved to exact lockfile
    /// versions. Scans every Cargo / npm manifest under `--path`.
    List {
        /// Project root to scan. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Include dev and build dependencies, not just direct ones.
        #[arg(long)]
        dev: bool,
    },
    /// Extract each dependency's docs from its on-disk copy and ingest
    /// them into memory as version-pinned `_dep_docs` chunks.
    Sync {
        /// Project root to scan. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Local-only: never touch the network. Currently always on —
        /// the web fallback is a later increment.
        #[arg(long)]
        offline: bool,
        /// Also ingest transitive dependencies that the project's own
        /// source actually imports (Cargo / npm).
        #[arg(long)]
        transitive: bool,
    },
    /// Re-ingest docs for dependencies whose manifest or lockfile
    /// changed since the last sync.
    Refresh {
        /// Project root to scan. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Only refresh manifests that actually changed — a fast no-op
        /// when everything is already fresh.
        #[arg(long)]
        if_stale: bool,
        /// Also ingest transitive dependencies that the project's own
        /// source actually imports (Cargo / npm).
        #[arg(long)]
        transitive: bool,
    },
    /// Ingest docs for one dependency. By default reads agent-supplied
    /// text (Path A — from a file or stdin); `--from-web` fetches over
    /// HTTP (Path B, requires the `web-docs` build feature).
    Ingest {
        /// Dependency as `name@version` (e.g. `tokio@1.40.0`).
        #[arg(long)]
        dep: String,
        /// Ecosystem: `cargo` or `npm`.
        #[arg(long, default_value = "cargo")]
        ecosystem: String,
        /// Read doc text from this file instead of stdin.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Fetch the docs over HTTP instead of reading a file/stdin.
        /// Requires a build with `--features web-docs`.
        #[arg(long)]
        from_web: bool,
    },
    /// Show the dependency-doc memory state: synced deps + manifest drift.
    Status {
        /// Project root to scan. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        path: PathBuf,
    },
}

/// Subcommands for the semantic answer cache (`axil cache …`).
///
/// Each variant marshals into the same `CliInvocation` the generic
/// external-subcommand path builds, so the typed surface here and the
/// Extension's own `handle_cli` stay in lockstep — the Extension remains the
/// single owner of put/get/stats/clear logic.
#[cfg(feature = "cache")]
#[derive(Subcommand)]
enum CacheCommand {
    /// Store a question/answer pair. Inline JSON positional or `-` for stdin:
    /// `{question, answer, code_refs?[], ttl?}`.
    Put {
        /// Inline JSON object, or `-` to read it from stdin.
        json: Option<String>,
    },
    /// Look up a cached answer for a semantically similar question.
    Get {
        /// The question to look up.
        question: String,
        /// Minimum similarity for a hit (default 0.92).
        #[arg(long)]
        threshold: Option<f32>,
        /// Maximum hits to return (default 1).
        #[arg(long)]
        top_k: Option<usize>,
    },
    /// Show cumulative hit / miss / eviction counters.
    Stats,
    /// Remove cached entries. Defaults to expired-only; `--all` wipes them.
    Clear {
        /// Remove every entry.
        #[arg(long)]
        all: bool,
        /// Remove only entries past their TTL (the default).
        #[arg(long)]
        expired: bool,
    },
}

#[derive(Subcommand)]
enum BrainCommand {
    /// Enable brain mode (persisted in axil.toml).
    Enable,
    /// Disable brain mode.
    Disable,
    /// Show brain state: mode, scopes, beliefs, pipeline stats.
    Status,
    /// Trigger offline learning cycle (consolidation + reflection).
    Reflect,
    /// Debug a memory: why-remembered + why-recalled + provenance.
    Debug {
        /// Record ID.
        id: String,
    },
    /// Run the BrainEval suite.
    Eval,
}

#[derive(Subcommand)]
enum RetentionCommand {
    /// Set retention policy for a scope.
    Set {
        /// Scope: session, agent, project, user, global.
        #[arg(long)]
        scope: String,
        /// Days to retain records.
        #[arg(long)]
        days: u64,
    },
    /// Show all retention policies.
    Show,
}

#[derive(Subcommand)]
enum WorkerCommand {
    /// Run all worker tasks (consolidation, connections, stale detection).
    /// With --brain, also runs belief revision, procedure/preference extraction, dedup.
    Run {
        /// Enable brain consolidation tasks.
        #[arg(long)]
        brain: bool,
    },
    /// Show the last worker run report.
    Status,
    /// Run maintenance in a loop for the given duration (seconds).
    Daemon {
        /// Interval between runs in seconds.
        #[arg(long, default_value = "300")]
        interval: u64,
        /// Total duration to run in seconds (0 = run once then exit).
        #[arg(long, default_value = "0")]
        duration: u64,
    },
}

#[derive(Subcommand)]
enum BranchCommand {
    /// Create a new branch (copy of the database).
    Create {
        /// Branch name (alphanumeric, hyphens, underscores).
        name: String,
    },
    /// List all branches.
    List,
    /// Delete a branch.
    Delete {
        /// Branch name to delete.
        name: String,
    },
    /// Compare a branch to the main database.
    Diff {
        /// Branch name to diff.
        name: String,
    },
    /// Switch to a branch (prints the path to use with AXIL_DB).
    Switch {
        /// Branch name to switch to.
        name: String,
    },
    /// Merge a branch back into the main database.
    Merge {
        /// Branch name to merge.
        name: String,
        /// Conflict resolution strategy: branch-wins (default), main-wins, keep-both.
        #[arg(long, default_value = "branch-wins")]
        strategy: String,
        /// Delete the branch after successful merge.
        #[arg(long)]
        delete: bool,
    },
}

#[derive(Subcommand)]
enum ReportCommand {
    /// Generate a field report about Axil problems.
    Generate,
    /// List all generated reports.
    List,
    /// Import a report file into the incoming directory.
    Import {
        /// Path to the report file, or use --from to pull from a project.
        path: Option<PathBuf>,
        /// Pull the latest report from a working project directory.
        #[arg(long)]
        from: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum SkillCommand {
    /// Install all skills to ~/.claude/skills/.
    Install {
        /// Install only a specific skill: memory, report, diagnose, optimize.
        #[arg(long)]
        only: Option<String>,
    },
    /// List installed skills and versions.
    List,
    /// Remove all installed skills.
    Uninstall,
}

/// MCP server management.
#[cfg(feature = "mcp")]
#[derive(Subcommand)]
enum McpCommand {
    /// Register the Axil MCP server in an agent's config file (idempotent
    /// merge — only the `axil` server entry is touched).
    Install {
        /// Target agent: claude-code | cursor | windsurf | codex | copilot |
        /// droid | qwen | antigravity | opencode.
        target: String,
        /// Print the config path and entry without writing.
        #[arg(long)]
        dry_run: bool,
    },
}

/// Agent lifecycle hook runtime.
#[derive(Subcommand)]
enum HookCommand {
    /// Execute one hook event: reads the event JSON on stdin, emits the
    /// dialect's response (JSON or injected context text) on stdout.
    /// Always exits 0 once the input parses — a memory hook must never
    /// break the agent loop.
    Run {
        /// Hook dialect — which agent's JSON contract to speak: claude,
        /// codex, copilot, droid, antigravity, or qwen.
        #[arg(long, default_value = "claude")]
        dialect: String,
        /// Override the event name (defaults to the input's own event field,
        /// e.g. `hook_event_name` in the claude dialect). Required for the
        /// antigravity dialect, whose payload carries no event name.
        #[arg(long)]
        event: Option<String>,
    },
    /// Debug probe: record the raw hook payload plus what the dialect parser
    /// extracted to `.axil/hook-capture.jsonl`, then run the normal loop.
    /// Wire it as an agent's hook command temporarily to confirm a dialect's
    /// field mappings against what the agent actually sends.
    Capture {
        /// Hook dialect to parse the payload as (see `run`).
        #[arg(long, default_value = "claude")]
        dialect: String,
        /// Override the event name (required for antigravity).
        #[arg(long)]
        event: Option<String>,
    },
}

/// Scheduled task operations (12.3).
#[derive(Subcommand)]
enum ScheduleOp {
    /// Install a recurring task. Name is one of: daily-brief, weekly-retro, monthly-retro.
    Install {
        /// Task name: daily-brief | weekly-retro | monthly-retro.
        name: String,
        /// Hour of day (0–23) for daily tasks. Default 8.
        #[arg(long, default_value = "8")]
        hour: u32,
        /// Minute of hour. Default 0.
        #[arg(long, default_value = "0")]
        minute: u32,
        /// Scheduler to use: auto (default, macos→launchd, linux→systemd), launchd, systemd, cron.
        #[arg(long, default_value = "auto")]
        scheduler: String,
        /// Print the plan without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// List installed axil scheduled tasks.
    List,
    /// Remove a scheduled task.
    Uninstall {
        name: String,
        #[arg(long)]
        dry_run: bool,
    },
}

// ── Workspace / consent / bridge / atlas subcommands ────

#[derive(Subcommand)]
enum WorkspaceOp {
    /// Scaffold a `.axil-workspace.toml` from sibling `.axil/` dirs.
    Init {
        /// Workspace name (label). Defaults to the cwd basename.
        #[arg(long)]
        name: Option<String>,
    },
    /// Show the current workspace topology and which member cwd resolves to.
    Status,
    /// List workspaces known to the global registry.
    List,
    /// Register an additional member DB in the manifest.
    Add {
        /// Path to the sibling `.axil` directory (or memory.axil file).
        path: PathBuf,
        /// Member label (defaults to the parent directory name).
        #[arg(long = "as")]
        as_label: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConsentOp {
    /// Set read/write consent on a single record.
    Set {
        /// Record ID (ULID).
        record_id: String,
        /// Read-consent scope: private | workspace | public | members:a,b | roles:role_ui
        #[arg(long)]
        read: Option<String>,
        /// Write-consent scope: source-only | workspace | members:a,b | roles:role_ui
        #[arg(long)]
        write: Option<String>,
    },
    /// Show the effective consent scopes on a record.
    Show {
        /// Record ID (ULID).
        record_id: String,
    },
    /// Set per-table default consent scopes.
    Default {
        /// Table name.
        #[arg(long)]
        table: String,
        /// Read scope: private | workspace | public | members:a,b | roles:r1,r2
        #[arg(long)]
        read: Option<String>,
        /// Write scope: source-only | workspace | members:a,b | roles:r1,r2
        #[arg(long)]
        write: Option<String>,
    },
    /// Export the audit log for consent changes.
    Audit {
        /// Filter to entries newer than this ISO timestamp.
        #[arg(long)]
        since: Option<String>,
        /// Export format: json (default) or csv.
        #[arg(long = "audit-format", default_value = "json")]
        audit_format: String,
    },
}

#[derive(Subcommand)]
enum BridgeOp {
    /// Assert a bridge from a local canonical id to a remote one.
    Add {
        /// Local canonical id.
        local: String,
        /// Remote spec: `<member>:<canonical>`.
        #[arg(long = "to")]
        to: String,
        /// Evidence: manual | scip_symbol | shared_uri | name_and_type.
        #[arg(long, default_value = "manual")]
        evidence: String,
        /// Confidence override (0.0–1.0).
        #[arg(long)]
        confidence: Option<f32>,
    },
    /// List known bridges.
    List {
        /// Filter by local canonical id.
        #[arg(long)]
        local: Option<String>,
        /// Filter by remote member label.
        #[arg(long)]
        member: Option<String>,
    },
    /// Re-check bridges against the current DB and mark dangling ones.
    Verify,
    /// Auto-bridge: scan siblings for SCIP-canonical-id matches and create
    /// high-confidence bridges. Weak (name-only) matches stay below the
    /// federation's min_bridge_confidence threshold.
    Auto {
        /// Limit the scan to specific member labels (comma-separated; `*` = all).
        #[arg(long, default_value = "*")]
        members: String,
        /// Print what would be created without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
}

// ─── Helper functions ───────────────────────────────────────────────────────

/// Every SCIP-indexable language in the repo, for `axil doctor`
/// suggestions and the ingest-scip not-found hint. Composes the bounded
/// recursive walk in [`scip_detect`], so subfolder-only languages
/// (`frontend/package.json`, `backend/pyproject.toml`) are seen too.
#[cfg(feature = "scip")]
fn detect_scip_indexable_languages(repo_root: &std::path::Path) -> Vec<&'static str> {
    scip_detect::detected_languages(&scip_detect::detect_scip_projects(repo_root))
}

/// RAII lock-file cleanup for background `scip refresh` runs.
/// Deletes the file on drop (success, error, panic) so a crashed
/// refresh doesn't leave a stale lock. Best-effort — never returns
/// errors during cleanup. Always owns the path; a manual foreground
/// `axil scip refresh` will clear an existing lock as a side effect,
/// which is fine because concurrent indexer runs are wasteful but
/// safe and the next stale check will redetect via mtime anyway.
// Shared by `axil scip refresh` and `axil maintain` background spawns.
struct LockGuard(PathBuf);

impl LockGuard {
    /// Drop-only guard — assumes the lock file already exists (the
    /// parent created it via [`LockGuard::try_acquire`]). Used by the
    /// scip child process for cleanup-on-exit.
    #[cfg(feature = "scip")]
    fn new(path: PathBuf) -> Self {
        LockGuard(path)
    }

    /// Atomically claim the lock by creating the file with
    /// `O_CREAT|O_EXCL`. Returns `Err(AlreadyExists)` if another
    /// process already owns the lock — caller bails out gracefully.
    ///
    /// Stale locks must be removed by the caller *before* invoking
    /// this (see the `STALE_LOCK_SECS` checks in `axil scip refresh
    /// --in-background` and `axil maintain --in-background`).
    fn try_acquire(path: PathBuf, content: &str) -> std::io::Result<Self> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        f.write_all(content.as_bytes())?;
        Ok(LockGuard(path))
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Indexer invocation for `axil scip refresh`. Returns
/// `(binary, args_template)` where `{out}` in any arg is replaced
/// with the resolved output path. None for unsupported languages.
///
/// Kept distinct from `suggest_scip_installers` — that returns a
/// shell-pasteable install+run line, this returns the run-only
/// `Command` shape we can spawn directly.
#[cfg(feature = "scip")]
fn scip_indexer_command(lang: &str) -> Option<(&'static str, Vec<&'static str>)> {
    match lang {
        "rust" => Some(("rust-analyzer", vec!["scip", ".", "--output", "{out}"])),
        "typescript" => Some(("scip-typescript", vec!["index", "--output", "{out}"])),
        "python" => Some(("scip-python", vec!["index", "--output", "{out}", "."])),
        "go" => Some(("scip-go", vec!["--output", "{out}"])),
        "java" => Some(("scip-java", vec!["index", "--output", "{out}"])),
        _ => None,
    }
}

/// Resolve `bin` to something `Command::new` can actually launch.
///
/// On Unix this is `bin` itself when `which` finds it. On Windows the
/// distinction matters: npm installs CLIs (scip-python, scip-typescript)
/// as `.cmd` shims, which `where` reports as present but CreateProcess
/// cannot execute — so a bare `Command::new("scip-python")` fails with
/// NotFound even though the probe said "installed". Resolve via `where`,
/// prefer a real `.exe`/`.com`, and fall back to the full path of a
/// `.cmd`/`.bat` shim (std routes those through cmd.exe with safe quoting).
#[cfg(feature = "scip")]
fn resolve_indexer_program(bin: &str) -> Option<std::path::PathBuf> {
    if cfg!(target_os = "windows") {
        let out = std::process::Command::new("where")
            .arg(bin)
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let hits: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        // Extensionless hits (Git-Bash shims) are unspawnable from
        // CreateProcess, so only native binaries and cmd/bat shims count.
        for exts in [&["exe", "com"][..], &["cmd", "bat"][..]] {
            for hit in &hits {
                let lower = hit.to_ascii_lowercase();
                if exts.iter().any(|e| lower.ends_with(&format!(".{e}"))) {
                    return Some(std::path::PathBuf::from(hit));
                }
            }
        }
        None
    } else {
        let found = std::process::Command::new("which")
            .arg(bin)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        found.then(|| std::path::PathBuf::from(bin))
    }
}

/// Returns true if `bin` is invokable — same resolution the spawn uses,
/// so `scip status` can never report an indexer the refresh can't run.
#[cfg(feature = "scip")]
fn binary_on_path(bin: &str) -> bool {
    resolve_indexer_program(bin).is_some()
}

/// Recognize known indexer crash signatures and return a targeted,
/// actionable hint. Currently: scip-python's Windows startup crash from
/// an unescaped `new RegExp(path.sep)` (path.sep is `\` on Windows — an
/// invalid regex), which kills the CLI before it indexes anything.
#[cfg(feature = "scip")]
fn indexer_crash_hint(bin: &str, stderr: &str) -> Option<String> {
    if bin == "scip-python"
        && stderr.contains("Invalid regular expression")
        && stderr.contains("PythonEnvironment")
    {
        return Some(
            "known scip-python bug on Windows (unescaped RegExp(path.sep) at startup). \
             Fix: patch the installed bundle — see docs/src/getting-started/installation.md#windows--scip-python \
             — or track https://github.com/sourcegraph/scip-python/issues for the upstream fix"
                .to_string(),
        );
    }
    None
}

#[cfg(feature = "scip")]
fn suggest_scip_installers(langs: &[&'static str]) -> String {
    let mut lines = Vec::new();
    for l in langs {
        let cmd = match *l {
            "rust" => "rustup component add rust-analyzer && rust-analyzer scip . --output .axil/index.scip",
            "python" => "pipx install scip-python && scip-python index --output .axil/index.scip .",
            "typescript" => "npm install -g @sourcegraph/scip-typescript && scip-typescript index --output .axil/index.scip",
            "go" => "go install github.com/sourcegraph/scip-go/cmd/scip-go@latest && scip-go --output .axil/index.scip",
            "java" => "brew install sourcegraph/scip/scip-java && scip-java index --output .axil/index.scip",
            _ => continue,
        };
        lines.push(format!("  {l}: {cmd}"));
    }
    format!(
        "install an indexer, then either run `axil scip refresh` (auto-detects language and ingests) or write to `.axil/index.scip` and run `axil ingest-scip`:\n{}",
        lines.join("\n")
    )
}

/// Pass-4 of `recall-for-entity`: when a workspace manifest exists,
/// load `_entity_bridges` rows whose `local_canonical` matches the
/// queried canonical id, filter by `federation.min_bridge_confidence`,
/// open each remote sibling, and pull mentions of the bridged
/// canonical id with provenance tags.
///
/// Returns `(remote_hits, trace_hops)`. Missing manifest → empty
/// result, silent. Unreachable siblings surface as `warnings` entries
/// inside the trace hops when `trace_graph=true`.
#[cfg(feature = "scip")]
fn follow_bridges_for_entity(
    db: &axil_core::Axil,
    db_path: &std::path::Path,
    canonical_id: &str,
    top_k: usize,
    trace_graph: bool,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let manifest = match axil_workspace::discover_manifest(db_path) {
        Ok(Some(m)) => m,
        Ok(None) => return Ok((Vec::new(), Vec::new())),
        Err(e) => anyhow::bail!("manifest load failed: {e}"),
    };
    let threshold = manifest.federation.min_bridge_confidence;

    let bridges = db
        .list_bridges(Some(canonical_id), None)
        .unwrap_or_default();
    let live: Vec<_> = bridges
        .into_iter()
        .filter(|r| {
            !r.data
                .get("dangling")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
                && r.data
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .map(|c| c as f32 >= threshold)
                    .unwrap_or(false)
        })
        .collect();

    let mut hits: Vec<Value> = Vec::new();
    let mut trace: Vec<Value> = Vec::new();

    for bridge in live {
        let remote_member_id = bridge
            .data
            .get("remote_member_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let remote_canonical = bridge
            .data
            .get("remote_canonical")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let confidence = bridge
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let evidence_kind = bridge
            .data
            .get("evidence")
            .and_then(|v| v.get("kind"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let Some((member_label, member)) = manifest
            .members
            .iter()
            .find(|(_, m)| m.id == remote_member_id)
        else {
            if trace_graph {
                trace.push(json!({
                    "remote_member_id": remote_member_id,
                    "remote_canonical": remote_canonical,
                    "confidence": confidence,
                    "evidence": evidence_kind,
                    "status": "member not in manifest",
                }));
            }
            continue;
        };
        let remote_path = manifest.member_db_abs(member);
        if !remote_path.exists() {
            if trace_graph {
                trace.push(json!({
                    "remote_member": member_label,
                    "remote_canonical": remote_canonical,
                    "confidence": confidence,
                    "evidence": evidence_kind,
                    "status": "remote db missing",
                }));
            }
            continue;
        }
        let Ok(remote_db) = open_with_all_detected(&remote_path) else {
            if trace_graph {
                trace.push(json!({
                    "remote_member": member_label,
                    "remote_canonical": remote_canonical,
                    "confidence": confidence,
                    "evidence": evidence_kind,
                    "status": "remote open failed",
                }));
            }
            continue;
        };

        // Find the remote entity row by canonical id, follow `mentions`
        // inbound. Each record is filtered against its own
        // `read_consent` so remote privates don't leak.
        let remote_entities = remote_db.list("_entities").unwrap_or_default();
        let Some(remote_entity) = remote_entities.into_iter().find(|r| {
            r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(remote_canonical)
        }) else {
            if trace_graph {
                trace.push(json!({
                    "remote_member": member_label,
                    "remote_canonical": remote_canonical,
                    "confidence": confidence,
                    "evidence": evidence_kind,
                    "status": "bridge_dangling_no_remote_entity",
                }));
            }
            continue;
        };
        let Some(gi) = remote_db.graph_index_ref() else {
            continue;
        };
        let Ok(mentions) = gi.edges(
            remote_entity.id.clone(),
            Some(axil_core::util::edge_types::MENTIONS),
            axil_core::Direction::In,
        ) else {
            continue;
        };

        let caller_ctx = axil_workspace::consent::MatchContext {
            source_workspace: &manifest.workspace.id,
            source_member: &member.id,
            caller_workspace: &manifest.workspace.id,
            caller_member: &String::new(),
            caller_roles: &Vec::new(),
            strict: false,
        };

        let mut surfaced = 0usize;
        for edge in mentions {
            if hits.len() >= top_k {
                break;
            }
            let Ok(Some(record)) = remote_db.get(&edge.from) else {
                continue;
            };
            if record.table.starts_with('_') {
                continue;
            }
            let read_consent: axil_workspace::consent::ReadConsent =
                serde_json::from_value(record.read_consent_raw()).unwrap_or_default();
            if !read_consent.allows(&caller_ctx) {
                continue;
            }
            hits.push(json!({
                "table": record.table,
                "id": record.id.to_string(),
                "summary": record.data.get("summary")
                    .or_else(|| record.data.get("error"))
                    .or_else(|| record.data.get("fact"))
                    .cloned()
                    .unwrap_or_else(|| truncate_value(&record.data, 150)),
                "importance": axil_core::importance::get_importance(&record.data),
                "source": "bridge",
                "source_member": member_label,
                "source_member_id": member.id,
                "bridge_confidence": confidence,
                "bridge_evidence": evidence_kind,
            }));
            surfaced += 1;
        }
        if trace_graph {
            trace.push(json!({
                "remote_member": member_label,
                "remote_canonical": remote_canonical,
                "confidence": confidence,
                "evidence": evidence_kind,
                "status": "followed",
                "surfaced": surfaced,
            }));
        }
    }

    Ok((hits, trace))
}

/// Pass 4 of `recall-for-file`: walk SCIP-grounded entity edges rooted
/// at the file's `_idx_files` row and surface memories attached to
/// entities defined in this file plus their 1-hop neighbors (callers,
/// callees, references, implementations). Deduped via `seen_ids`.
#[cfg(feature = "scip")]
fn scip_entity_pass(
    db: &axil_core::Axil,
    file: &str,
    short_name: &str,
    seen_ids: &mut std::collections::HashSet<String>,
    top_k: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(gi) = db.graph_index_ref() else {
        return out;
    };

    // Find the `_idx_files` row for this path. `recall-for-file` Pass 3
    // already has this machinery; we re-do it so the function works even
    // when `indexer` feature is off.
    let Ok(rows) = db.list("_idx_files") else {
        return out;
    };
    let file_row = rows.into_iter().find(|r| {
        r.data
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| p == file || p.ends_with(short_name))
            .unwrap_or(false)
    });
    let Some(file_row) = file_row else { return out };

    // Entities with `defined_in` edge into this file.
    let incoming = match gi.edges(
        file_row.id.clone(),
        Some(axil_scip::EDGE_DEFINED_IN),
        axil_core::Direction::In,
    ) {
        Ok(v) => v,
        Err(_) => return out,
    };
    let mut entity_ids: Vec<axil_core::RecordId> = incoming.into_iter().map(|e| e.from).collect();

    // Expand 1 hop across calls/references/implements so we can surface
    // memories about callers/callees of each entity in this file.
    let mut neighbors: Vec<axil_core::RecordId> = Vec::new();
    for eid in &entity_ids {
        for etype in &[
            axil_scip::EDGE_CALLS,
            axil_scip::EDGE_REFERENCES,
            axil_scip::EDGE_IMPLEMENTS,
            axil_scip::EDGE_TYPE_OF,
        ] {
            if let Ok(edges) = gi.edges(eid.clone(), Some(*etype), axil_core::Direction::Both) {
                for e in edges {
                    let other = if &e.from == eid { e.to } else { e.from };
                    neighbors.push(other);
                }
            }
        }
    }
    entity_ids.extend(neighbors);
    entity_ids.sort();
    entity_ids.dedup();

    // For each entity, follow `mentions` inbound edges to the memory
    // records that mention it, filtered to user tables.
    for eid in entity_ids.iter().take(top_k * 3) {
        let Ok(mentions) = gi.edges(
            eid.clone(),
            Some(axil_core::util::edge_types::MENTIONS),
            axil_core::Direction::In,
        ) else {
            continue;
        };
        for edge in mentions {
            let rid_str = edge.from.to_string();
            if !seen_ids.insert(rid_str.clone()) {
                continue;
            }
            let Ok(Some(record)) = db.get(&edge.from) else {
                continue;
            };
            if record.table.starts_with('_') {
                continue;
            }
            let confidence = edge
                .properties
                .get("confidence")
                .and_then(|v| v.as_str())
                .unwrap_or("direct")
                .to_string();
            out.push(json!({
                "table": record.table,
                "id": rid_str,
                "summary": record.data.get("summary")
                    .or_else(|| record.data.get("error"))
                    .or_else(|| record.data.get("fact"))
                    .cloned()
                    .unwrap_or_else(|| truncate_value(&record.data, 150)),
                "importance": axil_core::importance::get_importance(&record.data),
                "match": "entity_graph",
                "source": "entity_graph",
                "confidence": confidence,
            }));
            if out.len() >= top_k {
                return out;
            }
        }
    }
    out
}

/// Mirror of one JSON object the hook writes to `/tmp/axil-session-${SID}.problems`.
#[derive(Debug, Clone, serde::Deserialize)]
struct SessionProblem {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    subcommand: String,
    #[serde(default)]
    query: String,
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    stderr: String,
    #[serde(default)]
    at: String,
}

#[derive(Debug, Default)]
struct SessionProblemSummary {
    command_failures: usize,
    empty_recall: usize,
    empty_code_search: usize,
    empty_fts: usize,
    /// First few raw events (capped) to surface in the heal log.
    samples: Vec<Value>,
}

/// Read a JSONL file of session problems. Bad/blank lines are skipped — the
/// file is captured under tight time pressure in the hook so robustness wins
/// over strictness.
fn load_session_problems(path: &Path) -> Result<Vec<SessionProblem>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(p) = serde_json::from_str::<SessionProblem>(line) {
            out.push(p);
        }
    }
    Ok(out)
}

/// Build a hint object iff `count > 0`. Keeps the SessionHeal handler free
/// of three near-identical `if … { hints.push(json!{…}) }` blocks.
fn build_hint(kind: &str, count: usize, evidence: &str, recommendation: &str) -> Option<Value> {
    if count == 0 {
        return None;
    }
    Some(json!({
        "kind": kind,
        "evidence": format!("{} {}", count, evidence),
        "recommendation": recommendation,
    }))
}

/// Bucket session problems for the heal-log payload.
fn classify_session_problems(problems: &[SessionProblem]) -> SessionProblemSummary {
    let mut s = SessionProblemSummary::default();
    for p in problems {
        match p.kind.as_str() {
            "command_failure" => s.command_failures += 1,
            "empty_result" => match p.subcommand.as_str() {
                "recall" | "boot" | "recall-for-file" | "recall-for-entity" => s.empty_recall += 1,
                "code-search" | "code-context" => s.empty_code_search += 1,
                "fts" => s.empty_fts += 1,
                _ => s.empty_recall += 1,
            },
            _ => {}
        }
    }
    // Keep up to 5 samples — enough for context, small enough not to bloat the log.
    s.samples = problems
        .iter()
        .take(5)
        .map(|p| {
            json!({
                "kind": p.kind,
                "subcommand": p.subcommand,
                "query": p.query,
                "exit_code": p.exit_code,
                "stderr": truncate_str(&p.stderr, 200),
                "at": p.at,
            })
        })
        .collect();
    s
}

/// Resolution order:
/// 1. Explicit `--db` flag or `AXIL_DB` env var
/// 2. Auto-detect `.axil/memory.axil` walking up from cwd
fn require_db(db: &Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = db {
        return Ok(path.clone());
    }

    // Auto-detect .axil/memory.axil by walking up from cwd
    if let Some(path) = find_axil_dir() {
        return Ok(path);
    }

    anyhow::bail!(
        "database not found. Options:\n  \
         1. Run `axil install` in your project root to create .axil/memory.axil\n  \
         2. Use --db <path> or set AXIL_DB env var"
    )
}

/// Walk up from cwd looking for `.axil/memory.axil`.
fn find_axil_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(".axil").join("memory.axil");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Run the brain-eval suite against a fresh scratch database, cleaning up the
/// temp directory on both success and error paths. Shared by
/// `axil brain eval` and the top-level `axil brain-eval` command.
fn run_brain_eval_scratch() -> Result<Value> {
    // `tempfile::tempdir()` mkdir's a randomly-named directory with
    // permissions 0700 and returns a guard that recursively removes
    // it on Drop. The randomised path defeats local symlink attacks
    // that the previous `/tmp/axil-eval-<pid>` pattern was vulnerable
    // to (predictable name → attacker pre-creates a symlink → DB
    // file written elsewhere).
    let eval_dir = tempfile::Builder::new()
        .prefix("axil-eval-")
        .tempdir()
        .context("failed to create eval temp dir")?;
    let eval_path = eval_dir.path().join("eval.axil");

    // Build with FTS so the needle-retention retrieval eval has a backend to
    // recall over. The existing write/classify/scope cases don't need
    // it but happily share the DB. FTS-only (no embedder) keeps the eval offline,
    // deterministic, and model-free.
    #[cfg(feature = "fts")]
    let db = open_with_fts(&eval_path).context("failed to create eval database")?;
    #[cfg(not(feature = "fts"))]
    let db = Axil::open(&eval_path)
        .build()
        .context("failed to create eval database")?;

    let report = axil_core::run_brain_eval(&db).context("brain eval failed")?;
    let mut out = serde_json::to_value(&report).unwrap_or(json!(null));

    // Append the synthetic needle-retention retrieval eval. Guards the recall
    // path (recall@k == 100% on planted needles); requires the FTS backend.
    #[cfg(feature = "fts")]
    if let Some(obj) = out.as_object_mut() {
        let needle = axil_core::run_needle_eval(&db).context("needle eval failed")?;
        obj.insert(
            "retrieval".into(),
            serde_json::to_value(&needle).unwrap_or(json!(null)),
        );
    }

    // `eval_dir` Drop cleans up the directory on return.
    Ok(out)
}

/// Maximum bytes read from stdin (16 MB).
const MAX_STDIN_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum value for --top-k / --limit to prevent overflow and unbounded allocation.
const MAX_RESULT_LIMIT: usize = 10_000;

/// Add a pattern to .gitignore if not already present. Returns true if modified.
fn add_to_gitignore(path: &Path, pattern: &str) -> bool {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    if content.lines().any(|line| line.trim() == pattern) {
        return false;
    }
    let separator = if content.is_empty() || content.ends_with('\n') {
        ""
    } else {
        "\n"
    };
    let new_content = format!("{content}{separator}{pattern}\n");
    std::fs::write(path, new_content).is_ok()
}

/// Claude Code hook events Axil owns. New events must land here so install / uninstall / dry-run stay in sync.
const AXIL_HOOK_EVENTS: &[&str] = &["UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"];
/// Legacy hook scripts earlier installs wrote to `.claude/hooks/`. The brain
/// now lives in the binary (`axil hook run`); these names remain only so
/// install/sync/uninstall can clean old copies up.
const AXIL_HOOK_SCRIPTS: &[&str] = &["axil-brain.sh", "store-on-task-complete.sh"];
/// True when a hook-entry `command` field is Axil-owned — either the current
/// `axil hook run` form or one of the legacy shell scripts.
fn is_axil_hook_command(cmd: &str) -> bool {
    AXIL_HOOK_SCRIPTS.iter().any(|s| cmd.contains(s))
        || (cmd.contains("axil") && cmd.contains(" hook run"))
}

/// True when an `axil` executable resolves on PATH.
fn axil_is_on_path() -> bool {
    std::env::var_os("PATH")
        .map(|p| {
            std::env::split_paths(&p)
                .any(|d| d.join(format!("axil{}", std::env::consts::EXE_SUFFIX)).is_file())
        })
        .unwrap_or(false)
}

/// Handshake the PATH `axil` against the current hook contract by running
/// `axil hook run --help`: clap short-circuits `--help` to exit 0 before it
/// ever reads stdin, so a zero exit proves the `hook run` subcommand exists.
/// A binary predating the subcommand emits clap's usage error (exit 2), and a
/// missing/non-executable one fails to spawn — both surface as `false`. stdin
/// is nulled so an old binary that expected a payload sees immediate EOF
/// instead of blocking. Kept non-interactive and quiet (all stdio nulled).
fn probe_path_axil_hook_run() -> bool {
    std::process::Command::new("axil")
        .args(["hook", "run", "--help"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pick the executable string wired into agent hook configs. Bare `axil` is
/// only safe when a PATH copy passes `probe` — otherwise a stale released
/// binary (every one predates `hook run`) would be invoked as
/// `axil hook run --dialect …`, exit clap's usage code 2, and be read by the
/// agent as a BLOCKING PreToolUse/Stop hook error — vetoing every tool call.
/// On any doubt, pin `current_exe`'s absolute path; only when even that is
/// unavailable fall back to the bare name. `probe`/`current_exe` are injected
/// so the branch matrix is testable without a real PATH or subprocess.
fn resolve_axil_exe_with(
    on_path: bool,
    probe: impl FnOnce() -> bool,
    current_exe: impl FnOnce() -> Option<String>,
) -> String {
    if on_path && probe() {
        return "axil".to_string();
    }
    current_exe().unwrap_or_else(|| "axil".to_string())
}

/// The executable to reference from agent configs. Prefers bare `axil` when a
/// PATH copy proves it speaks the current `hook run` contract (portable across
/// machines sharing the repo); otherwise pins this binary's absolute path.
/// Memoized: the handshake spawns at most once per process, not per dialect.
fn resolved_axil_exe() -> String {
    use std::sync::OnceLock;
    static RESOLVED: OnceLock<String> = OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            resolve_axil_exe_with(axil_is_on_path(), probe_path_axil_hook_run, || {
                std::env::current_exe()
                    .ok()
                    .map(|p| p.display().to_string())
            })
        })
        .clone()
}

/// The command string wired into agent hook configs.
fn hook_run_command_for(dialect: &str) -> String {
    let exe = resolved_axil_exe();
    let exe = if exe.contains(' ') {
        format!("\"{exe}\"")
    } else {
        exe
    };
    format!("{exe} hook run --dialect {dialect}")
}

fn hook_run_command() -> String {
    hook_run_command_for("claude")
}

/// Merge Axil's hook entries into a standalone hooks file (Codex/Droid/
/// Copilot), preserving every non-Axil entry. `flat` selects Copilot's
/// shape (event → array of hook definitions) vs the Claude-style shape
/// (event → array of matcher groups each holding a "hooks" array).
/// `version` adds Copilot's required top-level `"version": 1`.
fn merge_hooks_file(
    path: &Path,
    events: &[(&str, Value)],
    flat: bool,
    version: Option<i64>,
) -> Result<bool> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut root: serde_json::Map<String, Value> = if existing.trim().is_empty() {
        serde_json::Map::new()
    } else {
        serde_json::from_str(&existing)
            .with_context(|| format!("{} is not valid JSON — fix it and rerun", path.display()))?
    };
    if let Some(v) = version {
        root.insert("version".to_string(), json!(v));
    }
    let hooks = root.entry("hooks".to_string()).or_insert_with(|| json!({}));
    let hooks_obj = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("'hooks' in {} is not an object", path.display()))?;

    for (event, ours) in events {
        let preserved: Vec<Value> = hooks_obj
            .get(*event)
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter(|entry| {
                        if flat {
                            // Copilot: each entry IS a hook definition with
                            // command/bash/powershell fields.
                            !["command", "bash", "powershell"].iter().any(|k| {
                                entry
                                    .get(*k)
                                    .and_then(Value::as_str)
                                    .map(is_axil_hook_command)
                                    .unwrap_or(false)
                            })
                        } else {
                            // Claude shape: matcher groups with inner hooks.
                            match entry.get("hooks").and_then(Value::as_array) {
                                None => true, // unrecognised — preserve
                                Some(inner) => !inner.iter().any(|h| {
                                    h.get("command")
                                        .and_then(Value::as_str)
                                        .map(is_axil_hook_command)
                                        .unwrap_or(false)
                                }),
                            }
                        }
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let mut merged = preserved;
        merged.extend(ours.as_array().cloned().unwrap_or_default());
        hooks_obj.insert((*event).to_string(), Value::Array(merged));
    }

    let next = serde_json::to_string_pretty(&Value::Object(root))? + "\n";
    let changed = existing != next;
    if changed {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, next)?;
    }
    Ok(changed)
}

/// Claude-shaped matcher-group entry holding one Axil command hook.
fn hook_group(cmd: &str, timeout: i64) -> Value {
    json!([{ "hooks": [{ "type": "command", "command": cmd, "timeout": timeout }] }])
}

/// Codex hooks: `.codex/hooks.json`, Claude-compatible event names and
/// shape. NOTE: Codex trust-hashes each hook definition — a changed
/// definition is silently skipped until the user re-trusts it via /hooks,
/// so this file must stay byte-stable across reinstalls (idempotent merge
/// guarantees that as long as the command string doesn't change).
fn install_codex_hooks(cwd: &Path) -> Result<bool> {
    let cmd = hook_run_command_for("codex");
    merge_hooks_file(
        &cwd.join(".codex").join("hooks.json"),
        &[
            ("SessionStart", hook_group(&cmd, 10)),
            ("UserPromptSubmit", hook_group(&cmd, 3)),
            ("PreToolUse", hook_group(&cmd, 10)),
            ("PostToolUse", hook_group(&cmd, 10)),
            ("Stop", hook_group(&cmd, 15)),
        ],
        false,
        None,
    )
}

/// Copilot CLI hooks: `.github/hooks/axil.json`. Events are registered in
/// PascalCase (the "VS Code compatible" format) so payloads arrive
/// snake_case WITH `hook_event_name` — self-describing input, same as the
/// other dialects. userPromptSubmitted is not registered: its output is
/// documented as ignored. `command` is the cross-platform field.
fn install_copilot_hooks(cwd: &Path) -> Result<bool> {
    let cmd = hook_run_command_for("copilot");
    let def =
        |timeout: i64| json!([{ "type": "command", "command": cmd.as_str(), "timeoutSec": timeout }]);
    merge_hooks_file(
        &cwd.join(".github").join("hooks").join("axil.json"),
        &[
            ("SessionStart", def(10)),
            ("PreToolUse", def(10)),
            ("PostToolUse", def(10)),
            ("Stop", def(15)),
            ("SessionEnd", def(15)),
        ],
        true,
        Some(1),
    )
}

/// Factory Droid hooks: `.factory/hooks.json`, byte-for-byte the Claude
/// Code contract (snake_case stdin, exit-2 blocks) with Droid tool names.
fn install_droid_hooks(cwd: &Path) -> Result<bool> {
    let cmd = hook_run_command_for("droid");
    merge_hooks_file(
        &cwd.join(".factory").join("hooks.json"),
        &[
            ("SessionStart", hook_group(&cmd, 10)),
            ("UserPromptSubmit", hook_group(&cmd, 3)),
            ("PreToolUse", hook_group(&cmd, 10)),
            ("PostToolUse", hook_group(&cmd, 10)),
            ("Stop", hook_group(&cmd, 15)),
            ("SessionEnd", hook_group(&cmd, 15)),
        ],
        false,
        None,
    )
}

/// Best-effort MCP registration for the full installers — the loop works
/// without it (the CLI is the write path), so failures are reported, not
/// fatal.
fn mcp_register_soft(cwd: &Path, target: &str) -> Value {
    #[cfg(feature = "mcp")]
    {
        mcp_register(cwd, target, false).unwrap_or_else(|e| json!({"error": e.to_string()}))
    }
    #[cfg(not(feature = "mcp"))]
    {
        let _ = (cwd, target);
        json!({"skipped": "binary built without the mcp feature"})
    }
}

/// Full Codex integration: hooks + project-scoped MCP + skills in the
/// cross-tool `.agents/skills/` layout (read by Codex, Antigravity, Zed,
/// and Amp). The AGENTS.md contract is written by the default install.
fn install_codex_full(cwd: &Path) -> Result<Value> {
    let hooks_written = install_codex_hooks(cwd)?;
    let skills_root = cwd.join(".agents").join("skills");
    let mut skills = Vec::new();
    for skill in ALL_SKILLS {
        write_skill(&skills_root, skill)?;
        skills.push(skill.name);
    }
    Ok(json!({
        "hooks": cwd.join(".codex").join("hooks.json").display().to_string(),
        "hooks_written": hooks_written,
        "skills_dir": skills_root.display().to_string(),
        "skills": skills,
        "mcp": mcp_register_soft(cwd, "codex"),
        "note": "Codex runs project hooks only after you trust the project AND the hook definitions (run /hooks inside Codex once)",
    }))
}

/// Full Copilot CLI integration: repo hooks + user-level MCP registration.
fn install_copilot_full(cwd: &Path) -> Result<Value> {
    let hooks_written = install_copilot_hooks(cwd)?;
    Ok(json!({
        "hooks": cwd.join(".github").join("hooks").join("axil.json").display().to_string(),
        "hooks_written": hooks_written,
        "mcp": mcp_register_soft(cwd, "copilot"),
        "note": "the same .github/hooks file is also loaded by the Copilot cloud agent from the cloned repo",
    }))
}

/// Full Factory Droid integration: project hooks + project MCP.
fn install_droid_full(cwd: &Path) -> Result<Value> {
    let hooks_written = install_droid_hooks(cwd)?;
    Ok(json!({
        "hooks": cwd.join(".factory").join("hooks.json").display().to_string(),
        "hooks_written": hooks_written,
        "mcp": mcp_register_soft(cwd, "droid"),
    }))
}

/// Build the Antigravity (`agy`) plugin: a directory with `plugin.json`
/// plus a root `hooks.json`, then register it via `agy plugin install`.
///
/// Verified on `agy` 1.1.0: agy does NOT read a loose `.agents/hooks.json`
/// — its extension unit is a *plugin* (`agy plugin install <dir>`), whose
/// hooks live in a `hooks.json` at the plugin root (matcher-group shape).
/// Payloads carry no event name, so each hook passes `--event`.
///
/// Returns (plugin_dir, registered) — `registered` is true only if the
/// `agy plugin install` actually ran (agy on PATH); otherwise the plugin
/// is staged and the caller surfaces the one-line manual command.
fn install_antigravity_plugin(cwd: &Path) -> Result<(PathBuf, bool)> {
    let plugin_dir = cwd.join(".agents").join("axil-plugin");
    std::fs::create_dir_all(&plugin_dir)?;
    std::fs::write(
        plugin_dir.join("plugin.json"),
        "{\n  \"name\": \"axil\",\n  \"version\": \"1.0.0\",\n  \"description\": \"Axil cognitive memory hooks\"\n}\n",
    )?;

    let cmd = |event: &str| format!("{} --event {event}", hook_run_command_for("antigravity"));
    let group = |event: &str, timeout: i64| {
        json!([{ "hooks": [ { "type": "command", "command": cmd(event), "timeout": timeout } ] }])
    };
    // PreInvocation, not SessionStart: Antigravity emits no session-start
    // event, and PreInvocation is the dialect's only context-injection
    // channel — its first fire carries the boot and later fires flush queued
    // context. Registering SessionStart would parse to nothing and leave
    // injection inert. Events must match parse_antigravity's accepted set.
    let hooks = json!({
        "hooks": {
            "PreInvocation": group("PreInvocation", 10),
            "PreToolUse":    group("PreToolUse", 10),
            "PostToolUse":   group("PostToolUse", 10),
            "Stop":          group("Stop", 15),
        }
    });
    std::fs::write(
        plugin_dir.join("hooks.json"),
        serde_json::to_string_pretty(&hooks)? + "\n",
    )?;

    // Best-effort registration. `agy` may not be on PATH (it installs to a
    // per-user dir); try the bare name and the known Windows location.
    let registered = ["agy", "agy.exe"]
        .iter()
        .map(|s| s.to_string())
        .chain(
            axil_core::home_dir()
                .map(|h| {
                    h.join("AppData")
                        .join("Local")
                        .join("agy")
                        .join("bin")
                        .join("agy.exe")
                        .display()
                        .to_string()
                })
                .into_iter(),
        )
        .any(|exe| {
            std::process::Command::new(&exe)
                .args(["plugin", "install"])
                .arg(&plugin_dir)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });
    Ok((plugin_dir, registered))
}

/// Qwen Code hooks: the top-level `hooks` key in `.qwen/settings.json`,
/// Claude-style event names and matcher-group shape — but timeouts in
/// MILLISECONDS. Project-level hooks require the folder to be trusted.
fn install_qwen_hooks(cwd: &Path) -> Result<bool> {
    let cmd = hook_run_command_for("qwen");
    merge_hooks_file(
        &cwd.join(".qwen").join("settings.json"),
        &[
            ("SessionStart", hook_group(&cmd, 10_000)),
            ("UserPromptSubmit", hook_group(&cmd, 3_000)),
            ("PreToolUse", hook_group(&cmd, 10_000)),
            ("PostToolUse", hook_group(&cmd, 10_000)),
            ("Stop", hook_group(&cmd, 15_000)),
            ("SessionEnd", hook_group(&cmd, 15_000)),
        ],
        false,
        None,
    )
}

/// Full Google Antigravity integration: always-on rule + cross-tool skills
/// + Gemini-lineage hooks + workspace MCP. AGENTS.md (default install) is
/// read natively as the context file.
fn install_antigravity_full(cwd: &Path, db_path: &Path) -> Result<Value> {
    // Rules are plain markdown files in .agents/rules/, loaded as
    // persistent prompt-level guidance.
    let rules_dir = cwd.join(".agents").join("rules");
    std::fs::create_dir_all(&rules_dir)?;
    std::fs::write(rules_dir.join("axil.md"), agent_instructions_cursor(db_path))?;

    // Same skills layout Codex reads (.agents/skills/<name>/SKILL.md).
    let skills_root = cwd.join(".agents").join("skills");
    let mut skills = Vec::new();
    for skill in ALL_SKILLS {
        write_skill(&skills_root, skill)?;
        skills.push(skill.name);
    }

    let (plugin_dir, registered) = install_antigravity_plugin(cwd)?;
    let note = if registered {
        "installed the axil plugin via `agy plugin install` (hooks now active)".to_string()
    } else {
        format!(
            "agy not found on PATH — register the hooks yourself: `agy plugin install {}`",
            plugin_dir.display()
        )
    };
    Ok(json!({
        "rules": rules_dir.join("axil.md").display().to_string(),
        "skills_dir": skills_root.display().to_string(),
        "skills": skills,
        "plugin_dir": plugin_dir.display().to_string(),
        "plugin_registered": registered,
        "mcp": mcp_register_soft(cwd, "antigravity"),
        "note": note,
    }))
}

/// Full Qwen Code integration: hooks + MCP in .qwen/settings.json, plus
/// AGENTS.md added to the context-file list so the shared contract loads.
fn install_qwen_full(cwd: &Path) -> Result<Value> {
    let hooks_written = install_qwen_hooks(cwd)?;
    let context_updated = qwen_add_agents_md_context(cwd)?;
    Ok(json!({
        "settings": cwd.join(".qwen").join("settings.json").display().to_string(),
        "hooks_written": hooks_written,
        "context_file_added": context_updated,
        "mcp": mcp_register_soft(cwd, "qwen"),
        "note": "Qwen ships its own LLM-driven auto-memory — set memory.enableManagedAutoMemory=false in ~/.qwen/settings.json to avoid double-capture alongside Axil",
    }))
}

/// Full OpenCode integration: a self-contained local plugin (no npm
/// dependency — OpenCode auto-loads `.opencode/plugins/*.ts`) plus the
/// MCP entry in opencode.json. The plugin is a thin event adapter that
/// shells out to this binary; all cognitive logic stays in the brain.
fn install_opencode_full(cwd: &Path) -> Result<Value> {
    let plugins_dir = cwd.join(".opencode").join("plugins");
    std::fs::create_dir_all(&plugins_dir)?;
    let plugin_path = plugins_dir.join("axil.ts");
    std::fs::write(&plugin_path, OPENCODE_PLUGIN_TEMPLATE)?;
    Ok(json!({
        "plugin": plugin_path.display().to_string(),
        "mcp": mcp_register_soft(cwd, "opencode"),
        "note": "the plugin shells out to the axil binary; OpenCode also reads the AGENTS.md contract (or CLAUDE.md via its Claude Code compatibility)",
    }))
}

/// Ensure `.qwen/settings.json` lists AGENTS.md in `context.fileName` so
/// Qwen loads the cross-tool contract alongside its native QWEN.md.
fn qwen_add_agents_md_context(cwd: &Path) -> Result<bool> {
    let path = cwd.join(".qwen").join("settings.json");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut root: serde_json::Map<String, Value> = if existing.trim().is_empty() {
        serde_json::Map::new()
    } else {
        serde_json::from_str(&existing)
            .with_context(|| format!("{} is not valid JSON — fix it and rerun", path.display()))?
    };
    let context = root.entry("context".to_string()).or_insert_with(|| json!({}));
    let context = context
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("'context' in {} is not an object", path.display()))?;
    let names = match context.get("fileName") {
        // Preserve an existing list; normalize a bare string to a list.
        Some(Value::Array(a)) => a.clone(),
        Some(Value::String(s)) => vec![json!(s)],
        _ => vec![json!("QWEN.md")],
    };
    if names.iter().any(|v| v.as_str() == Some("AGENTS.md")) {
        return Ok(false);
    }
    let mut names = names;
    names.push(json!("AGENTS.md"));
    context.insert("fileName".to_string(), Value::Array(names));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&Value::Object(root))? + "\n",
    )?;
    Ok(true)
}

/// Install Axil brain hooks into .claude/settings.json, merging with existing config.
///
/// Preserves all existing settings (permissions, other hooks). Only adds
/// Axil hook entries to the `hooks` object if not already present.
fn install_hooks_to_settings(path: &Path) -> Result<bool> {
    let mut settings: serde_json::Map<String, Value> = if path.exists() {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        serde_json::Map::new()
    };

    // Note: we always replace Axil hooks to ensure they stay up to date.
    // Non-Axil hooks in settings.json are preserved since we only overwrite
    // the PreToolUse/PostToolUse/Stop keys.

    // Get or create the hooks object — preserve existing hooks
    let hooks = settings
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}));
    let hooks_obj = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("settings.json 'hooks' is not an object"))?;

    // Define our hook entries. One command serves every event — the brain
    // dispatches on the event name in the input JSON.
    let cmd_owned = hook_run_command();
    let cmd = cmd_owned.as_str();
    let axil_hooks = [
        // 12.1: inject <context> block on every user prompt. Tight 3s cap —
        // the brain enforces its own 1.8s deadline inside `axil recall`.
        (
            "UserPromptSubmit",
            json!([{
                "hooks": [{
                    "type": "command",
                    "command": cmd,
                    "timeout": 3
                }]
            }]),
        ),
        (
            "PreToolUse",
            json!([{
                "hooks": [{
                    "type": "command",
                    "command": cmd,
                    "timeout": 10
                }]
            }]),
        ),
        (
            "PostToolUse",
            json!([
                {
                    "matcher": "Edit|Write",
                    "hooks": [{
                        "type": "command",
                        "command": cmd,
                        "async": true,
                        "timeout": 10
                    }]
                },
                {
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "command": cmd,
                        "async": true,
                        "timeout": 10
                    }]
                },
                {
                    // Fallback-capture: when Read happens shortly after an
                    // empty Axil recall, the hook stores a low-importance
                    // context row tying the missed query to the file:line
                    // range the agent opened. Async — never blocks the agent.
                    "matcher": "Read",
                    "hooks": [{
                        "type": "command",
                        "command": cmd,
                        "async": true,
                        "timeout": 5
                    }]
                },
                {
                    // Store reminder when a todo flips to completed. Matches
                    // TodoWrite — the tool stock Claude Code actually emits
                    // (the old TaskUpdate matcher never fired). Synchronous:
                    // the reminder must land before the agent moves on.
                    "matcher": "TodoWrite",
                    "hooks": [{
                        "type": "command",
                        "command": cmd,
                        "timeout": 5
                    }]
                }
            ]),
        ),
        // Stop must be SYNCHRONOUS — the hook returns {"decision":"block"}
        // when files were edited but no narrative was stored, which only
        // works if the harness waits for and reads stdout. Under async mode
        // the block decision is silently dropped.
        (
            "Stop",
            json!([{
                "hooks": [{
                    "type": "command",
                    "command": cmd,
                    "timeout": 15
                }]
            }]),
        ),
    ];

    // Merge our hook entries onto each event's existing array,
    // preserving any non-Axil hooks the user already had on the same
    // event (matcher rules, third-party hook scripts, etc.). For each
    // managed event:
    //
    //   1. Partition the existing array via `is_axil_hook_command` —
    //      Axil-owned matcher groups go in the "replace" set, anything
    //      else stays.
    //   2. Concatenate non-Axil matcher groups + our fresh `axil_hooks`
    //      entries → write back.
    //
    // This makes `axil hooks install` idempotent AND non-destructive.
    for (event, our_entries) in axil_hooks {
        let preserved: Vec<Value> = hooks_obj
            .get(event)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|matcher_group| {
                        // Strip only Axil-owned commands from the group's
                        // inner `hooks` array — never drop the whole group,
                        // or a user hook sharing a matcher with Axil's would
                        // be silently deleted. Keep the group if any non-Axil
                        // hook survives (or the shape is unrecognised).
                        let Some(inner) = matcher_group.get("hooks").and_then(|h| h.as_array())
                        else {
                            return Some(matcher_group.clone()); // unknown shape — preserve
                        };
                        let kept: Vec<Value> = inner
                            .iter()
                            .filter(|h| {
                                !h.get("command")
                                    .and_then(|c| c.as_str())
                                    .map(is_axil_hook_command)
                                    .unwrap_or(false)
                            })
                            .cloned()
                            .collect();
                        if kept.is_empty() {
                            return None; // the group was purely Axil's — drop it
                        }
                        let mut group = matcher_group.clone();
                        group["hooks"] = Value::Array(kept);
                        Some(group)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let our_arr = our_entries.as_array().cloned().unwrap_or_default();
        let mut merged = preserved;
        merged.extend(our_arr);
        hooks_obj.insert(event.to_string(), Value::Array(merged));
    }

    // Pre-seed `permissions.allow` so the agent never gets prompted on the
    // core read/write commands. Permission prompts teach avoidance — if the
    // agent ever sees a denial, it stops trying. The patterns cover store,
    // recall, boot, and a generic `Bash(axil:*)` catchall so subcommands
    // (since, fts, brief, retro, etc.) work without re-listing every verb.
    // Existing user-defined entries are preserved; we only add ours.
    let permissions = settings
        .entry("permissions".to_string())
        .or_insert_with(|| json!({}));
    if let Some(perms_obj) = permissions.as_object_mut() {
        let allow = perms_obj
            .entry("allow".to_string())
            .or_insert_with(|| json!([]));
        if let Some(allow_arr) = allow.as_array_mut() {
            for pattern in [
                "Bash(axil:*)",
                "Bash(axil store:*)",
                "Bash(axil recall:*)",
                "Bash(axil boot:*)",
                "Bash(axil since:*)",
                "Bash(axil fts:*)",
                "Bash(axil get:*)",
                "Bash(axil list:*)",
                "Bash(axil info:*)",
                "Bash(axil scip:*)",
            ] {
                let already = allow_arr.iter().any(|v| v.as_str() == Some(pattern));
                if !already {
                    allow_arr.push(Value::String(pattern.to_string()));
                }
            }
        }
    }

    std::fs::write(
        path,
        serde_json::to_string_pretty(&Value::Object(settings))?,
    )?;
    Ok(true)
}

/// Preview every file/entry that an `axil install` run would touch, without writing.
#[allow(clippy::too_many_arguments)]
fn dry_run_install_plan(
    cwd: &Path,
    claude_code: bool,
    codex: bool,
    copilot: bool,
    droid: bool,
    antigravity: bool,
    qwen: bool,
    opencode: bool,
    cursor: bool,
    windsurf: bool,
    aider: bool,
    agents_md: bool,
    all: bool,
    local: bool,
) -> Result<Value> {
    let axil_dir = cwd.join(".axil");
    let db_path = axil_dir.join("memory.axil");
    let mut files: Vec<String> = vec![
        axil_dir.display().to_string() + "/",
        db_path.display().to_string(),
        axil_dir.join("version").display().to_string(),
        format!(
            "{} (seed: pinned `rules` row marker={AXIL_FIRST_SEED_MARKER})",
            db_path.display()
        ),
    ];
    let mut gitignore_note = None;
    let gitignore_path = cwd.join(".gitignore");
    if !gitignore_path.exists()
        || !std::fs::read_to_string(&gitignore_path)
            .unwrap_or_default()
            .contains(".axil/")
    {
        gitignore_note = Some(format!(
            "would append `.axil/` to {}",
            gitignore_path.display()
        ));
    }
    if claude_code || all {
        let cc_dir = cwd.join(".claude");
        files.push(cc_dir.join("CLAUDE.md").display().to_string());
        files.push(format!(
            "{} (merged: hooks → `{}`)",
            cc_dir.join("settings.json").display(),
            hook_run_command()
        ));
        let skills_target = if local {
            cc_dir.join("skills")
        } else {
            skills_dir()?
        };
        for skill in ALL_SKILLS {
            files.push(
                skills_target
                    .join(skill.dir_name())
                    .join("SKILL.md")
                    .display()
                    .to_string(),
            );
        }
        if let Some(mem_dir) = claude_auto_memory_dir(cwd) {
            files.push(
                mem_dir
                    .join(AXIL_FIRST_FEEDBACK_FILENAME)
                    .display()
                    .to_string(),
            );
            files.push(mem_dir.join("MEMORY.md").display().to_string() + " (appended)");
        }
    }
    if codex || all {
        files.push(cwd.join(".codex").join("hooks.json").display().to_string() + " (merged)");
        files.push(cwd.join(".codex").join("config.toml").display().to_string() + " (MCP entry)");
        for skill in ALL_SKILLS {
            files.push(
                cwd.join(".agents")
                    .join("skills")
                    .join(skill.dir_name())
                    .join("SKILL.md")
                    .display()
                    .to_string(),
            );
        }
    }
    if copilot || all {
        files.push(
            cwd.join(".github")
                .join("hooks")
                .join("axil.json")
                .display()
                .to_string()
                + " (merged)",
        );
        files.push("~/.copilot/mcp-config.json (MCP entry)".to_string());
    }
    if droid || all {
        files.push(cwd.join(".factory").join("hooks.json").display().to_string() + " (merged)");
        files.push(cwd.join(".factory").join("mcp.json").display().to_string() + " (MCP entry)");
    }
    if antigravity || all {
        files.push(
            cwd.join(".agents").join("axil-plugin").display().to_string()
                + "/ (plugin.json + hooks.json → `agy plugin install`)",
        );
        files.push(cwd.join(".agents").join("rules").join("axil.md").display().to_string());
        files.push(
            cwd.join(".agents")
                .join("mcp_config.json")
                .display()
                .to_string()
                + " (MCP entry)",
        );
        for skill in ALL_SKILLS {
            files.push(
                cwd.join(".agents")
                    .join("skills")
                    .join(skill.dir_name())
                    .join("SKILL.md")
                    .display()
                    .to_string(),
            );
        }
    }
    if qwen || all {
        files.push(
            cwd.join(".qwen").join("settings.json").display().to_string()
                + " (merged: hooks + MCP + context.fileName)",
        );
    }
    if opencode || all {
        files.push(
            cwd.join(".opencode")
                .join("plugins")
                .join("axil.ts")
                .display()
                .to_string(),
        );
        files.push(cwd.join("opencode.json").display().to_string() + " (MCP entry)");
    }
    if cursor || all {
        files.push(
            cwd.join(".cursor")
                .join("rules")
                .join("axil.mdc")
                .display()
                .to_string(),
        );
    }
    if windsurf || all {
        files.push(cwd.join(".windsurfrules").display().to_string());
    }
    if aider || all {
        files.push(cwd.join("CONVENTIONS.md").display().to_string() + " (merged)");
        files.push(cwd.join(".aider.conf.yml").display().to_string() + " (read: key)");
    }
    if agents_md || all {
        files.push(cwd.join("AGENTS.md").display().to_string() + " (merged)");
    }
    Ok(json!({
        "dry_run": true,
        "would_write": files,
        "gitignore": gitignore_note,
        "hook_events_to_register": AXIL_HOOK_EVENTS,
    }))
}

/// Strip Axil's managed `<!-- AXIL:BEGIN/END -->` block from a markdown
/// file (AGENTS.md / CONVENTIONS.md). Deletes the file if nothing but the
/// block remains. Returns true if anything changed.
fn remove_axil_block(path: &Path, dry_run: bool) -> bool {
    const BEGIN: &str = "<!-- AXIL:BEGIN -->";
    const END: &str = "<!-- AXIL:END -->";
    let Ok(existing) = std::fs::read_to_string(path) else {
        return false;
    };
    let Some(start) = existing.find(BEGIN) else {
        return false;
    };
    let Some(end_rel) = existing[start..].find(END) else {
        return false;
    };
    let end = start + end_rel + END.len();
    let head = existing[..start].trim_end();
    let tail = existing[end..].trim_start();
    let remainder = if head.is_empty() {
        tail.to_string()
    } else if tail.is_empty() {
        head.to_string()
    } else {
        format!("{head}\n\n{tail}")
    };
    if dry_run {
        return true;
    }
    if remainder.trim().is_empty() {
        let _ = std::fs::remove_file(path);
    } else {
        let _ = std::fs::write(path, remainder + "\n");
    }
    true
}

/// Remove a top-level `<top_key>.axil` entry from a JSON config (MCP
/// registrations, and Antigravity's named `axil-brain` hook key when
/// top_key is empty → operates at the root). Returns true if removed.
fn remove_json_entry(path: &Path, top_key: &str, entry_key: &str, dry_run: bool) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(mut root) = serde_json::from_str::<serde_json::Map<String, Value>>(&content) else {
        return false;
    };
    let removed = if top_key.is_empty() {
        root.remove(entry_key).is_some()
    } else {
        root.get_mut(top_key)
            .and_then(|v| v.as_object_mut())
            .map(|obj| obj.remove(entry_key).is_some())
            .unwrap_or(false)
    };
    if removed && !dry_run {
        let _ = std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(root)).unwrap_or(content) + "\n",
        );
    }
    removed
}

/// Strip Axil hook entries from a Claude-shaped hooks JSON file (used by
/// Codex `.codex/hooks.json`, Droid `.factory/hooks.json`, and Qwen
/// `.qwen/settings.json`), keeping every non-Axil hook. Returns true if
/// anything changed.
fn remove_claude_style_hooks(path: &Path, dry_run: bool) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(mut root) = serde_json::from_str::<serde_json::Map<String, Value>>(&content) else {
        return false;
    };
    let Some(hooks) = root.get_mut("hooks").and_then(|v| v.as_object_mut()) else {
        return false;
    };
    let mut changed = false;
    let events: Vec<String> = hooks.keys().cloned().collect();
    for event in events {
        let Some(Value::Array(entries)) = hooks.get_mut(&event) else {
            continue;
        };
        let before = entries.len();
        entries.retain_mut(|entry| {
            if let Some(Value::Array(inner)) = entry.get_mut("hooks") {
                inner.retain(|h| {
                    h.get("command")
                        .and_then(|c| c.as_str())
                        .map(|s| !is_axil_hook_command(s))
                        .unwrap_or(true)
                });
                !inner.is_empty()
            } else {
                true
            }
        });
        if entries.len() != before {
            changed = true;
        }
        if entries.is_empty() {
            hooks.remove(&event);
        }
    }
    if changed && !dry_run {
        let _ = std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(root)).unwrap_or(content) + "\n",
        );
    }
    changed
}

/// Remove the Axil integrations for the six terminal agents + Cursor /
/// Aider / AGENTS.md that `axil install` writes outside `.claude/`. Best
/// effort: hook wirings first (those error on every event once the binary
/// is gone), then MCP entries and managed contract blocks. Files that are
/// entirely Axil's are deleted; shared configs have only the axil entry
/// stripped. Never touches the database.
fn uninstall_agent_integrations(cwd: &Path, dry_run: bool, removed: &mut Vec<String>) {
    let mut note = |path: PathBuf, hit: bool| {
        if hit {
            removed.push(path.display().to_string());
        }
    };
    let del = |path: PathBuf| -> bool {
        if !path.exists() {
            return false;
        }
        if !dry_run {
            let _ = std::fs::remove_file(&path);
        }
        true
    };
    let del_dir = |path: PathBuf| -> bool {
        if !path.is_dir() {
            return false;
        }
        if !dry_run {
            let _ = std::fs::remove_dir_all(&path);
        }
        true
    };

    // Codex: hooks + project MCP + shared .agents/skills.
    note(cwd.join(".codex/hooks.json"), remove_claude_style_hooks(&cwd.join(".codex/hooks.json"), dry_run));
    #[cfg(feature = "mcp")]
    {
        let codex_toml = cwd.join(".codex/config.toml");
        if let Ok(content) = std::fs::read_to_string(&codex_toml) {
            if let Ok(mut root) = content.parse::<toml::Table>() {
                let hit = root
                    .get_mut("mcp_servers")
                    .and_then(|v| v.as_table_mut())
                    .map(|t| t.remove("axil").is_some())
                    .unwrap_or(false);
                if hit && !dry_run {
                    let _ = std::fs::write(&codex_toml, toml::to_string_pretty(&root).unwrap_or(content));
                }
                note(codex_toml, hit);
            }
        }
    }

    // Copilot: the hooks file is entirely ours; MCP is a per-user global.
    note(cwd.join(".github/hooks/axil.json"), del(cwd.join(".github/hooks/axil.json")));
    #[cfg(feature = "mcp")]
    if let Some(home) = axil_core::home_dir() {
        let cfg = home.join(".copilot").join("mcp-config.json");
        note(cfg.clone(), remove_json_entry(&cfg, "mcpServers", "axil", dry_run));
    }

    // Droid: hooks + project MCP.
    note(cwd.join(".factory/hooks.json"), remove_claude_style_hooks(&cwd.join(".factory/hooks.json"), dry_run));
    note(cwd.join(".factory/mcp.json"), remove_json_entry(&cwd.join(".factory/mcp.json"), "mcpServers", "axil", dry_run));

    // Antigravity: plugin dir + rule + MCP (skills shared with Codex,
    // removed once below). Also strip the legacy `.agents/hooks.json`
    // axil-brain key from pre-plugin installs, and unregister the plugin
    // from agy if it's on PATH.
    note(cwd.join(".agents/axil-plugin"), del_dir(cwd.join(".agents/axil-plugin")));
    note(cwd.join(".agents/hooks.json"), remove_json_entry(&cwd.join(".agents/hooks.json"), "", "axil-brain", dry_run));
    note(cwd.join(".agents/rules/axil.md"), del(cwd.join(".agents/rules/axil.md")));
    note(cwd.join(".agents/mcp_config.json"), remove_json_entry(&cwd.join(".agents/mcp_config.json"), "mcpServers", "axil", dry_run));
    if !dry_run {
        for exe in ["agy", "agy.exe"] {
            let _ = std::process::Command::new(exe)
                .args(["plugin", "uninstall", "axil"])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }

    // Shared cross-tool skills under .agents/skills (Codex + Antigravity).
    for skill in ALL_SKILLS {
        note(
            cwd.join(".agents/skills").join(skill.dir_name()),
            del_dir(cwd.join(".agents/skills").join(skill.dir_name())),
        );
    }

    // Qwen: hooks + MCP in the same settings file.
    let qwen = cwd.join(".qwen/settings.json");
    let qwen_hooks = remove_claude_style_hooks(&qwen, dry_run);
    let qwen_mcp = remove_json_entry(&qwen, "mcpServers", "axil", dry_run);
    note(qwen, qwen_hooks || qwen_mcp);

    // OpenCode: the plugin is ours; strip MCP from whichever config exists.
    note(cwd.join(".opencode/plugins/axil.ts"), del(cwd.join(".opencode/plugins/axil.ts")));
    for name in ["opencode.json", "opencode.jsonc"] {
        note(cwd.join(name), remove_json_entry(&cwd.join(name), "mcp", "axil", dry_run));
    }

    // Cursor rule (ours), Aider + AGENTS.md managed blocks.
    note(cwd.join(".cursor/rules/axil.mdc"), del(cwd.join(".cursor/rules/axil.mdc")));
    note(cwd.join("CONVENTIONS.md"), remove_axil_block(&cwd.join("CONVENTIONS.md"), dry_run));
    note(cwd.join("AGENTS.md"), remove_axil_block(&cwd.join("AGENTS.md"), dry_run));
}

/// 12.1: remove Axil-owned hook entries from `.claude/settings.json` plus hook scripts + skills.
/// Preserves the database, CLAUDE.md, and any non-Axil settings. Safe to rerun.
fn uninstall_claude_code_files(cwd: &Path, dry_run: bool) -> Result<Value> {
    let claude_dir = cwd.join(".claude");
    let mut removed: Vec<String> = Vec::new();
    let skipped: Vec<String> = Vec::new();

    // 1. Hook scripts — drive from the shared catalog so new scripts removed automatically.
    for name in AXIL_HOOK_SCRIPTS {
        let p = claude_dir.join("hooks").join(name);
        if p.exists() {
            if !dry_run {
                if let Err(e) = std::fs::remove_file(&p) {
                    eprintln!("[uninstall] warn: failed to remove {}: {e}", p.display());
                }
            }
            removed.push(p.display().to_string());
        }
    }

    // 2. Axil skills — check both project-local (.claude/skills/, used by
    //    `axil install --local`) and the global skills dir (~/.claude/skills/).
    //    Auto-detected from disk so we clean up regardless of which mode the
    //    original install used.
    let mut skills_dirs: Vec<PathBuf> = vec![claude_dir.join("skills")];
    if let Ok(global) = skills_dir() {
        skills_dirs.push(global);
    }
    for skills_dir_path in &skills_dirs {
        for skill in ALL_SKILLS {
            // Current skills/<name>/SKILL.md layout plus the legacy flat file.
            let skill_dir = skills_dir_path.join(skill.dir_name());
            if skill_dir.is_dir() {
                if !dry_run {
                    if let Err(e) = std::fs::remove_dir_all(&skill_dir) {
                        eprintln!(
                            "[uninstall] warn: failed to remove {}: {e}",
                            skill_dir.display()
                        );
                    }
                }
                removed.push(skill_dir.display().to_string());
            }
            let p = skills_dir_path.join(skill.filename);
            if p.is_file() {
                if !dry_run {
                    if let Err(e) = std::fs::remove_file(&p) {
                        eprintln!("[uninstall] warn: failed to remove {}: {e}", p.display());
                    }
                }
                removed.push(p.display().to_string());
            }
        }
    }

    // 3. Settings.json — remove Axil hook entries only; keep everything else.
    let settings_path = claude_dir.join("settings.json");
    if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path).unwrap_or_default();
        let mut settings: serde_json::Map<String, Value> =
            serde_json::from_str(&content).unwrap_or_default();
        if let Some(hooks) = settings.get_mut("hooks").and_then(|v| v.as_object_mut()) {
            // Per-entry filter: strip Axil-owned commands from each array, keep user entries,
            // drop the event key only when every entry was ours.
            for event in AXIL_HOOK_EVENTS {
                let Some(Value::Array(entries)) = hooks.get_mut(*event) else {
                    continue;
                };
                let before = entries.len();
                entries.retain_mut(|entry| {
                    // Inside a single entry, filter its `.hooks[*]` array down to non-Axil commands.
                    if let Some(Value::Array(inner)) = entry.get_mut("hooks") {
                        inner.retain(|h| {
                            h.get("command")
                                .and_then(|c| c.as_str())
                                .map(|s| !is_axil_hook_command(s))
                                .unwrap_or(true) // unknown shape → keep (don't destroy user data)
                        });
                        !inner.is_empty()
                    } else {
                        true
                    }
                });
                let after = entries.len();
                if entries.is_empty() {
                    hooks.remove(*event);
                    removed.push(format!("{}:hooks.{}", settings_path.display(), event));
                } else if after < before {
                    removed.push(format!(
                        "{}:hooks.{} ({} Axil entries removed, {} user entries kept)",
                        settings_path.display(),
                        event,
                        before - after,
                        after
                    ));
                }
            }
        }
        if !dry_run {
            std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&Value::Object(settings))?,
            )?;
        }
    }

    // 4. The six terminal-agent integrations + Cursor/Aider/AGENTS.md.
    //    Leaving these behind means their hooks keep invoking `axil hook
    //    run` on every event — and hard-error once the binary is removed.
    let mut removed = removed;
    uninstall_agent_integrations(cwd, dry_run, &mut removed);

    Ok(json!({
        "uninstalled": !dry_run,
        "dry_run": dry_run,
        "removed": removed,
        "skipped": skipped,
        "note": "Database and CLAUDE.md preserved. Delete .axil/ manually if you want to wipe memory.",
    }))
}

// ── Axil-first lock-in (rule + auto-memory) ──────────────────────────────────
// `.claude/CLAUDE.md` instructions get skipped by Claude Code in some projects.
// To make the "use Axil first / store immediately" rule actually stick, install
// also writes it in two harness-loaded surfaces: (1) the Axil DB itself, as a
// pinned `rules` row that the boot Constraints section never drops; (2) the
// per-project Claude auto-memory at `~/.claude/projects/<slug>/memory/`, whose
// `MEMORY.md` index is loaded into every conversation. Both writes are
// idempotent: re-running install/update detects existing seeds and skips.

const AXIL_FIRST_SEED_MARKER: &str = "axil_first_v1";

const AXIL_FIRST_RULE_TEXT: &str = "Always consult Axil first. Before any \
repo-discovery (rg/grep/find/fd/ls/tree) or 'where/what/how/what changed' \
question, run `axil boot`, `axil recall`, `axil code-search`, `axil fts`, or \
`axil recall-for-file`, then verify against current files. After every \
completed unit of work (decision, fix, summary, gotcha), run `axil store …` \
immediately — never batch at the end. The .claude/CLAUDE.md guidance is \
mandatory, not optional.";

const AXIL_FIRST_FEEDBACK_FILENAME: &str = "feedback_axil_proactive.md";

/// Render the feedback file body using the actual project name so the
/// frontmatter `name`/`description` and the lead sentence read naturally
/// in any repo. Mirrors the wording the user landed on in their working
/// project so future sessions see the same phrasing they already validated.
fn render_axil_first_feedback_body(project_name: &str) -> String {
    format!(
        r#"---
name: Use axil proactively in {project_name}
description: This project enforces immediate (not batched) axil writes after each unit of work, plus axil recall before broad exploration
type: feedback
---
In the {project_name} project, axil (`.axil/memory.axil`) is the persistent brain. The project's `.claude/CLAUDE.md` is explicit: store **immediately**, not in batches; recall **before** broad discovery.

**Why:** Batching axil writes at session end defeats the point — future sessions miss decisions/architecture that happened mid-session. Project-level `.claude/CLAUDE.md` instructions get skipped sometimes, so this rule lives in two places: this auto-memory entry (loaded every conversation) and a pinned `rules` row in the project DB (`_seed: "axil_first_v1"`, surfaced by `axil boot`).

**How to apply:**
- Run `axil boot` or `axil recall "<topic>" --top-k 5` BEFORE broad repo exploration. Honor the AXIL search gate hook on first Bash use.
- After every concrete unit of work, store *before* responding to the user:
  - `axil store decisions ...` after a design choice
  - `axil store errors ...` after a bug/gotcha/fix
  - `axil store context '{{"type":"architecture",...}}'` after learning how something works
  - `axil checkpoint '{{"state":"<where things stand>","next_steps":["<remaining work>"]}}'` after finishing a task
- Don't wait for the user to say "update axil memory" — that means I already missed the moment.
"#
    )
}

const AXIL_FIRST_MEMORY_INDEX_LINE: &str =
    "- [Use axil proactively](feedback_axil_proactive.md) — store axil records immediately, not batched at session end; recall before broad exploration";

/// Insert a pinned "use Axil first / store immediately" rule into the `rules`
/// table so the boot Constraints section surfaces it on every session start.
/// Idempotent — checks for `_seed: AXIL_FIRST_SEED_MARKER` before writing.
/// Returns `true` when newly inserted, `false` when already present.
fn seed_axil_first_rule(db: &Axil) -> Result<bool> {
    if let Ok(existing) = db.list("rules") {
        if existing
            .iter()
            .any(|r| r.data.get("_seed").and_then(|v| v.as_str()) == Some(AXIL_FIRST_SEED_MARKER))
        {
            return Ok(false);
        }
    }
    db.insert(
        "rules",
        json!({
            "rule": AXIL_FIRST_RULE_TEXT,
            "_importance": 1.0,
            "_importance_pinned": true,
            "_seed": AXIL_FIRST_SEED_MARKER,
        }),
    )?;
    Ok(true)
}

/// Compute the Claude Code per-project auto-memory directory:
/// `~/.claude/projects/<absolute-path-with-slashes-replaced-by-dashes>/memory`.
/// Returns `None` when `HOME` isn't set.
fn claude_auto_memory_dir(project: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let canonical = project
        .canonicalize()
        .unwrap_or_else(|_| project.to_path_buf());
    let slug = canonical.to_string_lossy().replace('/', "-");
    Some(
        PathBuf::from(home)
            .join(".claude")
            .join("projects")
            .join(slug)
            .join("memory"),
    )
}

/// Write the Axil-first feedback file into Claude's per-project auto-memory
/// and ensure `MEMORY.md` indexes it. Both operations are idempotent — the
/// feedback file is only written when missing, and the index pointer is
/// appended only when not already present (matched by filename).
fn install_claude_auto_memory(cwd: &Path) -> Result<Value> {
    let Some(mem_dir) = claude_auto_memory_dir(cwd) else {
        return Ok(json!({"skipped": "HOME not set"}));
    };
    std::fs::create_dir_all(&mem_dir)
        .with_context(|| format!("failed to create {}", mem_dir.display()))?;

    let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let project_name = canonical_cwd
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");

    let feedback_file = mem_dir.join(AXIL_FIRST_FEEDBACK_FILENAME);
    let feedback_written = if !feedback_file.exists() {
        std::fs::write(
            &feedback_file,
            render_axil_first_feedback_body(project_name),
        )?;
        true
    } else {
        false
    };

    let memory_index = mem_dir.join("MEMORY.md");
    let existing = std::fs::read_to_string(&memory_index).unwrap_or_default();
    let index_updated = if !existing.contains(AXIL_FIRST_FEEDBACK_FILENAME) {
        let mut content = existing.clone();
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(AXIL_FIRST_MEMORY_INDEX_LINE);
        content.push('\n');
        std::fs::write(&memory_index, content)?;
        true
    } else {
        false
    };

    Ok(json!({
        "memory_dir": mem_dir.display().to_string(),
        "feedback_file": feedback_file.display().to_string(),
        "feedback_written": feedback_written,
        "memory_index": memory_index.display().to_string(),
        "memory_index_updated": index_updated,
    }))
}

/// Install/update Claude Code integration files (hook, CLAUDE.md, skills, settings.json).
/// When `force_claude_md` is true, always overwrites CLAUDE.md (update behavior).
/// When false, only writes if missing or doesn't contain Axil header (install behavior).
/// When `local_skills` is true, skills install to `<project>/.claude/skills/` instead
/// of `~/.claude/skills/` — fully self-contained per-project install.
fn install_claude_code_files(
    cwd: &Path,
    force_claude_md: bool,
    local_skills: bool,
) -> Result<Value> {
    let claude_dir = cwd.join(".claude");
    std::fs::create_dir_all(&claude_dir)?;

    // The brain hook lives in the binary (`axil hook run`) — no scripts to
    // write. Clean up scripts from older installs so nothing stale lingers
    // next to the settings.json entries that no longer reference them.
    let hooks_dir = claude_dir.join("hooks");
    for legacy in AXIL_HOOK_SCRIPTS {
        let p = hooks_dir.join(legacy);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }

    // CLAUDE.md
    let project_claude_md = cwd.join(".claude").join("CLAUDE.md");
    let instructions_written = if force_claude_md
        || !project_claude_md.exists()
        || !std::fs::read_to_string(&project_claude_md)
            .unwrap_or_default()
            .contains("Axil Agent Brain")
    {
        std::fs::write(&project_claude_md, CLAUDE_MD_TEMPLATE)?;
        true
    } else {
        false
    };

    // Skills — local (project-scoped) or global (user-level home dir).
    let skills_dir_path = if local_skills {
        cwd.join(".claude").join("skills")
    } else {
        skills_dir()?
    };
    std::fs::create_dir_all(&skills_dir_path)?;
    let mut installed_skills = Vec::new();
    for skill in ALL_SKILLS {
        write_skill(&skills_dir_path, skill)?;
        installed_skills.push(skill.name);
    }

    // Settings.json hooks
    let settings_path = cwd.join(".claude").join("settings.json");
    let hooks_configured = install_hooks_to_settings(&settings_path)?;

    // Per-project Claude auto-memory lock-in: writes feedback_axil_proactive.md
    // and updates MEMORY.md so the rule survives Claude ignoring CLAUDE.md.
    let auto_memory =
        install_claude_auto_memory(cwd).unwrap_or_else(|e| json!({"error": format!("{e}")}));

    Ok(json!({
        "skills_installed": installed_skills,
        "skills_dir": skills_dir_path.display().to_string(),
        "project_claude_md": project_claude_md.display().to_string(),
        "instructions_written": instructions_written,
        "hook_command": hook_run_command(),
        "hooks_configured": hooks_configured,
        "auto_memory": auto_memory,
    }))
}

/// Install/update agent framework integration files (Cursor, Windsurf,
/// Aider, AGENTS.md). Merges with existing user config instead of
/// overwriting. (Cody was removed: Sourcegraph discontinued it in 2025 —
/// its successor Amp reads the AGENTS.md contract.)
fn install_agent_integrations(
    cwd: &Path,
    db_path: &Path,
    cursor: bool,
    windsurf: bool,
    aider: bool,
    agents_md: bool,
) -> Result<Vec<&'static str>> {
    let mut installed = Vec::new();

    if cursor {
        // Cursor's rules live in a `.cursor/rules/` DIRECTORY of `*.mdc`
        // files — a plain file at that path is not a Cursor convention and
        // blocks Cursor from ever creating the directory. Earlier installs
        // wrote exactly that file; migrate ours away, but never delete a
        // file we didn't write.
        let rules_dir = cwd.join(".cursor").join("rules");
        if rules_dir.is_file() {
            let content = std::fs::read_to_string(&rules_dir).unwrap_or_default();
            if content.contains("Axil Agent Memory") {
                std::fs::remove_file(&rules_dir)?;
            } else {
                eprintln!(
                    "[install] warn: {} is a file (not the directory Cursor expects) and wasn't written by Axil — leaving it; skipping the Cursor rule",
                    rules_dir.display()
                );
            }
        }
        if !rules_dir.is_file() {
            std::fs::create_dir_all(&rules_dir)?;
            let body = format!(
                "---\ndescription: Axil agent memory rules\nalwaysApply: true\n---\n\n{}",
                agent_instructions_cursor(db_path)
            );
            std::fs::write(rules_dir.join("axil.mdc"), body)?;
            installed.push("cursor");
        }
    }

    if windsurf {
        std::fs::write(
            cwd.join(".windsurfrules"),
            agent_instructions_windsurf(db_path),
        )?;
        installed.push("windsurf");
    }

    if aider {
        install_aider_files(cwd, db_path)?;
        installed.push("aider");
    }

    if agents_md {
        install_codex_agents_md(cwd, db_path)?;
        installed.push("agents-md");
    }

    Ok(installed)
}

/// Read JSON from the argument, or from stdin if the value is "-".
fn read_json_input(data: &str) -> Result<Value> {
    if data == "-" {
        let mut buf = String::new();
        io::stdin()
            .take(MAX_STDIN_BYTES + 1)
            .read_to_string(&mut buf)
            .context("failed to read from stdin")?;
        if buf.len() as u64 > MAX_STDIN_BYTES {
            anyhow::bail!(
                "stdin input exceeds {} MB limit",
                MAX_STDIN_BYTES / (1024 * 1024)
            );
        }
        serde_json::from_str(&buf).context("invalid JSON")
    } else {
        serde_json::from_str(data).context("invalid JSON")
    }
}

/// Parse a where clause like "field=value", "field>value", "field>=value".
fn parse_where_clause(clause: &str) -> Result<(String, Op, Value)> {
    // Two-char operators must be checked before single-char to avoid partial matches.
    for op_str in &[">=", "<=", "!=", "=", ">", "<"] {
        if let Some(pos) = clause.find(op_str) {
            let field = clause[..pos].to_string();
            if field.is_empty() {
                anyhow::bail!("invalid where clause: field name must not be empty in '{clause}'");
            }
            let op: Op = op_str.parse().map_err(|e| anyhow::anyhow!("{e}"))?;
            let val_str = &clause[pos + op_str.len()..];
            let value = serde_json::from_str(val_str)
                .unwrap_or_else(|_| Value::String(val_str.to_string()));
            return Ok((field, op, value));
        }
    }
    anyhow::bail!("invalid where clause: {clause} (expected field=value, field>value, etc.)")
}

/// Parse a human-readable duration string (e.g. "3d", "1h", "30m", "90s") into seconds.
///
/// Not feature-gated: pure string parsing, called from ungated helpers
/// (`resolve_window_start`).
fn parse_duration(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("empty duration string");
    }
    let (num, multiplier) = if let Some(n) = s.strip_suffix('d') {
        (n, 86400u64)
    } else if let Some(n) = s.strip_suffix('h') {
        (n, 3600u64)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60u64)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1u64)
    } else {
        (s, 1u64)
    };
    let n: u64 = num.parse().context("invalid duration number")?;
    if n == 0 {
        anyhow::bail!("duration must be positive");
    }
    n.checked_mul(multiplier)
        .ok_or_else(|| anyhow::anyhow!("duration overflow"))
}

/// Parse an ISO 8601 date string into microseconds since epoch.
#[allow(dead_code)]
fn parse_datetime_us(s: &str) -> Result<i64> {
    // Try full RFC 3339 first, then date-only.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp_micros());
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return Ok(dt.timestamp_micros());
    }
    anyhow::bail!("invalid date: {s} (expected ISO 8601, e.g. 2026-03-24 or 2026-03-24T00:00:00Z)")
}

/// Check if a path has a `.json` extension.
fn is_json_file(path: &Path) -> bool {
    path.extension().map(|e| e == "json").unwrap_or(false)
}

/// Format a DateTime as RFC 3339.
fn format_dt(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// ─── Database opening helpers ───────────────────────────────────────────────

#[cfg(feature = "vector")]
fn parse_model(s: &str) -> Result<axil_vector::models::EmbeddingModel> {
    s.parse().map_err(|e| anyhow::anyhow!("{e}"))
}

/// Apply --explain and --profile flags to a QL query string.
#[cfg(feature = "ql")]
fn ql_apply_flags(input: &str, explain: bool, profile: bool) -> String {
    let effective = if explain {
        format!("EXPLAIN {input}")
    } else {
        input.to_string()
    };
    if profile {
        match axil_ql::parse(&effective) {
            Ok(q) if !q.has_profile() => format!("{effective} PROFILE"),
            _ => effective,
        }
    } else {
        effective
    }
}

/// Attach all auto-detected plugins to a builder.
/// Resolve embedding model from config or default to BgeSmall.
#[cfg(feature = "embed")]
fn resolve_embedding_model(db_path: &Path) -> axil_vector::models::EmbeddingModel {
    if let Ok(config) = load_config(db_path) {
        if let Some(name) = &config.database.embedding_model {
            if let Some(model) = axil_vector::models::EmbeddingModel::from_name(name) {
                return model;
            }
            eprintln!("warning: unknown embedding model '{name}' in axil.toml, using bge-small");
        }
    }
    axil_vector::models::EmbeddingModel::BgeSmall
}

// NOTE: `axil_mcp::attach_detected_engines` is the parallel implementation for
// the MCP server — keep the engine set + gating in sync (the vector/embed setup
// differs because the two crates resolve the embedder differently, but the
// graph/fts/timeseries/extension blocks are identical). A divergence here is the
// "MCP and CLI expose different engines for the same DB" bug.
/// Install the process-wide encryption cipher from the environment, once, before
/// any database is opened.
///
/// Key sources, in priority order (see [`axil_core::crypto`]):
/// 1. `AXIL_ENC_KEY` — 32 raw key bytes as hex (64 chars) or standard base64.
/// 2. `AXIL_ENC_KEY_FILE` — path to a key file (parsed hex/base64, else raw).
///
/// With no key set this is a no-op — the `encryption` feature is compiled in but
/// databases open with cleartext bodies. A *malformed* key (wrong length/encoding,
/// unreadable file) is a hard error rather than a silent cleartext fallback:
/// encryption was asked for and could not be honored.
///
/// Installing it as a process default (rather than attaching at each open site)
/// is what lets opens this code can't reach — core-internal `branch_merge`, the
/// workspace federation fan-out, the in-process MCP server — seal/unseal with the
/// same key. [`AxilBuilder::build`](axil_core::AxilBuilder::build) consults the
/// default when no explicit cipher is set. No-op without the `encryption` feature.
#[cfg(feature = "encryption")]
fn init_default_cipher() -> Result<()> {
    use axil_core::crypto::{Cipher, CryptoError};
    // Treat an empty AXIL_ENC_KEY_FILE as unset (mirrors Cipher::from_env's
    // empty-string guard for AXIL_ENC_KEY) — otherwise an empty value would
    // resolve to `from_key_file("")` and fail every command on a bogus read.
    let key_file = std::env::var_os("AXIL_ENC_KEY_FILE")
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from);
    match Cipher::resolve(key_file.as_deref()) {
        Ok(cipher) => {
            axil_core::crypto::set_default_cipher(cipher);
            Ok(())
        }
        Err(CryptoError::MissingKey) => Ok(()),
        Err(e) => Err(anyhow::Error::new(e)
            .context("failed to load encryption key (AXIL_ENC_KEY / AXIL_ENC_KEY_FILE)")),
    }
}

fn attach_detected_engines(mut builder: axil_core::AxilBuilder) -> Result<axil_core::AxilBuilder> {
    let path = builder.path().to_path_buf();
    // Config (from the db dir) governs both which Engines attach
    // (`[engines] disabled = ["vec","graph","ts","fts"]`) and which Extensions
    // register (`[extensions] disabled`). An Engine with a companion file on
    // disk but listed in `[engines] disabled` is skipped — the operator can keep
    // the data and turn the Engine off for a session without a rebuild.
    let config = path
        .parent()
        .and_then(|dir| axil_core::load_config_from(dir).ok())
        .unwrap_or_default();

    #[cfg(feature = "vector")]
    {
        if !config.is_engine_disabled("vec") {
            if let Ok(Some(_)) = axil_vector::read_stored_dimensions(&path) {
                // Attach vector index + embedder together so auto-embed on insert works.
                #[cfg(feature = "embed")]
                {
                    builder = builder
                        .with_embedder_model(resolve_embedding_model(&path))
                        .context("failed to open vector store with embedder")?;
                }
                #[cfg(not(feature = "embed"))]
                {
                    builder = builder
                        .with_vector_auto()
                        .context("failed to open vector store")?;
                }
            }
        }
    }

    #[cfg(feature = "graph")]
    {
        if !config.is_engine_disabled("graph") && axil_graph::has_graph_store(&path) {
            builder = builder
                .with_graph_engine()
                .context("failed to open graph store")?;
        }
    }

    #[cfg(feature = "timeseries")]
    {
        if !config.is_engine_disabled("ts") && axil_timeseries::has_timeseries_store(&path) {
            builder = builder
                .with_timeseries_engine()
                .context("failed to open timeseries store")?;
        }
    }

    #[cfg(feature = "fts")]
    {
        if !config.is_engine_disabled("fts") && axil_fts::has_fts_store(&path) {
            builder = builder
                .with_fts_engine()
                .context("failed to open FTS store")?;
        }
    }

    // Register every enabled built-in Extension from the central bundle —
    // one site for the CLI, the MCP server, and the audit test. The
    // `[extensions] disabled` filter is applied inside the bundle, so a
    // disabled Extension never reaches `db.extensions()` and its CLI/MCP
    // surface + boot_block vanish without a rebuild.
    builder = axil_bundle::register_builtin_extensions(builder, &config);

    Ok(builder)
}

/// Open a database with all detected plugins.
fn open_with_all_detected(path: &Path) -> Result<Axil> {
    let builder = attach_detected_engines(Axil::open(path))?;
    let db = builder.build().context("failed to open database")?;
    // Honor the `[healing] event_log` config flag (no-op unless the `event-log`
    // feature is compiled in). Off by default — opt-in write-amplifier.
    #[cfg(feature = "event-log")]
    {
        let cfg = path
            .parent()
            .and_then(|d| axil_core::config::load_config_from(d).ok())
            .unwrap_or_default();
        if cfg.healing.event_log {
            db.set_event_log_enabled(true);
        }
    }
    Ok(db)
}

/// Number of times a hot read command retries the writable open when the
/// single-writer lock is contended before falling back to a read-only open.
const BUSY_RETRY_ATTEMPTS: u32 = 3;

/// Base backoff between busy-retry attempts. Grows linearly per attempt
/// (~50ms, ~100ms, ~150ms), so a short-lived writer's lock is usually clear
/// before the read-only fallback is needed.
const BUSY_RETRY_BASE: std::time::Duration = std::time::Duration::from_millis(50);

/// True if an `anyhow` error chain bottoms out in
/// [`axil_core::AxilError::Busy`] — the single-writer lock is contended.
///
/// Open helpers wrap the core error with `.context(...)`, so the `Busy` is
/// nested in the chain rather than the outermost error; this walks the chain
/// to find it.
fn is_busy_chain(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|c| matches!(c.downcast_ref::<axil_core::AxilError>(), Some(e) if e.is_busy()))
}

/// Retry a database open that may hit the single-writer lock.
///
/// Runs `open` up to [`BUSY_RETRY_ATTEMPTS`] times, sleeping a short linear
/// backoff between attempts but only when the failure chain bottoms out in
/// [`axil_core::AxilError::Busy`] (another process holds the writable handle).
/// Any other error returns immediately. A still-contended result returns the
/// last `Busy` error so the caller can fall back to a read-only open.
fn retry_busy_open(mut open: impl FnMut() -> Result<Axil>) -> Result<Axil> {
    let mut last = None;
    for attempt in 0..BUSY_RETRY_ATTEMPTS {
        match open() {
            Ok(db) => return Ok(db),
            Err(e) if is_busy_chain(&e) => {
                if attempt + 1 < BUSY_RETRY_ATTEMPTS {
                    std::thread::sleep(BUSY_RETRY_BASE * (attempt + 1));
                }
                last = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::Error::from(axil_core::AxilError::Busy)))
}

/// Open for a hot read command, tolerating a concurrent writer.
///
/// Axil is single-writer: while another process holds the writable handle, a
/// normal open fails with [`axil_core::AxilError::Busy`]. This helper first
/// retries the full writable open (with detected engines) a few times with a
/// short linear backoff — most writers commit in well under that window. Only
/// if the writer is *still* active after the retry budget does it fall back to
/// a **core-only read-only** open, which succeeds only in the gap between
/// writer sessions (redb's shared lock can't coexist with a live writer's
/// exclusive lock). The fallback serves committed records (boot/get/list) but
/// attaches no companion engines, so vector/FTS-backed features degrade to what
/// the core store alone can answer.
fn open_read_command(path: &Path) -> Result<Axil> {
    match retry_busy_open(|| open_with_all_detected(path)) {
        Ok(db) => Ok(db),
        Err(e) if is_busy_chain(&e) => {
            // Writer still holds the lock after the retry budget — serve
            // committed records read-only without contending for the lock. No
            // companion engines are attached in this mode, so flag the
            // degradation rather than silently returning core-only results.
            eprintln!(
                "axil: another process holds the writer lock — opening read-only \
                 (engine-backed features like vector recall are unavailable until \
                 the writer is idle)."
            );
            Axil::open(path)
                .read_only(true)
                .build()
                .context("failed to open database read-only")
        }
        Err(e) => Err(e),
    }
}

/// Open with FTS for a hot read command, tolerating a concurrent writer.
///
/// Same bounded busy-retry as [`open_read_command`], but with no read-only
/// fallback: full-text search is engine-backed, and a core-only read-only
/// handle can't serve it. If the writer is still active after the retry budget
/// this surfaces a clear "busy" message rather than silently degrading to an
/// empty result.
#[cfg(feature = "fts")]
fn open_read_command_fts(path: &Path) -> Result<Axil> {
    match retry_busy_open(|| open_with_fts(path)) {
        Ok(db) => Ok(db),
        Err(e) if is_busy_chain(&e) => Err(anyhow::anyhow!(
            "database busy: another process holds the writer lock — retry shortly"
        )),
        Err(e) => Err(e),
    }
}

/// Like `open_with_all_detected`, but force-attaches the graph plugin
/// (creating `.axil.graph` if missing). SCIP ingest writes thousands
/// of `calls` / `references` / `implements` / `type_of` edges, and
/// `axil_scip::relate_once` silently no-ops them when
/// `graph_index_ref()` is `None` — so a graph-less DB ends up with an
/// `IngestReport` that claims thousands of edges while zero are
/// persisted. This helper guarantees the graph store exists before
/// the ingest runs and prints a one-line stderr notice on creation.
///
/// Cfg is just `feature = "scip"` because the `scip` feature already
/// requires `graph` (see `axil-cli/Cargo.toml`: `scip = ["dep:axil-scip", "graph"]`).
/// Adding `all(feature = "scip", feature = "graph")` here would be
/// redundant — and worse, would visually suggest the two are
/// independent, leaving a reader to wonder what happens in the
/// scip-without-graph build. There is no such build.
#[cfg(feature = "scip")]
fn open_for_scip_ingest(path: &Path) -> Result<Axil> {
    let creating_graph = !axil_graph::has_graph_store(path);
    let mut builder = attach_detected_engines(Axil::open(path))?;
    if creating_graph {
        builder = builder
            .with_graph_engine()
            .context("failed to create graph store for SCIP ingest")?;
        eprintln!(
            "axil ingest-scip: created graph store at {}",
            axil_core::companion_path(path, ".graph").display()
        );
    }
    builder.build().context("failed to open database")
}

/// Open a database with vector support (auto-detecting dimensions).
#[cfg(feature = "vector")]
fn open_with_vector(path: &Path, dimensions: Option<usize>) -> Result<Axil> {
    let builder = Axil::open(path);
    let db = match dimensions {
        Some(dims) => builder
            .with_vector(dims)
            .context("failed to open vector store")?,
        None => builder
            .with_vector_auto()
            .context("failed to auto-detect vector dimensions — use --dimensions")?,
    };
    db.build().context("failed to open database")
}

/// Open a database with embedder loaded (for recall/search commands).
#[cfg(feature = "embed")]
fn open_with_embedder(path: &Path) -> Result<Axil> {
    // Probe BEFORE attaching engines. `attach_detected_engines` opens the
    // vector store's redb file and holds it for the life of the builder, so a
    // probe placed after it collides with its own process's lock and reads a
    // healthy store as missing. Probe errors (lock held by another process,
    // corrupt metadata) propagate as-is instead of collapsing into "not found".
    if axil_vector::read_stored_dimensions(path)
        .context("failed to probe vector store")?
        .is_none()
    {
        // Require an existing vector store — don't silently create one on read operations.
        anyhow::bail!("no vector store found — run `axil init <path>` first to create one");
    }
    // The store exists, so `attach_detected_engines` attaches the vector index
    // + embedder itself (unless the operator disabled the engine in axil.toml).
    // Attaching a second embedder here would re-open the redb file this
    // process already holds and fail.
    let db = attach_detected_engines(Axil::open(path))?
        .build()
        .context("failed to open database")?;
    if !db.has_embedder() {
        anyhow::bail!(
            "the vector engine is disabled in axil.toml ([engines] disabled) — \
             re-enable it to run embedding commands"
        );
    }
    Ok(db)
}

/// Open with embedder if available, otherwise fall back to `open_with_all_detected`.
///
/// Used by commands that benefit from vector search but can still function without it.
#[cfg(feature = "indexer")]
fn open_with_best_effort(path: &Path) -> Result<Axil> {
    #[cfg(feature = "embed")]
    {
        if axil_vector::read_stored_dimensions(path)
            .ok()
            .flatten()
            .is_some()
        {
            if let Ok(db) = open_with_embedder(path) {
                return Ok(db);
            }
        }
    }
    open_with_all_detected(path)
}

/// Open with timeseries, auto-creating and backfilling if needed.
#[cfg(feature = "timeseries")]
fn open_with_timeseries(path: &Path) -> Result<Axil> {
    let is_new_ts = !axil_timeseries::has_timeseries_store(path);

    let mut builder = attach_detected_engines(Axil::open(path))?;

    if is_new_ts {
        builder = builder
            .with_timeseries_engine()
            .context("failed to open timeseries store")?;
    }

    let db = builder.build().context("failed to open database")?;

    if is_new_ts {
        let n = db
            .backfill_timeseries()
            .context("failed to backfill timeseries index")?;
        if n > 0 {
            eprintln!("Indexed {n} existing records into timeseries.");
        }
    }

    Ok(db)
}

/// Open with FTS, auto-creating if needed.
#[cfg(feature = "fts")]
fn open_with_fts(path: &Path) -> Result<Axil> {
    let mut builder = attach_detected_engines(Axil::open(path))?;

    if !axil_fts::has_fts_store(path) {
        builder = builder
            .with_fts_engine()
            .context("failed to create FTS store")?;
    }

    builder.build().context("failed to open database")
}

/// Open with all features enabled (for init).
#[cfg(feature = "vector")]
fn open_with_all_features(path: &Path, vector_dims: usize) -> Result<Axil> {
    let mut builder = Axil::open(path);
    builder = builder
        .with_vector(vector_dims)
        .context("failed to create vector store")?;
    open_with_all_features_inner(builder)
}

#[cfg(not(feature = "vector"))]
fn open_with_all_features(path: &Path) -> Result<Axil> {
    let builder = Axil::open(path);
    open_with_all_features_inner(builder)
}

fn open_with_all_features_inner(mut builder: axil_core::AxilBuilder) -> Result<Axil> {
    #[cfg(feature = "graph")]
    {
        builder = builder
            .with_graph_engine()
            .context("failed to create graph store")?;
    }

    #[cfg(feature = "timeseries")]
    {
        builder = builder
            .with_timeseries_engine()
            .context("failed to create timeseries store")?;
    }

    #[cfg(feature = "fts")]
    {
        builder = builder
            .with_fts_engine()
            .context("failed to create FTS store")?;
    }

    builder.build().context("failed to create database")
}

/// Load config from `axil.toml` in the same directory as the database file.
fn load_config(db_path: &Path) -> Result<axil_core::AxilConfig> {
    let dir = db_path.parent().unwrap_or(Path::new("."));
    axil_core::load_config_from(dir).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Wire up an LLM provider to an existing Axil database based on config.
///
/// This re-opens the database with LLM support. If no LLM config is found,
/// the database is returned as-is (all features degrade gracefully).
fn wire_llm(db: Axil, db_path: &Path) -> Result<Axil> {
    let config = load_config(db_path).unwrap_or_default();

    if !config.llm.is_configured() {
        return Ok(db);
    }

    // Re-open with LLM wired in.
    drop(db);

    #[cfg(feature = "llm-http")]
    {
        if let Some(http_llm) = axil_core::HttpLlm::from_config(&config.llm) {
            // `attach_detected_engines` already attaches the vector index +
            // embedder (with the configured model) when the store exists, so
            // --embed keeps working alongside --llm. Attaching a second
            // embedder here would re-open the redb file this process already
            // holds and fail.
            let builder = attach_detected_engines(Axil::open(db_path))?
                .with_llm(std::sync::Arc::new(http_llm))
                .with_llm_config(config.llm);
            return builder.build().context("failed to open database with LLM");
        }
    }

    let builder = attach_detected_engines(Axil::open(db_path))?;
    builder.build().context("failed to open database")
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    // The `Command` enum is very large; in debug builds the per-frame cost
    // of clap's generated parser overflows the OS default main-thread stack
    // (1 MiB on Windows — `axil --version` alone overflows it). Run the real
    // entry point on a worker thread with an explicit, generous stack so
    // behaviour is identical across platforms and build profiles.
    let code = std::thread::Builder::new()
        .name("axil-main".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(real_main)
        .expect("failed to spawn axil main thread")
        .join()
        .unwrap_or(EXIT_ERROR);
    std::process::exit(code);
}

/// Real entry point — runs on a large-stack worker thread (see `main`).
fn real_main() -> i32 {
    // The CLI is Axil's Tier-3 argv Adapter; drive it through that type. `main`
    // uses the inherent `dispatch` (which returns the process exit code) rather
    // than `Adapter::run` so exit-code fidelity is preserved.
    CliAdapter::new().dispatch()
}

/// Tier-3 [`Adapter`](axil_core::Adapter) for the command-line interface.
///
/// The CLI resolves its database per-subcommand (from `--db` / auto-detect; some
/// commands — `init`, `--help` — open none), so unlike a server Adapter it does
/// not run against a single pre-bound `Axil`. `bind` is accepted for contract
/// conformance but intentionally unused; the argv flow opens what each command
/// needs. This makes `axil-cli` a first-class Adapter alongside `axil-mcp` /
/// the AxilQL frontend / the HTTP example.
struct CliAdapter;

impl CliAdapter {
    fn new() -> Self {
        Self
    }

    /// Parse argv and dispatch, returning the process exit code. The CLI's real
    /// entry — `main` calls this directly because the `Adapter::run` signature
    /// (`Result<()>`) cannot carry the exit code.
    fn dispatch(self) -> i32 {
        let cli = Cli::parse();
        let out = Output {
            format: cli.format.clone(),
            quiet: cli.quiet,
            jsonl: cli.jsonl,
        };

        match run(cli, &out) {
            Ok(code) => code,
            Err(e) => {
                let err_json = json!({"error": format!("{e:#}")});
                eprintln!("{}", serde_json::to_string(&err_json).unwrap());
                EXIT_ERROR
            }
        }
    }
}

impl axil_core::Adapter for CliAdapter {
    fn id(&self) -> &str {
        "cli"
    }

    fn protocol(&self) -> axil_core::Protocol {
        axil_core::Protocol::Cli
    }

    fn bind(&mut self, _db: std::sync::Arc<axil_core::Axil>) -> axil_core::Result<()> {
        // See the type docs: the CLI self-resolves its database per subcommand,
        // so a pre-bound handle is accepted but unused.
        Ok(())
    }

    fn run(self) -> axil_core::Result<()> {
        // Trait-conformant entry: dispatch, mapping a non-zero exit code to an
        // error so a programmatic embedder sees failure. `main` prefers
        // `dispatch` to keep the exact exit code.
        match self.dispatch() {
            EXIT_OK => Ok(()),
            code => Err(axil_core::AxilError::plugin(format!(
                "CLI exited with code {code}"
            ))),
        }
    }
}

/// Run the CLI command. Returns the exit code.
fn run(cli: Cli, out: &Output) -> Result<i32> {
    let db_opt = cli.db;

    // Install the process-wide encryption cipher before any command opens a
    // database, so every open — including core-internal multi-DB ops and the
    // in-process MCP server — seals/unseals with the same key. No-op when no key
    // is configured or the `encryption` feature is off.
    #[cfg(feature = "encryption")]
    init_default_cipher()?;

    match cli.command {
        // ── Init ────────────────────────────────────────────────────
        Command::Init {
            path,
            #[cfg(feature = "vector")]
            vector_dims,
        } => {
            let db_path = path
                .or(db_opt.clone())
                .ok_or_else(|| anyhow::anyhow!("path required: axil init <path> or --db <path>"))?;

            #[cfg(feature = "vector")]
            let _db = open_with_all_features(&db_path, vector_dims)?;
            #[cfg(not(feature = "vector"))]
            let _db = open_with_all_features(&db_path)?;

            let mut features = vec!["core"];
            #[cfg(feature = "vector")]
            features.push("vector");
            #[cfg(feature = "graph")]
            features.push("graph");
            #[cfg(feature = "timeseries")]
            features.push("timeseries");
            #[cfg(feature = "fts")]
            features.push("fts");

            out.print(&json!({
                "path": db_path.display().to_string(),
                "created": true,
                "features": features,
            }));
            Ok(EXIT_OK)
        }

        // ── Install (project setup) ─────────────────────────────────
        // ── Brief / Retro / Schedule (12.3) ───────────────────────────
        Command::Brief {
            window,
            after,
            brief_format,
            budget,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let window_start = resolve_window_start(&window, after.as_deref())?;
            let report = generate_brief(&db, window_start, &window, budget)?;
            emit_brief(&report, &brief_format, &out);
            Ok(EXIT_OK)
        }
        Command::Retro {
            window,
            brief_format,
            save,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let window_start = resolve_window_start(&window, None)?;
            let report = generate_brief(&db, window_start, &window, None)?;
            emit_brief(&report, &brief_format, &out);
            if save {
                // Persist as markdown under .axil/reports/ and store as context record.
                let reports_dir = db_path
                    .parent()
                    .map(|p| p.join("reports"))
                    .unwrap_or_else(|| PathBuf::from("reports"));
                std::fs::create_dir_all(&reports_dir)?;
                let stamp = chrono::Utc::now().format("%Y-%m-%d").to_string();
                let out_path = reports_dir.join(format!("retro-{stamp}.md"));
                let md = render_brief_markdown(&report, /* is_retro = */ true);
                std::fs::write(&out_path, &md)?;
                let _ = db.insert(
                    "context",
                    json!({
                        "type": "retrospective",
                        "window": window,
                        "summary": report["narrative"].as_str().unwrap_or(""),
                        "report_path": out_path.display().to_string(),
                        "generated_at": chrono::Utc::now().to_rfc3339(),
                    }),
                );
                eprintln!("[retro] wrote {}", out_path.display());
            }
            Ok(EXIT_OK)
        }
        Command::Schedule { op } => match op {
            ScheduleOp::Install {
                name,
                hour,
                minute,
                scheduler,
                dry_run,
            } => {
                let db_path = require_db(&db_opt)?;
                let plan = plan_scheduled_task(&name, hour, minute, &scheduler, &db_path)?;
                if !dry_run {
                    install_scheduled_task(&plan)?;
                }
                out.print(&json!({
                    "installed": !dry_run,
                    "dry_run": dry_run,
                    "name": plan.name,
                    "scheduler": plan.scheduler,
                    "path": plan.install_path.display().to_string(),
                    "companion_path": plan.companion_path.as_ref().map(|p| p.display().to_string()),
                    "command": plan.command,
                }));
                Ok(EXIT_OK)
            }
            ScheduleOp::List => {
                let tasks = list_scheduled_tasks()?;
                out.print(&json!({ "tasks": tasks }));
                Ok(EXIT_OK)
            }
            ScheduleOp::Uninstall { name, dry_run } => {
                let removed = uninstall_scheduled_task(&name, dry_run)?;
                out.print(&json!({
                    "uninstalled": !dry_run,
                    "dry_run": dry_run,
                    "name": name,
                    "removed_paths": removed,
                }));
                Ok(EXIT_OK)
            }
        },

        // ── Ingest: bulk filesystem ingest (12.2) ──────────────────────
        #[cfg(feature = "indexer")]
        Command::Ingest {
            dir,
            recursive,
            ext,
            exclude,
            table,
            stats,
            resume,
            chunk_bytes,
            watch,
            interval,
        } => {
            let db_path = require_db(&db_opt)?;
            let root_canonical = dir
                .canonicalize()
                .with_context(|| format!("cannot resolve ingest dir: {}", dir.display()))?;
            if !root_canonical.is_dir() {
                anyhow::bail!("--dir must be a directory: {}", root_canonical.display());
            }

            // Parse --ext into an extension set.
            let exts: std::collections::HashSet<String> = ext
                .split(',')
                .map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();

            // --exclude is substring-based (documented as such). Stripping leading `*` / `/`
            // lets users paste glob-ish patterns; we keep the inner segment and match via contains().
            let exclude_patterns: Vec<String> = exclude
                .iter()
                .map(|p| p.trim_matches('*').trim_matches('/').to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Ingest state lives next to the DB so resume/skip survives across runs and ticks.
            let state_path = db_path
                .parent()
                .map(|p| p.join("ingest.state.json"))
                .unwrap_or_else(|| PathBuf::from("ingest.state.json"));

            // --stats is a pure dry-run plan: count files/bytes, no DB, no writes.
            if stats {
                let candidates = ingest_collect_candidates(
                    &root_canonical,
                    recursive,
                    &exts,
                    &exclude_patterns,
                );
                let prior_state: serde_json::Map<String, Value> = if state_path.exists() {
                    serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap_or_default())
                        .unwrap_or_default()
                } else {
                    serde_json::Map::new()
                };
                let total_bytes: u64 = candidates.iter().map(|(_, s)| s).sum();
                let already_indexed = candidates
                    .iter()
                    .filter(|(p, _)| prior_state.get(&p.display().to_string()).is_some())
                    .count();
                out.print(&json!({
                    "dry_run": true,
                    "root": root_canonical.display().to_string(),
                    "total_files": candidates.len(),
                    "total_bytes": total_bytes,
                    "already_indexed": already_indexed,
                    "extensions": exts.iter().cloned().collect::<Vec<_>>(),
                    "excludes": exclude_patterns,
                }));
                return Ok(EXIT_OK);
            }

            // Open DB once, with all detected plugins so auto-embed/auto-entity run on insert.
            // In watch mode the same handle is reused across every tick.
            let db = open_with_all_detected(&db_path)?;

            if watch {
                // Interval-poll watch (same style as `stats --watch`): no extra
                // crate dependency. Each tick re-scans the tree and runs the
                // incremental ingest; content-hash skipping means a tick only
                // touches new/changed files. State is always reused across ticks
                // (an explicit --resume is therefore redundant but harmless).
                let interval = std::time::Duration::from_secs(interval.max(1));
                out.status(&format!(
                    "[watch] ingesting {} every {}s — Ctrl+C to stop",
                    root_canonical.display(),
                    interval.as_secs()
                ));
                loop {
                    let candidates = ingest_collect_candidates(
                        &root_canonical,
                        recursive,
                        &exts,
                        &exclude_patterns,
                    );
                    let report = run_ingest_pass(
                        &db,
                        &root_canonical,
                        &candidates,
                        &state_path,
                        &table,
                        chunk_bytes,
                        true, // always reuse prior state across ticks
                        out,
                    )?;
                    let ingested = report
                        .get("files_ingested")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    // Stay quiet when nothing changed; one concise line otherwise.
                    if ingested > 0 {
                        let chunks = report
                            .get("chunks_written")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        out.status(&format!(
                            "[watch] tick: {ingested} new/changed file(s) ingested ({chunks} chunks)"
                        ));
                    }
                    std::thread::sleep(interval);
                }
            } else {
                let candidates = ingest_collect_candidates(
                    &root_canonical,
                    recursive,
                    &exts,
                    &exclude_patterns,
                );
                let report = run_ingest_pass(
                    &db,
                    &root_canonical,
                    &candidates,
                    &state_path,
                    &table,
                    chunk_bytes,
                    resume,
                    out,
                )?;
                out.print(&report);
                Ok(EXIT_OK)
            }
        }

        Command::InstallProject {
            claude_code,
            cursor,
            windsurf,
            cody,
            aider,
            codex,
            copilot,
            droid,
            antigravity,
            qwen,
            opencode,
            no_agents_md,
            all,
            agent,
            #[cfg(feature = "vector")]
            vector_dims,
            dry_run,
            uninstall,
            bootstrap,
            local,
        } => {
            let cwd_early = std::env::current_dir()?;

            if uninstall {
                let result = uninstall_claude_code_files(&cwd_early, dry_run)?;
                out.print(&result);
                return Ok(EXIT_OK);
            }
            if cody {
                out.status(
                    "note: --cody is a deprecated no-op — Sourcegraph discontinued Cody; its successor Amp reads the AGENTS.md contract (written by default).",
                );
            }
            if dry_run {
                let plan = dry_run_install_plan(
                    &cwd_early,
                    claude_code,
                    codex,
                    copilot,
                    droid,
                    antigravity,
                    qwen,
                    opencode,
                    cursor,
                    windsurf,
                    aider,
                    // AGENTS.md is on by default (cross-tool contract).
                    !no_agents_md,
                    all,
                    local,
                )?;
                out.print(&plan);
                return Ok(EXIT_OK);
            }

            // Bare `axil install` on a terminal → interactive wizard. Any
            // selection flag (or piped stdin / --quiet) means the caller has
            // already decided, so scripts and CI keep today's behavior.
            // `--no-agents-md` is a de-selection modifier, not a selection —
            // it must NOT suppress the wizard (passing it alone would
            // otherwise install nothing). It only flips the AGENTS.md default
            // on the non-wizard path below.
            let no_selection = !claude_code
                && !cursor
                && !windsurf
                && !aider
                && !codex
                && !copilot
                && !droid
                && !antigravity
                && !qwen
                && !opencode
                && !all
                && agent.is_none()
                && !bootstrap
                && !local;
            let mut choices = install_wizard::InstallChoices {
                claude_code,
                codex,
                copilot,
                droid,
                antigravity,
                qwen,
                opencode,
                cursor,
                windsurf,
                aider,
                agents_md: false, // resolved below
                bootstrap,
                local,
            };
            let mut wizard_ran = false;
            if no_selection {
                match install_wizard::maybe_run(&cwd_early, out.quiet)? {
                    install_wizard::WizardOutcome::Aborted => {
                        out.status("install aborted — nothing written");
                        return Ok(EXIT_OK);
                    }
                    install_wizard::WizardOutcome::Choices(c) => {
                        wizard_ran = true;
                        choices = c;
                    }
                    install_wizard::WizardOutcome::NotInteractive => {}
                }
            }
            // AGENTS.md managed block: on by default — it is the cross-tool
            // contract (Codex, OpenCode, Qwen Code, Copilot, Droid, …). The
            // wizard's toggle is an explicit user choice and wins; flag/CI
            // installs opt out only via --no-agents-md.
            let agents_md = if wizard_ran {
                choices.agents_md
            } else {
                !no_agents_md
            };
            let install_wizard::InstallChoices {
                claude_code,
                codex,
                copilot,
                droid,
                antigravity,
                qwen,
                opencode,
                cursor,
                windsurf,
                aider,
                bootstrap,
                local,
                ..
            } = choices;

            let axil_dir = cwd_early.join(".axil");
            let db_path = axil_dir.join("memory.axil");

            // Create .axil/ directory
            std::fs::create_dir_all(&axil_dir).context("failed to create .axil/ directory")?;

            // Write version file for `axil update` staleness detection
            let current_version = env!("CARGO_PKG_VERSION");
            std::fs::write(axil_dir.join("version"), current_version)?;

            // Create the database
            #[cfg(feature = "vector")]
            let _db = open_with_all_features(&db_path, vector_dims)?;
            #[cfg(not(feature = "vector"))]
            let _db = open_with_all_features(&db_path)?;

            // Seed the Axil-first rule into the freshly created DB so
            // `axil boot` surfaces it in every session's Constraints section.
            // The Constraints section is priority 1 — never dropped from boot.
            let rule_seeded = seed_axil_first_rule(&_db).unwrap_or(false);

            let mut result = json!({
                "path": db_path.display().to_string(),
                "directory": axil_dir.display().to_string(),
                "created": true,
                "rule_seeded": rule_seeded,
            });

            // Add .axil/ to .gitignore if not already there
            let gitignore_path = std::env::current_dir()?.join(".gitignore");
            let gitignore_updated = add_to_gitignore(&gitignore_path, ".axil/");
            result["gitignore_updated"] = json!(gitignore_updated);

            // Claude Code integration
            let cwd = cwd_early.clone();
            if claude_code || all {
                let cc_result = install_claude_code_files(&cwd, false, local)?;
                result["claude_code"] = cc_result.clone();
                out.status("Claude Code agent brain installed:");
                if let Some(skills) = cc_result.get("skills_installed").and_then(|v| v.as_array()) {
                    let names: Vec<&str> = skills.iter().filter_map(|v| v.as_str()).collect();
                    out.status(&format!("  Skills: {}", names.join(", ")));
                }
                out.status(&format!(
                    "  Hook: {} (wired in .claude/settings.json)",
                    cc_result["hook_command"].as_str().unwrap_or("?")
                ));
                out.status(&format!(
                    "  Instructions: {}",
                    cc_result["project_claude_md"].as_str().unwrap_or("?")
                ));
                out.status(&format!("  DB: {}", db_path.display()));
                if let Some(am) = cc_result.get("auto_memory") {
                    if let Some(file) = am.get("feedback_file").and_then(|v| v.as_str()) {
                        out.status(&format!("  Auto-memory: {}", file));
                    }
                }
                if rule_seeded {
                    out.status("  Boot rule: pinned Axil-first row seeded in `rules` table");
                }
            }

            // Generic agent integration
            if let Some(ref agent_type) = agent {
                let generic_skill = include_str!("skills/generic-agent.md");
                let instructions_path = axil_dir.join("agent-instructions.md");
                let instructions =
                    generic_skill.replace("./memory.axil", &db_path.display().to_string());
                std::fs::write(&instructions_path, &instructions)?;
                result["agent"] = json!({
                    "type": agent_type,
                    "instructions": instructions_path.display().to_string(),
                    "setup": format!("export AXIL_DB={}", db_path.display()),
                });
            }

            // Multi-agent framework support
            let mut agents_installed = install_agent_integrations(
                &cwd,
                &db_path,
                cursor || all,
                windsurf || all,
                aider || all,
                agents_md || all,
            )?;

            // Full terminal-agent loops (hooks + MCP; Codex also gets the
            // cross-tool skills). AGENTS.md above is their shared contract.
            if codex || all {
                result["codex"] = install_codex_full(&cwd)?;
                agents_installed.push("codex");
                out.status(
                    "Codex: .codex/hooks.json + .codex/config.toml (MCP) + .agents/skills/ — trust the project, then run /hooks in Codex once",
                );
            }
            if copilot || all {
                result["copilot"] = install_copilot_full(&cwd)?;
                agents_installed.push("copilot");
                out.status(
                    "Copilot CLI: .github/hooks/axil.json + ~/.copilot/mcp-config.json (also picked up by the Copilot cloud agent)",
                );
            }
            if droid || all {
                result["droid"] = install_droid_full(&cwd)?;
                agents_installed.push("droid");
                out.status("Droid: .factory/hooks.json + .factory/mcp.json");
            }
            if antigravity || all {
                result["antigravity"] = install_antigravity_full(&cwd, &db_path)?;
                agents_installed.push("antigravity");
                out.status(
                    "Antigravity: .agents/rules/axil.md + .agents/skills/ + hooks + mcp_config.json",
                );
            }
            if qwen || all {
                result["qwen"] = install_qwen_full(&cwd)?;
                agents_installed.push("qwen");
                out.status(
                    "Qwen Code: .qwen/settings.json (hooks + MCP + AGENTS.md context) — consider disabling memory.enableManagedAutoMemory to avoid double-capture",
                );
            }
            if opencode || all {
                result["opencode"] = install_opencode_full(&cwd)?;
                agents_installed.push("opencode");
                out.status("OpenCode: .opencode/plugins/axil.ts + opencode.json MCP entry");
            }

            if !agents_installed.is_empty() {
                result["agents_installed"] = json!(agents_installed);
                out.status(&format!(
                    "Agent integrations: {}",
                    agents_installed.join(", ")
                ));
            }

            // One-shot bootstrap: index structural proxies + kick off SCIP refresh.
            // Both steps are best-effort — failures are surfaced in the result JSON
            // but do not fail the install (the DB and agent wiring are already in place).
            drop(_db);
            if bootstrap {
                let mut bootstrap_report = serde_json::Map::new();

                #[cfg(feature = "indexer")]
                {
                    out.status("Bootstrap: indexing project (structural proxies)...");
                    let index_config = axil_core::load_config_from(&cwd)
                        .map(|c| c.index)
                        .unwrap_or_default();
                    let index_result = (|| -> anyhow::Result<Value> {
                        let db = open_with_all_detected(&db_path)?;
                        let indexer = axil_indexer::ProjectIndexer::new(&db, index_config)
                            .with_progress(make_index_progress(out.quiet));
                        let report = indexer
                            .index_full(&cwd)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        Ok(serde_json::to_value(&report)?)
                    })();
                    match index_result {
                        Ok(v) => {
                            bootstrap_report.insert("index".into(), v);
                            out.status("  Structural proxies indexed.");
                        }
                        Err(e) => {
                            bootstrap_report.insert("index_error".into(), json!(format!("{e}")));
                            out.status(&format!("  Index step failed: {e} (continuing)"));
                        }
                    }
                }
                #[cfg(not(feature = "indexer"))]
                {
                    bootstrap_report.insert(
                        "index_skipped".into(),
                        json!("indexer feature not compiled in"),
                    );
                }

                // SCIP refresh (background): re-exec the same binary so the existing
                // --in-background lock + spawn logic handles language detection,
                // missing-indexer no-op, and PID lock-file cleanup.
                let exe = std::env::current_exe().ok();
                if let Some(exe) = exe {
                    out.status("Bootstrap: kicking off SCIP refresh in background...");
                    let scip_result = std::process::Command::new(&exe)
                        .arg("--db")
                        .arg(&db_path)
                        .arg("scip")
                        .arg("refresh")
                        .arg("--if-stale")
                        .arg("--in-background")
                        .arg("--quiet")
                        .current_dir(&cwd)
                        .output();
                    match scip_result {
                        Ok(o) if o.status.success() => {
                            let stdout = String::from_utf8_lossy(&o.stdout);
                            let parsed: Value = serde_json::from_str(stdout.trim())
                                .unwrap_or_else(|_| json!({"raw": stdout.trim()}));
                            bootstrap_report.insert("scip".into(), parsed);
                            out.status("  SCIP refresh dispatched (runs in background).");
                        }
                        Ok(o) => {
                            let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
                            bootstrap_report.insert(
                                "scip_error".into(),
                                json!(format!("exit {}: {}", o.status, stderr)),
                            );
                            out.status(&format!("  SCIP refresh failed: {} (continuing)", stderr));
                        }
                        Err(e) => {
                            bootstrap_report
                                .insert("scip_error".into(), json!(format!("spawn failed: {e}")));
                            out.status(&format!("  SCIP refresh spawn failed: {e} (continuing)"));
                        }
                    }
                }

                result["bootstrap"] = Value::Object(bootstrap_report);
            }

            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Update ──────────────────────────────────────────────────
        Command::UpdateProject {
            claude_code,
            cursor,
            windsurf,
            cody,
            aider,
            codex,
            copilot,
            droid,
            antigravity,
            qwen,
            opencode,
            all,
        } => {
            let cwd = std::env::current_dir()?;
            let axil_dir = cwd.join(".axil");

            if !axil_dir.exists() {
                anyhow::bail!("No .axil/ directory found. Run `axil install` first.");
            }

            let db_path = axil_dir.join("memory.axil");
            let current_version = env!("CARGO_PKG_VERSION");
            let version_file = axil_dir.join("version");
            let installed_version = std::fs::read_to_string(&version_file).unwrap_or_default();
            let installed_version = installed_version.trim().to_string();

            let result_version = json!({
                "previous_version": if installed_version.is_empty() { "unknown" } else { &installed_version },
                "current_version": current_version,
            });

            if cody {
                out.status(
                    "note: --cody is a deprecated no-op — Sourcegraph discontinued Cody; its successor Amp reads the AGENTS.md contract.",
                );
            }

            // Same-version fast path applies only to the bare auto-detect
            // refresh. An explicit flag means "add/refresh this integration
            // now" — that must work without a version bump.
            let any_flag = claude_code
                || cursor
                || windsurf
                || aider
                || codex
                || copilot
                || droid
                || antigravity
                || qwen
                || opencode
                || all;
            if !any_flag && !installed_version.is_empty() && installed_version == current_version
            {
                out.status(&format!("Already up to date (v{})", current_version));
                let mut result = result_version;
                result["updated"] = json!([]);
                out.print(&result);
                return Ok(EXIT_OK);
            }

            let update_claude = claude_code || all;
            let update_cursor = cursor || all;
            let update_windsurf = windsurf || all;
            let update_aider = aider || all;
            let update_codex = codex || all;
            let update_copilot = copilot || all;
            let update_droid = droid || all;
            let update_antigravity = antigravity || all;
            let update_qwen = qwen || all;
            let update_opencode = opencode || all;

            // Auto-detect if no flags given: update whatever is already installed
            let auto_detect = !update_claude
                && !update_cursor
                && !update_windsurf
                && !update_aider
                && !update_codex
                && !update_copilot
                && !update_droid
                && !update_antigravity
                && !update_qwen
                && !update_opencode;

            let mut updated: Vec<&str> = Vec::new();

            // Claude Code (force_claude_md=true so update always refreshes).
            // Auto-detect whether the original install used --local by checking
            // for the canonical project-scoped memory skill; otherwise refresh
            // the global skills dir at ~/.claude/skills/.
            // Detect an existing Claude Code install either by the current
            // settings.json wiring or by a legacy hook script still on disk.
            let claude_installed = || {
                std::fs::read_to_string(cwd.join(".claude/settings.json"))
                    .map(|s| s.contains(" hook run") || s.contains("axil-brain.sh"))
                    .unwrap_or(false)
                    || cwd.join(".claude/hooks/axil-brain.sh").exists()
            };
            if update_claude || (auto_detect && claude_installed()) {
                let local_skills = cwd.join(".claude/skills/axil/SKILL.md").exists()
                    || cwd.join(".claude/skills/axil.md").exists();
                install_claude_code_files(&cwd, true, local_skills)?;

                // Backfill the Axil-first pinned rule into older DBs that were
                // installed before the seed existed. Idempotent; opens the DB
                // briefly and closes it — no impact when already seeded.
                if db_path.exists() {
                    let seed_attempt =
                        open_with_all_detected(&db_path).and_then(|db| seed_axil_first_rule(&db));
                    if let Ok(true) = seed_attempt {
                        out.status("Backfilled: pinned Axil-first rule in DB");
                    }
                }

                updated.push("claude-code");
                out.status("Updated: hook, CLAUDE.md, skills, settings.json, auto-memory");
            }

            // Other agents: auto-detect by checking if their config exists
            let do_cursor = update_cursor || (auto_detect && cwd.join(".cursor/rules").exists());
            let do_windsurf =
                update_windsurf || (auto_detect && cwd.join(".windsurfrules").exists());
            // `.aider.conf.yml` is the aider marker; a bare CONVENTIONS.md is
            // not — plenty of repos keep one with no aider involved.
            let do_aider = update_aider || (auto_detect && cwd.join(".aider.conf.yml").exists());
            // AGENTS.md is the shared contract for the terminal agents, so
            // refresh it whenever any of them is being updated (explicit flag
            // or --all) — not only on the bare auto-detect path, which
            // `--all`/`--codex` disable by setting auto_detect=false.
            let do_agents_md = update_codex
                || update_copilot
                || update_droid
                || update_antigravity
                || update_qwen
                || update_opencode
                || (auto_detect && cwd.join("AGENTS.md").exists());

            let agents = install_agent_integrations(
                &cwd,
                &db_path,
                do_cursor,
                do_windsurf,
                do_aider,
                do_agents_md,
            )?;
            for agent in &agents {
                out.status(&format!("Updated: {}", agent));
            }
            updated.extend(agents);

            // Full terminal-agent loops: refresh where installed or flagged.
            if update_codex || (auto_detect && cwd.join(".codex/hooks.json").exists()) {
                install_codex_full(&cwd)?;
                updated.push("codex");
                out.status("Updated: codex (.codex/hooks.json + MCP + .agents/skills)");
            }
            if update_copilot || (auto_detect && cwd.join(".github/hooks/axil.json").exists()) {
                install_copilot_full(&cwd)?;
                updated.push("copilot");
                out.status("Updated: copilot (.github/hooks/axil.json + MCP)");
            }
            if update_droid || (auto_detect && cwd.join(".factory/hooks.json").exists()) {
                install_droid_full(&cwd)?;
                updated.push("droid");
                out.status("Updated: droid (.factory/hooks.json + MCP)");
            }
            if update_antigravity || (auto_detect && cwd.join(".agents/rules/axil.md").exists()) {
                install_antigravity_full(&cwd, &db_path)?;
                updated.push("antigravity");
                out.status("Updated: antigravity (.agents/ rules + skills + hooks + MCP)");
            }
            if update_qwen
                || (auto_detect
                    && std::fs::read_to_string(cwd.join(".qwen/settings.json"))
                        .map(|s| s.contains("hook run"))
                        .unwrap_or(false))
            {
                install_qwen_full(&cwd)?;
                updated.push("qwen");
                out.status("Updated: qwen (.qwen/settings.json hooks + MCP)");
            }
            if update_opencode || (auto_detect && cwd.join(".opencode/plugins/axil.ts").exists())
            {
                install_opencode_full(&cwd)?;
                updated.push("opencode");
                out.status("Updated: opencode (.opencode/plugins/axil.ts + MCP)");
            }

            if updated.is_empty() {
                out.status("Nothing to update. No agent integrations detected.");
                out.status("Run `axil install --claude-code` (or --all) to set up first.");
            } else {
                std::fs::write(&version_file, current_version)?;
                out.status(&format!(
                    "Updated {} integration(s) to v{}",
                    updated.len(),
                    current_version
                ));
            }

            let mut result = result_version;
            result["updated"] = json!(updated);
            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Info ────────────────────────────────────────────────────
        Command::Info => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let info = db.info().context("failed to get database info")?;

            let mut features = Vec::new();
            for name in info.plugins.keys() {
                features.push(name.as_str());
            }

            let tables: Vec<Value> = info
                .tables
                .iter()
                .map(|(name, count)| json!({"name": name, "count": count}))
                .collect();

            out.print(&json!({
                "path": info.path.display().to_string(),
                "size_bytes": info.total_size,
                "record_count": info.total_records,
                "tables": tables,
                "features": features,
                "plugins": info.plugins,
            }));
            Ok(EXIT_OK)
        }

        // ── Tables ──────────────────────────────────────────────────
        Command::Tables => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let tables = db.tables_with_counts().context("failed to list tables")?;

            let values: Vec<Value> = tables
                .iter()
                .map(|(name, count)| json!({"name": name, "count": count}))
                .collect();

            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Features (binary build inspection, no DB needed) ────────
        Command::Features { wizard } => {
            if wizard {
                features::run_wizard(out.quiet)
            } else {
                out.print_array(&features::catalog_json());
                Ok(EXIT_OK)
            }
        }

        // ── Detect ──────────────────────────────────────────────────
        Command::Detect => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let results = axil_core::run_all_detectors(&db);
            let triggered: Vec<_> = results.iter().filter(|r| r.triggered).collect();
            out.print(&json!({
                "detectors": serde_json::to_value(&results).unwrap_or(json!([])),
                "issues_found": triggered.len(),
            }));
            Ok(EXIT_OK)
        }

        // ── Doctor ───────────────────────────────────────────────────
        Command::Doctor => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mut report = db.doctor().context("doctor failed")?;

            // Check if downloaded models have checksum sidecar files.
            // Full SHA256 verification is expensive (17-500MB per model), so doctor
            // only checks that checksums exist. Use `verify_model()` for full verification.
            #[cfg(feature = "vector")]
            if db.has_vector_index() {
                use axil_core::diagnostics::{CheckResult, Severity};
                let models_to_check = [
                    axil_vector::models::EmbeddingModel::BgeSmall,
                    axil_vector::models::EmbeddingModel::BgeSmallInt8,
                    axil_vector::models::EmbeddingModel::BgeBase,
                    axil_vector::models::EmbeddingModel::Nomic,
                ];
                for model in &models_to_check {
                    if axil_vector::download::is_model_available(model) {
                        let check_name = format!("model_integrity_{}", model.name());
                        match axil_vector::download::has_checksums(model) {
                            Ok(results) => {
                                let missing: Vec<_> = results
                                    .iter()
                                    .filter(|(_, ok)| !*ok)
                                    .map(|(f, _)| f.as_str())
                                    .collect();
                                if missing.is_empty() {
                                    report.add_check(CheckResult {
                                        name: check_name,
                                        status: Severity::Ok,
                                        detail: format!("{}: checksums present", model.name()),
                                        fix: None,
                                    });
                                } else {
                                    report.add_check(CheckResult {
                                        name: check_name,
                                        status: Severity::Warning,
                                        detail: format!(
                                            "{}: missing checksums for: {}",
                                            model.name(),
                                            missing.join(", "),
                                        ),
                                        fix: Some(format!("axil model-download {}", model.name())),
                                    });
                                }
                            }
                            Err(e) => {
                                eprintln!("  [warn] model check for {}: {e}", model.name());
                            }
                        }
                    }
                }
            }

            // Windows: verify the ONNX Runtime DLL setup up front — a wrong
            // setup otherwise panics deep inside ort on the first embed call
            // (System32 ships an ancient 1.10 DLL that shadows a missing
            // app-dir copy). Mirrors the runtime preflight in axil-vector.
            #[cfg(all(feature = "embed", target_os = "windows"))]
            {
                use axil_core::diagnostics::{CheckResult, Severity};
                match axil_vector::embed::preflight_ort_dll() {
                    Ok(()) => report.add_check(CheckResult {
                        name: "onnxruntime_dll".to_string(),
                        status: Severity::Ok,
                        detail: "ONNX Runtime DLL resolution looks correct".to_string(),
                        fix: None,
                    }),
                    Err(e) => report.add_check(CheckResult {
                        name: "onnxruntime_dll".to_string(),
                        status: Severity::Error,
                        detail: e,
                        fix: Some(
                            "place a >=1.22 onnxruntime.dll next to axil.exe (release \
                             archives bundle one; `cargo binstall axildb` sets this up)"
                                .to_string(),
                        ),
                    }),
                }
            }

            // B: SCIP detection block.
            #[cfg(feature = "scip")]
            {
                use axil_core::diagnostics::{CheckResult, Severity};
                // Look at the repo root (parent of .axil/) plus the .axil/ dir itself.
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let axil_dir = db_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or(cwd.clone());
                let repo_root = axil_dir.parent().unwrap_or(&axil_dir);
                let roots: Vec<&std::path::Path> = vec![repo_root, axil_dir.as_path()];
                let found = axil_scip::discover_scip_files(&roots);
                if found.is_empty() {
                    // Soft signal — only warn when the repo clearly contains code
                    // a SCIP indexer could handle. Otherwise stay Ok.
                    let langs = detect_scip_indexable_languages(repo_root);
                    if langs.is_empty() {
                        report.add_check(CheckResult {
                            name: "scip_index".to_string(),
                            status: Severity::Ok,
                            detail: "no SCIP index configured (non-code repo or not using SCIP)"
                                .to_string(),
                            fix: None,
                        });
                    } else {
                        report.add_check(CheckResult {
                            name: "scip_index".to_string(),
                            status: Severity::Warning,
                            detail: "no *.scip files found — code-graph enrichment is off"
                                .to_string(),
                            fix: Some(suggest_scip_installers(&langs)),
                        });
                    }
                } else {
                    for f in &found {
                        let age_days = f.modified_secs_ago / 86_400;
                        let detail = match axil_scip::inspect_scip(&f.path) {
                            Ok(r) => format!(
                                "{} ({} symbols, {} docs, indexer: {} {}, {}d old)",
                                f.path.display(),
                                r.symbol_count,
                                r.document_count,
                                if r.indexer_name.is_empty() {
                                    "?"
                                } else {
                                    &r.indexer_name
                                },
                                r.indexer_version,
                                age_days,
                            ),
                            Err(e) => format!("{}: decode failed: {e}", f.path.display()),
                        };
                        let severity = if age_days > 14 {
                            Severity::Warning
                        } else {
                            Severity::Ok
                        };
                        report.add_check(CheckResult {
                            name: "scip_index".to_string(),
                            status: severity,
                            detail,
                            fix: if age_days > 14 {
                                Some(
                                    "re-run your SCIP indexer; call edges may be stale".to_string(),
                                )
                            } else {
                                None
                            },
                        });
                    }
                }
            }

            let exit_code = report.exit_code();
            out.print(&serde_json::to_value(&report).unwrap());
            Ok(exit_code)
        }

        // ── Importance ────────────────────────────────────────────────
        Command::MemoryPressure { archive } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let now = chrono::Utc::now();
            let tier_config = axil_core::tiering::TierConfig::default();

            // Load project config so per-table decay half-lives (e.g. faster
            // decay for `errors`, slower for `preferences`) apply to tiering.
            let decay_cfg = detect_project_root(&db_path)
                .and_then(|r| axil_core::load_config_from(&r).ok())
                .map(|c| c.decay);

            // Collect all records and classify tiers
            let tables = db.tables().context("failed to list tables")?;
            let mut all_records = Vec::new();
            for table in &tables {
                if table.starts_with('_') {
                    continue;
                }
                all_records.extend(db.list(table).unwrap_or_default());
            }

            let info = db.info().context("failed to get DB info")?;

            // Single pass: classify tiers and collect archive candidates
            let mut stats = axil_core::tiering::TierStats {
                hot: 0,
                warm: 0,
                cold: 0,
                archived: 0,
                total: all_records.len(),
            };
            let mut archive_candidates = Vec::new();
            for r in &all_records {
                match axil_core::tiering::classify_tier(r, &tier_config, &now, decay_cfg.as_ref()) {
                    axil_core::tiering::MemoryTier::Hot => stats.hot += 1,
                    axil_core::tiering::MemoryTier::Warm => stats.warm += 1,
                    axil_core::tiering::MemoryTier::Cold => stats.cold += 1,
                    axil_core::tiering::MemoryTier::Archived => {
                        stats.archived += 1;
                        archive_candidates.push(r);
                    }
                }
            }

            let mut result = json!({
                "tiers": {
                    "hot": stats.hot,
                    "warm": stats.warm,
                    "cold": stats.cold,
                    "archived": stats.archived,
                    "total": stats.total,
                },
                "db_size_bytes": info.total_size,
                "db_size_human": axil_core::diagnostics::human_bytes(info.total_size),
                "archive_candidates": archive_candidates.len(),
                "pressure": if stats.total > 10000 || info.total_size > 100 * 1024 * 1024 {
                    "high"
                } else if stats.cold + stats.archived > stats.hot + stats.warm {
                    "medium"
                } else {
                    "low"
                },
            });

            // Auto-archive if requested
            if archive && !archive_candidates.is_empty() {
                let mut archived_count = 0usize;
                for record in &archive_candidates {
                    let mut data = record.data.clone();
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("_archived".to_string(), json!(true));
                        obj.insert("_archived_at".to_string(), json!(now.to_rfc3339()));
                    }
                    if db.update(&record.id, data).is_ok() {
                        archived_count += 1;
                    }
                }
                result["archived_now"] = json!(archived_count);
            }

            out.print(&result);
            Ok(EXIT_OK)
        }

        Command::Decay { dry_run } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let now = chrono::Utc::now();
            let default_half_life = axil_core::importance::DEFAULT_HALF_LIFE_DAYS;
            let threshold = axil_core::importance::ARCHIVE_THRESHOLD;

            // Per-table half-life overrides from axil.toml `[decay.tables]`.
            let decay_cfg = detect_project_root(&db_path)
                .and_then(|r| axil_core::load_config_from(&r).ok())
                .map(|c| c.decay);

            let tables = db.tables().context("failed to list tables")?;
            let mut results = Vec::new();
            let mut updated = 0usize;

            for table in &tables {
                if table.starts_with('_') {
                    continue;
                }
                let half_life = decay_cfg
                    .as_ref()
                    .map(|d| d.half_life_for(table))
                    .unwrap_or(default_half_life);
                for record in db.list(table).unwrap_or_default() {
                    if axil_core::importance::is_pinned(&record.data) {
                        continue;
                    }
                    let base = axil_core::importance::get_importance(&record.data);
                    let age_days = (now - record.created_at).num_seconds() as f64 / 86400.0;
                    let effective = axil_core::importance::effective_importance(
                        &record.data,
                        age_days,
                        half_life,
                    );
                    let stored_effective = record
                        .data
                        .get("_effective_importance")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(base);
                    let changed = (effective - stored_effective).abs() > 0.01;
                    if effective < threshold || changed {
                        results.push(json!({
                            "id": record.id.to_string(),
                            "table": record.table,
                            "base_importance": base,
                            "effective_importance": effective,
                            "age_days": age_days as u64,
                            "half_life_days": half_life,
                            "below_archive": effective < threshold,
                        }));
                    }
                    if !dry_run && changed && record.data.get("_importance").is_some() {
                        let mut data = record.data.clone();
                        axil_core::importance::apply_decay(&mut data, age_days, half_life);
                        if db.update(&record.id, data).is_ok() {
                            updated += 1;
                        }
                    }
                }
            }

            let archive_candidates = results
                .iter()
                .filter(|r| r["below_archive"].as_bool().unwrap_or(false))
                .count();

            out.print(&json!({
                "dry_run": dry_run,
                "updated": updated,
                "archive_candidates": archive_candidates,
                "records": results,
            }));
            Ok(EXIT_OK)
        }

        Command::Importance { id, pin, unpin } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let record = db
                .get(&rid)?
                .ok_or_else(|| anyhow::anyhow!("record not found: {id}"))?;

            if pin {
                let mut data = record.data.clone();
                if let Some(obj) = data.as_object_mut() {
                    obj.insert(
                        "_importance".to_string(),
                        json!(axil_core::importance::PINNED_IMPORTANCE),
                    );
                    obj.insert("_importance_pinned".to_string(), json!(true));
                }
                db.update(&rid, data).context("failed to pin importance")?;
                out.print(&json!({"id": id, "importance": 1.0, "pinned": true}));
            } else if unpin {
                let mut data = record.data.clone();
                let score = axil_core::importance::compute_importance(&data);
                if let Some(obj) = data.as_object_mut() {
                    obj.insert("_importance".to_string(), json!(score));
                    obj.remove("_importance_pinned");
                }
                db.update(&rid, data)
                    .context("failed to unpin importance")?;
                out.print(&json!({"id": id, "importance": score, "pinned": false}));
            } else {
                let breakdown = axil_core::importance::compute_importance_breakdown(&record.data);
                let current = axil_core::importance::get_importance(&record.data);
                let pinned = axil_core::importance::is_pinned(&record.data);
                out.print(&json!({
                    "id": id,
                    "importance": current,
                    "pinned": pinned,
                    "breakdown": breakdown,
                }));
            }
            Ok(EXIT_OK)
        }

        // ── Stats ───────────────────────────────────────────────────
        Command::Stats {
            table,
            watch,
            activation,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            if activation {
                let config = axil_core::ActivationConfig::default();
                let act_stats = db
                    .activation_stats(table.as_deref(), &config)
                    .context("activation stats failed")?;
                out.print(&serde_json::to_value(&act_stats).unwrap());
                return Ok(EXIT_OK);
            }

            if let Some(interval_secs) = watch {
                let interval = std::time::Duration::from_secs(interval_secs.max(1));
                loop {
                    // Clear screen for watch mode.
                    eprint!("\x1B[2J\x1B[H");
                    let stats = db.stats(table.as_deref()).context("stats failed")?;
                    out.print(&serde_json::to_value(&stats).unwrap());
                    out.status(&format!(
                        "(refreshing every {interval_secs}s — Ctrl+C to stop)"
                    ));
                    std::thread::sleep(interval);
                }
            } else {
                let stats = db.stats(table.as_deref()).context("stats failed")?;
                out.print(&serde_json::to_value(&stats).unwrap());
                Ok(EXIT_OK)
            }
        }

        // ── Bench ───────────────────────────────────────────────────
        Command::Bench { save, compare } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            if compare {
                // Load last saved benchmark and current run, then compare.
                let current = db.bench().context("bench failed")?;
                let previous = db
                    .query()
                    .table("_bench_results")
                    .order_by("created_at", axil_core::SortDirection::Desc)
                    .limit(1)
                    .exec()
                    .ok()
                    .and_then(|r| r.into_iter().next());

                if let Some(prev_record) = previous {
                    let mut comparison = Vec::new();
                    for bench in &current.benchmarks {
                        let prev_ops = prev_record
                            .data
                            .get("benchmarks")
                            .and_then(|b| b.as_array())
                            .and_then(|arr| {
                                arr.iter().find(|b| {
                                    b.get("name").and_then(|n| n.as_str()) == Some(&bench.name)
                                })
                            })
                            .and_then(|b| b.get("ops_per_sec").and_then(|v| v.as_f64()));

                        let change = prev_ops.map(|prev| {
                            let pct = ((bench.ops_per_sec - prev) / prev) * 100.0;
                            json!({"name": bench.name, "ops_per_sec": bench.ops_per_sec, "prev_ops_per_sec": prev, "change_pct": (pct * 10.0).round() / 10.0})
                        }).unwrap_or_else(|| {
                            json!({"name": bench.name, "ops_per_sec": bench.ops_per_sec, "prev_ops_per_sec": null, "change_pct": null})
                        });
                        comparison.push(change);
                    }
                    out.print(&json!({
                        "current": serde_json::to_value(&current).unwrap(),
                        "comparison": comparison,
                    }));
                } else {
                    out.status("No previous benchmark found — showing current results only.");
                    out.print(&serde_json::to_value(&current).unwrap());
                }
            } else {
                let report = db.bench().context("bench failed")?;
                out.print(&serde_json::to_value(&report).unwrap());

                if save {
                    let data = serde_json::to_value(&report).unwrap();
                    let record = db
                        .insert("_bench_results", data)
                        .context("failed to save benchmark")?;
                    out.status(&format!("Benchmark saved as {}", record.id));
                }
            }
            Ok(EXIT_OK)
        }

        // ── Slow Queries ────────────────────────────────────────────
        Command::SlowQueries {
            limit,
            after,
            clear,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            if clear {
                db.clear_slow_queries();
                out.print(&json!({"cleared": true}));
            } else {
                let entries = db.slow_queries(limit, after.as_deref());
                let values: Vec<Value> = entries
                    .iter()
                    .map(|e| serde_json::to_value(e).unwrap())
                    .collect();
                out.print_array(&values);
            }
            Ok(EXIT_OK)
        }

        // ── Explain ──────────────────────────────────────────────────
        Command::Explain {
            table,
            where_clauses,
            similar,
            traverse,
            limit,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mut qb = db.query().table(&table).limit(limit);

            for clause in &where_clauses {
                let (field, op, value) = parse_where_clause(clause)?;
                qb = qb.where_field(&field, op, value);
            }

            if let Some(ref text) = similar {
                qb = qb.similar_to(text, limit);
            }

            if let Some(ref path) = traverse {
                qb = qb.traverse(path);
            }

            let plan = qb.explain();
            out.print(&serde_json::to_value(&plan).unwrap());
            Ok(EXIT_OK)
        }

        // ── Audit Log ───────────────────────────────────────────────
        Command::Log {
            limit,
            after,
            table,
            op,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let entries = db.audit_log(limit, after.as_deref(), table.as_deref(), op.as_deref());
            let values: Vec<Value> = entries
                .iter()
                .map(|e| serde_json::to_value(e).unwrap())
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Compact ─────────────────────────────────────────────────
        Command::Compact { drop_engine } => {
            let db_path = require_db(&db_opt)?;
            if let Some(engine) = drop_engine {
                // Never open the DB here — the Engine being dropped may be
                // orphaned or version-incompatible, which is exactly why it's
                // being removed. This is a pure file operation on the companion.
                let report = axil_core::drop_engine_companion(&db_path, &engine)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                out.print(&serde_json::to_value(&report).unwrap());
                Ok(EXIT_OK)
            } else {
                let db = open_with_all_detected(&db_path)?;
                let report = db.compact().context("compact failed")?;
                out.print(&serde_json::to_value(&report).unwrap());
                Ok(EXIT_OK)
            }
        }

        // ── Health Report ───────────────────────────────────────────
        Command::HealthReport {
            brief,
            save,
            compare,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let report = db.report().context("report generation failed")?;
            let report_json = serde_json::to_value(&report).unwrap();

            if compare {
                // Compare with last saved report.
                let saved = db.list("_health_reports")?;
                if let Some(last) = saved.last() {
                    let last_score = last.data.get("score").and_then(|v| v.as_u64()).unwrap_or(0);
                    let diff = report.score as i64 - last_score as i64;
                    out.print(&json!({
                        "current_score": report.score,
                        "previous_score": last_score,
                        "score_diff": diff,
                        "current_health": report.overall_health,
                        "previous_health": last.data.get("overall_health").and_then(|v| v.as_str()).unwrap_or("unknown"),
                        "previous_at": last.data.get("generated_at").and_then(|v| v.as_str()).unwrap_or("unknown"),
                        "trend": if diff > 0 { "improving" } else if diff < 0 { "declining" } else { "stable" },
                    }));
                } else {
                    out.print(&json!({
                        "current_score": report.score,
                        "note": "no previous report saved — run with --save first",
                    }));
                }
            } else if brief {
                out.print(&json!({
                    "overall_health": report.overall_health,
                    "score": report.score,
                    "summary": report.summary,
                }));
            } else {
                out.print(&report_json);
            }

            if save {
                db.insert("_health_reports", report_json)
                    .context("failed to save health report")?;
                out.status("health report saved");
            }

            Ok(EXIT_OK)
        }

        // ── Trends ──────────────────────────────────────────────────
        Command::Trends { days } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let report = db.trends(days).context("trends failed")?;
            out.print(&serde_json::to_value(&report).unwrap());
            Ok(EXIT_OK)
        }

        // ── Snapshot ────────────────────────────────────────────────
        Command::Snapshot => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let entry = db.snapshot_metrics().context("snapshot failed")?;
            out.print(&serde_json::to_value(&entry).unwrap());
            Ok(EXIT_OK)
        }

        // ── Maintain (opportunistic, time-gated) ────────────────────
        Command::Maintain {
            if_stale,
            in_background,
            dry_run,
        } => {
            let db_path = require_db(&db_opt)?;
            let config = load_config(&db_path).unwrap_or_default();
            let m = &config.maintenance;
            let axil_dir = db_path
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            let lock_path = axil_dir.join("maintain.lock");

            // Opportunistic mode honors the [maintenance] auto switch; an
            // explicit `axil maintain` (no --if-stale) always runs.
            if if_stale && !m.auto {
                out.print(&json!({
                    "ok": true, "skipped": true,
                    "reason": "maintenance.auto disabled", "ran": [],
                }));
                return Ok(EXIT_OK);
            }

            // Background: re-exec detached so the brain hook never blocks.
            // `nohup` matches the scip-refresh detachment idiom — a plain
            // spawn can catch SIGHUP when the parent exits. The lock is
            // claimed atomically (O_CREAT|O_EXCL) so two concurrent fires
            // can't both spawn; an mtime gate reclaims a crashed run's lock.
            if in_background {
                const STALE_LOCK_SECS: u64 = 300;
                if let Ok(md) = std::fs::metadata(&lock_path) {
                    if let Ok(modified) = md.modified() {
                        if let Ok(elapsed) = modified.elapsed() {
                            if elapsed.as_secs() < STALE_LOCK_SECS {
                                out.print(&json!({
                                    "ok": true, "skipped": true,
                                    "reason": "already_running",
                                    "lock_age_seconds": elapsed.as_secs(),
                                }));
                                return Ok(EXIT_OK);
                            }
                            // Stale (prior run crashed) — reclaim it.
                            let _ = std::fs::remove_file(&lock_path);
                        }
                    }
                }
                if let Some(parent) = lock_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                // Atomically claim the lock BEFORE spawning (shared
                // `LockGuard::try_acquire`, same as scip-refresh). If a racing
                // invocation created it between our mtime check and here, bail
                // the guard a plain `fs::write` would miss. `mem::forget`
                // keeps the file for the detached child, which removes it on exit.
                let pid = std::process::id().to_string();
                match LockGuard::try_acquire(lock_path.clone(), &pid) {
                    Ok(g) => std::mem::forget(g),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        out.print(&json!({
                            "ok": true, "skipped": true,
                            "reason": "already_running_race",
                        }));
                        return Ok(EXIT_OK);
                    }
                    Err(e) => {
                        return Err(anyhow::Error::new(e).context("failed to claim maintain lock"));
                    }
                }
                let exe =
                    std::env::current_exe().context("failed to resolve current executable")?;
                let db_abs = db_path.canonicalize().unwrap_or_else(|_| db_path.clone());
                let mut parts = vec!["nohup".to_string(), shell_quote(&exe.to_string_lossy())];
                parts.push("--db".to_string());
                parts.push(shell_quote(&db_abs.display().to_string()));
                parts.push("maintain".to_string());
                if if_stale {
                    parts.push("--if-stale".to_string());
                }
                if dry_run {
                    parts.push("--dry-run".to_string());
                }
                parts.push(format!(
                    ">/dev/null 2>{}",
                    shell_quote(&axil_dir.join("maintain.log").to_string_lossy())
                ));
                parts.push("</dev/null &".to_string());
                let shell_cmd = parts.join(" ");
                match std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&shell_cmd)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                {
                    Ok(_) => out.print(&json!({ "ok": true, "spawned": true })),
                    Err(e) => {
                        let _ = std::fs::remove_file(&lock_path);
                        return Err(
                            anyhow::Error::new(e).context("failed to spawn background maintain")
                        );
                    }
                }
                return Ok(EXIT_OK);
            }

            // Foreground: run each due task. Only SAFE, ADDITIVE tasks run
            // automatically — `snapshot` (trend metrics) and `health-report
            // --save`. Downsampling is deliberately excluded: `db.heal`'s
            // downsample purges (deletes) records past the retention window,
            // so auto-firing it from the hook would silently destroy memory.
            // Keep downsampling explicit via `axil heal`.
            let db = open_with_all_detected(&db_path)?;
            let now_secs = chrono::Utc::now().timestamp();
            let runs = db.list("_maintenance_runs").unwrap_or_default();
            let last_run = |task: &str| -> Option<i64> {
                runs.iter()
                    .filter(|r| r.data.get("task").and_then(|v| v.as_str()) == Some(task))
                    .map(|r| r.created_at.timestamp())
                    .max()
            };
            let due = |task: &str, every: &str, default: u64| -> bool {
                let secs = axil_core::config::parse_duration_secs(every).unwrap_or(default);
                !if_stale || axil_core::config::is_due(last_run(task), now_secs, secs)
            };
            // Record a completed task (resets its cadence). Staleness reads the
            // row's intrinsic `created_at`, so no timestamp field is stored.
            // Drop any prior row for the same task first, so `_maintenance_runs`
            // holds one row per task and can't grow unbounded over the DB's life.
            let mark = |task: &str| {
                for r in runs
                    .iter()
                    .filter(|r| r.data.get("task").and_then(|v| v.as_str()) == Some(task))
                {
                    let _ = db.delete(&r.id);
                }
                let _ = db.insert("_maintenance_runs", json!({ "task": task }));
            };
            // Trim an append-only table to its `keep` most-recent rows.
            let trim_latest = |table: &str, keep: usize| {
                if let Ok(mut rows) = db.list(table) {
                    if rows.len() > keep {
                        rows.sort_by_key(|r| r.created_at);
                        for r in rows.iter().take(rows.len() - keep) {
                            let _ = db.delete(&r.id);
                        }
                    }
                }
            };
            let mut ran: Vec<&str> = Vec::new();
            let mut skipped: Vec<&str> = Vec::new();
            let mut errors: Vec<serde_json::Value> = Vec::new();

            // snapshot — additive. Errors are collected, never propagated via
            // `?`, so one failing task can't abort the others or leak the lock.
            if due("snapshot", &m.snapshot_every, 86_400) {
                if dry_run {
                    ran.push("snapshot");
                } else {
                    match db.snapshot_metrics() {
                        Ok(_) => {
                            mark("snapshot");
                            ran.push("snapshot");
                        }
                        Err(e) => errors.push(json!({"task": "snapshot", "error": e.to_string()})),
                    }
                }
            } else {
                skipped.push("snapshot");
            }

            // health-report --save — additive.
            if due("health_report", &m.health_report_every, 604_800) {
                if dry_run {
                    ran.push("health_report");
                } else {
                    match db.report() {
                        Ok(report) => {
                            let _ = db
                                .insert("_health_reports", serde_json::to_value(&report).unwrap());
                            // Bound the report table (~1 year of weekly reports).
                            trim_latest("_health_reports", 52);
                            mark("health_report");
                            ran.push("health_report");
                        }
                        Err(e) => {
                            errors.push(json!({"task": "health_report", "error": e.to_string()}))
                        }
                    }
                }
            } else {
                skipped.push("health_report");
            }

            // Clear the background-spawn lock (no-op for a direct foreground run).
            let _ = std::fs::remove_file(&lock_path);

            out.print(&json!({
                "ok": errors.is_empty(),
                "dry_run": dry_run,
                "if_stale": if_stale,
                "ran": ran,
                "skipped": skipped,
                "errors": errors,
            }));
            Ok(EXIT_OK)
        }

        // ── Store ───────────────────────────────────────────────────
        Command::Store {
            table,
            json_data,
            #[cfg(feature = "embed")]
            embed,
            entities,
            llm,
            agent,
            code_ref,
        } => {
            let db_path = require_db(&db_opt)?;

            #[cfg(feature = "embed")]
            let db = if embed.is_some() {
                open_with_embedder(&db_path)?
            } else {
                open_with_all_detected(&db_path)?
            };
            #[cfg(not(feature = "embed"))]
            let db = open_with_all_detected(&db_path)?;

            // Wire up LLM if --llm flag is set and config exists.
            let db = if llm { wire_llm(db, &db_path)? } else { db };

            let mut data = read_json_input(&json_data)?;

            // Merge --entities into the data as metadata.
            if let Some(ref entities_json) = entities {
                let ents: Value = serde_json::from_str(entities_json)
                    .context("--entities must be a JSON array")?;
                if let Some(obj) = data.as_object_mut() {
                    obj.insert("_entities".to_string(), ents);
                }
            }

            // Tag with agent name if --agent is set.
            if let Some(ref agent_name) = agent {
                if let Some(obj) = data.as_object_mut() {
                    obj.insert("_agent".to_string(), json!(agent_name));
                }
            }

            #[cfg(feature = "indexer")]
            if !code_ref.is_empty() {
                let mut refs: Vec<Value> = Vec::with_capacity(code_ref.len());
                for spec in &code_ref {
                    match resolve_code_ref(&db, spec) {
                        Ok(Some(v)) => refs.push(v),
                        Ok(None) => {
                            eprintln!(
                                "warning: --code-ref '{spec}' did not match any proxy or path"
                            );
                        }
                        Err(e) => {
                            eprintln!("warning: --code-ref '{spec}' resolution error: {e}");
                        }
                    }
                }
                if !refs.is_empty() {
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("code_refs".to_string(), Value::Array(refs));
                    }
                }
            }
            // --code-ref resolves against the indexer's proxy table; without
            // that feature there is nothing to resolve against.
            #[cfg(not(feature = "indexer"))]
            if !code_ref.is_empty() {
                eprintln!("warning: --code-ref requires the `indexer` feature; ignored");
            }

            // LLM-enhanced entity extraction.
            if llm {
                // Extract text from common fields for entity extraction.
                let text_for_extraction = data
                    .as_object()
                    .map(|obj| {
                        obj.iter()
                            .filter(|(k, _)| !k.starts_with('_'))
                            .filter_map(|(_, v)| v.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();

                if !text_for_extraction.is_empty() {
                    let extracted = db.extract_entities_enhanced(&text_for_extraction);
                    if !extracted.is_empty() && entities.is_none() {
                        let entity_names: Vec<Value> = extracted
                            .iter()
                            .map(
                                |e| json!({"name": e.name, "type": format!("{:?}", e.entity_type)}),
                            )
                            .collect();
                        if let Some(obj) = data.as_object_mut() {
                            obj.insert("_entities".to_string(), json!(entity_names));
                        }
                    }
                }
            }

            let record = db.insert(&table, data).context("insert failed")?;

            // Auto-embed if requested.
            #[cfg(feature = "embed")]
            if let Some(ref fields) = embed {
                for field in fields.split(',') {
                    let field = field.trim();
                    if !field.is_empty() {
                        db.embed_field(&record.id, field)
                            .with_context(|| format!("failed to embed field '{field}'"))?;
                    }
                }
            }

            let mut result = json!({
                "id": record.id.to_string(),
                "table": record.table,
                "created_at": format_dt(&record.created_at),
            });

            // Include LLM usage if LLM was used.
            if llm {
                let usage = db.llm_usage();
                if usage.calls > 0 {
                    result.as_object_mut().unwrap().insert(
                        "llm_usage".to_string(),
                        serde_json::to_value(&usage).unwrap(),
                    );
                }
            }

            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Batch Insert ──────────────────────────────────────────────
        Command::BatchInsert { table, json_data } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            let items: Vec<Value> = if json_data == "-" {
                // Read from stdin — support both JSON array and JSONL (one object per line).
                let mut buf = String::new();
                io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)
                    .context("failed to read from stdin")?;
                if buf.len() as u64 > MAX_STDIN_BYTES {
                    anyhow::bail!("stdin input exceeds size limit");
                }
                // Try parsing as a JSON array first.
                if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&buf) {
                    arr
                } else {
                    // Fall back to JSONL: one JSON object per line.
                    buf.lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| {
                            serde_json::from_str(l).with_context(|| {
                                format!(
                                    "invalid JSON line: {}",
                                    l.chars().take(80).collect::<String>()
                                )
                            })
                        })
                        .collect::<Result<Vec<Value>>>()?
                }
            } else {
                let raw = read_json_input(&json_data)?;
                match raw {
                    Value::Array(arr) => arr,
                    other => vec![other],
                }
            };

            let records = db
                .insert_batch(&table, items)
                .context("batch insert failed")?;

            let ids: Vec<Value> = records
                .iter()
                .map(|r| json!({"id": r.id.to_string(), "table": r.table}))
                .collect();
            out.print(&json!({"inserted": ids.len(), "records": ids}));
            Ok(EXIT_OK)
        }

        // ── Get ─────────────────────────────────────────────────────
        Command::Get { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_read_command(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;

            match db.get(&rid).context("get failed")? {
                Some(record) => {
                    out.print(&record_to_json(&record));
                    Ok(EXIT_OK)
                }
                None => {
                    eprintln!("{{\"error\":\"not found\",\"id\":{}}}", json!(id));
                    Ok(EXIT_NOT_FOUND)
                }
            }
        }

        // ── Update ──────────────────────────────────────────────────
        Command::Update { id, json_data } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let data = read_json_input(&json_data)?;

            let record = db.update(&rid, data).context("update failed")?;
            out.print(&record_to_json(&record));
            Ok(EXIT_OK)
        }

        // ── Delete ──────────────────────────────────────────────────
        Command::Delete { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;

            if db.delete(&rid).context("delete failed")? {
                out.print(&json!({"deleted": true, "id": id}));
                Ok(EXIT_OK)
            } else {
                eprintln!("{{\"error\":\"not found\",\"id\":{}}}", json!(id));
                Ok(EXIT_NOT_FOUND)
            }
        }

        // ── List ────────────────────────────────────────────────────
        Command::List {
            table,
            limit,
            offset,
            where_clauses,
            agent,
            include_archived,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_read_command(&db_path)?;

            let filter_archived = |records: Vec<axil_core::Record>| -> Vec<axil_core::Record> {
                if include_archived {
                    records
                } else {
                    records
                        .into_iter()
                        .filter(|r| {
                            !r.data
                                .get("_archived")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                        })
                        .collect()
                }
            };

            if where_clauses.is_empty() && limit.is_none() && offset.is_none() && agent.is_none() {
                let records = filter_archived(db.list(&table).context("list failed")?);
                let values: Vec<Value> = records.iter().map(record_to_json).collect();
                out.print_array(&values);
            } else {
                // Use query builder for filtered list.
                let mut qb = db.query().table(&table);
                for clause in &where_clauses {
                    let (field, op, value) = parse_where_clause(clause)?;
                    qb = qb.where_field(&field, op, value);
                }
                if let Some(ref agent_name) = agent {
                    qb = qb.where_field("_agent", Op::Eq, json!(agent_name));
                }
                if let Some(n) = limit {
                    qb = qb.limit(n);
                }
                if let Some(n) = offset {
                    qb = qb.offset(n);
                }
                let results = filter_archived(qb.exec().context("query failed")?);
                let values: Vec<Value> = results.iter().map(record_to_json).collect();
                out.print_array(&values);
            }
            Ok(EXIT_OK)
        }

        // ── Portable export / import ─────────────────────────────────
        Command::Export {
            out: out_file,
            tables,
            since,
            include_system,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_read_command(&db_path)?;

            let since_dt = match since {
                Some(ref s) => Some(
                    DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .with_context(|| format!("--since must be ISO 8601: '{s}'"))?,
                ),
                None => None,
            };
            let opts = axil_core::ExportOptions {
                tables: if tables.is_empty() {
                    None
                } else {
                    Some(tables)
                },
                since: since_dt,
                include_system,
            };

            let stats = if let Some(ref path) = out_file {
                let file = std::fs::File::create(path)
                    .with_context(|| format!("failed to create '{}'", path.display()))?;
                let mut w = io::BufWriter::new(file);
                let stats = axil_core::export_to_writer(&db, &opts, &mut w)
                    .context("export failed")?;
                w.flush().ok();
                stats
            } else {
                let stdout = io::stdout();
                let mut w = io::BufWriter::new(stdout.lock());
                let stats = axil_core::export_to_writer(&db, &opts, &mut w)
                    .context("export failed")?;
                w.flush().ok();
                stats
            };

            // The JSONL itself is the payload; report the summary on stderr so a
            // piped-to-file export stays a clean stream.
            out.status(&format!(
                "exported {} record(s), {} edge(s) across {} table(s){}",
                stats.records,
                stats.edges,
                stats.tables,
                out_file
                    .as_ref()
                    .map(|p| format!(" to {}", p.display()))
                    .unwrap_or_default()
            ));
            Ok(EXIT_OK)
        }

        Command::Import {
            file,
            dedup,
            dry_run,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            let opts = axil_core::ImportOptions { dedup, dry_run };

            let result = if file == "-" {
                let stdin = io::stdin();
                axil_core::import_from_reader(&db, &opts, stdin.lock())
            } else {
                let f = std::fs::File::open(&file)
                    .with_context(|| format!("failed to open '{file}'"))?;
                axil_core::import_from_reader(&db, &opts, io::BufReader::new(f))
            };

            // A mid-stream failure still committed everything before it (import
            // is fail-fast with partial state) — surface the partial report so
            // the accounting isn't lost with the error, then fail below.
            let (report, interrupted) = match result {
                Ok(report) => (report, None),
                Err(axil_core::AxilError::ImportInterrupted { report, source }) => {
                    (*report, Some(source.to_string()))
                }
                Err(e) => return Err(e).context("import failed"),
            };

            out.print(&json!({
                "dry_run": dry_run,
                "interrupted": interrupted.is_some(),
                "imported": report.imported,
                "overwritten": report.overwritten,
                "skipped_id": report.skipped_id,
                "skipped_dup": report.skipped_dup,
                "superseded": report.superseded,
                "edges_created": report.edges_created,
                "edges_skipped": report.edges_skipped,
                "edges_remapped": report.edges_remapped,
                "id_remapped": report.id_remapped,
                "embeddings": report.embeddings,
            }));
            // An import that replaces or demotes existing records should never
            // pass silently — the report counts it, but say it out loud too.
            if report.overwritten > 0 {
                eprintln!(
                    "warning: {} record(s) overwritten in place (same id already existed) — \
                     re-importing a stale export replaces newer local data; use --dedup to skip instead",
                    report.overwritten
                );
            }
            if report.superseded > 0 {
                eprintln!(
                    "warning: {} local record(s) superseded by newer imported near-duplicates",
                    report.superseded
                );
            }
            if let Some(source) = interrupted {
                eprintln!(
                    "error: import interrupted after partial write — the counts above are what \
                     was committed before the failure: {source}"
                );
                anyhow::bail!("import interrupted: {source}");
            }
            // Imported-but-unembedded records are invisible to semantic recall;
            // say so now with the fix, instead of letting it surface later as
            // mysteriously weaker recall.
            match &report.embeddings {
                Some(axil_core::EmbeddingVerification::Verified { missing, .. })
                    if *missing > 0 =>
                {
                    eprintln!(
                        "warning: {missing} imported record(s) have no embedding — \
                         semantic recall won't see them until `axil heal --reindex`"
                    );
                }
                Some(axil_core::EmbeddingVerification::EngineUnavailable { affected })
                    if *affected > 0 =>
                {
                    eprintln!(
                        "warning: imported {affected} record(s) with no embedder attached — \
                         run `axil heal --reindex` once embeddings are available"
                    );
                }
                _ => {}
            }
            Ok(EXIT_OK)
        }

        // ── Recall ──────────────────────────────────────────────────
        #[cfg(feature = "embed")]
        Command::Recall {
            query,
            top_k,
            after,
            before,
            alpha,
            fresh_only,
            explain,
            feedback: _feedback_flag,
            table: table_filter,
            r#type: type_filter,
            budget,
            recall_format,
            no_dedup,
            no_widen,
            agent: agent_filter,
            min_importance,
            scope,
            min_confidence,
            timeout_ms,
            rerank,
            expand,
            expand_neighbors,
            no_cascade,
            profile,
        } => {
            let db_path = require_db(&db_opt)?;
            if top_k > MAX_RESULT_LIMIT {
                anyhow::bail!("--top-k exceeds maximum of {MAX_RESULT_LIMIT}");
            }
            let recall_start = std::time::Instant::now();
            let deadline = timeout_ms.map(|ms| recall_start + std::time::Duration::from_millis(ms));
            let deadline_exceeded = || -> bool {
                deadline
                    .map(|d| std::time::Instant::now() > d)
                    .unwrap_or(false)
            };

            #[allow(unused_mut)]
            let mut db = open_read_command(&db_path)?;

            // Expand BEFORE the similarity fetch so every downstream stage sees the same query.
            let query = if expand {
                expand_query(&db, &query, expand_neighbors)
            } else {
                query
            };

            // Resolve project root and config once for freshness + auto-refresh.
            #[cfg(feature = "indexer")]
            let (resolved_root, resolved_config) = {
                let root = detect_project_root(&db_path);
                let cfg = root
                    .as_ref()
                    .and_then(|r| axil_core::load_config_from(r).ok());
                (root, cfg)
            };

            // Compute stale file set for freshness annotations and --fresh-only filtering.
            let stale_paths: std::collections::HashSet<String> = {
                #[cfg(feature = "indexer")]
                {
                    match (&resolved_root, &resolved_config) {
                        (Some(root), Some(cfg)) => {
                            axil_indexer::freshness::stale_file_paths(&db, root, &cfg.index)
                        }
                        _ => std::collections::HashSet::new(),
                    }
                }
                #[cfg(not(feature = "indexer"))]
                {
                    std::collections::HashSet::new()
                }
            };

            // Parse time filters.
            let after_dt = after
                .as_deref()
                .map(|s| {
                    parse_datetime_us(s)
                        .map(|us| DateTime::from_timestamp_micros(us).unwrap_or_default())
                })
                .transpose()?;
            let before_dt = before
                .as_deref()
                .map(|s| {
                    parse_datetime_us(s)
                        .map(|us| DateTime::from_timestamp_micros(us).unwrap_or_default())
                })
                .transpose()?;

            // Both paths use db.recall() for multi-signal scoring
            let alpha = alpha.clamp(0.0, 1.0);
            let scope_filter: Vec<String> = if let Some(s) = scope.as_deref() {
                let mut out = Vec::new();
                for raw in s.split(',') {
                    let token = raw.trim();
                    if token.is_empty() {
                        continue;
                    }
                    let parsed = axil_core::MemoryScope::parse(token).ok_or_else(|| {
                        anyhow::anyhow!(
                            "invalid scope: {token}. Valid: session, agent, project, user, global"
                        )
                    })?;
                    out.push(parsed.to_string());
                }
                out
            } else {
                Vec::new()
            };
            let cfg = axil_core::RecallConfig {
                weights: axil_core::ScoreWeights {
                    vector: alpha,
                    recency: 1.0 - alpha,
                    ..Default::default()
                },
                scope_filter,
                min_confidence,
                min_importance,
                // Enable QTC on the default recall path — re-scoring the top
                // candidates against their stored chunk embeddings lifts
                // hit-rate from ~92% to ~97% (LongMemEval-S) without adding
                // query-time embedder calls when chunks were precomputed at
                // insert.
                qtc: Some(axil_core::scoring::QtcConfig::default()),
                // Collapse near-duplicate hits before truncation so top-k slots
                // aren't spent on restated memories. `--no-dedup`
                // restores the old behavior (used for before/after baselines).
                dedup: axil_core::scoring::DedupConfig {
                    enabled: !no_dedup,
                    // Widen k when the kept top-k compresses far better than the
                    // candidate pool — a diverse cluster was cut. `--no-widen`
                    // restores the plain top-k cut.
                    completeness_widen: !no_widen,
                    ..Default::default()
                },
                ..Default::default()
            };
            // Keep a copy for cascade fallbacks — the primary recall consumes `cfg`.
            let cfg_for_cascade = cfg.clone();

            // Deadline-bounded recall: when --timeout-ms is set we run db.recall in a worker
            // thread and wait up to the remaining budget on a channel. If it doesn't finish
            // in time we return empty partial results and abandon the thread — the CLI process
            // exits shortly afterward and tears down any lingering work.
            let mut recall_results = if deadline_exceeded() {
                Vec::new()
            } else if let Some(d) = deadline {
                use std::sync::Arc;
                let db_arc: Arc<axil_core::Axil> = Arc::new(db);
                let db_handle = db_arc.clone();
                let q = query.clone();
                let cfg_clone = cfg.clone();
                let tk = top_k;
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(db_handle.recall(&q, tk, Some(cfg_clone)));
                });
                let remaining = d.saturating_duration_since(std::time::Instant::now());
                let result = match rx.recv_timeout(remaining) {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => {
                        eprintln!(
                            "[axil recall] timeout_ms={} exceeded inside db.recall; \
                                   returning partial results",
                            timeout_ms.unwrap_or(0)
                        );
                        Vec::new()
                    }
                };
                // Reclaim `db` only if the worker already dropped its clone; otherwise we keep
                // the Arc alive for the rest of the handler by leaving it leaked (the process
                // is about to exit anyway and the leak is bounded to this invocation).
                db = Arc::try_unwrap(db_arc).unwrap_or_else(|arc| {
                    // SAFETY fallback: we can't take Axil out of the Arc when the worker thread
                    // still holds it. Leak this arc and open a fresh handle for the remainder
                    // of the pipeline — this path only runs on timeout with a stuck worker, so
                    // the cost of re-opening the DB is an acceptable worst-case fallback.
                    std::mem::forget(arc);
                    open_with_all_detected(&db_path).expect("reopen after timeout fallback")
                });
                result
            } else {
                db.recall(&query, top_k, Some(cfg))?
            };

            // Normalize the --type facet filter once (case-insensitive,
            // trimmed). Matched against `data.type` as a plain where-clause —
            // no index, no scoring change. Records without a `type` field are
            // excluded when --type is set.
            let type_filter = type_filter.map(|t| t.trim().to_lowercase());

            // Apply table, type, time, and freshness filters.
            recall_results.retain(|rr| {
                if let Some(ref tf) = table_filter {
                    if rr.record.table != *tf {
                        return false;
                    }
                }
                if let Some(ref tf) = type_filter {
                    let record_type = rr
                        .record
                        .data
                        .get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_lowercase());
                    if record_type.as_deref() != Some(tf.as_str()) {
                        return false;
                    }
                }
                if let Some(after_dt) = after_dt {
                    if rr.record.created_at < after_dt {
                        return false;
                    }
                }
                if let Some(before_dt) = before_dt {
                    if rr.record.created_at > before_dt {
                        return false;
                    }
                }
                if fresh_only && !stale_paths.is_empty() {
                    let path = rr
                        .record
                        .data
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if stale_paths.contains(path) {
                        return false;
                    }
                }
                if let Some(ref af) = agent_filter {
                    let record_agent = rr
                        .record
                        .data
                        .get("_agent")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if record_agent != af.as_str() {
                        return false;
                    }
                }
                if let Some(min_imp) = min_importance {
                    let imp = axil_core::importance::get_importance(&rr.record.data);
                    if imp < min_imp {
                        return false;
                    }
                }
                // Exclude archived records by default
                if rr
                    .record
                    .data
                    .get("_archived")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    return false;
                }
                true
            });

            // Cascade: when primary recall + filters returned nothing, try
            // progressively cheaper-then-broader fallbacks before declaring
            // empty. Each rung honors the same deadline. First non-empty
            // wins. Tagged via stderr so callers (session-heal hook) can
            // distinguish "real miss" from "matched on rung N".
            let mut cascade_rung: Option<&'static str> = None;
            if !no_cascade && recall_results.is_empty() && !deadline_exceeded() {
                // Re-applies the same post-filter set as the inline retain
                // above so cascade results respect the user's --table /
                // --after / --before / --agent / --fresh-only / --min-importance
                // intent. Defined as a local helper to avoid duplicating the
                // 30-line predicate three times.
                let apply_filters = |rs: &mut Vec<axil_core::scoring::RecallResult>| {
                    rs.retain(|rr| {
                        if let Some(ref tf) = table_filter {
                            if rr.record.table != *tf {
                                return false;
                            }
                        }
                        if let Some(ref tf) = type_filter {
                            let record_type = rr
                                .record
                                .data
                                .get("type")
                                .and_then(|v| v.as_str())
                                .map(|s| s.trim().to_lowercase());
                            if record_type.as_deref() != Some(tf.as_str()) {
                                return false;
                            }
                        }
                        if let Some(after_dt) = after_dt {
                            if rr.record.created_at < after_dt {
                                return false;
                            }
                        }
                        if let Some(before_dt) = before_dt {
                            if rr.record.created_at > before_dt {
                                return false;
                            }
                        }
                        if fresh_only && !stale_paths.is_empty() {
                            let path = rr
                                .record
                                .data
                                .get("path")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if stale_paths.contains(path) {
                                return false;
                            }
                        }
                        if let Some(ref af) = agent_filter {
                            let record_agent = rr
                                .record
                                .data
                                .get("_agent")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if record_agent != af.as_str() {
                                return false;
                            }
                        }
                        if let Some(min_imp) = min_importance {
                            let imp = axil_core::importance::get_importance(&rr.record.data);
                            if imp < min_imp {
                                return false;
                            }
                        }
                        if rr
                            .record
                            .data
                            .get("_archived")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            return false;
                        }
                        true
                    });
                };

                // Rung 1: drop QTC + min_confidence + scope_filter on the
                // recall config and re-run. Many `empty_recall` events come
                // from a too-strict QTC cutoff or a default min_confidence;
                // user-set table/time/importance filters still apply via
                // apply_filters so explicit intent is preserved.
                if !deadline_exceeded() {
                    let mut cfg_relaxed = cfg_for_cascade.clone();
                    cfg_relaxed.qtc = None;
                    cfg_relaxed.min_confidence = None;
                    cfg_relaxed.scope_filter.clear();
                    if let Ok(mut r1) = db.recall(&query, top_k, Some(cfg_relaxed)) {
                        apply_filters(&mut r1);
                        if !r1.is_empty() {
                            recall_results = r1;
                            cascade_rung = Some("filters_relaxed");
                        }
                    }
                }

                // Rung 2: query expansion (alias synonyms + 1-hop graph
                // neighbors). Skipped when the user already passed --expand
                // since the primary recall already saw the expanded query.
                if cascade_rung.is_none() && !expand && !deadline_exceeded() {
                    let expanded = expand_query(&db, &query, expand_neighbors);
                    if expanded != query {
                        if let Ok(mut r2) =
                            db.recall(&expanded, top_k, Some(cfg_for_cascade.clone()))
                        {
                            apply_filters(&mut r2);
                            if !r2.is_empty() {
                                recall_results = r2;
                                cascade_rung = Some("expanded");
                            }
                        }
                    }
                }

                // Rung 3: FTS fallback. Catches exact-text queries
                // (file/symbol names, error strings, config keys) the vector
                // index missed because it semantically clustered them away.
                if cascade_rung.is_none() && !deadline_exceeded() {
                    if let Ok(fts_hits) = db.search_text(&query, top_k.saturating_mul(2)) {
                        let mut r3: Vec<axil_core::scoring::RecallResult> = fts_hits
                            .into_iter()
                            .map(|(rec, score)| axil_core::scoring::RecallResult {
                                record: rec,
                                score,
                                explanation: axil_core::scoring::ScoreExplanation::new(
                                    vec![("fts".to_string(), score)],
                                    "FTS fallback (vector recall returned empty)".to_string(),
                                ),
                            })
                            .collect();
                        apply_filters(&mut r3);
                        if !r3.is_empty() {
                            r3.truncate(top_k);
                            recall_results = r3;
                            cascade_rung = Some("fts");
                        }
                    }
                }

                if let Some(rung) = cascade_rung {
                    eprintln!(
                        "[recall] cascade rung={} (primary recall returned empty)",
                        rung
                    );
                }
            }

            // --profile: surface the identifier-aware query classification and
            // whether the FTS rank tilt was applied. Printed to stderr so stdout
            // stays valid (JSON / context-block). The tilt only fires on
            // identifier queries; natural-language recall reports the class with
            // no tilt note. Classifying here (rather than reading it off a
            // result) means the profile prints even on an empty result set.
            if profile {
                let qc = axil_core::classify_query(&query);
                if qc.is_identifier() {
                    let applied = recall_results.iter().any(|rr| {
                        rr.explanation
                            .signals
                            .iter()
                            .any(|(name, _)| name == "fts_identifier_tilt")
                    });
                    let note = if applied {
                        "FTS tilt applied"
                    } else {
                        "FTS tilt eligible (no FTS hit to tilt)"
                    };
                    eprintln!("[recall] query_class={} ({note})", qc.tag());
                } else {
                    eprintln!("[recall] query_class={} (pure RRF, no tilt)", qc.tag());
                }
            }

            let values: Vec<Value> = recall_results
                .iter()
                .map(|rr| {
                    let mut v = scored_to_json(&rr.record, rr.score);
                    // Include explanation when --explain is set
                    if explain {
                        let signals: Value = rr
                            .explanation
                            .signals
                            .iter()
                            .map(|(name, val)| json!({name: val}))
                            .collect();
                        v["explanation"] = json!({
                            "signals": signals,
                            "summary": rr.explanation.summary,
                            "query_class": rr.explanation.query_class,
                        });
                    }
                    // Include feedback flag when --feedback is set
                    if _feedback_flag {
                        v["has_prior_feedback"] =
                            json!(db.feedback_store().has_feedback(&rr.record.id));
                    }
                    if !stale_paths.is_empty() {
                        let path = rr
                            .record
                            .data
                            .get("path")
                            .and_then(|p| p.as_str())
                            .unwrap_or("");
                        let is_fresh = !stale_paths.contains(path);
                        v["fresh"] = json!(is_fresh);
                        if !is_fresh {
                            v["stale_reason"] = json!("file modified since index");
                        }
                    }
                    v
                })
                .collect();

            // Skip rerank when the caller's deadline has already passed — reranking is
            // quality sugar, never worth blowing the budget the hook promised.
            let values = if deadline_exceeded() {
                values
            } else {
                match rerank {
                    RerankMode::Off => values,
                    RerankMode::CrossEncoder => {
                        #[cfg(feature = "rerank")]
                        {
                            let mut vals = values;
                            let rerank_config = axil_indexer::rerank::RerankConfig {
                                enabled: true,
                                ..Default::default()
                            };
                            if let Err(e) =
                                axil_indexer::rerank::rerank(&query, &mut vals, &rerank_config)
                            {
                                eprintln!(
                                    "[rerank] cross-encoder failed: {e} — returning RRF order"
                                );
                            }
                            vals
                        }
                        #[cfg(not(feature = "rerank"))]
                        {
                            eprintln!("[rerank] cross-encoder not compiled in (build with --features rerank)");
                            values
                        }
                    }
                    RerankMode::Llm => {
                        let mut vals = values;
                        if !db.has_llm() {
                            eprintln!(
                                "[rerank] no LLM configured — returning RRF order. \
                                   Run `axil llm config` to configure."
                            );
                        } else if let Err(e) = rerank_via_llm(&db, &query, &mut vals) {
                            eprintln!("[rerank] LLM rerank failed: {e} — returning RRF order");
                        }
                        vals
                    }
                }
            };

            let recall_ms = recall_start.elapsed().as_secs_f64() * 1000.0;
            db.record_slow_query(&format!("recall {query}"), recall_ms, values.len());

            // Apply format and budget
            match recall_format {
                RecallFormat::ContextBlock => {
                    // Plain-text block — not JSON — so bypass out.print_array entirely.
                    // The block is a self-contained `<context>…</context>` element
                    // that the UserPromptSubmit hook injects verbatim every turn, so
                    // nothing may be printed after it (a trailing line would land
                    // outside the wrapper). Each line already carries `(id=…)`, so the
                    // expand path is discoverable without an extra footer here.
                    let block = format_context_block(&values, budget);
                    if !block.is_empty() {
                        print!("{block}");
                    }
                }
                RecallFormat::Oneline => {
                    let formatted = format_recall_results(&values, &recall_format, budget);
                    for line in &formatted {
                        if let Some(s) = line.as_str() {
                            println!("{s}");
                        } else {
                            println!("{}", serde_json::to_string(line).unwrap_or_default());
                        }
                    }
                }
                _ => {
                    let formatted = format_recall_results(&values, &recall_format, budget);
                    out.print_array(&formatted);
                    // Compact output omits full record bodies. Surface the
                    // expand path on stderr so stdout stays valid JSON.
                    if matches!(recall_format, RecallFormat::Compact) && !formatted.is_empty() {
                        eprintln!("[axil] compact view — expand any hit with: axil get <id>");
                    }
                }
            }

            // 12.1: emit deadline-exceeded marker on stderr so hooks can log silently if desired.
            if deadline.is_some() && deadline_exceeded() {
                eprintln!(
                    "[axil recall] timeout_ms={} exceeded; returned partial results",
                    timeout_ms.unwrap_or(0)
                );
            }

            // Auto-refresh: if stale results were returned, trigger incremental re-index.
            #[cfg(feature = "indexer")]
            if fresh_only && !stale_paths.is_empty() {
                let hit_stale = recall_results.iter().any(|rr| {
                    rr.record
                        .data
                        .get("path")
                        .and_then(|p| p.as_str())
                        .map(|p| stale_paths.contains(p))
                        .unwrap_or(false)
                });
                if hit_stale {
                    if let (Some(root), Some(cfg)) = (&resolved_root, &resolved_config) {
                        if cfg.runtime.auto_refresh {
                            out.status("Stale results detected — refreshing index...");
                            let indexer = axil_indexer::ProjectIndexer::new(&db, cfg.index.clone())
                                .with_progress(make_index_progress(out.quiet));
                            let _ = indexer.index_incremental(root);
                        }
                    }
                }
            }

            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::CodeSearch {
            query,
            top_k,
            json,
            trace_graph,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_read_command(&db_path)?;
            // Fetch a larger pool so non-proxy hits (project/file index)
            // don't crowd proxies out before we filter.
            let pool = top_k.saturating_mul(5).max(15);
            let mut proxies: Vec<axil_indexer::recall::RecallResult> = if trace_graph {
                let rwr = axil_indexer::recall::recall_with_related(&db, &query, pool, top_k)
                    .context("code-search failed")?;
                let mut all: Vec<axil_indexer::recall::RecallResult> = rwr
                    .primary
                    .into_iter()
                    .filter(|r| r.source == "proxy")
                    .collect();
                all.extend(rwr.graph_neighbors);
                all
            } else {
                axil_indexer::recall::recall(&db, &query, pool)
                    .context("code-search failed")?
                    .into_iter()
                    .filter(|r| r.source == "proxy")
                    .collect()
            };
            // Cap final list at top_k to keep output compact.
            if proxies.len() > top_k {
                proxies.truncate(top_k);
            }
            if json {
                let v = serde_json::to_value(&proxies).unwrap_or(json!([]));
                out.print(&v);
            } else if proxies.is_empty() {
                println!(
                    "(no code proxies matched — run `axil index .` if you have not indexed yet)"
                );
            } else {
                for r in &proxies {
                    let pointer = match (r.path.as_deref(), r.line_start, r.symbol.as_deref()) {
                        (Some(p), Some(l), Some(s)) => format!("{p}:{l} {s}"),
                        (Some(p), Some(l), None) => format!("{p}:{l}"),
                        (Some(p), None, Some(s)) => format!("{p} {s}"),
                        (Some(p), None, None) => p.to_string(),
                        _ => r.id.clone(),
                    };
                    let why = r.why.as_deref().unwrap_or("matched code proxy");
                    println!("{pointer} — {why}");
                }
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::CodeContext {
            task,
            budget,
            context_format,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            // Honor an explicit --budget; otherwise auto-size by indexed repo size.
            let budget = budget.unwrap_or_else(|| axil_indexer::recall::auto_context_budget(&db));
            let opts = axil_indexer::recall::ContextOptions {
                max_tokens: budget,
                task: Some(task.clone()),
                ..Default::default()
            };
            let value = axil_indexer::recall::context(&db, &opts).context("code-context failed")?;
            match context_format.as_str() {
                "json" => out.print(&value),
                // `compact` (default): lean pointer lines, ~10× smaller than
                // the JSON bundle — the right shape for locating code.
                _ => print!("{}", axil_indexer::recall::render_context_compact(&value)),
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::ExplainCodeHit { id, query } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            // Try direct record id, then proxy_id lookup.
            let parsed_id = axil_core::RecordId::from_string(&id).ok();
            let record = match parsed_id
                .as_ref()
                .and_then(|rid| db.get(rid).ok().flatten())
            {
                Some(r) => r,
                None => {
                    let all = db
                        .list(axil_indexer::TABLE_CODE_PROXIES)
                        .context("list proxies")?;
                    match all.into_iter().find(|r| {
                        r.data.get("proxy_id").and_then(|v| v.as_str()) == Some(id.as_str())
                    }) {
                        Some(r) => r,
                        None => {
                            anyhow::bail!("no proxy with record id or proxy_id '{id}'");
                        }
                    }
                }
            };
            let mut explanation = json!({
                "proxy": record.data,
                "matches": [],
            });
            if let Some(q) = query {
                let mut matches: Vec<&'static str> = Vec::new();
                if let Ok(vec_hits) = db.similar_to(&q, 50) {
                    if vec_hits.iter().any(|(r, _)| r.id == record.id) {
                        matches.push(axil_indexer::recall::WHY_VECTOR);
                    }
                }
                if let Ok(fts_hits) = db.search_text(&q, 50) {
                    if fts_hits.iter().any(|(r, _)| r.id == record.id) {
                        matches.push(axil_indexer::recall::WHY_FTS);
                    }
                }
                let q_lower = q.to_lowercase();
                let path = record
                    .data
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let symbol = record
                    .data
                    .get("symbol")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let breadcrumb = record
                    .data
                    .get("breadcrumb")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if path.to_lowercase().contains(&q_lower)
                    || symbol.to_lowercase().contains(&q_lower)
                    || breadcrumb.to_lowercase().contains(&q_lower)
                {
                    matches.push(axil_indexer::recall::WHY_PATH_BOOST);
                }
                explanation["matches"] = json!(matches);
            }
            out.print(&explanation);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::CodeRecallBench {
            cases,
            top_k,
            bench_format,
            save,
            regression_gate,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let cases_vec: Vec<axil_indexer::code_recall_eval::EvalCase> = match cases {
                Some(p) => {
                    let raw = std::fs::read_to_string(&p)
                        .with_context(|| format!("read cases file {p}"))?;
                    serde_json::from_str(&raw).context("parse cases JSON")?
                }
                None => axil_indexer::code_recall_eval::axil_dogfood_cases(),
            };
            let report = axil_indexer::code_recall_eval::run_bench(&db, &cases_vec, top_k);

            if let Some(path) = save.as_deref() {
                let json = axil_indexer::code_recall_eval::report_to_json(&report);
                let pretty =
                    serde_json::to_string_pretty(&json).context("serialize bench report")?;
                std::fs::write(path, pretty)
                    .with_context(|| format!("write bench report to {path}"))?;
            }

            match bench_format.as_str() {
                "markdown" | "md" => {
                    print!(
                        "{}",
                        axil_indexer::code_recall_eval::render_markdown_table(&report)
                    );
                }
                "json" => {
                    out.print(&axil_indexer::code_recall_eval::report_to_json(&report));
                }
                _ => {
                    print!(
                        "{}",
                        axil_indexer::code_recall_eval::render_plain_table(&report)
                    );
                }
            }

            if let Some(baseline_path) = regression_gate.as_deref() {
                let raw = std::fs::read_to_string(baseline_path)
                    .with_context(|| format!("read baseline {baseline_path}"))?;
                let baseline: axil_indexer::code_recall_eval::BenchReport =
                    serde_json::from_str(&raw).context("parse baseline JSON")?;
                let regressions =
                    axil_indexer::code_recall_eval::compare_for_gate(&baseline, &report);
                if !regressions.is_empty() {
                    eprintln!("regression gate FAILED:");
                    for msg in &regressions {
                        eprintln!("  - {msg}");
                    }
                    return Ok(EXIT_BENCH_REGRESSION);
                }
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::ContextSavings {
            task,
            tasks,
            top_k,
            savings_format,
            save,
        } => {
            use axil_indexer::context_savings as cs;
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let task_specs: Vec<cs::TaskSpec> = if let Some(path) = tasks.as_deref() {
                let raw = std::fs::read_to_string(path)
                    .with_context(|| format!("read tasks file {path}"))?;
                serde_json::from_str(&raw).context("parse tasks JSON")?
            } else if !task.is_empty() {
                task.into_iter().map(cs::TaskSpec::new).collect()
            } else {
                cs::default_tasks()
            };

            let report = cs::measure(&db, &task_specs, top_k);

            if let Some(path) = save.as_deref() {
                let pretty = serde_json::to_string_pretty(&cs::report_to_json(&report))
                    .context("serialize savings report")?;
                std::fs::write(path, pretty)
                    .with_context(|| format!("write savings report to {path}"))?;
            }

            match savings_format.as_str() {
                "markdown" | "md" => print!("{}", cs::render_markdown(&report)),
                "json" => out.print(&cs::report_to_json(&report)),
                _ => print!("{}", cs::render_plain(&report)),
            }
            Ok(EXIT_OK)
        }

        // ── Search (semantic) ───────────────────────────────────────
        #[cfg(feature = "embed")]
        Command::Search { query, top_k } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_embedder(&db_path)?;

            let start = std::time::Instant::now();
            let results = db
                .similar_to(&query, top_k)
                .context("similarity search failed")?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;

            let values: Vec<Value> = results
                .iter()
                .map(|(r, score)| scored_to_json(r, *score))
                .collect();

            db.record_slow_query(&format!("search {query}"), ms, values.len());
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── FTS ─────────────────────────────────────────────────────
        #[cfg(feature = "fts")]
        Command::Fts { query, limit } => {
            let db_path = require_db(&db_opt)?;
            let db = open_read_command_fts(&db_path)?;

            let start = std::time::Instant::now();
            let results = db
                .search_text(&query, limit)
                .context("full-text search failed")?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;

            let values: Vec<Value> = results
                .iter()
                .map(|(r, score)| scored_to_json(r, *score))
                .collect();

            db.record_slow_query(&format!("fts {query}"), ms, values.len());
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "fts")]
        Command::FtsOptimize => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_fts(&db_path)?;
            db.fts_optimize().context("FTS optimize failed")?;
            out.print(&json!({"status": "ok", "action": "fts_optimize"}));
            Ok(EXIT_OK)
        }

        // ── Link ────────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Link {
            from_id,
            edge_type,
            to_id,
            props,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let from = RecordId::from_string(&from_id).context("invalid source record ID")?;
            let to = RecordId::from_string(&to_id).context("invalid target record ID")?;
            let properties: Option<Value> = props
                .map(|p| serde_json::from_str(&p).context("invalid JSON properties"))
                .transpose()?;

            let edge_id = db.relate(&from, &edge_type, &to, properties)?;

            out.print(&json!({
                "edge_id": edge_id.to_string(),
                "from": from_id,
                "to": to_id,
                "edge_type": edge_type,
            }));
            Ok(EXIT_OK)
        }

        // ── Unlink ──────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Unlink { edge_id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let eid = RecordId::from_string(&edge_id).context("invalid edge ID")?;

            if db.unrelate(&eid).context("unlink failed")? {
                out.print(&json!({"deleted": true, "edge_id": edge_id}));
                Ok(EXIT_OK)
            } else {
                eprintln!(
                    "{{\"error\":\"edge not found\",\"edge_id\":{}}}",
                    json!(edge_id)
                );
                Ok(EXIT_NOT_FOUND)
            }
        }

        // ── Neighbors ───────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Neighbors {
            id,
            edge_type,
            direction,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;

            let results = db.neighbors(&rid, edge_type.as_deref(), direction.into())?;

            let values: Vec<Value> = results.iter().map(record_to_json).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Traverse ────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Traverse { id, path_expr } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let start = RecordId::from_string(&id).context("invalid start record ID")?;

            let results = db.traverse(&start, &path_expr)?;

            let values: Vec<Value> = results.iter().map(record_to_json).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Session ─────────────────────────────────────────────────
        Command::Session { command: sess_cmd } => {
            let db_path = require_db(&db_opt)?;
            run_session(sess_cmd, &db_path, out)
        }

        // ── Query ───────────────────────────────────────────────────
        Command::Query {
            table,
            where_clauses,
            order_by,
            direction,
            limit,
            offset,
            profile,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mut qb = db.query().table(&table);

            for clause in &where_clauses {
                let (field, op, value) = parse_where_clause(clause)?;
                qb = qb.where_field(&field, op, value);
            }

            if let Some(ref field) = order_by {
                qb = qb.order_by(field, direction.into());
            }

            if let Some(n) = limit {
                qb = qb.limit(n);
            }

            if let Some(n) = offset {
                qb = qb.offset(n);
            }

            if profile {
                let (results, prof) = qb.exec_profiled().context("query failed")?;
                db.record_slow_query(
                    &format!("query {} --profile", table),
                    prof.total_ms,
                    results.len(),
                );
                let values: Vec<Value> = results.iter().map(record_to_json).collect();
                out.print(&json!({
                    "results": values,
                    "profile": serde_json::to_value(&prof).unwrap(),
                }));
            } else {
                let start = std::time::Instant::now();
                let results = qb.exec().context("query failed")?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                db.record_slow_query(&format!("query {table}"), ms, results.len());
                let values: Vec<Value> = results.iter().map(record_to_json).collect();
                out.print_array(&values);
            }
            Ok(EXIT_OK)
        }

        // ── AxilQL ──────────────────────────────────────────────────
        #[cfg(feature = "ql")]
        Command::Ql {
            query_str,
            interactive,
            explain,
            profile,
        } => {
            let db_path = require_db(&db_opt)?;
            let builder = attach_detected_engines(Axil::open(&db_path))?;
            let db = builder.build().context("failed to open database")?;

            // Interactive REPL mode
            if interactive || query_str.is_none() {
                eprintln!("AxilQL interactive mode — type queries, Ctrl+D to exit");
                eprintln!(
                    "  RECALL \"text\" TOP k | FIND \"text\" | COUNT | GET <id> | EXPLAIN <query>"
                );

                let history_path = db_path
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .join(".axil_ql_history");
                let mut rl =
                    rustyline::DefaultEditor::new().context("failed to initialize line editor")?;
                let _ = rl.load_history(&history_path);

                loop {
                    match rl.readline("axil> ") {
                        Ok(line) => {
                            let line = line.trim();
                            if line.is_empty() {
                                continue;
                            }
                            let _ = rl.add_history_entry(line);

                            let effective = ql_apply_flags(line, explain, profile);
                            match axil_ql::run(&db, &effective) {
                                Ok(result) => {
                                    db.record_slow_query(
                                        &format!(
                                            "ql: {}",
                                            effective.chars().take(80).collect::<String>()
                                        ),
                                        result.elapsed_ms,
                                        result.count,
                                    );
                                    eprintln!("{}", serde_json::to_string_pretty(&result).unwrap());
                                }
                                Err(e) => {
                                    let err_resp = axil_ql::ErrorResponse::from(&e);
                                    eprintln!(
                                        "error: {}",
                                        serde_json::to_string_pretty(&err_resp).unwrap()
                                    );
                                }
                            }
                        }
                        Err(rustyline::error::ReadlineError::Interrupted) => continue,
                        Err(rustyline::error::ReadlineError::Eof) => break,
                        Err(e) => {
                            eprintln!("readline error: {e}");
                            break;
                        }
                    }
                }

                let _ = rl.save_history(&history_path);
                return Ok(EXIT_OK);
            }

            // Single query mode
            let query_str = query_str.unwrap();
            let input = if query_str == "-" {
                let mut buf = String::new();
                io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)
                    .context("failed to read from stdin")?;
                buf.trim().to_string()
            } else {
                query_str
            };

            let effective = ql_apply_flags(&input, explain, profile);
            match axil_ql::run(&db, &effective) {
                Ok(result) => {
                    db.record_slow_query(
                        &format!("ql: {}", effective.chars().take(80).collect::<String>()),
                        result.elapsed_ms,
                        result.count,
                    );
                    out.print(&serde_json::to_value(&result).unwrap());
                    Ok(EXIT_OK)
                }
                Err(e) => {
                    let err_resp = axil_ql::ErrorResponse::from(&e);
                    out.print(&serde_json::to_value(&err_resp).unwrap());
                    Ok(EXIT_ERROR)
                }
            }
        }

        // ── Since ───────────────────────────────────────────────────
        #[cfg(feature = "timeseries")]
        Command::Since { duration, table } => {
            let db_path = require_db(&db_opt)?;
            let secs = parse_duration(&duration)?;
            let db = open_with_timeseries(&db_path)?;

            let records = db.since(table.as_deref(), secs)?;

            let values: Vec<Value> = records.iter().map(record_to_json).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Timeline ────────────────────────────────────────────────
        #[cfg(feature = "timeseries")]
        Command::Timeline { table, limit } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_timeseries(&db_path)?;

            let records = db.timeline(table.as_deref(), limit)?;

            let values: Vec<Value> = records.iter().map(record_to_json).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Diff ────────────────────────────────────────────────────
        #[cfg(feature = "timeseries")]
        Command::Diff { since, table } => {
            let db_path = require_db(&db_opt)?;
            let secs = parse_duration(&since)?;
            let db = open_with_timeseries(&db_path)?;

            let created = db.since(table.as_deref(), secs)?;
            let created_ids: std::collections::HashSet<_> =
                created.iter().map(|r| r.id.clone()).collect();

            let updated = db.changed_since(table.as_deref(), secs)?;

            let modified: Vec<_> = updated
                .into_iter()
                .filter(|r| !created_ids.contains(&r.id))
                .collect();

            out.print(&json!({
                "since": since,
                "created": created.len(),
                "modified": modified.len(),
                "created_records": created.iter().map(record_to_json).collect::<Vec<_>>(),
                "modified_records": modified.iter().map(record_to_json).collect::<Vec<_>>(),
            }));
            Ok(EXIT_OK)
        }

        // ── Activity ────────────────────────────────────────────────
        #[cfg(feature = "timeseries")]
        Command::Activity { days, table } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_timeseries(&db_path)?;

            if !db.has_timeseries_index() {
                anyhow::bail!("timeseries index not available");
            }

            let now_us = chrono::Utc::now().timestamp_micros();
            let start_us = now_us.saturating_sub(
                i64::try_from(days)
                    .ok()
                    .and_then(|d| d.checked_mul(86_400_000_000))
                    .ok_or_else(|| anyhow::anyhow!("duration overflow: {days} days"))?,
            );

            let buckets = db.count_by_bucket(
                table.as_deref(),
                axil_core::TimeBucket::Day,
                start_us,
                now_us,
            )?;

            let counts: std::collections::BTreeMap<String, usize> = buckets
                .into_iter()
                .map(|(us, count)| {
                    let date = chrono::DateTime::from_timestamp_micros(us)
                        .unwrap_or_default()
                        .format("%Y-%m-%d")
                        .to_string();
                    (date, count)
                })
                .collect();
            out.print(&json!(counts));
            Ok(EXIT_OK)
        }

        // ── Heal ────────────────────────────────────────────────────
        Command::Heal {
            compact,
            reindex,
            orphans,
            dry_run,
        } => {
            let db_path = require_db(&db_opt)?;
            let config = load_config(&db_path)?;
            let db = open_with_all_detected(&db_path)?;

            // If specific flags are given, run only those actions
            if compact || reindex || orphans {
                let mut actions = Vec::new();

                if dry_run {
                    // Dry-run: report what would happen without modifying anything
                    let problems = db.detect_problems();
                    for p in &problems {
                        if p.auto_fixable {
                            let dominated = match p.detector.as_str() {
                                "expired_records" | "superseded_records" | "storage_bloat" => {
                                    compact
                                }
                                "vector_deletion_ratio" | "index_size_mismatch"
                                | "missing_embeddings" | "missing_fts" => reindex,
                                "orphaned_edges" => orphans,
                                _ => compact || orphans,
                            };
                            if dominated {
                                actions.push(json!({
                                    "action": p.detector,
                                    "result": format!("[dry-run] would fix: {}", p.message),
                                }));
                            }
                        }
                    }
                } else {
                    if compact {
                        let compact_report = db.compact().context("compact failed")?;
                        if compact_report.compacted {
                            actions.push(json!({
                                "action": "compact",
                                "result": format!(
                                    "purged {} expired, {} superseded, {} orphaned edges, {} orphaned vectors",
                                    compact_report.purged_expired,
                                    compact_report.purged_superseded,
                                    compact_report.cleaned_orphaned_edges,
                                    compact_report.cleaned_orphaned_vectors,
                                ),
                            }));
                        }
                    }

                    if orphans && !compact {
                        // --orphans without --compact: only clean orphaned entries,
                        // do not purge expired/superseded records
                        let oe = db.clean_orphaned_edges();
                        let ov = db.clean_orphaned_vectors();
                        if oe > 0 || ov > 0 {
                            actions.push(json!({
                                "action": "clean_orphans",
                                "result": format!(
                                    "removed {} orphaned edges, {} orphaned vectors", oe, ov
                                ),
                            }));
                        }
                    }

                    if reindex && db.has_vector_index() {
                        let rebuild_report =
                            db.vector_rebuild().context("vector rebuild failed")?;
                        actions.push(json!({
                            "action": "vector_rebuild",
                            "result": format!(
                                "{} -> {}, removed {} tombstones",
                                rebuild_report.old_size, rebuild_report.new_size,
                                rebuild_report.deleted_removed,
                            ),
                        }));
                    }

                    if reindex {
                        // Re-embed / re-index records lost by a torn insert
                        // (committed to storage but missing their vector
                        // embedding and/or FTS document). vector_rebuild above
                        // only compacts tombstones — it cannot regenerate a
                        // missing embedding.
                        let (reembedded, refts) =
                            db.reembed_missing().context("reembed missing failed")?;
                        if reembedded > 0 || refts > 0 {
                            actions.push(json!({
                                "action": "reembed_missing",
                                "result": format!(
                                    "restored {} missing embeddings, {} missing FTS docs",
                                    reembedded, refts
                                ),
                            }));
                        }
                    }
                }

                out.print(&json!({
                    "healed": !dry_run && !actions.is_empty(),
                    "actions": actions,
                }));
            } else {
                // Full heal
                let report = db
                    .heal_all(&config.healing, dry_run)
                    .context("heal failed")?;
                out.print(&serde_json::to_value(&report).unwrap());
            }

            if !dry_run {
                run_worker_and_report(&db, out);
            }

            // Also run timeseries heal if available
            #[cfg(feature = "timeseries")]
            if !dry_run
                && !compact
                && !reindex
                && !orphans
                && db.has_timeseries_index()
                && config.timeseries.auto_downsample
            {
                let _ = db.heal(&config.timeseries);
            }

            Ok(EXIT_OK)
        }

        // ── Session-Heal ────────────────────────────────────────────
        Command::SessionHeal {
            problems_file,
            session,
            dry_run,
            quiet,
        } => {
            let db_path = require_db(&db_opt)?;
            let config = load_config(&db_path)?;
            let db = open_with_all_detected(&db_path)?;

            let problems = problems_file
                .as_ref()
                .map(|p| load_session_problems(p))
                .transpose()
                .context("failed to read problems file")?
                .unwrap_or_default();
            let problem_summary = classify_session_problems(&problems);
            let detected = db.detect_problems();
            let auto_fixable = detected.iter().filter(|p| p.auto_fixable).count();

            let (healed, heal_actions): (bool, Vec<Value>) = if !dry_run && auto_fixable > 0 {
                let report = db
                    .heal_all(&config.healing, false)
                    .context("session-heal: heal_all failed")?;
                let actions = report
                    .actions
                    .iter()
                    .map(|a| json!({"action": a.action, "result": a.result}))
                    .collect();
                (report.healed, actions)
            } else {
                (false, Vec::new())
            };

            let hints: Vec<Value> = [
                build_hint("stale_structural_index",
                    problem_summary.empty_code_search + problem_summary.empty_fts,
                    "code-search/fts query/queries returned no results this session",
                    "axil index .  # refresh structural code index"),
                build_hint("low_memory_coverage", problem_summary.empty_recall,
                    "recall query/queries returned no results — memory may be sparse for these topics",
                    "store more decisions/context as work progresses"),
                build_hint("axil_command_failures", problem_summary.command_failures,
                    "axil invocation(s) exited non-zero this session",
                    "review errors logged in _heal_log; run 'axil doctor'"),
            ].into_iter().flatten().collect();

            let nothing_to_log = problems.is_empty() && detected.is_empty();
            let log_id = if !dry_run && !nothing_to_log {
                let payload = json!({
                    "type": "session_heal",
                    "session": session.as_deref().unwrap_or(""),
                    "dry_run": dry_run,
                    "session_problems": {
                        "total": problems.len(),
                        "command_failures": problem_summary.command_failures,
                        "empty_recall": problem_summary.empty_recall,
                        "empty_code_search": problem_summary.empty_code_search,
                        "empty_fts": problem_summary.empty_fts,
                        "samples": problem_summary.samples,
                    },
                    "db_problems": &detected,
                    "auto_fixable_count": auto_fixable,
                    "healed": healed,
                    "actions": heal_actions,
                    "hints": hints,
                });
                let rec = db.insert("_heal_log", payload).context("write _heal_log")?;
                Some(rec.id.to_string())
            } else {
                None
            };

            if !quiet {
                eprintln!(
                    "🧠 Axil session-heal: {} session-problem(s), {} db problem(s), {} fixed{}",
                    problems.len(),
                    detected.len(),
                    heal_actions.len(),
                    if dry_run { " [dry-run]" } else { "" },
                );
            }

            out.print(&json!({
                "session": session,
                "session_problems": problems.len(),
                "db_problems": detected.len(),
                "auto_fixable": auto_fixable,
                "healed": healed,
                "actions": heal_actions,
                "hints": hints,
                "log_id": log_id,
                "dry_run": dry_run,
            }));
            Ok(EXIT_OK)
        }

        // ── Embed ───────────────────────────────────────────────────
        #[cfg(feature = "embed")]
        Command::Embed { id, field, model } => {
            let db_path = require_db(&db_opt)?;
            let embedding_model = parse_model(&model)?;
            let db = Axil::open(&db_path)
                .with_embedder_model(embedding_model)
                .context("failed to open vector store with embedder")?
                .build()
                .context("failed to open database")?;

            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            db.embed_field(&rid, &field).context("embed failed")?;

            out.print(&json!({
                "embedded": true,
                "id": id,
                "field": field,
            }));
            Ok(EXIT_OK)
        }

        // ── AddVector ───────────────────────────────────────────────
        #[cfg(feature = "vector")]
        Command::AddVector {
            id,
            vector,
            dimensions,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_vector(&db_path, dimensions)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let vec: Vec<f32> = serde_json::from_str(&vector).context("invalid vector JSON")?;
            db.add_vector(&rid, &vec).context("add_vector failed")?;

            out.print(&json!({
                "added": true,
                "id": id,
                "dimensions": vec.len(),
            }));
            Ok(EXIT_OK)
        }

        // ── SearchVector ────────────────────────────────────────────
        #[cfg(feature = "vector")]
        Command::SearchVector {
            vector,
            top_k,
            dimensions,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_vector(&db_path, dimensions)?;
            let vec: Vec<f32> = serde_json::from_str(&vector).context("invalid vector JSON")?;
            let results = db
                .similar_to_vector(&vec, top_k)
                .context("vector search failed")?;

            let values: Vec<Value> = results
                .iter()
                .map(|(r, score)| scored_to_json(r, *score))
                .collect();

            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Edges ───────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Edges { id, direction } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;

            let edges = db.edges(&rid, None, direction.into())?;

            let values: Vec<Value> = edges
                .iter()
                .map(|e| {
                    json!({
                        "edge_id": e.id.to_string(),
                        "from": e.from.to_string(),
                        "edge_type": e.edge_type,
                        "to": e.to.to_string(),
                        "properties": e.properties,
                        "created_at": e.created_at,
                    })
                })
                .collect();

            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── IndexText ───────────────────────────────────────────────
        #[cfg(feature = "fts")]
        Command::IndexText { id, field, text } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_fts(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;

            let content = match text {
                Some(t) => t,
                None => {
                    let record = db
                        .get(&rid)
                        .context("get failed")?
                        .ok_or_else(|| anyhow::anyhow!("record not found: {id}"))?;
                    record
                        .data
                        .get(&field)
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            anyhow::anyhow!("field '{field}' not found or not a string")
                        })?
                        .to_string()
                }
            };
            db.index_text(&rid, &field, &content)?;

            out.print(&json!({
                "indexed": true,
                "id": id,
                "field": field,
            }));
            Ok(EXIT_OK)
        }

        // ── Config ───────────────────────────────────────────────────
        Command::Config { command: cfg_cmd } => run_config(cfg_cmd, out),

        Command::Extensions(ext_cmd) => run_extensions(ext_cmd, out),

        #[cfg(feature = "wasm-host")]
        Command::Ext(ext_cmd) => run_ext(ext_cmd, &db_opt, out),

        Command::External(tokens) => run_extension_dispatch(tokens, &db_opt, out),

        // ── Report ──────────────────────────────────────────────────
        Command::Report { command: rep_cmd } => {
            let db_path = db_opt;
            run_report(rep_cmd, &db_path, out)
        }

        // ── Skill ───────────────────────────────────────────────────
        Command::Skill { command: sk_cmd } => run_skill(sk_cmd, out),
        Command::Hook {
            command: HookCommand::Run { dialect, event },
        } => hook_brain::run(&dialect, event.as_deref()),
        Command::Hook {
            command: HookCommand::Capture { dialect, event },
        } => hook_brain::capture(&dialect, event.as_deref()),

        // ── Model management ────────────────────────────────────────
        #[cfg(feature = "vector")]
        Command::ModelDownload { model } => {
            let embedding_model = parse_model(&model)?;
            out.status(&format!(
                "Downloading model: {} ({} dims, {})",
                embedding_model.name(),
                embedding_model.dimensions(),
                embedding_model.approx_size(),
            ));

            let dir = axil_vector::download::download_model(&embedding_model)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            out.print(&json!({
                "model": embedding_model.name(),
                "dimensions": embedding_model.dimensions(),
                "path": dir.display().to_string(),
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "vector")]
        Command::ModelList => {
            let models = axil_vector::download::list_models();
            let values: Vec<Value> = models
                .iter()
                .map(|(model, path, size)| {
                    json!({
                        "name": model.name(),
                        "dimensions": model.dimensions(),
                        "size_bytes": size,
                        "path": path.display().to_string(),
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "vector")]
        Command::ModelRemove { model } => {
            let embedding_model = parse_model(&model)?;
            axil_vector::download::remove_model(&embedding_model)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            out.print(&json!({
                "removed": true,
                "model": embedding_model.name(),
            }));
            Ok(EXIT_OK)
        }

        // ── Reembed ──────────────────────────────────────────────────
        #[cfg(feature = "embed")]
        Command::Reembed {
            model,
            field,
            table,
        } => {
            let db_path = require_db(&db_opt)?;

            let embedding_model = parse_model(&model)?;
            let new_dims = embedding_model.dimensions();

            // Step 1: Open DB without vector to collect record IDs + text.
            let db = Axil::open(&db_path)
                .build()
                .context("failed to open database")?;

            // Gather records that need re-embedding.
            let tables: Vec<String> = if let Some(ref t) = table {
                vec![t.clone()]
            } else {
                db.tables().context("failed to list tables")?
            };

            let mut work: Vec<(RecordId, String)> = Vec::new();
            for tbl in &tables {
                for record in db.list(tbl).unwrap_or_default() {
                    if let Some(text) = record.data.get(&field).and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            work.push((record.id.clone(), text.to_string()));
                        }
                    }
                }
            }

            if work.is_empty() {
                out.print(&json!({
                    "reembedded": 0,
                    "model": embedding_model.name(),
                    "field": field,
                    "message": "no records have a non-empty string in the specified field",
                }));
                return Ok(EXIT_OK);
            }

            out.status(&format!(
                "Re-embedding {} records with {} ({} dims)...",
                work.len(),
                embedding_model.name(),
                new_dims,
            ));

            // Step 2: Close DB, delete old .vec file, re-open with new model.
            drop(db);
            let vec_path = axil_vector::vector_db_path(&db_path);
            match std::fs::remove_file(&vec_path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(anyhow::anyhow!(e).context("failed to remove old vector store"))
                }
            }

            let db = Axil::open(&db_path)
                .with_embedder_model(embedding_model.clone())
                .context("failed to initialize embedder")?
                .build()
                .context("failed to open database with new vector store")?;

            // Step 3: Re-embed all collected records.
            let mut success = 0usize;
            let mut errors = 0usize;
            let total = work.len();
            for (i, (rid, text)) in work.iter().enumerate() {
                match db.embed_text(rid, text) {
                    Ok(()) => success += 1,
                    Err(e) => {
                        eprintln!("  [skip] {rid}: {e}");
                        errors += 1;
                    }
                }
                if (i + 1) % 100 == 0 {
                    eprintln!("  [{}/{total}] re-embedded...", i + 1);
                }
            }

            out.print(&json!({
                "reembedded": success,
                "errors": errors,
                "model": embedding_model.name(),
                "dimensions": new_dims,
                "field": field,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "embed")]
        Command::ModelAdd {
            name,
            path,
            dimensions,
            pooling,
            max_seq_len,
        } => {
            use axil_vector::models::{EmbeddingModel, PoolingStrategy};

            let pool = match pooling.as_str() {
                "cls" => PoolingStrategy::Cls,
                "mean" => PoolingStrategy::Mean,
                other => anyhow::bail!("unknown pooling strategy: {other}. Use 'cls' or 'mean'."),
            };

            let model_path = std::fs::canonicalize(&path).context("model path does not exist")?;

            let model = EmbeddingModel::Custom {
                path: model_path.clone(),
                dimensions,
                pooling: pool,
                max_seq_len,
            };

            // Validate by loading the model
            let _embedder =
                axil_vector::embed::Embedder::new(model).map_err(|e| anyhow::anyhow!("{e}"))?;

            // Persist to ~/.axil/custom_models.json so later commands can find it by name.
            axil_vector::models::save_custom_model(
                &name,
                &model_path,
                dimensions,
                pool,
                max_seq_len,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;

            out.print(&json!({
                "registered": true,
                "name": name,
                "path": model_path.display().to_string(),
                "dimensions": dimensions,
                "pooling": pooling,
                "max_seq_len": max_seq_len,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "embed")]
        Command::ModelBench { models, count } => {
            use axil_vector::models::EmbeddingModel;
            use std::time::Instant;

            let sample_texts = [
                "Fixed authentication timeout in the login flow for mobile clients",
                "The database migration failed silently when running in production",
                "Added retry logic with exponential backoff for the payment gateway",
                "Refactored the caching layer to use a two-tier approach with Redis and local LRU",
                "User reported that search results are inconsistent after the last deployment",
            ];

            let model_names: Vec<String> = if let Some(ref m) = models {
                m.split(',').map(|s| s.trim().to_string()).collect()
            } else {
                // Benchmark all available built-in models
                let mut available = Vec::new();
                for name in ["bge-small", "bge-small-int8", "bge-base", "nomic", "bge-m3"] {
                    if let Ok(model) = name.parse::<EmbeddingModel>() {
                        if axil_vector::download::is_model_available(&model) {
                            available.push(name.to_string());
                        }
                    }
                }
                if available.is_empty() {
                    anyhow::bail!("no models available. Run `axil model-download` first.");
                }
                available
            };

            let mut results = Vec::new();
            for model_name in &model_names {
                let model: EmbeddingModel = model_name
                    .parse()
                    .map_err(|e: String| anyhow::anyhow!("{e}"))?;

                if !axil_vector::download::is_model_available(&model) {
                    out.status(&format!("skipping {} (not downloaded)", model_name));
                    continue;
                }

                out.status(&format!("benchmarking {}...", model_name));
                let embedder = axil_vector::embed::Embedder::new(model.clone())
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                let start = Instant::now();
                for i in 0..count {
                    let text = sample_texts[i % sample_texts.len()];
                    // black_box prevents the optimizer from eliding the embed call.
                    std::hint::black_box(embedder.embed(text).map_err(|e| anyhow::anyhow!("{e}"))?);
                }
                let elapsed = start.elapsed();

                let avg_ms = elapsed.as_secs_f64() * 1000.0 / count as f64;
                let throughput = count as f64 / elapsed.as_secs_f64();

                results.push(json!({
                    "model": model_name,
                    "dimensions": model.dimensions(),
                    "count": count,
                    "total_ms": format!("{:.1}", elapsed.as_secs_f64() * 1000.0),
                    "avg_ms": format!("{:.2}", avg_ms),
                    "throughput_per_sec": format!("{:.1}", throughput),
                    "approx_size": model.approx_size(),
                }));
            }

            out.print_array(&results);
            Ok(EXIT_OK)
        }

        // ── Project indexer commands ──────────────────────────────────
        #[cfg(feature = "indexer")]
        Command::Index {
            path,
            full,
            dry_run,
        } => {
            let root = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());

            // Load config from axil.toml near the project root (no DB needed)
            let config = axil_core::load_config_from(&root)
                .map(|c| c.index)
                .unwrap_or_default();

            if dry_run {
                let project_type = axil_indexer::scanner::detect_project_type(&root);
                let files = axil_indexer::scanner::scan_files(&root, &config);
                out.print(&json!({
                    "dry_run": true,
                    "project_type": project_type.as_str(),
                    "files_to_index": files.len(),
                    "files": files.iter().map(|f| &f.rel_path).collect::<Vec<_>>(),
                }));
                return Ok(EXIT_OK);
            }

            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let indexer = axil_indexer::ProjectIndexer::new(&db, config)
                .with_progress(make_index_progress(out.quiet));

            let result = if full || !indexer.has_index() {
                indexer
                    .index_full(&root)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            } else {
                indexer
                    .index_incremental(&root)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            out.print(&serde_json::to_value(&result)?);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::Reindex {
            path,
            full,
            no_scip,
            wait,
        } => {
            let root = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());

            let config = axil_core::load_config_from(&root)
                .map(|c| c.index)
                .unwrap_or_default();

            // ── 1. Structural proxy index (foreground, incremental) ──
            let db_path = require_db(&db_opt)?;
            let proxies = {
                let db = open_with_all_detected(&db_path)?;
                let indexer = axil_indexer::ProjectIndexer::new(&db, config)
                    .with_progress(make_index_progress(out.quiet));
                let result = if full || !indexer.has_index() {
                    indexer
                        .index_full(&root)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    indexer
                        .index_incremental(&root)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                };
                // Drop the DB handle (end of scope) before spawning the SCIP
                // child so the worker can take the redb write lock cleanly.
                serde_json::to_value(&result)?
            };

            // ── 2. SCIP code-graph refresh (background by default) ──
            // Pin SCIP to the same `root` the proxy index used (via --root)
            // so the two layers never refresh different trees when the DB
            // lives outside the indexed project.
            let scip = if no_scip {
                json!({ "status": "skipped", "reason": "no_scip" })
            } else {
                reindex_scip(&db_path, &root, full, wait, out.quiet)
            };

            out.print(&json!({ "proxies": proxies, "scip": scip }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::RecallIndex { query, top_k } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let results = axil_indexer::recall::recall(&db, &query, top_k)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let values: Vec<Value> = results
                .iter()
                .map(|r| serde_json::to_value(r).unwrap())
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::Context {
            max_tokens,
            focus,
            diff,
            depth,
            task,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_best_effort(&db_path)?;

            let project_root = detect_project_root(&db_path);
            let index_config = project_root
                .as_ref()
                .and_then(|root| axil_core::load_config_from(root).ok())
                .map(|c| c.index);

            // Auto-reindex if stale and auto_index is enabled
            if let (Some(ref root), Some(ref cfg)) = (&project_root, &index_config) {
                if cfg.auto_index {
                    let report = axil_indexer::check_freshness(&db, root, cfg);
                    if report.status != axil_indexer::FreshnessStatus::Fresh {
                        out.status(&format!(
                            "Index is {} — auto-reindexing...",
                            report.status.as_str()
                        ));
                        let indexer = axil_indexer::ProjectIndexer::new(&db, cfg.clone())
                            .with_progress(make_index_progress(out.quiet));
                        let _ = indexer.index_incremental(root);
                    }
                }
            }

            let ctx_depth = axil_indexer::ContextDepth::parse(&depth);
            let effective_max = max_tokens.unwrap_or_else(|| ctx_depth.default_max_tokens());

            let opts = axil_indexer::ContextOptions {
                max_tokens: effective_max,
                focus: focus
                    .map(|f| f.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default(),
                diff,
                depth: ctx_depth,
                task,
                project_root: project_root.clone(),
                index_config: index_config.clone(),
            };
            let result =
                axil_indexer::recall::context(&db, &opts).map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&result);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::IndexStats => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let project_root = detect_project_root(&db_path);
            let index_config = project_root
                .as_ref()
                .and_then(|root| axil_core::load_config_from(root).ok())
                .map(|c| c.index);
            let result =
                axil_indexer::recall::stats(&db, project_root.as_deref(), index_config.as_ref())
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Agent runtime commands (4e) ──────────────────────────────
        #[cfg(feature = "indexer")]
        Command::Ask {
            query,
            top_k,
            parallel,
            strategy,
            plan,
            rerank,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_best_effort(&db_path)?;
            let db = std::sync::Arc::new(db);

            let mut result = if plan {
                let qp = axil_indexer::plan_query(&query);
                axil_indexer::ask::execute_plan(&db, &qp, &query, top_k)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            } else if parallel {
                let allowed =
                    strategy.map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
                // Use existing runtime or create one that stays alive through block_on.
                let owned_rt;
                let handle = match tokio::runtime::Handle::try_current() {
                    Ok(h) => h,
                    Err(_) => {
                        owned_rt = tokio::runtime::Runtime::new()
                            .context("failed to create tokio runtime")?;
                        owned_rt.handle().clone()
                    }
                };
                handle
                    .block_on(axil_indexer::ask::ask_parallel(
                        std::sync::Arc::clone(&db),
                        &query,
                        top_k,
                        allowed,
                    ))
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            } else {
                axil_indexer::ask::ask(&db, &query, top_k).map_err(|e| anyhow::anyhow!("{e}"))?
            };

            // Apply cross-encoder reranking if requested.
            if rerank {
                let rerank_config = axil_indexer::rerank::RerankConfig {
                    enabled: true,
                    ..Default::default()
                };
                if let Err(e) =
                    axil_indexer::rerank::rerank(&query, &mut result.results, &rerank_config)
                {
                    out.status(&format!("Reranking skipped: {e}"));
                }
            }

            let _ = axil_indexer::log_query(
                &db,
                &result.intent.to_string(),
                &query,
                result.results.len(),
                result.tokens,
            );
            out.print(&serde_json::to_value(&result)?);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::Rule { command: rule_cmd } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match rule_cmd {
                RuleCommand::Set { key, value } => {
                    axil_indexer::set_rule(&db, &key, &value, "user")
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    out.print(&json!({"ok": true, "key": key}));
                }
                RuleCommand::Get { key } => {
                    match axil_indexer::get_rule(&db, &key).map_err(|e| anyhow::anyhow!("{e}"))? {
                        Some(rule) => out.print(&serde_json::to_value(&rule)?),
                        None => {
                            out.print(&json!({"error": format!("rule '{}' not found", key)}));
                            return Ok(EXIT_NOT_FOUND);
                        }
                    }
                }
                RuleCommand::List => {
                    let rules =
                        axil_indexer::list_rules(&db).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let values: Vec<Value> = rules
                        .iter()
                        .map(|r| serde_json::to_value(r).unwrap())
                        .collect();
                    out.print_array(&values);
                }
                RuleCommand::Delete { key } => {
                    let deleted =
                        axil_indexer::delete_rule(&db, &key).map_err(|e| anyhow::anyhow!("{e}"))?;
                    out.print(&json!({"deleted": deleted, "key": key}));
                }
                RuleCommand::Extract { path } => {
                    let root = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    let extracted = axil_indexer::auto_extract_rules(&db, &root)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    out.print(&json!({"extracted": extracted.len(), "source": "detected"}));
                }
                RuleCommand::Distill {
                    file,
                    min_evidence,
                    max,
                    dry_run,
                } => {
                    let directives =
                        axil_indexer::distill::distill_directives(&db, min_evidence, max)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let directive_vals: Vec<Value> = directives
                        .iter()
                        .map(|d| {
                            json!({
                                "directive": d.directive,
                                "frequency": d.frequency,
                                "last_seen": d.last_seen.to_rfc3339(),
                                "impact": d.impact,
                            })
                        })
                        .collect();

                    if dry_run {
                        out.print(&json!({
                            "dry_run": true,
                            "directives": directive_vals,
                            "claude_md": file.display().to_string(),
                            "block_preview": axil_indexer::distill::render_block(&directives),
                        }));
                        return Ok(EXIT_OK);
                    }

                    let claude_md_changed =
                        axil_indexer::distill::write_claude_md(&file, &directives)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let pinned = axil_indexer::distill::persist_rules(&db, &directives)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    out.print(&json!({
                        "directives": directive_vals,
                        "claude_md": file.display().to_string(),
                        "claude_md_changed": claude_md_changed,
                        "rules_pinned": pinned,
                    }));
                }
            }
            Ok(EXIT_OK)
        }

        #[cfg(all(feature = "indexer", feature = "graph"))]
        Command::Impact { path, reverse } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let report = if reverse {
                axil_indexer::reverse_impact(&db, &path)
            } else {
                axil_indexer::impact(&db, &path)
            }
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let count = report.direct_dependents.len() + report.transitive_dependents.len();
            let _ = axil_indexer::log_query(&db, "graph", &path, count, 0);
            out.print(&serde_json::to_value(&report)?);
            Ok(EXIT_OK)
        }

        #[cfg(all(feature = "indexer", feature = "graph"))]
        Command::Why { path_a, path_b } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let chain = axil_indexer::why_connected(&db, &path_a, &path_b)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let query = format!("{} -> {}", path_a, path_b);
            let _ = axil_indexer::log_query(&db, "graph", &query, chain.len(), 0);
            if chain.is_empty() {
                out.print(&json!({"connected": false, "message": "no connection found"}));
            } else {
                out.print(&json!({"connected": true, "path": chain}));
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::Analytics { days } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let result =
                axil_indexer::get_analytics(&db, days).map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&result);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "indexer")]
        Command::Prefetch {
            intent,
            max_tokens,
            file,
        } => {
            let db_path = require_db(&db_opt)?;
            let cache_key = file.as_deref().unwrap_or(&intent);

            // Try cache first (30 min TTL).
            if let Some(cached) = axil_indexer::load_cached(&db_path, cache_key, 30) {
                out.print(&serde_json::to_value(&cached)?);
                return Ok(EXIT_OK);
            }

            let db = open_with_best_effort(&db_path)?;

            let result = if let Some(ref file_path) = file {
                axil_indexer::prefetch_file(&db, file_path, max_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            } else {
                axil_indexer::prefetch(&db, &intent, max_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            // Save to cache for subsequent calls.
            axil_indexer::save_cache(&db_path, cache_key, &result);

            let _ = axil_indexer::log_query(
                &db,
                "prefetch",
                cache_key,
                result.sections.len(),
                result.total_tokens,
            );
            out.print(&serde_json::to_value(&result)?);
            Ok(EXIT_OK)
        }

        // ── Agent memory commands ──────────────────────────────────────
        #[cfg(feature = "memory")]
        Command::Know {
            entity,
            fact,
            source,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let record = mem
                .semantic()
                .know(&entity, &fact, source.as_deref())
                .context("failed to store fact")?;

            out.print(&json!({
                "id": record.id.to_string(),
                "entity": entity,
                "fact": fact,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::KnowAbout { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let knowledge = mem
                .semantic()
                .about(&entity)
                .context("failed to query entity")?;
            out.print(&knowledge.to_json());
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::KnowList { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let facts = mem
                .semantic()
                .list_facts(entity.as_deref())
                .context("failed to list facts")?;

            let values: Vec<Value> = facts
                .iter()
                .map(|r| {
                    json!({
                        "id": r.id.to_string(),
                        "entity": r.data.get("entity").cloned().unwrap_or(json!(null)),
                        "fact": r.data.get("fact").cloned().unwrap_or(json!(null)),
                        "created_at": format_dt(&r.created_at),
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::Learn {
            pattern_name,
            description,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let record = mem
                .procedural()
                .learn(&pattern_name, &description, None)
                .context("failed to store procedure")?;

            let confidence = record
                .data
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            out.print(&json!({
                "id": record.id.to_string(),
                "pattern_name": pattern_name,
                "confidence": confidence,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::How { task, top_k } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let results = mem
                .procedural()
                .how(&task, top_k)
                .context("failed to search procedures")?;

            let values: Vec<Value> = results
                .iter()
                .map(|(r, score)| {
                    json!({
                        "id": r.id.to_string(),
                        "pattern_name": r.data.get("pattern_name").cloned().unwrap_or(json!(null)),
                        "description": r.data.get("description").cloned().unwrap_or(json!(null)),
                        "confidence": r.data.get("confidence").cloned().unwrap_or(json!(0.0)),
                        "score": score,
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::Episodes { outcome, limit } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let outcome_filter = outcome
                .as_deref()
                .map(|s| s.parse::<axil_memory::Outcome>())
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let episodes = mem
                .episodic()
                .list(outcome_filter, limit)
                .context("failed to list episodes")?;

            let values: Vec<Value> = episodes.iter().map(|r| {
                json!({
                    "id": r.id.to_string(),
                    "summary": r.data.get("summary").cloned().unwrap_or(json!(null)),
                    "outcome": r.data.get("outcome").cloned().unwrap_or(json!(null)),
                    "created_at": format_dt(&r.created_at),
                    "duration_secs": r.data.get("duration_secs").cloned().unwrap_or(json!(null)),
                })
            }).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::EpisodesSimilar { query, top_k } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let results = mem
                .episodic()
                .similar(&query, top_k)
                .context("failed to search episodes")?;

            let values: Vec<Value> = results
                .iter()
                .map(|(r, score)| {
                    json!({
                        "id": r.id.to_string(),
                        "summary": r.data.get("summary").cloned().unwrap_or(json!(null)),
                        "outcome": r.data.get("outcome").cloned().unwrap_or(json!(null)),
                        "score": score,
                        "created_at": format_dt(&r.created_at),
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::Remember {
            query,
            top_k,
            max_tokens,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let opts = axil_memory::RecallOptions {
                top_k,
                max_tokens,
                ..Default::default()
            };

            let results = mem
                .remember(&query, opts)
                .context("failed to search memories")?;

            let total_tokens: usize = results.iter().map(|r| r.tokens).sum();
            let values: Vec<Value> = results
                .iter()
                .map(|r| {
                    json!({
                        "type": r.memory_type.to_string(),
                        "id": r.scored.record.id,
                        "score": r.scored.final_score,
                        "similarity": r.scored.similarity,
                        "recency": r.scored.recency,
                        "data": r.scored.record.data,
                        "tokens": r.tokens,
                    })
                })
                .collect();

            out.print(&json!({
                "results": values,
                "tokens": total_tokens,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::History { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let records = mem
                .semantic()
                .history(&entity)
                .context("failed to get history")?;

            let values: Vec<Value> = records
                .iter()
                .map(|r| {
                    let superseded = r
                        .data
                        .get("_meta")
                        .and_then(|m| m.get("superseded"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    json!({
                        "id": r.id.to_string(),
                        "fact": r.data.get("fact").cloned().unwrap_or(json!(null)),
                        "superseded": superseded,
                        "created_at": format_dt(&r.created_at),
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::TtlSet { id, duration } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let secs = parse_duration(&duration)?;
            if secs > i64::MAX as u64 {
                anyhow::bail!("duration too large");
            }
            let dur = chrono::Duration::seconds(secs as i64);
            mem.ttl().set_ttl(&rid, dur).context("failed to set TTL")?;

            out.print(&json!({"ok": true, "id": id, "expires_in": duration}));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::TtlClear { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            mem.ttl().clear_ttl(&rid).context("failed to clear TTL")?;

            out.print(&json!({"ok": true, "id": id, "ttl": null}));
            Ok(EXIT_OK)
        }

        // ── Intelligent Database ─────────────────────────────
        #[cfg(all(feature = "embed", feature = "graph"))]
        Command::AutoLink { id, threshold } => {
            let db_path = require_db(&db_opt)?;
            if !(0.5..=1.0).contains(&threshold) {
                anyhow::bail!("--threshold must be between 0.5 and 1.0");
            }
            let db = open_with_embedder(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let report = db
                .auto_link(&rid, Some(threshold))
                .context("auto-link failed")?;
            out.print(&json!({
                "ok": true,
                "id": id,
                "entities_found": report.entities_found,
                "edges_created": report.edges_created,
                "similarity_links": report.similarity_links,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(all(feature = "embed", feature = "graph"))]
        Command::DetectConflicts { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_embedder(&db_path)?;
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            let conflicts = db
                .detect_conflicts(&rid)
                .context("conflict detection failed")?;
            let values: Vec<Value> = conflicts
                .iter()
                .map(|c| match c {
                    axil_core::ConflictResult::Novel => json!({"type": "novel"}),
                    axil_core::ConflictResult::Supersedes {
                        old_record_id,
                        similarity,
                    } => json!({
                        "type": "supersedes",
                        "old_record_id": old_record_id.as_str(),
                        "similarity": similarity,
                    }),
                    axil_core::ConflictResult::Contradicts {
                        existing_record_id,
                        similarity,
                    } => json!({
                        "type": "contradicts",
                        "existing_record_id": existing_record_id.as_str(),
                        "similarity": similarity,
                    }),
                })
                .collect();
            out.print(&json!({"id": id, "conflicts": values}));
            Ok(EXIT_OK)
        }

        #[cfg(all(feature = "embed", feature = "graph"))]
        Command::Consolidate { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_embedder(&db_path)?;
            match db.consolidate_entity(&entity)? {
                Some(cf) => {
                    out.print(&json!({
                        "entity": cf.entity,
                        "summary": cf.summary,
                        "source_count": cf.source_ids.len(),
                        "latest_at": cf.latest_at.to_rfc3339(),
                    }));
                }
                None => {
                    out.print(&json!({"entity": entity, "error": "entity not found"}));
                }
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "graph")]
        Command::EntityHistory { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let history = db.entity_history(&entity)?;
            let values: Vec<Value> = history
                .iter()
                .map(|(record, status)| {
                    let mut v = record_to_json(record);
                    v["status"] = json!(status);
                    v
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::EntityAlias { entity, alias } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            mem.semantic()
                .add_alias(&entity, &alias)
                .context("failed to add entity alias")?;

            out.print(&json!({
                "ok": true,
                "entity": entity,
                "alias": alias,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::EntityResolve {
            name,
            fuzzy,
            strategy,
            context,
            session_id,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);

            if fuzzy {
                let strat: axil_memory::DisambiguationStrategy =
                    strategy.parse().unwrap_or_default();
                let opts = axil_memory::DisambiguationOptions {
                    strategy: strat,
                    context_terms: context
                        .map(|c| c.split(',').map(|s| s.trim().to_string()).collect())
                        .unwrap_or_default(),
                    session_id,
                };
                let matches = mem.semantic().resolve_with_strategy(&name, &opts)?;
                let match_json: Vec<Value> = matches.iter().map(|m| m.to_json()).collect();
                let needs_review: Vec<&axil_memory::EntityMatch> =
                    matches.iter().filter(|m| m.needs_review()).collect();
                out.print(&json!({
                    "name": name,
                    "strategy": strategy,
                    "matches": match_json,
                    "needs_review": needs_review.len(),
                    "auto_resolve": matches.first().map(|m| m.is_auto_resolve()).unwrap_or(false),
                }));
            } else {
                match mem.semantic().resolve(&name)? {
                    Some(canonical) => {
                        let aliases = mem.semantic().aliases(&canonical)?;
                        out.print(&json!({
                            "name": name,
                            "canonical": canonical,
                            "aliases": aliases,
                        }));
                    }
                    None => {
                        let aliases = mem.semantic().aliases(&name)?;
                        out.print(&json!({
                            "name": name,
                            "canonical": name,
                            "is_alias": false,
                            "aliases": aliases,
                        }));
                    }
                }
            }
            Ok(EXIT_OK)
        }

        #[cfg(feature = "memory")]
        Command::EntityMerge { target, source } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);
            let moved = mem
                .semantic()
                .merge(&target, &source)
                .context("entity merge failed")?;
            out.print(&json!({
                "merged": true,
                "target": target,
                "source": source,
                "facts_moved": moved,
            }));
            Ok(EXIT_OK)
        }

        // ── LLM commands ────────────────────────────────
        Command::Llm { command: llm_cmd } => {
            let db_path = require_db(&db_opt)?;
            match llm_cmd {
                LlmCommand::Test => {
                    let db = wire_llm(open_with_all_detected(&db_path)?, &db_path)?;
                    if !db.has_llm() {
                        out.print(&json!({
                            "ok": false,
                            "error": "no LLM configured — set llm.endpoint, llm.model, and llm.api_key in axil.toml or set AXIL_LLM_API_KEY env var",
                        }));
                        return Ok(EXIT_ERROR);
                    }

                    // Send a simple test prompt.
                    match db.llm_complete("Say 'hello' and nothing else.") {
                        Ok(response) => {
                            out.print(&json!({
                                "ok": true,
                                "model": db.llm_model_name(),
                                "response": response.text.trim(),
                                "input_tokens": response.input_tokens,
                                "output_tokens": response.output_tokens,
                            }));
                            Ok(EXIT_OK)
                        }
                        Err(e) => {
                            out.print(&json!({
                                "ok": false,
                                "model": db.llm_model_name(),
                                "error": format!("{e}"),
                            }));
                            Ok(EXIT_ERROR)
                        }
                    }
                }
                LlmCommand::Usage => {
                    let db = wire_llm(open_with_all_detected(&db_path)?, &db_path)?;
                    let usage = db.llm_usage();
                    out.print(&json!({
                        "model": db.llm_model_name(),
                        "has_llm": db.has_llm(),
                        "calls": usage.calls,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "estimated_cost_usd": usage.estimated_cost_usd,
                        "fallback_count": usage.fallback_count,
                    }));
                    Ok(EXIT_OK)
                }
                LlmCommand::Config => {
                    let dir = db_path.parent().unwrap_or(Path::new("."));
                    let config = axil_core::load_config_from(dir).unwrap_or_default();
                    let llm_config = &config.llm;
                    out.print(&json!({
                        "endpoint": llm_config.endpoint,
                        "model": llm_config.model,
                        "has_api_key": llm_config.resolved_api_key().is_some(),
                        "is_configured": llm_config.is_configured(),
                        "limits": {
                            "max_calls_per_minute": llm_config.limits.max_calls_per_minute,
                            "max_tokens_per_session": llm_config.limits.max_tokens_per_session,
                            "budget_usd_per_day": llm_config.limits.budget_usd_per_day,
                        },
                        "cost_per_1m_input": llm_config.cost_per_1m_input,
                        "cost_per_1m_output": llm_config.cost_per_1m_output,
                    }));
                    Ok(EXIT_OK)
                }
            }
        }

        Command::WarmUp => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let report = db.warm_up().context("warm-up failed")?;
            out.print(&json!({
                "ok": true,
                "warmed_up": report.warmed_up,
                "vector_rebuilt": report.vector_rebuilt,
            }));
            Ok(EXIT_OK)
        }

        // ── SCIP ingest ────────────────────────────────
        #[cfg(feature = "scip")]
        Command::IngestScip {
            path,
            dry_run,
            watch,
        } => {
            let db_path = require_db(&db_opt)?;
            let path_was_inferred = path.is_none();
            let paths: Vec<PathBuf> = match path {
                Some(p) => vec![p],
                None => {
                    let axil_dir = db_path
                        .parent()
                        .expect("require_db returns .axil/memory.axil");
                    let repo_root = axil_dir.parent().unwrap_or(axil_dir);

                    // Canonical location wins over freshness. A repo
                    // can carry stale per-crate or experimental
                    // *.scip files whose mtime is newer than
                    // `.axil/index.scip` — picking newest-by-mtime
                    // would ingest the wrong index. Honor
                    // `.axil/index.scip` first, then the per-language
                    // `index-<lang>*.scip` set a polyglot `scip
                    // refresh` writes (all of them), then `./index.scip`
                    // for back-compat (pre-Phase-14 layout), then
                    // fall back to discovery for anyone who wrote
                    // a different filename.
                    let canonical = axil_dir.join("index.scip");
                    let legacy = repo_root.join("index.scip");
                    if canonical.exists() {
                        vec![canonical]
                    } else {
                        // The canonical file short-circuits above, so the
                        // per-language read_dir only runs when needed.
                        let mut per_lang: Vec<PathBuf> = std::fs::read_dir(axil_dir)
                            .map(|entries| {
                                entries
                                    .flatten()
                                    .map(|e| e.path())
                                    .filter(|p| {
                                        p.is_file()
                                            && p.file_name()
                                                .and_then(|n| n.to_str())
                                                .map(|n| {
                                                    n.starts_with("index-") && n.ends_with(".scip")
                                                })
                                                .unwrap_or(false)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        per_lang.sort();
                        if !per_lang.is_empty() {
                            per_lang
                        } else if legacy.exists() {
                            vec![legacy]
                        } else {
                            let mut found = axil_scip::discover_scip_files(&[axil_dir, repo_root]);
                            found.sort_by_key(|f| f.modified_secs_ago);
                            match found.into_iter().next() {
                                Some(f) => vec![f.path],
                                None => {
                                    let langs = detect_scip_indexable_languages(repo_root);
                                    return Err(anyhow::anyhow!(
                                        "no *.scip file found in {} or {}. {}",
                                        axil_dir.display(),
                                        repo_root.display(),
                                        suggest_scip_installers(&langs),
                                    ));
                                }
                            }
                        }
                    }
                }
            };
            if path_was_inferred {
                // Make the choice visible so users (and tests) can
                // confirm which *.scip got picked when multiple are
                // present in the tree.
                for p in &paths {
                    eprintln!("axil ingest-scip: using {}", p.display());
                }
            }
            if watch {
                // Watch follows one file; a polyglot inferred set is
                // ambiguous, so demand an explicit path.
                let path = match paths.as_slice() {
                    [single] => single.clone(),
                    many => anyhow::bail!(
                        "--watch follows a single file but {} were inferred; pass the path explicitly",
                        many.len()
                    ),
                };
                // Minimal watch loop: wait for the file to stabilize, ingest,
                // then poll for fresh mtimes. Cheap enough without depending on notify.
                let mut last_ingested_mtime: Option<std::time::SystemTime> = None;
                let stabilize_timeout = std::time::Duration::from_secs(30);
                eprintln!("axil ingest-scip --watch: waiting for {}…", path.display());
                loop {
                    match std::fs::metadata(&path) {
                        Ok(md) => {
                            let mtime = md.modified().ok();
                            if mtime != last_ingested_mtime {
                                if !axil_scip::wait_for_stable(&path, stabilize_timeout)? {
                                    eprintln!(
                                        "axil ingest-scip --watch: index still being written after {}s; skipping cycle",
                                        stabilize_timeout.as_secs()
                                    );
                                } else {
                                    let db = open_for_scip_ingest(&db_path)?;
                                    let report = axil_scip::ingest_scip_opts(
                                        &db,
                                        &path,
                                        axil_scip::IngestOptions { dry_run },
                                    )?;
                                    out.print(
                                        &serde_json::to_value(&report)
                                            .unwrap_or(json!({"ok":true})),
                                    );
                                    last_ingested_mtime = mtime;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("axil ingest-scip --watch: stat failed: {e}");
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            } else {
                let db = open_for_scip_ingest(&db_path)?;
                // A live refresh lock means a detached background `scip
                // refresh` may be mid-write on these very files; pay the
                // stabilization wait only in that case.
                let refresh_lock_live = db_path
                    .parent()
                    .map(|d| d.join("scip-refresh.lock"))
                    .and_then(|l| std::fs::metadata(l).ok())
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.elapsed().ok())
                    .map(|e| e.as_secs() < 600)
                    .unwrap_or(false);
                let mut reports: Vec<Value> = Vec::new();
                let mut ok_count = 0usize;
                for p in &paths {
                    if refresh_lock_live {
                        let _ = axil_scip::wait_for_stable(p, std::time::Duration::from_secs(30));
                    }
                    // Per-file capture: one unreadable file (e.g. a write
                    // race) must not abort the rest after earlier files
                    // already mutated the DB, nor eat the whole report.
                    match axil_scip::ingest_scip_opts(&db, p, axil_scip::IngestOptions { dry_run })
                    {
                        Ok(report) => {
                            ok_count += 1;
                            let mut v = serde_json::to_value(&report).unwrap_or(json!({"ok":true}));
                            if let Some(obj) = v.as_object_mut() {
                                obj.insert("path".into(), json!(p.display().to_string()));
                            }
                            reports.push(v);
                        }
                        Err(e) => reports.push(json!({
                            "ok": false,
                            "path": p.display().to_string(),
                            "error": e.to_string(),
                        })),
                    }
                }
                if ok_count == 0 {
                    let errs: Vec<&str> = reports
                        .iter()
                        .filter_map(|r| r.get("error").and_then(Value::as_str))
                        .collect();
                    anyhow::bail!("ingest failed for every file:\n  {}", errs.join("\n  "));
                }
                let all_ok = ok_count == reports.len();
                let mut out_value = if reports.len() == 1 {
                    reports.pop().expect("len checked")
                } else {
                    json!({ "ok": all_ok, "ingested": reports })
                };
                // Backfill `_idx_code_proxies` rows that were built before
                // SCIP arrived. Best-effort, only on real (non-dry) ingest;
                // surfaces counts so downstream tooling can act on
                // `proxy_backfill.upgraded > 0`.
                if !dry_run {
                    if let Ok(bf) = axil_indexer::proxy::backfill_canonical_ids_from_scip(&db) {
                        if let Some(obj) = out_value.as_object_mut() {
                            obj.insert(
                                "proxy_backfill".to_string(),
                                serde_json::to_value(&bf).unwrap_or(json!({})),
                            );
                        }
                    }
                }
                out.print(&out_value);
                Ok(EXIT_OK)
            }
        }

        #[cfg(feature = "deps")]
        Command::Deps(action) => {
            // try Extension dispatch first; fall back
            // to the hardcoded `run_deps` for any subcommand the
            // Extension doesn't claim. DocsExtension::handle_cli
            // currently claims only `deps status`; the rest stay on
            // the hardcoded path.
            if let Some(exit) = try_deps_extension_dispatch(&action, &db_opt, out)? {
                return Ok(exit);
            }
            run_deps(action, &db_opt, out)
        }

        #[cfg(feature = "deps")]
        Command::DepDocs {
            query,
            dep,
            top_k,
            include_superseded,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let hits =
                axil_docs::query_dep_docs(&db, &query, top_k, dep.as_deref(), include_superseded)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&serde_json::to_value(&hits)?);
            Ok(EXIT_OK)
        }

        // Marshal clap args → CliInvocation; CheckpointExtension owns write/read.
        #[cfg(feature = "checkpoint")]
        Command::Checkpoint {
            arg,
            session,
            is_final,
        } => run_checkpoint_extension(arg, session, is_final, &db_opt, out),

        // Marshal clap args → CliInvocation; CacheExtension owns put/get/stats/clear.
        #[cfg(feature = "cache")]
        Command::Cache(action) => run_cache_extension(action, &db_opt, out),

        #[cfg(feature = "scip")]
        Command::Scip(action) => {
            let db_path = require_db(&db_opt)?;
            let axil_dir = db_path
                .parent()
                .expect("require_db returns .axil/memory.axil")
                .to_path_buf();
            let repo_root = axil_dir.parent().unwrap_or(&axil_dir).to_path_buf();

            match action {
                ScipCommand::Status => {
                    let detected = scip_detect::detect_scip_projects(&repo_root);
                    let langs = scip_detect::detected_languages(&detected);
                    let polyglot = detected.len() > 1;
                    // One row per (language, project dir) pair, with the
                    // output file `scip refresh` would write for it.
                    let mut projects: Vec<Value> = Vec::new();
                    for p in &detected {
                        let (bin, _args) = match scip_indexer_command(p.language) {
                            Some(c) => c,
                            None => continue,
                        };
                        // Same naming/age/label rules refresh uses, so the
                        // two commands can't disagree on where an index is.
                        let expected =
                            scip_detect::expected_output(p, polyglot, &axil_dir, &repo_root);
                        projects.push(json!({
                            "language": p.language,
                            "dir": scip_detect::rel_label(&p.dir, &repo_root),
                            "indexer": bin,
                            "on_path": binary_on_path(bin),
                            "output": expected.display().to_string(),
                            "output_exists": expected.is_file(),
                            "age_seconds": scip_detect::age_secs(&expected),
                        }));
                    }
                    let found =
                        axil_scip::discover_scip_files(&[axil_dir.as_path(), repo_root.as_path()]);
                    let files: Vec<Value> = found.iter().map(|f| {
                        let summary = axil_scip::inspect_scip(&f.path).ok();
                        json!({
                            "path": f.path.display().to_string(),
                            "age_seconds": f.modified_secs_ago,
                            "indexer_name": summary.as_ref().map(|s| s.indexer_name.clone()),
                            "indexer_version": summary.as_ref().map(|s| s.indexer_version.clone()),
                            "symbol_count": summary.as_ref().map(|s| s.symbol_count),
                            "document_count": summary.as_ref().map(|s| s.document_count),
                        })
                    }).collect();
                    out.print(&json!({
                        "ok": true,
                        "repo_root": repo_root.display().to_string(),
                        "axil_dir": axil_dir.display().to_string(),
                        "polyglot": polyglot,
                        "detected_languages": langs,
                        "projects": projects,
                        "scip_files": files,
                        "install_hint": suggest_scip_installers(&langs),
                    }));
                    Ok(EXIT_OK)
                }
                ScipCommand::Refresh {
                    root: root_override,
                    language,
                    output,
                    skip_ingest,
                    dry_run,
                    if_stale,
                    max_age_days,
                    in_background,
                } => {
                    // One refresh covers every detected project: each
                    // `(language, project dir)` pair gets its own indexer
                    // run (cwd = project dir) and its own output file, so
                    // `frontend/` TypeScript and `backend/` Python are both
                    // kept fresh by the same brain-hook call.
                    const DEFAULT_OUTPUT: &str = ".axil/index.scip";
                    let output_is_custom = output.as_path() != Path::new(DEFAULT_OUTPUT);

                    // Indexers run with cwd = project dir, so output paths
                    // (and the dirs they're derived from) must be absolute.
                    // `--root` overrides where we SCAN for projects (so the
                    // caller can point at a tree the DB doesn't live under);
                    // output files still anchor to the DB's `.axil` dir.
                    let repo_root_abs = match &root_override {
                        Some(r) => scip_detect::absolutize(r),
                        None => scip_detect::absolutize(&repo_root),
                    };
                    let axil_dir_abs = scip_detect::absolutize(&axil_dir);

                    let all_projects = scip_detect::detect_scip_projects(&repo_root_abs);
                    // Polyglot repos use per-language output names so one
                    // language's run can't clobber another's. Single-project
                    // repos keep the legacy `index.scip` name (no migration).
                    let polyglot = all_projects.len() > 1;
                    // An explicit --language keeps the old hard-error
                    // contract (missing indexer aborts with the install
                    // hint), no matter how many project dirs the language
                    // resolves to.
                    let explicit = language.is_some();

                    let mut projects = all_projects.clone();
                    if let Some(l) = language.as_deref() {
                        let l = scip_detect::normalize_language(l).ok_or_else(|| {
                            anyhow::anyhow!(
                                "unsupported --language '{l}'. Supported: rust, python, typescript, go, java"
                            )
                        })?;
                        projects.retain(|p| p.language == l);
                        if projects.is_empty() {
                            // No marker for the forced language — keep the
                            // old escape hatch: run its indexer at the root.
                            projects.push(scip_detect::ScipProject {
                                language: l,
                                dir: repo_root_abs.clone(),
                            });
                        }
                    }
                    if projects.is_empty() {
                        anyhow::bail!(
                            "no language detected in {} (or any subfolder). Pass --language explicitly, or add a marker file (Cargo.toml, package.json, pyproject.toml, go.mod, pom.xml).",
                            repo_root_abs.display()
                        );
                    }
                    // Languages detected but excluded by --language, kept in
                    // the JSON so an agent knows to re-run for the rest.
                    let skipped_languages: Vec<&'static str> =
                        scip_detect::detected_languages(&all_projects)
                            .into_iter()
                            .filter(|l| !projects.iter().any(|p| p.language == *l))
                            .collect();

                    if output_is_custom && projects.len() > 1 {
                        anyhow::bail!(
                            "--output names a single file but {} projects were detected; add --language (and run once per language) to use a custom output path",
                            projects.len()
                        );
                    }

                    let targets: Vec<(scip_detect::ScipProject, PathBuf)> = projects
                        .iter()
                        .map(|p| {
                            let out_path = if output_is_custom {
                                if output.is_absolute() {
                                    output.clone()
                                } else {
                                    repo_root_abs.join(&output)
                                }
                            } else {
                                scip_detect::expected_output(
                                    p,
                                    polyglot,
                                    &axil_dir_abs,
                                    &repo_root_abs,
                                )
                            };
                            (p.clone(), out_path)
                        })
                        .collect();

                    let threshold = max_age_days.saturating_mul(86_400);
                    let is_fresh = |out_path: &Path| {
                        scip_detect::age_secs(out_path)
                            .map(|a| a < threshold)
                            .unwrap_or(false)
                    };
                    // One skip-entry shape shared by the fast path and the
                    // run loop, so consumers see a single schema.
                    let fresh_entry = |p: &scip_detect::ScipProject, out_path: &Path| {
                        json!({
                            "language": p.language,
                            "dir": scip_detect::rel_label(&p.dir, &repo_root_abs),
                            "status": "skipped",
                            "reason": "fresh",
                            "age_seconds": scip_detect::age_secs(out_path),
                            "output": out_path.display().to_string(),
                        })
                    };
                    let missing_indexer_entry = |p: &scip_detect::ScipProject, bin: &str| {
                        json!({
                            "language": p.language,
                            "dir": scip_detect::rel_label(&p.dir, &repo_root_abs),
                            "status": "skipped",
                            "reason": "indexer_missing",
                            "indexer": bin,
                            "hint": suggest_scip_installers(&[p.language]),
                        })
                    };
                    // Old single-output consumers read `language`/`output`
                    // at the top level — keep that shape when exactly one
                    // project is reported.
                    fn flat_merge_single(result: &mut Value, flat: Option<Value>) {
                        let Some(Value::Object(flat)) = flat else {
                            return;
                        };
                        if let Some(obj) = result.as_object_mut() {
                            for (k, v) in flat {
                                obj.entry(k).or_insert(v);
                            }
                        }
                    }

                    // Cheap fast-path for the brain hook: return before any
                    // DB or indexer work when no project is actionable —
                    // output fresh, or (in auto mode) indexer not installed.
                    // Without the installed check, one uninstallable
                    // language would defeat this path forever and spawn a
                    // useless background child on every session.
                    if if_stale {
                        let mut skips: Vec<Value> = Vec::new();
                        let mut actionable = 0usize;
                        for (p, out_path) in &targets {
                            if is_fresh(out_path) {
                                skips.push(fresh_entry(p, out_path));
                            } else if explicit {
                                // Forced language: let the run loop bail
                                // with the install hint if needed.
                                actionable += 1;
                            } else {
                                let (bin, _) = scip_indexer_command(p.language)
                                    .expect("languages come from MARKERS");
                                if binary_on_path(bin) {
                                    actionable += 1;
                                } else {
                                    skips.push(missing_indexer_entry(p, bin));
                                }
                            }
                        }
                        if actionable == 0 {
                            let all_fresh = skips
                                .iter()
                                .all(|e| e.get("reason") == Some(&json!("fresh")));
                            let flat = (skips.len() == 1).then(|| skips[0].clone());
                            let mut result = json!({
                                "ok": true,
                                "skipped": true,
                                "reason": if all_fresh { "fresh" } else { "no_actionable_projects" },
                                "max_age_seconds": threshold,
                                "polyglot": polyglot,
                                "projects": skips,
                            });
                            flat_merge_single(&mut result, flat);
                            out.print(&result);
                            return Ok(EXIT_OK);
                        }
                    }

                    // Background spawn: re-exec ourselves without
                    // `--in-background` so the parent can return
                    // immediately. Lock file at `.axil/scip-refresh.lock`
                    // prevents concurrent rust-analyzer runs (it's PID-checked
                    // so a stale lock from a crashed run is ignored).
                    if in_background {
                        let lock_path = axil_dir.join("scip-refresh.lock");
                        // Lock is "live" if its mtime is within the
                        // typical refresh window. Five minutes is well
                        // past rust-analyzer's worst-case observed time
                        // on this repo (~24s) but short enough that a
                        // crashed run unblocks within one work-break.
                        // Avoids needing libc::kill or a process probe.
                        const STALE_LOCK_SECS: u64 = 300;
                        if let Ok(md) = std::fs::metadata(&lock_path) {
                            if let Ok(modified) = md.modified() {
                                if let Ok(elapsed) = modified.elapsed() {
                                    if elapsed.as_secs() < STALE_LOCK_SECS {
                                        let pid = std::fs::read_to_string(&lock_path)
                                            .ok()
                                            .and_then(|s| s.trim().parse::<u32>().ok());
                                        out.print(&json!({
                                            "ok": true,
                                            "skipped": true,
                                            "reason": "already_running",
                                            "lock_pid": pid,
                                            "lock_age_seconds": elapsed.as_secs(),
                                        }));
                                        return Ok(EXIT_OK);
                                    }
                                    // Stale — remove so try_acquire below
                                    // can claim atomically.
                                    let _ = std::fs::remove_file(&lock_path);
                                }
                            }
                        }
                        let exe = std::env::current_exe()
                            .context("failed to resolve current executable")?;
                        let mut child_args: Vec<String> = vec![
                            "--db".into(),
                            db_path.display().to_string(),
                            "scip".into(),
                            "refresh".into(),
                            "--max-age-days".into(),
                            max_age_days.to_string(),
                        ];
                        if let Some(r) = root_override.as_deref() {
                            // Propagate the scan-root override so the detached
                            // worker scans the same tree this parent detected.
                            child_args.push("--root".into());
                            child_args.push(r.display().to_string());
                        }
                        if output_is_custom {
                            // A custom output is single-project (validated
                            // above); default outputs are re-derived per
                            // project by the child.
                            child_args.push("--output".into());
                            child_args.push(targets[0].1.display().to_string());
                        }
                        if let Some(l) = language.as_deref() {
                            child_args.push("--language".into());
                            child_args.push(l.into());
                        }
                        if skip_ingest {
                            child_args.push("--skip-ingest".into());
                        }
                        if dry_run {
                            child_args.push("--dry-run".into());
                        }
                        if if_stale {
                            // Propagate so the child refreshes only the
                            // stale subset of projects. No mtime race:
                            // the lock claimed below makes this refresh
                            // the only writer of these files.
                            child_args.push("--if-stale".into());
                        }
                        // Do NOT propagate --in-background: that's the
                        // whole point of the spawn.

                        // Atomically claim the lock BEFORE spawning so
                        // two concurrent `axil scip refresh
                        // --in-background` invocations can't both pass
                        // the stale-check and double-spawn. If the
                        // create_new fails with AlreadyExists, another
                        // process raced past the stale check between
                        // our check and our claim — bail gracefully.
                        // The child's LockGuard later owns drop-cleanup.
                        match LockGuard::try_acquire(lock_path.clone(), "spawning") {
                            Ok(g) => {
                                // Prevent Drop from deleting the file
                                // here in the parent — the child owns
                                // the lifetime.
                                std::mem::forget(g);
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                                out.print(&json!({
                                    "ok": true,
                                    "skipped": true,
                                    "reason": "already_running_race",
                                }));
                                return Ok(EXIT_OK);
                            }
                            Err(e) => {
                                return Err(anyhow::Error::new(e)
                                    .context("failed to claim scip-refresh lock"))
                            }
                        }
                        let stderr_log = axil_dir.join("scip-refresh.log");
                        // Spawn through a shell-level `nohup` wrapper.
                        // Direct std::process::Command::spawn was being
                        // killed mid-rust-analyzer when the parent
                        // process exited — even with `process_group(0)`
                        // the child died ~9KB into rust-analyzer's
                        // metadata phase. `nohup` is the canonical
                        // Unix way to disown a child from the parent's
                        // controlling terminal and SIGHUP propagation,
                        // and it's POSIX-standard so always available.
                        // Quoting: paths shell-escaped with single
                        // quotes and embedded `'` doubled to `'\''`.
                        fn sh_quote(s: &str) -> String {
                            let escaped = s.replace('\'', "'\\''");
                            format!("'{escaped}'")
                        }
                        let mut shell_parts: Vec<String> =
                            vec!["nohup".into(), sh_quote(&exe.to_string_lossy())];
                        for a in &child_args {
                            shell_parts.push(sh_quote(a));
                        }
                        shell_parts.push(format!(
                            ">/dev/null 2>{}",
                            sh_quote(&stderr_log.to_string_lossy())
                        ));
                        shell_parts.push("</dev/null &".into());
                        shell_parts.push("echo $!".into());
                        let shell_cmd = shell_parts.join(" ");
                        let mut cmd = std::process::Command::new("sh");
                        cmd.arg("-c")
                            .arg(&shell_cmd)
                            // Use the absolutized root: a relative `--db`
                            // (e.g. `.axil/memory.axil`) makes `repo_root`
                            // an empty path, and `current_dir("")` fails the
                            // spawn with ENOENT.
                            .current_dir(&repo_root_abs)
                            .stdin(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .stdout(std::process::Stdio::piped());
                        let mut child = match cmd.spawn() {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = std::fs::remove_file(&lock_path);
                                return Err(anyhow::Error::new(e)
                                    .context("failed to spawn background refresh"));
                            }
                        };
                        // Wait for the shell to background the real
                        // worker and print its PID, then exit. Should
                        // be instant — we're not waiting for the
                        // worker itself.
                        let mut bg_pid_text = String::new();
                        if let Some(mut stdout) = child.stdout.take() {
                            use std::io::Read;
                            let _ = stdout.read_to_string(&mut bg_pid_text);
                        }
                        let _ = child.wait();
                        let bg_pid: u32 = bg_pid_text.trim().parse().unwrap_or(0);
                        let pid_for_lock = if bg_pid != 0 { bg_pid } else { child.id() };
                        let _ = std::fs::write(&lock_path, pid_for_lock.to_string());
                        out.print(&json!({
                            "ok": true,
                            "spawned": true,
                            "pid": pid_for_lock,
                            "lock": lock_path.display().to_string(),
                            "log": stderr_log.display().to_string(),
                        }));
                        return Ok(EXIT_OK);
                    }

                    // From here on: synchronous foreground path. If we
                    // were spawned with a lock, clear it on exit.
                    let lock_path = axil_dir.join("scip-refresh.lock");
                    let _lock_guard = LockGuard::new(lock_path);

                    let mut entries: Vec<Value> = Vec::new();
                    // (entry index, output path) of successful runs, for ingest.
                    let mut produced: Vec<(usize, PathBuf)> = Vec::new();

                    for (project, out_path) in &targets {
                        let lang = project.language;
                        let dir_label = scip_detect::rel_label(&project.dir, &repo_root_abs);

                        if if_stale && is_fresh(out_path) {
                            entries.push(fresh_entry(project, out_path));
                            continue;
                        }

                        let (bin, args_template) = scip_indexer_command(lang)
                            .expect("languages come from MARKERS or were validated above");

                        let program = match resolve_indexer_program(bin) {
                            Some(p) => p,
                            None => {
                                if explicit {
                                    anyhow::bail!(
                                        "indexer `{bin}` not found on PATH. {}",
                                        suggest_scip_installers(&[lang])
                                    );
                                }
                                entries.push(missing_indexer_entry(project, bin));
                                continue;
                            }
                        };

                        if let Some(parent) = out_path.parent() {
                            std::fs::create_dir_all(parent).with_context(|| {
                                format!("failed to create {}", parent.display())
                            })?;
                        }

                        let out_str = out_path.display().to_string();
                        let resolved_args: Vec<String> = args_template
                            .iter()
                            .map(|a| {
                                if *a == "{out}" {
                                    out_str.clone()
                                } else {
                                    (*a).to_string()
                                }
                            })
                            .collect();

                        eprintln!(
                            "axil scip refresh: running `{bin} {}` (cwd: {})",
                            resolved_args.join(" "),
                            project.dir.display()
                        );
                        // Show a spinner while the external indexer runs (TTY only,
                        // off when --quiet). enable_steady_tick handles the redraw
                        // on its own thread; .status() below blocks until the child
                        // finishes, after which we finish_and_clear so the JSON
                        // output that follows is the only thing left on screen.
                        let spinner = {
                            use std::io::IsTerminal;
                            if !out.quiet && std::io::stderr().is_terminal() {
                                let s = indicatif::ProgressBar::new_spinner();
                                s.set_draw_target(indicatif::ProgressDrawTarget::stderr());
                                s.set_style(
                                    indicatif::ProgressStyle::with_template(
                                        "  {spinner:.cyan} {msg} {elapsed}",
                                    )
                                    .unwrap(),
                                );
                                s.set_message(format!("{bin} indexing {dir_label}..."));
                                s.enable_steady_tick(std::time::Duration::from_millis(120));
                                Some(s)
                            } else {
                                None
                            }
                        };
                        let started = std::time::Instant::now();
                        // Capture rather than inherit: on failure the child's
                        // stderr is both forwarded and inspected for known
                        // crash signatures (e.g. scip-python's Windows bug).
                        let run = std::process::Command::new(&program)
                            .args(&resolved_args)
                            .current_dir(&project.dir)
                            .output();
                        if let Some(s) = spinner {
                            s.finish_and_clear();
                        }
                        let indexer_secs = started.elapsed().as_secs_f64();

                        let error = match run {
                            Err(e) => Some(format!(
                                "failed to spawn `{bin}` ({}): {e}",
                                program.display()
                            )),
                            Ok(ref out_run) if !out_run.status.success() => {
                                let child_err = String::from_utf8_lossy(&out_run.stderr);
                                if !child_err.is_empty() {
                                    eprint!("{child_err}");
                                }
                                let mut msg = format!(
                                    "`{bin}` exited with status {} (cwd: {})",
                                    out_run.status,
                                    project.dir.display()
                                );
                                if let Some(hint) = indexer_crash_hint(bin, &child_err) {
                                    msg.push_str("\n  hint: ");
                                    msg.push_str(&hint);
                                }
                                Some(msg)
                            }
                            Ok(_) => {
                                let scip_size =
                                    std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
                                if scip_size == 0 {
                                    Some(format!(
                                        "`{bin}` succeeded but produced an empty file at {}",
                                        out_path.display()
                                    ))
                                } else {
                                    produced.push((entries.len(), out_path.clone()));
                                    entries.push(json!({
                                        "language": lang,
                                        "dir": dir_label,
                                        "status": "refreshed",
                                        "indexer": bin,
                                        "indexer_args": resolved_args,
                                        "output": out_str,
                                        "scip_bytes": scip_size,
                                        "indexer_seconds": indexer_secs,
                                    }));
                                    None
                                }
                            }
                        };
                        if let Some(msg) = error {
                            // The old single-target contract aborts on a
                            // failed run; the multi-project sweep records
                            // it and lets the all-failed gate below decide.
                            if explicit && targets.len() == 1 {
                                anyhow::bail!("{msg}. Output (if any) at {}", out_path.display());
                            }
                            entries.push(json!({
                                "language": lang,
                                "dir": dir_label,
                                "status": "failed",
                                "error": msg,
                                "output": out_path.display().to_string(),
                            }));
                        }
                    }

                    let mut proxy_backfill: Option<Value> = None;
                    if !skip_ingest && !produced.is_empty() {
                        let db = open_for_scip_ingest(&db_path)?;
                        for (entry_idx, scip_path) in &produced {
                            match axil_scip::ingest_scip_opts(
                                &db,
                                scip_path,
                                axil_scip::IngestOptions { dry_run },
                            ) {
                                Ok(report) => {
                                    if let Some(obj) = entries[*entry_idx].as_object_mut() {
                                        obj.insert(
                                            "ingest".into(),
                                            serde_json::to_value(&report)
                                                .unwrap_or(json!({"ok":true})),
                                        );
                                    }
                                }
                                Err(e) => {
                                    // Self-heal: a fresh-but-unindexed file
                                    // would pass the --if-stale check until
                                    // it ages out, leaving a silent gap in
                                    // the code graph — drop it so the next
                                    // refresh re-indexes and re-ingests.
                                    let _ = std::fs::remove_file(scip_path);
                                    if let Some(obj) = entries[*entry_idx].as_object_mut() {
                                        obj.insert(
                                            "ingest_error".into(),
                                            json!(format!(
                                                "{e} (output removed so the next refresh retries)"
                                            )),
                                        );
                                    }
                                }
                            }
                        }
                        // Match `IngestScip`'s post-ingest backfill so
                        // `scip refresh` and `ingest-scip` produce
                        // identical DB state.
                        if !dry_run {
                            if let Ok(bf) =
                                axil_indexer::proxy::backfill_canonical_ids_from_scip(&db)
                            {
                                proxy_backfill =
                                    Some(serde_json::to_value(&bf).unwrap_or(json!({})));
                            }
                        }
                    }

                    // Failure accounting derives from the entries — the one
                    // source of truth for per-project outcomes.
                    let failure_msgs: Vec<&str> = entries
                        .iter()
                        .filter_map(|e| {
                            e.get("error")
                                .or_else(|| e.get("ingest_error"))
                                .and_then(Value::as_str)
                        })
                        .collect();
                    let succeeded = entries
                        .iter()
                        .filter(|e| {
                            e.get("status").and_then(Value::as_str) == Some("refreshed")
                                && e.get("ingest_error").is_none()
                        })
                        .count();
                    if succeeded == 0 && !failure_msgs.is_empty() {
                        anyhow::bail!(
                            "scip refresh failed for every runnable project:\n  {}",
                            failure_msgs.join("\n  ")
                        );
                    }

                    // The legacy single-file index may be the only on-disk
                    // copy of a language this run did not touch (explicit
                    // --language, failed run, missing indexer) — retire it
                    // only when the per-language set fully covers every
                    // detected project.
                    let fully_covered = !explicit
                        && entries.iter().all(|e| {
                            let status = e.get("status").and_then(Value::as_str);
                            (status == Some("refreshed") && e.get("ingest_error").is_none())
                                || (status == Some("skipped")
                                    && e.get("reason").and_then(Value::as_str) == Some("fresh"))
                        });

                    let failed = failure_msgs.len();
                    let flat = (entries.len() == 1).then(|| entries[0].clone());
                    let mut result = json!({
                        "ok": failed == 0,
                        "polyglot": polyglot,
                        "refreshed": produced.len(),
                        "failed": failed,
                        "skipped_languages": skipped_languages,
                        "projects": entries,
                    });
                    flat_merge_single(&mut result, flat);

                    if polyglot
                        && !output_is_custom
                        && !dry_run
                        && !produced.is_empty()
                        && fully_covered
                    {
                        let legacy = axil_dir_abs.join("index.scip");
                        if legacy.is_file() && std::fs::remove_file(&legacy).is_ok() {
                            if let Some(obj) = result.as_object_mut() {
                                obj.insert(
                                    "legacy_output_removed".into(),
                                    json!(legacy.display().to_string()),
                                );
                            }
                        }
                    }
                    if let Some(bf) = proxy_backfill {
                        if let Some(obj) = result.as_object_mut() {
                            obj.insert("proxy_backfill".into(), bf);
                        }
                    }

                    out.print(&result);
                    Ok(EXIT_OK)
                }
            }
        }

        #[cfg(feature = "scip")]
        Command::EntityResolveScoped { name, scopes } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let refs: Vec<&str> = scopes.iter().map(String::as_str).collect();
            // Always append "global" as the last-resort scope if not given.
            let mut walked: Vec<&str> = refs.clone();
            if !walked.iter().any(|s| *s == "global") {
                walked.push("global");
            }
            let resolved = db.resolve_entity_alias(&name, &walked)?;
            let canonical =
                resolved.unwrap_or_else(|| axil_core::entity::natural_canonical_id(&name));
            out.print(&json!({
                "ok": true,
                "name": name,
                "canonical_id": canonical,
                "scopes_walked": walked,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "scip")]
        Command::EntityMergeCanonical { from, to } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let moved = db.merge_entities(&from, &to)?;
            out.print(&json!({
                "ok": true,
                "from": from,
                "to": to,
                "edges_and_aliases_moved": moved,
            }));
            Ok(EXIT_OK)
        }

        // ── Extract Entities ────────────────────────────────
        Command::ExtractEntities { text } => {
            let input = if text == "-" {
                let mut buf = String::new();
                io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)?;
                if buf.len() as u64 > MAX_STDIN_BYTES {
                    return Err(anyhow::anyhow!(
                        "stdin exceeds {} MB cap",
                        MAX_STDIN_BYTES / (1024 * 1024)
                    ));
                }
                buf
            } else {
                text
            };
            let entities = axil_core::extract_entities(&input);
            let values: Vec<Value> = entities
                .iter()
                .map(|e| {
                    json!({
                        "name": e.name,
                        "type": format!("{:?}", e.entity_type).to_lowercase(),
                        "source": e.source_text,
                    })
                })
                .collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Boot ───────────────────────────────────────────
        Command::Boot {
            budget,
            boot_format,
            topic,
            files,
            entities,
            error,
            schema,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = std::sync::Arc::new(open_read_command(&db_path)?);
            // Register installed WASM plugins so their boot_block contributions
            // render in `axil boot`, like a native Extension's.
            register_installed_plugins(&db, &db_path);

            // Opt-in stable schema; legacy flat JSON below stays the default.
            if schema.as_deref() == Some("v1") {
                let opts = axil_core::BootOptions {
                    token_budget: budget,
                    topic: topic.clone(),
                    scope: None,
                };
                let ctx = db.boot(opts).context("boot failed")?;
                out.print(&serde_json::to_value(&ctx).expect("BootContext is always serializable"));
                return Ok(EXIT_OK);
            }

            let mut sections = serde_json::Map::new();

            // Pinned rules + high-importance constraints. These are loaded
            // FIRST so the agent sees them before any other context. The
            // narrative renderer prints them at the top, never dropped.
            let mut rules = db.list("rules").unwrap_or_default();
            rules.retain(|r| {
                axil_core::importance::is_pinned(&r.data)
                    || axil_core::importance::get_importance(&r.data) >= 0.9
            });
            if !rules.is_empty() {
                let rule_vals: Vec<Value> = rules
                    .iter()
                    .map(|r| {
                        json!({
                            "id": r.id.to_string(),
                            "rule": r.data.get("rule").cloned().unwrap_or(Value::Null),
                        })
                    })
                    .collect();
                sections.insert("rules".into(), json!(rule_vals));
            }

            // Last 3 sessions
            let sessions = db.storage().list("_sessions", 3, 0).unwrap_or_default();
            if !sessions.is_empty() {
                let session_vals: Vec<Value> = sessions
                    .iter()
                    .map(|r| {
                        let mut v = json!({"id": r.id.to_string()});
                        if let Some(obj) = r.data.as_object() {
                            for (k, val) in obj {
                                v[k] = Value::clone(val);
                            }
                        }
                        v
                    })
                    .collect();
                sections.insert("recent_sessions".into(), json!(session_vals));
            }

            // Top 5 decisions by importance (not just recency)
            let mut decisions = db.list("decisions").unwrap_or_default();
            decisions.sort_by(|a, b| {
                let ia = axil_core::importance::get_importance(&a.data);
                let ib = axil_core::importance::get_importance(&b.data);
                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
            });
            decisions.truncate(5);
            if !decisions.is_empty() {
                let dec_vals: Vec<Value> = decisions
                    .iter()
                    .map(|r| truncate_record_json(r, 200))
                    .collect();
                sections.insert("decisions".into(), json!(dec_vals));
            }

            // Top 5 errors by importance
            let mut errors = db.list("errors").unwrap_or_default();
            errors.sort_by(|a, b| {
                let ia = axil_core::importance::get_importance(&a.data);
                let ib = axil_core::importance::get_importance(&b.data);
                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
            });
            errors.truncate(5);
            if !errors.is_empty() {
                let err_vals: Vec<Value> = errors
                    .iter()
                    .map(|r| truncate_record_json(r, 200))
                    .collect();
                sections.insert("errors".into(), json!(err_vals));
            }

            // Current beliefs (top 5, non-doubted)
            let bs = axil_core::beliefs::BeliefSystem::new(&db);
            if let Ok(beliefs) = bs.list(None, false) {
                if !beliefs.is_empty() {
                    let belief_vals: Vec<Value> = beliefs
                        .iter()
                        .take(5)
                        .map(|b| {
                            json!({
                                "statement": b.statement,
                                "confidence": b.confidence,
                            })
                        })
                        .collect();
                    sections.insert("beliefs".into(), json!(belief_vals));
                }
            }

            // Procedural memory: high-confidence learned patterns
            #[cfg(feature = "memory")]
            {
                let mem = axil_memory::AgentMemory::new(&db);
                const BOOT_PROCEDURE_MIN_CONFIDENCE: f64 = 0.6;
                let procedures = mem.procedural().list().unwrap_or_default();
                let high_conf: Vec<Value> = procedures
                    .iter()
                    .filter(|r| {
                        r.data
                            .get("confidence")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0)
                            >= BOOT_PROCEDURE_MIN_CONFIDENCE
                    })
                    .take(5)
                    .map(|r| {
                        json!({
                            "name": r.data.get("name").cloned().unwrap_or(Value::Null),
                            "confidence": r.data.get("confidence").cloned().unwrap_or(Value::Null),
                        })
                    })
                    .collect();
                if !high_conf.is_empty() {
                    sections.insert("procedures".into(), json!(high_conf));
                }

                // Consolidated entity knowledge (top entities by fact count)
                let all_entities = db.list("_entities").unwrap_or_default();
                let mut entity_counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for r in &all_entities {
                    if let Some(name) = r.data.get("entity").and_then(|v| v.as_str()) {
                        *entity_counts.entry(name.to_string()).or_default() += 1;
                    }
                }
                let mut top_entities: Vec<_> =
                    entity_counts.into_iter().filter(|(_, c)| *c >= 2).collect();
                top_entities.sort_by(|a, b| b.1.cmp(&a.1));
                if !top_entities.is_empty() {
                    let entity_vals: Vec<Value> = top_entities
                        .iter()
                        .take(5)
                        .filter_map(|(name, count)| {
                            let knowledge = mem.semantic().about(name).ok()?;
                            let summary = knowledge
                                .consolidated_summary()
                                .unwrap_or_else(|| format!("{count} facts"));
                            Some(json!({
                                "entity": name,
                                "facts": count,
                                "consolidated": summary,
                            }))
                        })
                        .collect();
                    if !entity_vals.is_empty() {
                        sections.insert("key_entities".into(), json!(entity_vals));
                    }
                }

                // User preferences and rules
                let prefs = mem.preference().list().unwrap_or_default();
                if !prefs.is_empty() {
                    let pref_vals: Vec<Value> = prefs
                        .iter()
                        .take(5)
                        .map(|r| {
                            json!({
                                "key": r.data.get("key").cloned().unwrap_or(Value::Null),
                                "value": r.data.get("value").cloned().unwrap_or(Value::Null),
                            })
                        })
                        .collect();
                    sections.insert("preferences".into(), json!(pref_vals));
                }
            }

            // Architecture context
            let ctx_records = db.storage().list("context", 5, 0).unwrap_or_default();
            let arch: Vec<_> = ctx_records
                .iter()
                .filter(|r| r.data.get("type").and_then(Value::as_str) == Some("architecture"))
                .collect();
            if !arch.is_empty() {
                let arch_vals: Vec<Value> =
                    arch.iter().map(|r| truncate_record_json(r, 200)).collect();
                sections.insert("architecture".into(), json!(arch_vals));
            }

            // Topic-focused recall — uses Query-Time Chunk picking so
            // boot queries match the quality ceiling (~97% hit-rate on
            // LongMemEval-S). The embedder cost is amortized by stored
            // chunk vectors from insert time.
            if let Some(ref topic_query) = topic {
                #[cfg(feature = "embed")]
                {
                    let db_embed = open_with_embedder(&db_path)?;
                    let mut cfg = axil_core::RecallConfig::default();
                    cfg.qtc = Some(axil_core::scoring::QtcConfig::default());
                    if let Ok(results) = db_embed.recall(topic_query, 5, Some(cfg)) {
                        let topic_vals: Vec<Value> = results
                            .iter()
                            .map(|rr| {
                                json!({
                                    "id": rr.record.id.to_string(),
                                    "score": round4(rr.score),
                                    "table": rr.record.table,
                                    "summary": rr.record.data.get("summary")
                                        .or_else(|| rr.record.data.get("description"))
                                        .cloned()
                                        .unwrap_or_else(|| truncate_value(&rr.record.data, 200)),
                                })
                            })
                            .collect();
                        sections.insert("topic_recall".into(), json!(topic_vals));
                    }
                }
                #[cfg(not(feature = "embed"))]
                {
                    let _ = topic_query;
                    sections.insert(
                        "topic_recall".into(),
                        json!("embed feature required for topic recall"),
                    );
                }
            }

            // Active patterns (non-dismissed)
            #[cfg(feature = "memory")]
            {
                let engine = axil_memory::patterns::PatternEngine::new(&db);
                if let Ok(patterns) = engine.list(None) {
                    if !patterns.is_empty() {
                        let pat_vals: Vec<Value> = patterns
                            .iter()
                            .take(3)
                            .map(|p| {
                                json!({
                                    "name": p.name,
                                    "type": p.pattern_type.as_str(),
                                    "description": p.description,
                                    "frequency": p.frequency,
                                })
                            })
                            .collect();
                        sections.insert("active_patterns".into(), json!(pat_vals));
                    }
                }
            }

            // Dependency-doc freshness — flag manifests whose docs drifted.
            // Only nags once the project has actually run a `deps sync`.
            #[cfg(feature = "deps")]
            {
                let synced = db
                    .list(axil_docs::TABLE_DEP_MANIFESTS)
                    .map(|rows| !rows.is_empty())
                    .unwrap_or(false);
                if synced {
                    if let Some(root) = detect_project_root(&db_path) {
                        let stale = axil_docs::detect_manifests(&root)
                            .iter()
                            .filter(|m| {
                                matches!(
                                    axil_docs::manifest_drift(&db, m),
                                    Ok(d) if d.needs_sync()
                                )
                            })
                            .count();
                        if stale > 0 {
                            sections.insert(
                                "dep_docs_freshness".into(),
                                json!({
                                    "stale_manifests": stale,
                                    "recommendation": "run `axil deps refresh --if-stale`",
                                }),
                            );
                        }
                    }
                }
            }

            // Project structure and freshness from file index (if indexed)
            #[cfg(feature = "indexer")]
            {
                // Freshness check
                let project_root = detect_project_root(&db_path);
                if let Some(ref root) = project_root {
                    let idx_config = load_config(&db_path).map(|c| c.index).unwrap_or_default();
                    let freshness =
                        axil_indexer::freshness::check_freshness(&db, root, &idx_config);
                    if freshness.status != axil_indexer::freshness::FreshnessStatus::Fresh {
                        sections.insert(
                            "index_freshness".into(),
                            json!({
                                "status": freshness.status.as_str(),
                                "changed_files": freshness.changed_files,
                                "new_files": freshness.new_files,
                                "recommendation": freshness.recommendation,
                            }),
                        );
                    }
                }

                let project = db.list("_idx_project").unwrap_or_default();
                if !project.is_empty() {
                    let proj = &project[0];
                    sections.insert("project".into(), json!({
                        "name": proj.data.get("name").cloned().unwrap_or(Value::Null),
                        "type": proj.data.get("project_type").cloned().unwrap_or(Value::Null),
                        "files": proj.data.get("file_count").cloned().unwrap_or(Value::Null),
                        "modules": proj.data.get("module_count").cloned().unwrap_or(Value::Null),
                    }));

                    // Top modules by file count
                    let mut modules = db.list("_idx_modules").unwrap_or_default();
                    modules.sort_by(|a, b| {
                        let ca = a
                            .data
                            .get("file_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let cb = b
                            .data
                            .get("file_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        cb.cmp(&ca)
                    });
                    if !modules.is_empty() {
                        let mod_vals: Vec<Value> = modules.iter().take(5).map(|m| {
                            json!({
                                "path": m.data.get("path").cloned().unwrap_or(Value::Null),
                                "files": m.data.get("file_count").cloned().unwrap_or(Value::Null),
                                "summary": m.data.get("summary").cloned().unwrap_or(Value::Null),
                            })
                        }).collect();
                        sections.insert("modules".into(), json!(mod_vals));
                    }
                }
            }

            // Context-aware push: surface memories relevant to current context
            let has_context = files.is_some() || entities.is_some() || error.is_some();
            if has_context {
                let mut context_hits: Vec<Value> = Vec::new();

                // File context: find decisions/errors mentioning these files
                if let Some(ref file_list) = files {
                    let file_names: Vec<&str> = file_list.split(',').map(|s| s.trim()).collect();
                    let mut seen_ids = std::collections::HashSet::new();

                    // Pass 1: exact text matches
                    for table in &["decisions", "errors", "context"] {
                        for record in db.list(table).unwrap_or_default() {
                            let text = axil_core::util::value_text(&record.data).to_lowercase();
                            for file_name in &file_names {
                                let needle = file_name.to_lowercase();
                                let short = std::path::Path::new(file_name)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(file_name)
                                    .to_lowercase();
                                if text.contains(&needle) || text.contains(&short) {
                                    seen_ids.insert(record.id.to_string());
                                    context_hits.push(json!({
                                        "source": "file_context",
                                        "file": file_name,
                                        "table": record.table,
                                        "id": record.id.to_string(),
                                        "summary": record.data.get("summary")
                                            .or_else(|| record.data.get("error"))
                                            .cloned()
                                            .unwrap_or_else(|| truncate_value(&record.data, 150)),
                                    }));
                                    break;
                                }
                            }
                        }
                    }

                    // Pass 2: vector similarity for semantically related memories
                    if context_hits.len() < 5 && db.has_vector_index() && db.has_embedder() {
                        for file_name in &file_names {
                            let short = std::path::Path::new(file_name)
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(file_name);
                            let query = format!(
                                "{} {}",
                                short,
                                file_name.replace('/', " ").replace('_', " ")
                            );
                            if let Ok(results) = db.similar_to(&query, 3) {
                                for (record, score) in &results {
                                    if *score < 0.3 {
                                        continue;
                                    }
                                    if record.table.starts_with('_') {
                                        continue;
                                    }
                                    let id_str = record.id.to_string();
                                    if seen_ids.contains(&id_str) {
                                        continue;
                                    }
                                    seen_ids.insert(id_str.clone());
                                    context_hits.push(json!({
                                        "source": "file_context_semantic",
                                        "file": file_name,
                                        "table": record.table,
                                        "id": id_str,
                                        "summary": record.data.get("summary")
                                            .or_else(|| record.data.get("error"))
                                            .cloned()
                                            .unwrap_or_else(|| truncate_value(&record.data, 150)),
                                        "similarity": round4(*score),
                                    }));
                                }
                            }
                        }
                    }
                }

                // Entity context: find facts about mentioned entities
                if let Some(ref entity_list) = entities {
                    let entity_names: Vec<&str> =
                        entity_list.split(',').map(|s| s.trim()).collect();
                    #[cfg(feature = "memory")]
                    {
                        let mem = axil_memory::AgentMemory::new(&db);
                        for entity in &entity_names {
                            if let Ok(knowledge) = mem.semantic().about(entity) {
                                for fact in knowledge.facts.iter().take(3) {
                                    context_hits.push(json!({
                                        "source": "entity_context",
                                        "entity": entity,
                                        "fact": fact.data.get("fact").cloned().unwrap_or(Value::Null),
                                        "id": fact.id.to_string(),
                                    }));
                                }
                            }
                        }
                    }
                }

                // Error context: find similar past errors
                if let Some(ref error_text) = error {
                    let error_lower = error_text.to_lowercase();
                    let error_words: Vec<&str> = error_lower
                        .split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();
                    for record in db.list("errors").unwrap_or_default() {
                        let text = serde_json::to_string(&record.data)
                            .unwrap_or_default()
                            .to_lowercase();
                        let matching = error_words.iter().filter(|w| text.contains(**w)).count();
                        if matching >= 2 || (error_words.len() == 1 && matching == 1) {
                            context_hits.push(json!({
                                "source": "error_context",
                                "id": record.id.to_string(),
                                "error": record.data.get("error").cloned().unwrap_or(Value::Null),
                                "fix": record.data.get("fix").cloned().unwrap_or(Value::Null),
                            }));
                        }
                    }
                }

                if !context_hits.is_empty() {
                    context_hits.truncate(10);
                    sections.insert("context_push".into(), json!(context_hits));
                }
            }

            // Recent cross-agent changes from the durable semantic event log.
            // Off by default (feature-gated, write-amplifier); when enabled this
            // surfaces what other agents committed — belief revisions, decision
            // supersessions, error fixes, checkpoint writes — so boot replays
            // the delta, not just this agent's own history.
            #[cfg(feature = "event-log")]
            if db.event_log_enabled() {
                if let Ok(events) = db.recall_delta(None, None, 10) {
                    if !events.is_empty() {
                        let change_vals: Vec<Value> = events
                            .iter()
                            .map(|e| {
                                json!({
                                    "cursor": e.cursor,
                                    "kind": e.kind,
                                    "table": e.table,
                                    "record_id": e.record_id,
                                    "agent_id": e.agent_id,
                                })
                            })
                            .collect();
                        sections.insert("recent_changes".into(), json!(change_vals));
                    }
                }
            }

            // Render Extension boot_blocks first so high-signal blocks
            // (e.g. "Resume Here") land before rules / sessions / decisions.
            // Shape is `Array<{id, text}>` — a Map would silently sort
            // alphabetically (BTreeMap-backed) and break the registration-
            // order contract on `collect_extension_blocks`.
            let ext_blocks = axil_core::collect_extension_blocks(&db);
            if !ext_blocks.is_empty() {
                let blocks_arr: Vec<Value> = ext_blocks
                    .into_iter()
                    .map(|(id, text)| json!({ "id": id, "text": text }))
                    .collect();
                sections.insert("extension_blocks".into(), Value::Array(blocks_arr));
            }

            let boot_data = Value::Object(sections);

            match boot_format {
                BootFormat::Narrative => {
                    let narrative = boot_to_narrative(&boot_data);
                    let output = if let Some(max_tokens) = budget {
                        let max_bytes = max_tokens * 4;
                        if narrative.len() > max_bytes {
                            format!("{}...", &narrative[..max_bytes])
                        } else {
                            narrative
                        }
                    } else {
                        narrative
                    };
                    println!("{output}");
                }
                BootFormat::Compact => {
                    let compact = compact_boot_json(&boot_data);
                    let output = apply_token_budget(&compact, budget);
                    out.print(&output);
                }
                BootFormat::Json => {
                    let output = apply_token_budget(&boot_data, budget);
                    out.print(&output);
                }
            }
            Ok(EXIT_OK)
        }

        // ── Intent-native writes (Track B) ────────────────────────
        Command::Capture(sub) => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let source = axil_core::WriteSource::Cli;

            let result = match sub {
                CaptureCmd::Decision {
                    summary,
                    reason,
                    files,
                    agent_id,
                    external_id,
                    force_new,
                } => {
                    let file_vec = parse_comma_list(&files);
                    let file_refs: Vec<&str> = file_vec.iter().map(String::as_str).collect();
                    db.remember_decision(axil_core::DecisionInput {
                        summary: &summary,
                        reason: reason.as_deref(),
                        files: if file_refs.is_empty() {
                            None
                        } else {
                            Some(file_refs.as_slice())
                        },
                        agent_id: agent_id.as_deref(),
                        external_id: external_id.as_deref(),
                        force_new,
                        source,
                    })
                    .context("remember_decision failed")?
                }
                CaptureCmd::Error {
                    error,
                    root_cause,
                    fix,
                    files,
                    agent_id,
                    external_id,
                    force_new,
                } => {
                    let file_vec = parse_comma_list(&files);
                    let file_refs: Vec<&str> = file_vec.iter().map(String::as_str).collect();
                    db.remember_error(axil_core::ErrorInput {
                        error: &error,
                        root_cause: root_cause.as_deref(),
                        fix: fix.as_deref(),
                        files: if file_refs.is_empty() {
                            None
                        } else {
                            Some(file_refs.as_slice())
                        },
                        agent_id: agent_id.as_deref(),
                        external_id: external_id.as_deref(),
                        force_new,
                        source,
                    })
                    .context("remember_error failed")?
                }
            };

            out.print(&json!({
                "id": result.id.to_string(),
                "is_new": result.is_new,
                "superseded": result.superseded.iter().map(ToString::to_string).collect::<Vec<_>>(),
            }));
            Ok(EXIT_OK)
        }

        Command::Prefer { key, value } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            // Treat the raw CLI string as JSON when parseable so callers can
            // store numbers, objects, and arrays; fall back to a string.
            let parsed =
                serde_json::from_str::<serde_json::Value>(&value).unwrap_or_else(|_| json!(value));
            let result = db
                .set_preference(&key, parsed)
                .context("set_preference failed")?;
            out.print(&json!({
                "id": result.id.to_string(),
                "is_new": result.is_new,
                "key": key,
            }));
            Ok(EXIT_OK)
        }

        Command::CloseSession { id, summary } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let result = db
                .close_session(&id, summary.as_deref())
                .context("close_session failed")?;
            out.print(&json!({
                "id": result.id.to_string(),
                "session_id": id,
                "is_new": result.is_new,
            }));
            Ok(EXIT_OK)
        }

        // ── Beliefs ───────────────────────────────────────────────
        Command::Beliefs {
            topic,
            all,
            generate,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let bs = axil_core::beliefs::BeliefSystem::new(&db);

            if generate {
                let new = bs.auto_generate().context("failed to generate beliefs")?;
                out.print(&json!({
                    "generated": new.len(),
                    "beliefs": new.iter().map(|r| {
                        json!({
                            "id": r.id.to_string(),
                            "statement": r.data.get("statement").cloned().unwrap_or(Value::Null),
                        })
                    }).collect::<Vec<_>>(),
                }));
            } else {
                let beliefs = bs
                    .list(topic.as_deref(), all)
                    .context("failed to list beliefs")?;
                let values: Vec<Value> = beliefs
                    .iter()
                    .map(|b| {
                        json!({
                            "id": b.id,
                            "statement": b.statement,
                            "confidence": b.confidence,
                            "source": b.source,
                            "doubted": b.doubted,
                        })
                    })
                    .collect();
                out.print_array(&values);
            }
            Ok(EXIT_OK)
        }

        Command::Believe { statement } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let bs = axil_core::beliefs::BeliefSystem::new(&db);
            let record = bs.believe(&statement).context("failed to store belief")?;
            out.print(&json!({
                "id": record.id.to_string(),
                "statement": statement,
                "confidence": 1.0,
            }));
            Ok(EXIT_OK)
        }

        Command::Doubt { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let bs = axil_core::beliefs::BeliefSystem::new(&db);
            let rid = RecordId::from_string(&id).context("invalid record ID")?;
            bs.doubt(&rid).context("failed to doubt belief")?;
            out.print(&json!({"id": id, "doubted": true}));
            Ok(EXIT_OK)
        }

        // ── Auto-capture ──────────────────────────────────────────
        Command::AutoCapture {
            text,
            dry_run,
            min_confidence,
            source,
        } => {
            let input = if text == "-" {
                let mut buf = String::new();
                io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)
                    .context("failed to read from stdin")?;
                buf
            } else {
                text
            };

            let captures = axil_core::auto_capture::analyze(&input, &source);
            let actionable: Vec<_> = captures
                .iter()
                .filter(|c| c.confidence >= min_confidence)
                .collect();

            if dry_run || actionable.is_empty() {
                let all_json: Vec<Value> = captures
                    .iter()
                    .map(|c| {
                        json!({
                            "type": c.capture_type,
                            "summary": c.summary,
                            "confidence": c.confidence,
                            "would_store": c.confidence >= min_confidence,
                        })
                    })
                    .collect();
                out.print(&json!({
                    "dry_run": dry_run,
                    "captures": all_json,
                    "actionable": actionable.len(),
                }));
            } else {
                let db_path = require_db(&db_opt)?;
                let db = open_with_all_detected(&db_path)?;
                let mut stored = Vec::new();
                for cap in &actionable {
                    let (table, data) = axil_core::auto_capture::capture_to_record(cap);
                    match db.insert(&table, data) {
                        Ok(record) => stored.push(json!({
                            "id": record.id.to_string(),
                            "table": table,
                            "type": cap.capture_type,
                            "summary": cap.summary,
                        })),
                        Err(e) => eprintln!("  [skip] failed to store: {e}"),
                    }
                }
                out.print(&json!({
                    "stored": stored.len(),
                    "records": stored,
                }));
            }
            Ok(EXIT_OK)
        }

        // ── Brain banner ──────────────────────────────────────────
        Command::BrainBanner { style } => {
            let db_path = require_db(&db_opt)?;
            let db = Axil::open(&db_path)
                .build()
                .context("failed to open database")?;
            let config = load_config(&db_path).unwrap_or_default();

            let banner_style = style.as_deref().unwrap_or(&config.brain.banner);

            // Gather counts
            let decisions = db.list("decisions").map(|r| r.len()).unwrap_or(0);
            let errors = db.list("errors").map(|r| r.len()).unwrap_or(0);
            let sessions = db
                .storage()
                .list("_sessions", usize::MAX, 0)
                .map(|r| r.len())
                .unwrap_or(0);
            let beliefs = db.list("_beliefs").map(|r| r.len()).unwrap_or(0);

            let banner = match banner_style {
                "box" => format!(
                    "┌─────────────────────────────┐\n\
                     │ 🧠 AXIL BRAIN               │\n\
                     │ Session context loaded       │\n\
                     └─────────────────────────────┘"
                ),
                "ascii" => format!(
                    "    ╭───────────────────╮\n\
                     🧠 │  A X I L  B R A I N │\n\
                     ╰───────────────────╯"
                ),
                "status" => format!(
                    "🧠 axil brain\n\
                     ∙ {decisions} decisions ∙ {errors} errors ∙ {sessions} sessions ∙ {beliefs} beliefs"
                ),
                "bold" => format!(
                    "━━━ 🧠 AXIL BRAIN ━━━━━━━━━━━━━━━━━━━━\n\
                       {decisions} decisions │ {errors} errors │ {sessions} sessions\n\
                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                _ => format!(
                    "🧠 axil brain ∙ {decisions} decisions ∙ {errors} errors ∙ {sessions} sessions"
                ),
            };

            eprintln!("{banner}");
            Ok(EXIT_OK)
        }

        // ── Recall for entity ──────────────────────────
        #[cfg(feature = "scip")]
        Command::RecallForEntity {
            entity,
            depth,
            edge_types,
            scopes,
            trace_graph,
            top_k,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            // Resolve the entity input. If it already looks like a SCIP
            // symbol (contains spaces — SCIP symbols have them by design)
            // or a `provisional:` id, treat as canonical. Otherwise resolve
            // via the alias table.
            let looks_canonical = entity.contains(' ') || entity.starts_with("provisional:");
            let canonical_id = if looks_canonical {
                entity.clone()
            } else {
                let mut walked: Vec<&str> = scopes.iter().map(String::as_str).collect();
                if !walked.iter().any(|s| *s == "global") {
                    walked.push("global");
                }
                db.resolve_entity_alias(&entity, &walked)?
                    .unwrap_or_else(|| axil_core::entity::natural_canonical_id(&entity))
            };

            // Locate the `_entities` row.
            let entities = db.list("_entities").unwrap_or_default();
            let entity_row = entities.into_iter().find(|r| {
                r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(canonical_id.as_str())
            });
            let Some(entity_row) = entity_row else {
                out.print(&json!({
                    "ok": true,
                    "entity": entity,
                    "canonical_id": canonical_id,
                    "matches": 0,
                    "results": [],
                    "note": "no entity row found — is SCIP ingested?",
                }));
                return Ok(EXIT_OK);
            };

            let traversal_types: Vec<&str> = match edge_types.as_deref() {
                Some(csv) => csv.split(',').map(str::trim).collect(),
                None => vec![
                    axil_scip::EDGE_CALLS,
                    axil_scip::EDGE_REFERENCES,
                    axil_scip::EDGE_IMPLEMENTS,
                    axil_scip::EDGE_TYPE_OF,
                ],
            };

            let Some(gi) = db.graph_index_ref() else {
                out.print(&json!({
                    "ok": false,
                    "error": "graph index not available — enable the `graph` feature",
                }));
                return Ok(EXIT_OK);
            };

            // BFS out to `depth` hops, collecting visited entity ids +
            // per-hop trace when requested.
            let mut frontier: Vec<(axil_core::RecordId, usize, String, String)> = vec![(
                entity_row.id.clone(),
                0,
                "seed".to_string(),
                "direct".to_string(),
            )];
            let mut visited: std::collections::HashSet<axil_core::RecordId> =
                [entity_row.id.clone()].into_iter().collect();
            let mut trace: Vec<Value> = Vec::new();
            let mut collected: Vec<axil_core::RecordId> = vec![entity_row.id.clone()];

            while let Some((nid, d, via_edge, confidence)) = frontier.pop() {
                if trace_graph {
                    trace.push(json!({
                        "entity_id": nid.to_string(),
                        "depth": d,
                        "via": via_edge,
                        "confidence": confidence,
                        "layer": "_entities",
                    }));
                }
                if d >= depth {
                    continue;
                }
                for etype in &traversal_types {
                    if let Ok(edges) =
                        gi.edges(nid.clone(), Some(*etype), axil_core::Direction::Both)
                    {
                        for e in edges {
                            let other = if e.from == nid {
                                e.to.clone()
                            } else {
                                e.from.clone()
                            };
                            if visited.insert(other.clone()) {
                                let conf = e
                                    .properties
                                    .get("confidence")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("direct")
                                    .to_string();
                                frontier.push((other.clone(), d + 1, etype.to_string(), conf));
                                collected.push(other);
                            }
                        }
                    }
                }
            }

            // For every collected entity, follow `mentions` inbound to
            // real memory records.
            let mut hits: Vec<Value> = Vec::new();
            let mut seen_records = std::collections::HashSet::new();
            for eid in &collected {
                let Ok(mentions) = gi.edges(
                    eid.clone(),
                    Some(axil_core::util::edge_types::MENTIONS),
                    axil_core::Direction::In,
                ) else {
                    continue;
                };
                for edge in mentions {
                    let rid_str = edge.from.to_string();
                    if !seen_records.insert(rid_str.clone()) {
                        continue;
                    }
                    let Ok(Some(record)) = db.get(&edge.from) else {
                        continue;
                    };
                    if record.table.starts_with('_') {
                        continue;
                    }
                    hits.push(json!({
                        "table": record.table,
                        "id": rid_str,
                        "summary": record.data.get("summary")
                            .or_else(|| record.data.get("error"))
                            .or_else(|| record.data.get("fact"))
                            .cloned()
                            .unwrap_or_else(|| truncate_value(&record.data, 150)),
                        "importance": axil_core::importance::get_importance(&record.data),
                        "source": "entity_graph",
                    }));
                    if hits.len() >= top_k {
                        break;
                    }
                }
                if hits.len() >= top_k {
                    break;
                }
            }

            // Pass-4: follow `_entity_bridges` to sibling
            // members where the bridge confidence clears the workspace's
            // `federation.min_bridge_confidence` threshold. Remote
            // mentions are pulled with provenance tags. No-op when no
            // workspace manifest exists — solo-DB behavior byte-for-byte.
            let mut bridged_hops: Vec<Value> = Vec::new();
            if hits.len() < top_k {
                match follow_bridges_for_entity(
                    &db,
                    &db_path,
                    &canonical_id,
                    top_k - hits.len(),
                    trace_graph,
                ) {
                    Ok((mut bridged_hits, hops)) => {
                        for h in bridged_hits.drain(..) {
                            let id_str = h
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if !id_str.is_empty() && !seen_records.insert(id_str) {
                                continue;
                            }
                            hits.push(h);
                            if hits.len() >= top_k {
                                break;
                            }
                        }
                        bridged_hops = hops;
                    }
                    Err(e) => eprintln!("bridge traversal skipped: {e}"),
                }
            }

            let mut result = json!({
                "ok": true,
                "entity": entity,
                "canonical_id": canonical_id,
                "matches": hits.len(),
                "results": hits,
                "entities_walked": collected.len(),
                "edge_types": traversal_types,
            });
            if trace_graph {
                if let Some(obj) = result.as_object_mut() {
                    obj.insert("trace".to_string(), Value::Array(trace));
                    if !bridged_hops.is_empty() {
                        obj.insert("bridges".to_string(), Value::Array(bridged_hops));
                    }
                }
            }
            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Recall for file ────────────────────────────────────────
        Command::RecallForFile { file, top_k } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            let file_lower = file.to_lowercase();
            let short_name = std::path::Path::new(&file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&file)
                .to_lowercase();

            let mut seen_ids = std::collections::HashSet::new();
            let mut hits: Vec<Value> = Vec::new();

            // Pass 1: exact text matches (fast, precise)
            for table in &["decisions", "errors", "context"] {
                for record in db.list(table).unwrap_or_default() {
                    let text = axil_core::util::value_text(&record.data).to_lowercase();
                    if text.contains(&file_lower) || text.contains(&short_name) {
                        seen_ids.insert(record.id.to_string());
                        hits.push(json!({
                            "table": record.table,
                            "id": record.id.to_string(),
                            "summary": record.data.get("summary")
                                .or_else(|| record.data.get("error"))
                                .or_else(|| record.data.get("fact"))
                                .cloned()
                                .unwrap_or_else(|| truncate_value(&record.data, 150)),
                            "importance": axil_core::importance::get_importance(&record.data),
                            "match": "exact",
                        }));
                    }
                }
            }

            // Pass 2: vector similarity (semantic, finds related memories)
            if hits.len() < top_k && db.has_vector_index() && db.has_embedder() {
                let query = format!(
                    "file {} {}",
                    short_name,
                    file.replace('/', " ").replace('_', " ")
                );
                if let Ok(results) = db.similar_to(&query, top_k * 2) {
                    for (record, score) in &results {
                        let id_str = record.id.to_string();
                        if seen_ids.contains(&id_str) {
                            continue;
                        }
                        if record.table.starts_with('_') {
                            continue;
                        }
                        if *score < 0.3 {
                            continue;
                        }
                        seen_ids.insert(id_str.clone());
                        hits.push(json!({
                            "table": record.table,
                            "id": id_str,
                            "summary": record.data.get("summary")
                                .or_else(|| record.data.get("error"))
                                .or_else(|| record.data.get("fact"))
                                .cloned()
                                .unwrap_or_else(|| truncate_value(&record.data, 150)),
                            "importance": axil_core::importance::get_importance(&record.data),
                            "match": "semantic",
                            "similarity": round4(*score),
                        }));
                    }
                }
            }

            // Sort: exact matches first, then by importance
            hits.sort_by(|a, b| {
                let a_exact = a["match"].as_str() == Some("exact");
                let b_exact = b["match"].as_str() == Some("exact");
                if a_exact != b_exact {
                    return b_exact.cmp(&a_exact);
                }
                let ia = a["importance"].as_f64().unwrap_or(0.0);
                let ib = b["importance"].as_f64().unwrap_or(0.0);
                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Pass 3: related files from index graph (imports, same module)
            #[cfg(feature = "indexer")]
            {
                let idx_files = db.list("_idx_files").unwrap_or_default();
                // Find the indexed file record matching this path
                let idx_match = idx_files.iter().find(|r| {
                    r.data
                        .get("path")
                        .and_then(|v| v.as_str())
                        .map(|p| p == file || p.ends_with(&short_name))
                        .unwrap_or(false)
                });
                if let Some(idx_record) = idx_match {
                    // Get graph neighbors (files that import/are imported by this file)
                    if db.has_graph_index() {
                        if let Ok(neighbors) =
                            db.neighbors(&idx_record.id, None, axil_core::Direction::Both)
                        {
                            let related: Vec<Value> = neighbors.iter()
                                .filter(|n| n.table == "_idx_files")
                                .take(5)
                                .map(|n| json!({
                                    "path": n.data.get("path").cloned().unwrap_or(Value::Null),
                                    "language": n.data.get("language").cloned().unwrap_or(Value::Null),
                                    "summary": n.data.get("summary").cloned().unwrap_or(Value::Null),
                                }))
                                .collect();
                            if !related.is_empty() {
                                hits.push(json!({
                                    "match": "related_files",
                                    "file": file,
                                    "related": related,
                                }));
                            }
                        }
                    }
                    // Include the file's own summary from the index
                    if let Some(summary) = idx_record.data.get("summary").and_then(|v| v.as_str()) {
                        if !summary.is_empty() {
                            hits.push(json!({
                                "match": "file_index",
                                "file": file,
                                "summary": summary,
                                "imports": idx_record.data.get("imports").cloned().unwrap_or(json!([])),
                                "exports": idx_record.data.get("exports").cloned().unwrap_or(json!([])),
                            }));
                        }
                    }
                }

                // Impact analysis: what breaks if this file changes
                let impact = axil_indexer::impact::impact(&db, &file);
                if let Ok(report) = impact {
                    if !report.direct_dependents.is_empty() {
                        hits.push(json!({
                            "match": "impact_analysis",
                            "risk": report.risk,
                            "direct_dependents": report.direct_dependents,
                            "transitive_dependents": report.transitive_dependents,
                            "affected_modules": report.affected_modules,
                        }));
                    }
                }
            }

            // Pass 4: entity-level neighbors via SCIP edges.
            // For each entity whose `defined_in` edge points at this file,
            // surface memories attached to callers/callees/references.
            #[cfg(feature = "scip")]
            if db.has_graph_index() {
                let mut pass4 = scip_entity_pass(&db, &file, &short_name, &mut seen_ids, top_k);
                hits.append(&mut pass4);
            }

            // Pass 5: dependency docs for the file's imports.
            // Scans the file's import statements; for each that maps to a
            // synced dependency, surfaces a representative doc chunk.
            #[cfg(feature = "deps")]
            {
                if let Some(root) = detect_project_root(&db_path) {
                    let imports = imports_in_file(&root.join(&file));
                    if !imports.is_empty() {
                        let synced: std::collections::HashSet<String> = db
                            .list(axil_docs::TABLE_DEPS)
                            .unwrap_or_default()
                            .iter()
                            .filter_map(|r| {
                                r.data
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .map(String::from)
                            })
                            .collect();
                        let mut surfaced = std::collections::HashSet::new();
                        for import in imports {
                            if !synced.contains(&import) || !surfaced.insert(import.clone()) {
                                continue;
                            }
                            if let Ok(chunks) =
                                axil_docs::query_dep_docs(&db, &import, 1, Some(&import), false)
                            {
                                if let Some(top) = chunks.into_iter().next() {
                                    hits.push(json!({
                                        "match": "dep_docs",
                                        "dep": top.dep_name,
                                        "version": top.dep_version,
                                        "section": top.section_path,
                                        "content": top.content,
                                    }));
                                }
                            }
                        }
                    }
                }
            }

            hits.truncate(top_k);

            out.print(&json!({
                "file": file,
                "matches": hits.len(),
                "results": hits,
            }));
            Ok(EXIT_OK)
        }

        // ── Observe (unified write entry point) ──────────
        Command::Observe {
            text,
            stdin,
            file,
            source,
            source_ref,
            scope,
            kind,
            table,
            hints,
            agent,
            format,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;

            // Read text from the appropriate input.
            let obs_text = if stdin {
                let mut buf = String::new();
                io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)
                    .context("failed to read from stdin")?;
                if buf.len() as u64 > MAX_STDIN_BYTES {
                    return Err(anyhow::anyhow!(
                        "stdin exceeds {} MB cap",
                        MAX_STDIN_BYTES / (1024 * 1024)
                    ));
                }
                buf
            } else if let Some(ref path) = file {
                std::fs::read_to_string(path)
                    .with_context(|| format!("failed to read file: {path}"))?
            } else if let Some(ref t) = text {
                t.clone()
            } else {
                anyhow::bail!("provide text, --stdin, or --file");
            };

            let mem_source = axil_core::MemorySource::parse(&source);

            // Build observation.
            let mut obs = axil_core::Observation::from_text(obs_text).with_source(mem_source);

            if let Some(ref sr) = source_ref {
                obs = obs.with_source_ref(sr.as_str());
            }
            if let Some(ref s) = scope {
                if let Some(parsed) = axil_core::MemoryScope::parse(s) {
                    obs = obs.with_scope(parsed);
                } else {
                    anyhow::bail!(
                        "invalid scope: {s}. Valid: session, agent, project, user, global"
                    );
                }
            }
            if let Some(ref k) = kind {
                obs = obs.with_hint(k.as_str());
            }
            if let Some(ref t) = table {
                obs = obs.with_table(t.as_str());
            }
            if let Some(ref h) = hints {
                for hint in h.split(',') {
                    obs = obs.with_hint(hint.trim());
                }
            }
            if let Some(ref a) = agent {
                obs = obs.with_agent(a.as_str());
            }

            // Run the decision pipeline.
            let outcome = axil_core::remember(&db, obs).context("decision pipeline failed")?;

            match format.as_str() {
                "quiet" => {
                    // Just print the action and record ID.
                    match &outcome.action {
                        axil_core::PipelineAction::Stored => {
                            if let Some(ref r) = outcome.record {
                                println!("{}", r.id);
                            }
                        }
                        axil_core::PipelineAction::Updated { existing_id } => {
                            println!("updated:{existing_id}");
                        }
                        axil_core::PipelineAction::Superseded { old_id } => {
                            if let Some(ref r) = outcome.record {
                                println!("superseded:{old_id}->{}", r.id);
                            }
                        }
                        axil_core::PipelineAction::Ignored => {
                            eprintln!("ignored: {}", outcome.reason);
                        }
                    }
                }
                "verbose" => {
                    out.print(&json!({
                        "action": outcome.action,
                        "record_id": outcome.record.as_ref().map(|r| r.id.to_string()),
                        "memory_type": outcome.memory_type.to_string(),
                        "scope": outcome.scope.to_string(),
                        "importance": outcome.importance,
                        "confidence": outcome.confidence,
                        "entities": outcome.entities.iter().map(|e| json!({
                            "name": e.name,
                            "type": e.entity_type,
                        })).collect::<Vec<_>>(),
                        "reason": outcome.reason,
                        "latency_us": outcome.latency_us,
                    }));
                }
                _ => {
                    // Default JSON: compact.
                    out.print(&json!({
                        "action": outcome.action,
                        "id": outcome.record.as_ref().map(|r| r.id.to_string()),
                        "type": outcome.memory_type.to_string(),
                        "scope": outcome.scope.to_string(),
                        "importance": outcome.importance,
                        "reason": outcome.reason,
                    }));
                }
            }
            Ok(EXIT_OK)
        }

        // ── Inspect memory ──────────────────────────────────
        Command::InspectMemory { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let record = db
                .get(&record_id)?
                .ok_or_else(|| anyhow::anyhow!("record not found: {id}"))?;

            // Build provenance view.
            let source_kind = record
                .data
                .get("_source")
                .and_then(|s| s.get("kind"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let source_ref = record
                .data
                .get("_source")
                .and_then(|s| s.get("ref"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let scope = record
                .data
                .get("_scope")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let confidence = record
                .data
                .get("_confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let importance = axil_core::importance::get_importance(&record.data);
            let memory_type = record
                .data
                .get("_memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let superseded = record
                .data
                .get("_superseded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let supersedes = record.data.get("_supersedes").cloned();
            let contradicts = record.data.get("_contradicts").cloned();
            let entities = record.data.get("_entities").cloned();

            let trust_tier = axil_core::classify_trust(&record.data).to_string();

            out.print(&json!({
                "id": id,
                "table": record.table,
                "created_at": format_dt(&record.created_at),
                "updated_at": format_dt(&record.updated_at),
                "provenance": {
                    "source_kind": source_kind,
                    "source_ref": source_ref,
                    "scope": scope,
                    "memory_type": memory_type,
                    "confidence": confidence,
                    "importance": importance,
                    "trust_tier": trust_tier,
                },
                "links": {
                    "superseded": superseded,
                    "supersedes": supersedes,
                    "contradicts": contradicts,
                },
                "entities": entities,
                "data": record.data,
            }));
            Ok(EXIT_OK)
        }

        // ── Trace memory ────────────────────────────────────
        Command::TraceMemory { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let record = db
                .get(&record_id)?
                .ok_or_else(|| anyhow::anyhow!("record not found: {id}"))?;

            // BFS walk of the full supersedes DAG (a record may supersede
            // multiple ancestors — keeping only the first would drop branches
            // of the provenance chain). Visited set prevents cycles; the hop
            // bound caps cost on pathological histories.
            const MAX_HOPS: usize = 10;
            let mut chain = Vec::new();
            let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut frontier: Vec<axil_core::Record> = vec![record];
            let mut depth = 0;

            while !frontier.is_empty() && depth <= MAX_HOPS {
                let mut next_frontier: Vec<axil_core::Record> = Vec::new();
                for current in frontier.drain(..) {
                    let id_str = current.id.to_string();
                    if !visited.insert(id_str.clone()) {
                        continue;
                    }
                    chain.push(json!({
                        "id": id_str,
                        "depth": depth,
                        "summary": current.data.get("summary")
                            .or_else(|| current.data.get("fact"))
                            .or_else(|| current.data.get("statement"))
                            .cloned()
                            .unwrap_or_else(|| json!("(no summary)")),
                        "source": current.data.get("_source").cloned().unwrap_or(json!(null)),
                        "created_at": format_dt(&current.created_at),
                        "superseded": current.data.get("_superseded").and_then(|v| v.as_bool()).unwrap_or(depth > 0),
                    }));

                    if depth == MAX_HOPS {
                        continue;
                    }
                    if let Some(arr) = current.data.get("_supersedes").and_then(|v| v.as_array()) {
                        for prev_val in arr {
                            let Some(prev_id) = prev_val.as_str() else {
                                continue;
                            };
                            if visited.contains(prev_id) {
                                continue;
                            }
                            let Ok(rid) = axil_core::RecordId::from_string(prev_id) else {
                                continue;
                            };
                            if let Ok(Some(prev)) = db.get(&rid) {
                                next_frontier.push(prev);
                            }
                        }
                    }
                }
                frontier = next_frontier;
                depth += 1;
            }

            out.print(&json!({
                "record_id": id,
                "chain_length": chain.len(),
                "chain": chain,
            }));
            Ok(EXIT_OK)
        }

        // ── Brain mode ───────────────────────────────────
        Command::Brain(cmd) => {
            let db_path = require_db(&db_opt)?;
            match cmd {
                BrainCommand::Enable => {
                    let config_path = axil_core::find_config_file(&db_path).unwrap_or_else(|| {
                        db_path
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .join("axil.toml")
                    });
                    axil_core::set_config_value(&config_path, "brain.enabled", "true")
                        .map_err(|e| anyhow::anyhow!(e))?;
                    out.print(
                        &json!({"brain_mode": "enabled", "config": config_path.to_string_lossy()}),
                    );
                    Ok(EXIT_OK)
                }
                BrainCommand::Disable => {
                    let config_path = axil_core::find_config_file(&db_path).unwrap_or_else(|| {
                        db_path
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .join("axil.toml")
                    });
                    axil_core::set_config_value(&config_path, "brain.enabled", "false")
                        .map_err(|e| anyhow::anyhow!(e))?;
                    out.print(&json!({"brain_mode": "disabled"}));
                    Ok(EXIT_OK)
                }
                BrainCommand::Status => {
                    let db = open_with_all_detected(&db_path)?;
                    let config = load_config(&db_path).unwrap_or_default();

                    let beliefs_records = db.list("_beliefs").context("read _beliefs")?;
                    let beliefs = beliefs_records.len();
                    let doubted = beliefs_records
                        .iter()
                        .filter(|r| {
                            r.data
                                .get("doubted")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                        })
                        .count();
                    let decisions = db.list("decisions").context("read decisions")?.len();
                    let errors = db.list("errors").context("read errors")?.len();
                    let sessions = db
                        .storage()
                        .list("_sessions", usize::MAX, 0)
                        .context("read _sessions")?
                        .len();
                    let self_notes = db.list("_self_memory").context("read _self_memory")?.len();
                    let project_model = db
                        .list("_project_model")
                        .context("read _project_model")?
                        .len();
                    let user_rules = db
                        .list("_user_contract")
                        .context("read _user_contract")?
                        .len();

                    let policy =
                        axil_core::brain::memory_policy_show(&db).context("read memory policy")?;

                    out.print(&json!({
                        "brain_mode": config.brain.enabled,
                        "banner_style": config.brain.banner,
                        "memory": {
                            "beliefs": beliefs,
                            "beliefs_doubted": doubted,
                            "decisions": decisions,
                            "errors": errors,
                            "sessions": sessions,
                        },
                        "brain_features": {
                            "self_notes": self_notes,
                            "project_model_entries": project_model,
                            "user_contract_rules": user_rules,
                        },
                        "safety": {
                            "pinned_records": policy.get("pinned_records").and_then(|v| v.as_u64()).unwrap_or(0),
                            "redacted_records": policy.get("redacted_records").and_then(|v| v.as_u64()).unwrap_or(0),
                            "retention_policies": policy.get("retention_policies").cloned().unwrap_or(json!([])),
                        },
                    }));
                    Ok(EXIT_OK)
                }
                BrainCommand::Reflect => {
                    let db = open_with_all_detected(&db_path)?;
                    let worker = axil_core::AxilWorker::new(&db).with_brain();
                    let report = worker.run().context("brain reflection failed")?;
                    out.print(&json!({
                        "reflected": true,
                        "consolidated_entities": report.consolidated_entities,
                        "stale_beliefs": report.stale_beliefs,
                        "candidate_procedures": report.candidate_procedures,
                        "candidate_preferences": report.candidate_preferences,
                        "duplicate_clusters": report.duplicate_clusters,
                        "duration_ms": report.duration_ms,
                    }));
                    Ok(EXIT_OK)
                }
                BrainCommand::Debug { id } => {
                    let db = open_with_all_detected(&db_path)?;
                    let record_id =
                        axil_core::RecordId::from_string(&id).context("invalid record ID")?;

                    // Combine why-remembered + why-revised + provenance.
                    let remembered = axil_core::why_remembered(&db, &record_id)
                        .context("failed to explain memory")?;
                    let revised = axil_core::why_revised(&db, &record_id)
                        .context("failed to explain revision")?;
                    let record = db
                        .get(&record_id)?
                        .ok_or_else(|| anyhow::anyhow!("record not found"))?;
                    let provenance = axil_core::extract_provenance(&record.data);

                    out.print(&json!({
                        "id": id,
                        "remembered": serde_json::to_value(&remembered).ok(),
                        "revised": serde_json::to_value(&revised).ok(),
                        "provenance": serde_json::to_value(&provenance).ok(),
                    }));
                    Ok(EXIT_OK)
                }
                BrainCommand::Eval => {
                    let report = run_brain_eval_scratch()?;
                    out.print(&report);
                    Ok(EXIT_OK)
                }
            }
        }

        // ── Brain eval ────────────────────────────────────
        Command::BrainEval => {
            let report = run_brain_eval_scratch()?;
            out.print(&report);
            Ok(EXIT_OK)
        }

        // ── Redact ───────────────────────────────────────
        Command::Redact { id, field } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let _ = axil_core::brain::redact_field(&db, &record_id, &field)
                .context("failed to redact field")?;
            out.print(&json!({"id": id, "field": field, "redacted": true}));
            Ok(EXIT_OK)
        }

        // ── Retention ─────────────────────────────────────
        Command::Retention { command } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match command {
                RetentionCommand::Set { scope, days } => {
                    let _ = axil_core::brain::set_retention(&db, &scope, days)
                        .context("failed to set retention")?;
                    out.print(&json!({"scope": scope, "days": days, "set": true}));
                }
                RetentionCommand::Show => {
                    let policies = axil_core::brain::get_retention_policies(&db)
                        .context("failed to get retention policies")?;
                    out.print(&policies);
                }
            }
            Ok(EXIT_OK)
        }

        // ── Pin/Unpin ─────────────────────────────────────
        Command::Pin { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let _ =
                axil_core::brain::pin_record(&db, &record_id).context("failed to pin record")?;
            out.print(&json!({"id": id, "pinned": true}));
            Ok(EXIT_OK)
        }

        Command::Unpin { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let _ = axil_core::brain::unpin_record(&db, &record_id)
                .context("failed to unpin record")?;
            out.print(&json!({"id": id, "pinned": false}));
            Ok(EXIT_OK)
        }

        // ── Memory policy ─────────────────────────────────
        Command::MemoryPolicy => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let policy = axil_core::brain::memory_policy_show(&db)
                .context("failed to show memory policy")?;
            out.print(&policy);
            Ok(EXIT_OK)
        }

        // ── Self memory ──────────────────────────────────
        Command::SelfMemory(cmd) => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match cmd {
                SelfCommand::Note { text, category } => {
                    let record = axil_core::self_note(&db, &text, Some(&category))
                        .context("failed to store self note")?;
                    out.print(&json!({"id": record.id.to_string(), "stored": true}));
                }
                SelfCommand::Profile => {
                    let profile =
                        axil_core::self_profile(&db).context("failed to get self profile")?;
                    out.print(&profile);
                }
            }
            Ok(EXIT_OK)
        }

        // ── Project model ─────────────────────────────────
        Command::ProjectModel(cmd) => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match cmd {
                ProjectModelCommand::Set { key, value } => {
                    let record = axil_core::project_model_set(&db, &key, &value)
                        .context("failed to set project model entry")?;
                    out.print(&json!({"id": record.id.to_string(), "key": key, "stored": true}));
                }
                ProjectModelCommand::Show => {
                    let model = axil_core::project_model_show(&db)
                        .context("failed to show project model")?;
                    out.print(&model);
                }
                ProjectModelCommand::Generate => {
                    let generated = axil_core::project_model_generate(&db)
                        .context("failed to generate project model")?;
                    out.print(&json!({"generated": generated.len()}));
                }
            }
            Ok(EXIT_OK)
        }

        // ── User contract ─────────────────────────────────
        Command::UserContract(cmd) => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match cmd {
                UserContractCommand::Add { rule } => {
                    let record = axil_core::user_contract_set(&db, &rule)
                        .context("failed to add user contract rule")?;
                    out.print(&json!({"id": record.id.to_string(), "stored": true}));
                }
                UserContractCommand::List => {
                    let rules = axil_core::user_contract_list(&db)
                        .context("failed to list user contract")?;
                    out.print(&json!({"count": rules.len(), "rules": rules}));
                }
            }
            Ok(EXIT_OK)
        }

        // ── Why remembered ──────────────────────────────────
        Command::WhyRemembered { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let explanation =
                axil_core::why_remembered(&db, &record_id).context("failed to explain memory")?;
            out.print(&serde_json::to_value(&explanation).unwrap_or(json!(null)));
            Ok(EXIT_OK)
        }

        // ── Why recalled ────────────────────────────────────
        Command::WhyRecalled { id, query } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let explanation = axil_core::why_recalled(&db, &query, &record_id)
                .context("failed to explain recall")?;
            out.print(&serde_json::to_value(&explanation).unwrap_or(json!(null)));
            Ok(EXIT_OK)
        }

        // ── Why revised ─────────────────────────────────────
        Command::WhyRevised { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let explanation =
                axil_core::why_revised(&db, &record_id).context("failed to explain revision")?;
            out.print(&serde_json::to_value(&explanation).unwrap_or(json!(null)));
            Ok(EXIT_OK)
        }

        // ── Revise beliefs ───────────────────────────────────
        Command::ReviseBeliefs { text } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let result = axil_core::revise_beliefs(&db, &text).context("belief revision failed")?;
            out.print(&json!({
                "actions": result.actions,
                "updated_count": result.updated_beliefs.len(),
                "doubted_count": result.doubted_beliefs.len(),
            }));
            Ok(EXIT_OK)
        }

        // ── Belief history ──────────────────────────────────
        Command::BeliefHistory { topic } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let beliefs =
                axil_core::belief_history(&db, &topic).context("belief history lookup failed")?;
            let values: Vec<Value> = beliefs
                .iter()
                .map(|b| {
                    json!({
                        "id": b.id,
                        "statement": b.statement,
                        "confidence": b.confidence,
                        "source": b.source,
                        "doubted": b.doubted,
                        "created_at": b.created_at,
                    })
                })
                .collect();
            out.print(&json!({
                "topic": topic,
                "count": values.len(),
                "beliefs": values,
            }));
            Ok(EXIT_OK)
        }

        // ── Verify ───────────────────────────────────────────
        Command::Verify { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = axil_core::RecordId::from_string(&id).context("invalid record ID")?;
            let updated =
                axil_core::verify_record(&db, &record_id).context("failed to verify record")?;
            out.print(&json!({
                "id": id,
                "verified": true,
                "trust_tier": axil_core::classify_trust(&updated.data).to_string(),
            }));
            Ok(EXIT_OK)
        }

        // ── Migrate provenance ──────────────────────────────
        Command::MigrateProvenance => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let count =
                axil_core::migrate_provenance_all(&db).context("provenance migration failed")?;
            out.print(&json!({
                "migrated": count,
                "status": if count > 0 { "updated" } else { "already_current" },
            }));
            Ok(EXIT_OK)
        }

        // ── MCP server ─────────────────────────────────────────────
        #[cfg(feature = "mcp")]
        Command::Mcp {
            command: Some(McpCommand::Install { target, dry_run }),
            ..
        } => mcp_install(&std::env::current_dir()?, &target, dry_run, out),
        #[cfg(feature = "mcp")]
        Command::Mcp {
            command: None,
            otel_endpoint,
        } => {
            // Fail loudly if --otel-endpoint is used without the otel feature.
            #[cfg(not(feature = "otel"))]
            if otel_endpoint.is_some() {
                anyhow::bail!(
                    "--otel-endpoint requires the `otel` feature. \
                     Rebuild with: cargo build --features otel"
                );
            }

            // Initialize OpenTelemetry if endpoint is provided.
            let _otel_guard = if let Some(ref endpoint) = otel_endpoint {
                out.status(&format!("OpenTelemetry export → {endpoint}"));
                Some(
                    axil_core::otel::init_otel(endpoint)
                        .map_err(|e| anyhow::anyhow!("OTel init failed: {e}"))?,
                )
            } else {
                None
            };

            let db_path = require_db(&db_opt)?;
            // Use embedder so recall/similar_to works over MCP.
            #[cfg(feature = "embed")]
            let db = open_with_best_effort(&db_path)?;
            #[cfg(not(feature = "embed"))]
            let db = open_with_all_detected(&db_path)?;
            let db = std::sync::Arc::new(db);
            // Register installed WASM plugins so their mcp_tools appear in
            // tools/list and tools/call reaches them — not only via the plugin's
            // own CLI command.
            register_installed_plugins(&db, &db_path);
            // Drive the MCP server through its Tier-3 Adapter contract: bind a
            // shared Axil, then run the blocking serve loop (it owns the tokio
            // runtime internally).
            use axil_core::Adapter as _;
            let mut adapter = axil_mcp::McpAdapter::new();
            adapter
                .bind(db)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            adapter.run().map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
            Ok(EXIT_OK)
        }

        // ── Worker ─────────────────────────────────────────────────
        Command::Worker { command } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            match command {
                WorkerCommand::Run { brain } => {
                    let worker = if brain {
                        axil_core::AxilWorker::new(&db).with_brain()
                    } else {
                        axil_core::AxilWorker::new(&db)
                    };
                    let report = worker.run().context("worker run failed")?;
                    out.print(&serde_json::to_value(&report).unwrap());
                    Ok(EXIT_OK)
                }
                WorkerCommand::Status => {
                    let worker = axil_core::AxilWorker::new(&db);
                    match worker.last_run().context("failed to read worker status")? {
                        Some(report) => {
                            out.print(&serde_json::to_value(&report).unwrap());
                            Ok(EXIT_OK)
                        }
                        None => {
                            out.print(&json!({"status": "no worker runs yet"}));
                            Ok(EXIT_OK)
                        }
                    }
                }
                WorkerCommand::Daemon { interval, duration } => {
                    let db_arc = std::sync::Arc::new(db);
                    let iv = std::time::Duration::from_secs(interval);

                    if duration == 0 {
                        // Single run via maintenance thread (run once then stop).
                        let w = axil_core::AxilWorker::new(&db_arc);
                        let report = w.run().context("worker run failed")?;
                        out.print(&serde_json::to_value(&report).unwrap());
                    } else {
                        let mt = axil_core::MaintenanceThread::start(db_arc, iv);
                        eprintln!(
                            "Maintenance daemon started (interval={}s, duration={}s)",
                            interval, duration
                        );
                        std::thread::sleep(std::time::Duration::from_secs(duration));
                        let reports = mt.stop();
                        out.print(&json!({
                            "runs": reports.len(),
                            "reports": serde_json::to_value(&reports).unwrap_or(json!([])),
                        }));
                    }
                    Ok(EXIT_OK)
                }
            }
        }

        // ── Branch ─────────────────────────────────────────────────
        Command::Branch { command } => {
            let db_path = require_db(&db_opt)?;

            match command {
                BranchCommand::Create { name } => {
                    // Open the live handle so the branch is point-in-time
                    // consistent: holding it takes redb's exclusive write lock
                    // (no other process can mutate), and `branch_create`
                    // flushes the engines and closes the handles before copying.
                    let db = open_with_all_detected(&db_path)?;
                    let branch_path = db
                        .branch_create(&name)
                        .context("failed to create branch")?;
                    out.print(&json!({
                        "branch": name,
                        "path": branch_path.display().to_string(),
                    }));
                    Ok(EXIT_OK)
                }
                BranchCommand::List => {
                    let branches =
                        axil_core::branch_list(&db_path).context("failed to list branches")?;
                    let values: Vec<Value> = branches.iter().map(|b| json!({"name": b})).collect();
                    out.print_array(&values);
                    Ok(EXIT_OK)
                }
                BranchCommand::Delete { name } => {
                    axil_core::branch_delete(&db_path, &name).context("failed to delete branch")?;
                    out.print(&json!({"deleted": name}));
                    Ok(EXIT_OK)
                }
                BranchCommand::Diff { name } => {
                    let diff =
                        axil_core::branch_diff(&db_path, &name).context("failed to diff branch")?;
                    out.print(&serde_json::to_value(&diff).unwrap());
                    Ok(EXIT_OK)
                }
                BranchCommand::Switch { name } => {
                    let bp = axil_core::branch_switch(&db_path, &name)
                        .context("failed to switch branch")?;
                    let bp_str = bp.display().to_string();
                    out.print(&json!({
                        "branch": name,
                        "path": bp_str,
                        "hint": format!("export AXIL_DB=\"{bp_str}\""),
                    }));
                    Ok(EXIT_OK)
                }
                BranchCommand::Merge {
                    name,
                    strategy,
                    delete,
                } => {
                    let merge_strategy: axil_core::MergeStrategy =
                        strategy.parse().unwrap_or_default();
                    let report = axil_core::branch_merge(&db_path, &name, merge_strategy)
                        .context("failed to merge branch")?;
                    if report.indexes_need_rebuild {
                        eprintln!("Note: merged records are not yet indexed. Run `axil heal` to rebuild vector/graph/FTS indexes.");
                    }
                    out.print(&serde_json::to_value(&report).unwrap());

                    if delete {
                        axil_core::branch_delete(&db_path, &name)
                            .context("failed to delete branch after merge")?;
                    }
                    Ok(EXIT_OK)
                }
            }
        }

        // ── HTTP API server ───────────────────────────────────────────
        #[cfg(feature = "http")]
        Command::Serve { host, port } => {
            let db_path = require_db(&db_opt)?;
            #[cfg(feature = "embed")]
            let db = open_with_best_effort(&db_path)?;
            #[cfg(not(feature = "embed"))]
            let db = open_with_all_detected(&db_path)?;
            let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
            rt.block_on(http_server::serve(db, &host, port))
                .context("HTTP server error")?;
            Ok(EXIT_OK)
        }

        // ── Reflect ───────────────────────────────────────────────────
        #[cfg(feature = "memory")]
        Command::Reflect {
            topic,
            scope,
            store,
            llm,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let db = if llm { wire_llm(db, &db_path)? } else { db };
            let parsed_scope: axil_memory::ReflectScope = scope
                .parse()
                .map_err(|e: axil_core::AxilError| anyhow::anyhow!(e))?;
            let engine = axil_memory::ReflectEngine::new(&db);
            let mut report = engine
                .reflect(topic.as_deref(), parsed_scope)
                .context("reflect failed")?;

            let llm_enhanced = llm && db.has_llm();
            if llm && !db.has_llm() {
                eprintln!("warning: --llm requested but no LLM configured in axil.toml — using heuristic mode only");
            }
            if llm_enhanced && !report.insights.is_empty() {
                let context = format!(
                    "Memories analyzed: {}\n\nHeuristic insights:\n{}",
                    report.memories_analyzed,
                    report
                        .insights
                        .iter()
                        .enumerate()
                        .map(|(i, s)| format!("{}. {s}", i + 1))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                let prompt = format!(
                    "Given these initial pattern insights from agent memory analysis, \
                     synthesize 2-3 deeper insights that connect patterns or suggest actions. \
                     Be concise and actionable.\n\n{context}\n\n\
                     Return a JSON array of insight strings. Example: [\"insight one\", \"insight two\"]"
                );
                let schema_hint = r#"["string"]"#;
                match db.llm_extract_json(&prompt, schema_hint) {
                    Ok(response) => {
                        if let Ok(enhanced) = serde_json::from_str::<Vec<String>>(&response.text) {
                            for insight in enhanced {
                                report.insights.push(format!("[llm] {insight}"));
                            }
                        }
                    }
                    Err(_) => {
                        // Graceful fallback — heuristic insights are still valid.
                    }
                }
            }

            let mut result = report.to_json();
            result["llm_enhanced"] = json!(llm_enhanced);

            // Store insights as a record if --store given
            if let Some(table) = &store {
                let record = db
                    .insert(
                        table,
                        json!({
                            "type": "reflection",
                            "topic": report.topic,
                            "scope": scope,
                            "insights": report.insights,
                            "memories_analyzed": report.memories_analyzed,
                            "llm_enhanced": llm_enhanced,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                        }),
                    )
                    .context("failed to store reflection")?;
                result["stored_id"] = json!(record.id.to_string());
                result["stored_table"] = json!(table);
            }

            out.print(&result);
            Ok(EXIT_OK)
        }

        // ── Connections ───────────────────────────────────────────────
        #[cfg(all(feature = "memory", feature = "graph"))]
        Command::Connections { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);
            let knowledge = mem
                .semantic()
                .about(&entity)
                .context("failed to query entity")?;

            let mut connections = Vec::new();
            for fact in &knowledge.facts {
                if let Ok(neighbors) = db.neighbors(&fact.id, None, Direction::Both) {
                    for n in &neighbors {
                        if let Some(name) = n.data.get("entity").and_then(|v| v.as_str()) {
                            if name != entity {
                                connections.push(json!({
                                    "entity": name,
                                    "via_record": fact.id.to_string(),
                                }));
                            }
                        }
                    }
                }
            }
            connections.sort_by(|a, b| a["entity"].as_str().cmp(&b["entity"].as_str()));
            connections.dedup_by(|a, b| a["entity"] == b["entity"]);

            out.print(&json!({
                "entity": entity,
                "connections": connections,
                "related_entities": knowledge.related_entities,
            }));
            Ok(EXIT_OK)
        }

        // ── Profile ───────────────────────────────────────────────────
        #[cfg(feature = "memory")]
        Command::Profile { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mem = axil_memory::AgentMemory::new(&db);
            let knowledge = mem
                .semantic()
                .about(&entity)
                .context("failed to query entity")?;
            out.print(&knowledge.to_json());
            Ok(EXIT_OK)
        }

        // ── Patterns ──────────────────────────────────────────────────
        #[cfg(feature = "memory")]
        Command::Patterns {
            pattern_type,
            dismiss,
            detect,
        } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let engine = axil_memory::PatternEngine::new(&db);

            if let Some(name) = dismiss {
                let dismissed = engine.dismiss(&name).context("dismiss failed")?;
                out.print(&json!({"dismissed": dismissed, "pattern": name}));
                return Ok(EXIT_OK);
            }

            if detect {
                let patterns = engine.detect().context("pattern detection failed")?;
                let stored = engine.store_patterns(&patterns).context("store failed")?;
                let values: Vec<Value> = patterns.iter().map(|p| p.to_json()).collect();
                out.print(&json!({
                    "detected": values.len(),
                    "stored": stored,
                    "patterns": values,
                }));
                return Ok(EXIT_OK);
            }

            let pt = pattern_type
                .as_deref()
                .map(|s| s.parse::<axil_memory::PatternType>())
                .transpose()
                .map_err(|e: axil_core::AxilError| anyhow::anyhow!(e))?;
            let patterns = engine.list(pt).context("list patterns failed")?;
            let values: Vec<Value> = patterns.iter().map(|p| p.to_json()).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }

        // ── Infer ─────────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::Infer { entity } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let engine = axil_core::InferenceEngine::new(&db);

            let entity_id = if let Some(ref name) = entity {
                let records = db.list("_entities")?;
                records
                    .iter()
                    .find(|r| r.data.get("entity").and_then(|v| v.as_str()) == Some(name))
                    .map(|r| r.id.clone())
            } else {
                None
            };

            let facts = engine
                .infer_and_store(entity_id.as_ref())
                .context("inference failed")?;
            let values: Vec<Value> = facts
                .iter()
                .map(|f| serde_json::to_value(f).unwrap_or(json!({})))
                .collect();
            out.print(&json!({
                "inferred_facts": values.len(),
                "facts": values,
            }));
            Ok(EXIT_OK)
        }

        // ── Why-fact ──────────────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::WhyFact { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = RecordId::from_string(&id).context("invalid record ID")?;
            let engine = axil_core::InferenceEngine::new(&db);
            match engine.why(&record_id).context("why-fact failed")? {
                Some(chain) => out.print(&json!({
                    "id": id,
                    "reasoning": chain,
                })),
                None => out.print(&json!({
                    "id": id,
                    "reasoning": null,
                    "note": "not an inferred fact",
                })),
            }
            Ok(EXIT_OK)
        }

        // ── Fact confirm/reject ──────────────────────────────────────────
        #[cfg(feature = "graph")]
        Command::FactConfirm { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = RecordId::from_string(&id).context("invalid record ID")?;
            let engine = axil_core::InferenceEngine::new(&db);
            let confirmed = engine.confirm(&record_id).context("confirm failed")?;
            out.print(&json!({
                "id": id,
                "confirmed": confirmed,
            }));
            Ok(EXIT_OK)
        }

        #[cfg(feature = "graph")]
        Command::FactReject { id } => {
            let db_path = require_db(&db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let record_id = RecordId::from_string(&id).context("invalid record ID")?;
            let engine = axil_core::InferenceEngine::new(&db);
            let rejected = engine.reject(&record_id).context("reject failed")?;
            out.print(&json!({
                "id": id,
                "rejected": rejected,
            }));
            Ok(EXIT_OK)
        }

        // ── ──────────────────────────────────────────────────
        Command::Workspace { op } => workspace::handle_workspace(op, &db_opt, out),
        Command::Consent { op } => workspace::handle_consent(op, &db_opt, out),
        Command::Bridge { op } => workspace::handle_bridge(op, &db_opt, out),
        #[cfg(feature = "embed")]
        Command::RecallAcross {
            query,
            across,
            top_k,
            strict_consent,
            trace,
            oneline,
        } => {
            if top_k > MAX_RESULT_LIMIT {
                anyhow::bail!("--top-k exceeds maximum of {MAX_RESULT_LIMIT}");
            }
            workspace::handle_recall_across(
                &db_opt,
                out,
                &query,
                &across,
                top_k,
                strict_consent,
                trace,
                oneline,
            )
        }
        Command::TraceRecord { target } => workspace::handle_trace_record(&db_opt, out, &target),
    }
}

/// Run the background worker and report results if any work was done.
fn run_worker_and_report(db: &axil_core::Axil, out: &Output) {
    let worker = axil_core::AxilWorker::new(db);
    match worker.run() {
        Ok(wr) => {
            if wr.consolidated_entities > 0 || wr.new_connections > 0 || wr.inferred_facts > 0 {
                out.status(&format!(
                    "worker: consolidated={}, connections={}, inferred={}",
                    wr.consolidated_entities, wr.new_connections, wr.inferred_facts
                ));
            }
        }
        Err(e) => out.status(&format!("warning: worker failed: {e}")),
    }
}

// ─── Project root detection ─────────────────────────────────────────────────

/// Walk up from the database path to find the project root.
/// Looks for Cargo.toml, package.json, pyproject.toml, go.mod, etc.
///
/// Not feature-gated: pure `std::path` walking with no indexer dependency,
/// called from ungated handlers (`memory-pressure`, `decay`).
fn detect_project_root(db_path: &Path) -> Option<PathBuf> {
    let mut dir = db_path.parent()?.to_path_buf();
    let markers = [
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        ".git",
    ];
    loop {
        for marker in &markers {
            if dir.join(marker).exists() {
                return Some(dir);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    // Fallback: db parent directory
    db_path.parent().map(|p| p.to_path_buf())
}

// ─── dependency documentation memory ──────────────────────────────

/// Scan a source file for the external packages it imports.
///
/// Best-effort and heuristic, dispatched by file extension. Rust,
/// Python and JavaScript/TypeScript are covered; relative, local and
/// standard-library imports are excluded.
#[cfg(feature = "deps")]
fn imports_in_file(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        match ext.as_str() {
            "rs" => {
                if let Some(rest) = line.strip_prefix("use ") {
                    let head = rest.split([':', ' ', ';', '{']).next().unwrap_or("");
                    if !head.is_empty()
                        && !matches!(head, "crate" | "super" | "self" | "std" | "core" | "alloc")
                    {
                        out.push(head.to_string());
                    }
                }
            }
            "py" => {
                let module = line
                    .strip_prefix("import ")
                    .or_else(|| line.strip_prefix("from "));
                if let Some(rest) = module {
                    let head = rest.split(['.', ' ', ',']).next().unwrap_or("");
                    if !head.is_empty() && !head.starts_with('.') {
                        out.push(head.to_string());
                    }
                }
            }
            "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => {
                if let Some(pkg) = js_import_target(line) {
                    out.push(pkg);
                }
            }
            _ => {}
        }
    }
    out
}

/// Extract the package an ES-module / CommonJS line imports.
#[cfg(feature = "deps")]
fn js_import_target(line: &str) -> Option<String> {
    if !line.contains("import") && !line.contains("require") {
        return None;
    }
    let spec = js_first_quoted(line)?;
    if spec.starts_with('.') || spec.starts_with('/') {
        return None; // relative / absolute path, not a package
    }
    // Keep `@scope/name`; otherwise take the first path segment.
    let pkg = if spec.starts_with('@') {
        spec.splitn(3, '/').take(2).collect::<Vec<_>>().join("/")
    } else {
        spec.split('/').next().unwrap_or(spec).to_string()
    };
    Some(pkg)
}

/// The first single- or double-quoted string in a line.
#[cfg(feature = "deps")]
fn js_first_quoted(line: &str) -> Option<&str> {
    for quote in ['"', '\''] {
        if let Some(start) = line.find(quote) {
            let rest = &line[start + 1..];
            if let Some(end) = rest.find(quote) {
                return Some(&rest[..end]);
            }
        }
    }
    None
}

/// Resolve every manifest's dependencies and dedup them by
/// (ecosystem, name, version) — a Cargo workspace declares the same
/// dependency across many members. Path (first-party) dependencies are
/// excluded: dep-docs target external registry libraries only.
#[cfg(feature = "deps")]
/// P3 follow-up — wrapper around the relocated
/// `axil_docs::collect_unique_deps`. Adapts the `DocsError` return
/// to `anyhow::Result` so the rest of `axil-cli` stays unchanged.
fn deps_collect_unique(
    manifests: &[axil_docs::DetectedManifest],
    include_dev: bool,
) -> Result<Vec<axil_docs::Dependency>> {
    axil_docs::collect_unique_deps(manifests, include_dev).map_err(|e| anyhow::anyhow!("{e}"))
}

/// P3 follow-up — wrapper around the relocated
/// `axil_docs::ingest_manifests`. Adapts `DocsError` to `anyhow`.
#[cfg(feature = "deps")]
fn deps_ingest_manifests(
    db: &Axil,
    manifests: &[axil_docs::DetectedManifest],
    project_root: &Path,
    transitive: bool,
) -> Result<Value> {
    axil_docs::ingest_manifests(db, manifests, project_root, transitive)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Wrapper around `axil_docs::sweep_removed_for_manifests` that
/// adapts the `DocsError` return into `anyhow::Result`.
#[cfg(feature = "deps")]
fn deps_sweep_removed(
    db: &Axil,
    all_manifests: &[axil_docs::DetectedManifest],
) -> Result<Vec<String>> {
    axil_docs::sweep_removed_for_manifests(db, all_manifests).map_err(|e| anyhow::anyhow!("{e}"))
}

/// `axil deps` — dependency documentation memory.
///
/// Route a `axil deps …` invocation through the registered
/// `DocsExtension` (or any other Extension that claims the `deps`
/// top-level command) via [`axil_core::dispatch_cli`].
///
/// Returns `Ok(Some(exit_code))` if an Extension handled the call —
/// the helper has already written stdout/stderr through `out`, so the
/// caller just propagates the exit code.
///
/// Returns `Ok(None)` if no Extension claimed the call; the caller
/// then falls through to the hardcoded [`run_deps`] handler. This is
/// the load-bearing Path C fallback semantic.
///
/// The DB is only opened for variants that need it (every variant
/// except `List`, which is a pure filesystem scan over manifests).
/// `List` without an open DB falls back to `run_deps` directly, which
/// preserves the original contract that `axil deps list` works
/// outside an initialised `.axil` project.
#[cfg(feature = "deps")]
fn try_deps_extension_dispatch(
    action: &DepsCommand,
    db_opt: &Option<PathBuf>,
    out: &Output,
) -> Result<Option<i32>> {
    // Map the typed clap variant back to a flat CliInvocation. Only
    // include variants whose Extension migration has landed; other
    // variants return None to skip the DB open entirely.
    let invocation = match action {
        DepsCommand::Status { path } => axil_core::CliInvocation {
            command_path: vec!["deps".into(), "status".into()],
            args: vec!["--path".into(), path.display().to_string()],
            stdin: None,
        },
        DepsCommand::List { path, dev } => {
            let mut args = vec!["--path".into(), path.display().to_string()];
            if *dev {
                args.push("--dev".into());
            }
            axil_core::CliInvocation {
                command_path: vec!["deps".into(), "list".into()],
                args,
                stdin: None,
            }
        }
        DepsCommand::Ingest {
            dep,
            ecosystem,
            file,
            from_web,
        } => {
            let mut args = vec![
                "--dep".into(),
                dep.clone(),
                "--ecosystem".into(),
                ecosystem.clone(),
            ];
            if let Some(p) = file {
                args.push("--file".into());
                args.push(p.display().to_string());
            }
            if *from_web {
                args.push("--from-web".into());
            }
            // Capture stdin when no file path is provided so the
            // Extension can read it via invocation.stdin. Mirrors the
            // pre-migration `run_deps` `DepsCommand::Ingest` behavior.
            // Cap at MAX_STDIN_BYTES to prevent OOM on a piped giant.
            let stdin = if file.is_none() && !*from_web {
                use std::io::Read;
                let mut buf = String::new();
                match std::io::stdin()
                    .take(MAX_STDIN_BYTES + 1)
                    .read_to_string(&mut buf)
                {
                    Ok(_) if buf.len() as u64 > MAX_STDIN_BYTES => {
                        return Err(anyhow::anyhow!(
                            "stdin exceeds {} MB cap",
                            MAX_STDIN_BYTES / (1024 * 1024)
                        ));
                    }
                    Ok(_) => Some(buf),
                    Err(_) => None,
                }
            } else {
                None
            };
            axil_core::CliInvocation {
                command_path: vec!["deps".into(), "ingest".into()],
                args,
                stdin,
            }
        }
        DepsCommand::Sync {
            path,
            offline,
            transitive,
        } => {
            let mut args = vec!["--path".into(), path.display().to_string()];
            if *offline {
                args.push("--offline".into());
            }
            if *transitive {
                args.push("--transitive".into());
            }
            axil_core::CliInvocation {
                command_path: vec!["deps".into(), "sync".into()],
                args,
                stdin: None,
            }
        }
        DepsCommand::Refresh {
            path,
            if_stale,
            transitive,
        } => {
            let mut args = vec!["--path".into(), path.display().to_string()];
            if *if_stale {
                args.push("--if-stale".into());
            }
            if *transitive {
                args.push("--transitive".into());
            }
            axil_core::CliInvocation {
                command_path: vec!["deps".into(), "refresh".into()],
                args,
                stdin: None,
            }
        }
    };

    // `deps list` is a pure manifest scan — pre-Phase-17 it worked
    // outside an initialised `.axil` project. Preserve that by
    // falling back to `run_deps` when no DB is configured.
    let db_path = match (db_opt.as_ref(), action) {
        (Some(p), _) => p.clone(),
        (None, DepsCommand::List { .. }) => return Ok(None),
        (None, _) => require_db(db_opt)?,
    };
    let db = open_with_all_detected(&db_path)?;
    let consumed_stdin = invocation.stdin.is_some();
    match axil_core::dispatch_cli(&db, &db.extensions(), &invocation)? {
        axil_core::Dispatch::Handled(output) => {
            // Re-parse the stdout payload as JSON so Output's
            // format-aware printer (Json / Pretty / Table) still
            // applies. If the Extension returns non-JSON, fall back
            // to a raw write.
            if !output.stdout.is_empty() {
                match serde_json::from_str::<Value>(&output.stdout) {
                    Ok(v) => out.print(&v),
                    Err(_) => println!("{}", output.stdout),
                }
            }
            if !output.stderr.is_empty() {
                eprint!("{}", output.stderr);
            }
            Ok(Some(output.exit_code))
        }
        // If we ate stdin into `invocation.stdin` but no Extension
        // claimed the call, the hardcoded `run_deps` fallback would
        // re-read stdin and get EOF — silently ingesting an empty
        // doc. Fail loudly instead. Today no migrated variant returns
        // NotHandled, so this branch is a safety net for future
        // Extensions that surface the `deps` command without claiming
        // every variant.
        axil_core::Dispatch::NotHandled if consumed_stdin => Err(anyhow::anyhow!(
            "Extension declined `axil deps {}` after stdin was consumed — \
             refusing to fall back to the hardcoded handler with empty input",
            invocation
                .command_path
                .get(1)
                .map(|s| s.as_str())
                .unwrap_or(""),
        )),
        axil_core::Dispatch::NotHandled => Ok(None),
    }
}

/// route `axil checkpoint …` through the CheckpointExtension's
/// CLI surface via Path C dispatch. `arg` is the optional positional
/// (inline JSON, `-` for stdin, or the literal `show`); `session` and
/// `is_final` are clap-parsed flags. When the positional is missing or
/// equals `-`, stdin is captured into the invocation so the Extension
/// can read it.
///
/// Returns an error if the `checkpoint` Extension isn't registered (e.g.
/// a custom build with the feature stripped). When Path C declines
/// (shouldn't happen for a recognised checkpoint invocation), surfaces a
/// clear error rather than silently doing nothing.
#[cfg(feature = "checkpoint")]
fn run_checkpoint_extension(
    arg: Option<String>,
    session: Option<String>,
    is_final: bool,
    db_opt: &Option<PathBuf>,
    out: &Output,
) -> Result<i32> {
    use std::io::Read;

    // Build command_path: `checkpoint show` is the read sub; anything
    // else (including no positional) is the write sub.
    let (command_path, positional): (Vec<String>, Option<String>) = match arg.as_deref() {
        Some("show") => (vec!["checkpoint".into(), "show".into()], None),
        Some(other) => (vec!["checkpoint".into()], Some(other.to_string())),
        None => (vec!["checkpoint".into()], None),
    };

    // Marshal flags + the positional into a flat args vec the
    // Extension can parse with its own named_arg / flag_set helpers.
    let mut args: Vec<String> = Vec::new();
    if let Some(p) = positional.as_deref() {
        args.push(p.to_string());
    }
    if is_final {
        args.push("--final".into());
    }
    if let Some(ref s) = session {
        args.push("--session".into());
        args.push(s.clone());
    }

    // Capture stdin when the user passed `-` or no positional at all
    // (the implicit-stdin shorthand `cat foo.json | axil checkpoint`).
    // Stay quiet on a TTY to avoid hanging when nothing is piped.
    let stdin = if positional.as_deref() == Some("-")
        || (positional.is_none() && command_path.len() == 1)
    {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            None
        } else {
            let mut buf = String::new();
            const MAX: u64 = 4 * 1024 * 1024;
            if std::io::stdin()
                .take(MAX + 1)
                .read_to_string(&mut buf)
                .is_ok()
                && (buf.len() as u64) <= MAX
                && !buf.trim().is_empty()
            {
                Some(buf)
            } else {
                None
            }
        }
    } else {
        None
    };

    let invocation = axil_core::CliInvocation {
        command_path,
        args,
        stdin,
    };

    let db_path = require_db(db_opt)?;
    let db = open_with_all_detected(&db_path)?;
    match axil_core::dispatch_cli(&db, &db.extensions(), &invocation)? {
        axil_core::Dispatch::Handled(output) => {
            if !output.stdout.is_empty() {
                match serde_json::from_str::<Value>(&output.stdout) {
                    Ok(v) => out.print(&v),
                    Err(_) => println!("{}", output.stdout),
                }
            }
            if !output.stderr.is_empty() {
                eprint!("{}", output.stderr);
            }
            Ok(output.exit_code)
        }
        axil_core::Dispatch::NotHandled => Err(anyhow::anyhow!(
            "axil checkpoint: CheckpointExtension declined the call — \
             is the `checkpoint` feature compiled in?",
        )),
    }
}

/// Route `axil cache …` through the CacheExtension's CLI surface via Path C
/// dispatch, mirroring `run_checkpoint_extension`. The typed clap subcommand
/// exists only so `cache` shows in `axil --help`; it marshals into the same
/// `CliInvocation` the generic external-subcommand path would build, so the
/// Extension stays the single owner of the cache logic. `cache put` captures
/// piped stdin when the payload is `-` or omitted.
#[cfg(feature = "cache")]
fn run_cache_extension(cmd: CacheCommand, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    let mut stdin: Option<String> = None;
    let (sub, args): (&str, Vec<String>) = match cmd {
        CacheCommand::Put { json } => {
            // Read piped stdin for the `-` and bare-`put` shorthands, matching
            // the Extension's `read_payload` contract.
            if matches!(json.as_deref(), Some("-") | None) {
                stdin = read_piped_stdin();
            }
            ("put", json.into_iter().collect())
        }
        CacheCommand::Get {
            question,
            threshold,
            top_k,
        } => {
            let mut args = vec![question];
            if let Some(t) = threshold {
                args.push("--threshold".into());
                args.push(t.to_string());
            }
            if let Some(k) = top_k {
                args.push("--top-k".into());
                args.push(k.to_string());
            }
            ("get", args)
        }
        CacheCommand::Stats => ("stats", Vec::new()),
        CacheCommand::Clear { all, expired } => {
            let mut args = Vec::new();
            if all {
                args.push("--all".into());
            }
            if expired {
                args.push("--expired".into());
            }
            ("clear", args)
        }
    };

    let invocation = axil_core::CliInvocation {
        command_path: vec!["cache".into(), sub.into()],
        args,
        stdin,
    };

    let db_path = require_db(db_opt)?;
    let db = open_with_all_detected(&db_path)?;
    match axil_core::dispatch_cli(&db, &db.extensions(), &invocation)? {
        axil_core::Dispatch::Handled(output) => {
            if !output.stdout.is_empty() {
                match serde_json::from_str::<Value>(&output.stdout) {
                    Ok(v) => out.print(&v),
                    Err(_) => println!("{}", output.stdout),
                }
            }
            if !output.stderr.is_empty() {
                eprint!("{}", output.stderr);
            }
            Ok(output.exit_code)
        }
        axil_core::Dispatch::NotHandled => Err(anyhow::anyhow!(
            "axil cache: CacheExtension declined the call — \
             is the `cache` feature compiled in?",
        )),
    }
}

#[cfg(feature = "deps")]
fn run_deps(cmd: DepsCommand, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    match cmd {
        DepsCommand::List { path, dev } => {
            let manifests = axil_docs::detect_manifests(&path);
            let list: Vec<Value> = deps_collect_unique(&manifests, dev)?
                .iter()
                .map(|d| {
                    json!({
                        "name": d.name,
                        "ecosystem": d.ecosystem.as_str(),
                        "kind": d.kind.as_str(),
                        "declared_range": d.declared_range,
                        "version": d.version,
                        "pinned": d.version.is_some(),
                    })
                })
                .collect();
            out.print(&json!({
                "manifests": manifests.len(),
                "dependencies": list.len(),
                "deps": list,
            }));
            Ok(EXIT_OK)
        }
        DepsCommand::Sync {
            path,
            offline,
            transitive,
        } => {
            let _ = offline; // local-only is currently the only mode
            let db_path = require_db(db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let manifests = axil_docs::detect_manifests(&path);
            let mut summary = deps_ingest_manifests(&db, &manifests, &path, transitive)?;
            summary["removed"] = json!(deps_sweep_removed(&db, &manifests)?);
            summary["manifests"] = json!(manifests.len());
            out.print(&summary);
            Ok(EXIT_OK)
        }
        DepsCommand::Refresh {
            path,
            if_stale,
            transitive,
        } => {
            let db_path = require_db(db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let all = axil_docs::detect_manifests(&path);

            let mut drift_report: Vec<Value> = Vec::new();
            let mut stale: Vec<axil_docs::DetectedManifest> = Vec::new();
            for manifest in &all {
                let drift =
                    axil_docs::manifest_drift(&db, manifest).map_err(|e| anyhow::anyhow!("{e}"))?;
                drift_report.push(json!({
                    "path": manifest.path.display().to_string(),
                    "drift": drift.as_str(),
                }));
                if drift.needs_sync() {
                    stale.push(manifest.clone());
                }
            }

            // `--if-stale` fast exit: nothing changed since the last sync.
            if if_stale && stale.is_empty() {
                out.print(&json!({
                    "manifests": all.len(),
                    "refreshed": 0,
                    "status": "fresh",
                }));
                return Ok(EXIT_OK);
            }

            let to_sync = if if_stale { &stale } else { &all };
            let mut summary = deps_ingest_manifests(&db, to_sync, &path, transitive)?;
            summary["removed"] = json!(deps_sweep_removed(&db, &all)?);
            summary["manifests"] = json!(all.len());
            summary["refreshed"] = json!(to_sync.len());
            summary["drift"] = json!(drift_report);
            out.print(&summary);
            Ok(EXIT_OK)
        }
        DepsCommand::Ingest {
            dep,
            ecosystem,
            file,
            from_web,
        } => {
            let db_path = require_db(db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let (name, version) = dep
                .rsplit_once('@')
                .ok_or_else(|| anyhow::anyhow!("--dep must be name@version"))?;
            let eco = axil_docs::Ecosystem::from_str(&ecosystem)
                .ok_or_else(|| anyhow::anyhow!("unknown ecosystem: {ecosystem}"))?;
            let dependency = axil_docs::Dependency {
                name: name.to_string(),
                ecosystem: eco,
                kind: axil_docs::DepKind::Direct,
                declared_range: "agent".to_string(),
                version: Some(version.to_string()),
            };
            let (text, source): (String, &str) = if from_web {
                #[cfg(feature = "web-docs")]
                {
                    let fetched = axil_docs::fetch_web_doc(&dependency)
                        .ok_or_else(|| anyhow::anyhow!("web fetch returned no docs for {dep}"))?;
                    (fetched, "web")
                }
                #[cfg(not(feature = "web-docs"))]
                {
                    return Err(anyhow::anyhow!(
                        "--from-web requires the `web-docs` feature \
                         (rebuild with --features web-docs)"
                    ));
                }
            } else {
                let t = match &file {
                    Some(p) => std::fs::read_to_string(p)
                        .with_context(|| format!("reading {}", p.display()))?,
                    None => {
                        use std::io::Read;
                        let mut buf = String::new();
                        std::io::stdin()
                            .take(MAX_STDIN_BYTES + 1)
                            .read_to_string(&mut buf)
                            .context("reading doc text from stdin")?;
                        if buf.len() as u64 > MAX_STDIN_BYTES {
                            return Err(anyhow::anyhow!(
                                "stdin exceeds {} MB cap",
                                MAX_STDIN_BYTES / (1024 * 1024)
                            ));
                        }
                        buf
                    }
                };
                (t, "agent")
            };
            let n = axil_docs::ingest_dep_docs(
                &db,
                &dependency,
                &text,
                source,
                axil_docs::DEFAULT_MAX_CHUNKS_PER_DEP,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&json!({
                "dep": name,
                "version": version,
                "ecosystem": eco.as_str(),
                "chunks": n,
                "source": source,
            }));
            Ok(EXIT_OK)
        }
        DepsCommand::Status { path } => {
            let db_path = require_db(db_opt)?;
            let db = open_with_all_detected(&db_path)?;
            let mut deps: Vec<Value> = db
                .list(axil_docs::TABLE_DEPS)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .iter()
                .map(|r| r.data.clone())
                .collect();
            deps.sort_by(|a, b| {
                let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
                an.cmp(bn)
            });
            let mut manifests: Vec<Value> = Vec::new();
            for manifest in &axil_docs::detect_manifests(&path) {
                let drift =
                    axil_docs::manifest_drift(&db, manifest).map_err(|e| anyhow::anyhow!("{e}"))?;
                manifests.push(json!({
                    "path": manifest.path.display().to_string(),
                    "ecosystem": manifest.ecosystem.as_str(),
                    "drift": drift.as_str(),
                }));
            }
            out.print(&json!({
                "synced_deps": deps.len(),
                "deps": deps,
                "manifests": manifests,
            }));
            Ok(EXIT_OK)
        }
    }
}

// ─── Session commands ───────────────────────────────────────────────────────

const SESSION_TABLE: &str = "_sessions";
const SESSION_EDGE_TYPE: &str = "session_contains";
const SESSION_ACTIVE: &str = "active";
const SESSION_ENDED: &str = "ended";

fn run_session(cmd: SessionCommand, db_path: &Path, out: &Output) -> Result<i32> {
    match cmd {
        SessionCommand::Start { meta } => {
            let db = open_with_all_detected(db_path)?;
            let now = chrono::Utc::now();

            let mut data = json!({
                "status": SESSION_ACTIVE,
                "started_at": now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                "record_count": 0,
            });

            if let Some(ref meta_json) = meta {
                let meta_val: Value =
                    serde_json::from_str(meta_json).context("invalid meta JSON")?;
                data["meta"] = meta_val;
            }

            let record = db
                .insert(SESSION_TABLE, data)
                .context("failed to start session")?;

            out.print(&json!({
                "session_id": record.id.to_string(),
                "started_at": format_dt(&record.created_at),
            }));
            Ok(EXIT_OK)
        }

        SessionCommand::End {
            session_id,
            summary,
        } => {
            // Try to open with embedder for auto-embed; fall back gracefully.
            #[cfg(feature = "embed")]
            let (db, has_embedder) = if summary.is_some() {
                match open_with_embedder(db_path) {
                    Ok(db) => (db, true),
                    Err(_) => (open_with_all_detected(db_path)?, false),
                }
            } else {
                (open_with_all_detected(db_path)?, false)
            };
            #[cfg(not(feature = "embed"))]
            let db = open_with_all_detected(db_path)?;

            let sid = RecordId::from_string(&session_id).context("invalid session ID")?;

            let session = db
                .get(&sid)
                .context("get failed")?
                .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;

            let now = chrono::Utc::now();
            let mut data = session.data.clone();
            data["status"] = json!(SESSION_ENDED);
            data["ended_at"] = json!(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));

            if let Some(ref s) = summary {
                data["summary"] = json!(s);
            }

            // Count linked records if graph is available.
            let record_count = if db.has_graph_index() {
                db.neighbors(&sid, Some(SESSION_EDGE_TYPE), Direction::Out)
                    .map(|n| n.len())
                    .unwrap_or(0)
            } else {
                data.get("record_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize
            };
            data["record_count"] = json!(record_count);

            let updated = db.update(&sid, data).context("update failed")?;

            // Auto-embed the summary field if embedder is available.
            #[cfg(feature = "embed")]
            if has_embedder && summary.is_some() {
                if let Err(e) = db.embed_field(&sid, "summary") {
                    out.status(&format!("warning: failed to embed summary: {e}"));
                }
            }

            run_worker_and_report(&db, out);

            out.print(&json!({
                "session_id": session_id,
                "ended_at": format_dt(&updated.updated_at),
                "records": record_count,
            }));
            Ok(EXIT_OK)
        }

        SessionCommand::Log {
            session_id,
            table,
            json_data,
            #[cfg(feature = "embed")]
            embed,
        } => {
            #[cfg(feature = "embed")]
            let db = if embed.is_some() {
                open_with_embedder(db_path)?
            } else {
                open_with_all_detected(db_path)?
            };
            #[cfg(not(feature = "embed"))]
            let db = open_with_all_detected(db_path)?;

            let sid = RecordId::from_string(&session_id).context("invalid session ID")?;

            // Verify session exists and get current data for record_count.
            let session = db
                .get(&sid)
                .context("get failed")?
                .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;

            let data = read_json_input(&json_data)?;
            let record = db.insert(&table, data).context("insert failed")?;

            // Auto-embed if requested.
            #[cfg(feature = "embed")]
            if let Some(ref fields) = embed {
                for field in fields.split(',') {
                    let field = field.trim();
                    if !field.is_empty() {
                        db.embed_field(&record.id, field)
                            .with_context(|| format!("failed to embed field '{field}'"))?;
                    }
                }
            }

            // Link to session via graph if available.
            let linked = if db.has_graph_index() {
                db.relate(&sid, SESSION_EDGE_TYPE, &record.id, None)?;
                true
            } else {
                false
            };

            // Increment record_count so session end works without graph.
            let count = session
                .data
                .get("record_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                + 1;
            let mut session_data = session.data.clone();
            session_data["record_count"] = json!(count);
            db.update(&sid, session_data)
                .context("update session count")?;

            out.print(&json!({
                "id": record.id.to_string(),
                "session_id": session_id,
                "table": table,
                "linked": linked,
            }));
            Ok(EXIT_OK)
        }

        SessionCommand::List { active } => {
            let db = Axil::open(db_path)
                .build()
                .context("failed to open database")?;

            let records = if active {
                db.query()
                    .table(SESSION_TABLE)
                    .where_field("status", Op::Eq, json!(SESSION_ACTIVE))
                    .exec()
                    .context("query sessions failed")?
            } else {
                db.list(SESSION_TABLE).context("list sessions failed")?
            };

            let values: Vec<Value> = records
                .iter()
                .map(|r| {
                    let mut v = json!({
                        "session_id": r.id.to_string(),
                        "started_at": format_dt(&r.created_at),
                        "status": r.data.get("status").cloned().unwrap_or(json!("unknown")),
                    });
                    if let Some(meta) = r.data.get("meta") {
                        v["meta"] = meta.clone();
                    }
                    if let Some(ended) = r.data.get("ended_at") {
                        v["ended_at"] = ended.clone();
                    }
                    if let Some(summary) = r.data.get("summary") {
                        v["summary"] = summary.clone();
                    }
                    if let Some(count) = r.data.get("record_count") {
                        v["records"] = count.clone();
                    }
                    v
                })
                .collect();

            out.print_array(&values);
            Ok(EXIT_OK)
        }

        SessionCommand::History { session_id } => {
            let db = open_with_all_detected(db_path)?;
            let sid = RecordId::from_string(&session_id).context("invalid session ID")?;

            if !db.has_graph_index() {
                // No graph plugin — return empty array (consistent with session log returning linked=false).
                out.print_array(&[]);
                return Ok(EXIT_OK);
            }

            let records = db.neighbors(&sid, Some(SESSION_EDGE_TYPE), Direction::Out)?;

            let values: Vec<Value> = records.iter().map(record_to_json).collect();
            out.print_array(&values);
            Ok(EXIT_OK)
        }
    }
}

// ─── Config commands ────────────────────────────────────────────────────────

/// Build a [`CliInvocation`] from raw external-subcommand tokens, using the
/// owning Extension's declared [`CliSurface`] to split leading subcommand-path
/// tokens from arguments. Pure (no DB) so it is unit-testable.
///
/// `tokens[0]` is always the top-level command. If the surface declares a
/// nested subcommand whose name matches `tokens[1]`, that token joins the
/// command path; everything after is `args`. This reproduces what a bespoke
/// per-Extension marshaler did, but generically from the declared surface.
fn build_extension_invocation(
    tokens: &[String],
    surface: Option<&axil_core::CliSurface>,
    stdin: Option<String>,
) -> axil_core::CliInvocation {
    let mut command_path: Vec<String> = Vec::new();
    let mut idx = 0;
    if let Some(cmd) = tokens.first() {
        command_path.push(cmd.clone());
        idx = 1;
        if let (Some(surface), Some(next)) = (surface, tokens.get(1)) {
            if surface.subcommands.iter().any(|s| &s.name == next) {
                command_path.push(next.clone());
                idx = 2;
            }
        }
    }
    axil_core::CliInvocation::new(command_path, tokens[idx..].to_vec(), stdin)
}

/// Read piped stdin if present (non-TTY), so Extensions supporting the `-`
/// convention work over the generic dispatch path. Returns `None` at a TTY.
fn read_piped_stdin() -> Option<String> {
    use std::io::{IsTerminal, Read};
    if std::io::stdin().is_terminal() {
        return None;
    }
    let mut buf = String::new();
    match std::io::stdin().read_to_string(&mut buf) {
        Ok(n) if n > 0 => Some(buf),
        _ => None,
    }
}

/// Load config from the database's own directory — where engines, extensions,
/// and `[plugins.<key>]` grants all live — rather than the process cwd, so a
/// plugin's capability grants (and `[extensions] disabled`) are read from the
/// same place as everything else even when `axil --db /abs/path` is run from an
/// unrelated cwd.
fn plugin_config_for_db(db_path: &Path) -> axil_core::AxilConfig {
    db_path
        .parent()
        .and_then(|dir| axil_core::load_config_from(dir).ok())
        .unwrap_or_default()
}

/// Register any installed WASM plugins into an already-open database so their
/// `boot_block` / `mcp_tools` / `recall_for_file` flow through like native
/// Extensions — not only when the plugin's own CLI subcommand is invoked. A
/// failed plugin is quarantined inside `load_and_register`, never fatal. No-op
/// without the `wasm-host` feature.
fn register_installed_plugins(db: &std::sync::Arc<Axil>, db_path: &Path) {
    #[cfg(feature = "wasm-host")]
    {
        let config = plugin_config_for_db(db_path);
        let dir = wasm_plugins::plugins_dir(db_path);
        let _ = wasm_plugins::load_and_register(db, &dir, &config);
    }
    #[cfg(not(feature = "wasm-host"))]
    let _ = (db, db_path);
}

/// Generic dispatch for a command no built-in subcommand claimed: route it to
/// whichever registered Extension owns it. The Extension's command surface is
/// resolved from the bundle **without** opening the DB, so a genuine typo fails
/// fast (no DB open) exactly like before; only a real Extension command pays to
/// open the database and dispatch.
fn run_extension_dispatch(
    tokens: Vec<String>,
    db_opt: &Option<PathBuf>,
    _out: &Output,
) -> Result<i32> {
    let command = tokens
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no command given"))?;

    let db_path = require_db(db_opt)?;
    // Read config (disabled extensions + plugin grants) from the db's own dir,
    // consistent with where engines/extensions read it.
    let config = plugin_config_for_db(&db_path);
    let builtin = axil_bundle::builtin_extension_surfaces(&config)
        .into_iter()
        .find(|s| s.command == command);

    // Resolve which extension owns this command + open the database.
    //
    // Without `wasm-host`, a command no built-in claims is unknown — fail fast
    // (no DB open for a typo). With `wasm-host`, it might belong to a loaded
    // WASM plugin, so the DB is opened and `.axil/plugins/` scanned to find out.
    #[cfg(not(feature = "wasm-host"))]
    let (db, surface) = {
        let Some(surface) = builtin else {
            anyhow::bail!(
                "unknown command `{command}` — run `axil --help` for built-in commands, \
                 or `axil extensions list` to see loaded Extensions"
            );
        };
        (open_with_all_detected(&db_path)?, surface)
    };

    #[cfg(feature = "wasm-host")]
    let (db, surface) = {
        let db = std::sync::Arc::new(open_with_all_detected(&db_path)?);
        let _ = wasm_plugins::load_and_register(
            &db,
            &wasm_plugins::plugins_dir(&db_path),
            &config,
        );
        let surface = builtin.or_else(|| {
            db.extensions()
                .iter()
                .find_map(|e| e.cli_commands().filter(|s| s.command == command))
        });
        let Some(surface) = surface else {
            anyhow::bail!(
                "unknown command `{command}` — run `axil --help`, `axil extensions list` for \
                 built-ins, or `axil ext list` for WASM plugins"
            );
        };
        (db, surface)
    };

    let stdin = read_piped_stdin();
    let invocation = build_extension_invocation(&tokens, Some(&surface), stdin);

    match axil_core::dispatch_cli(&db, &db.extensions(), &invocation)
        .map_err(|e| anyhow::anyhow!("{e}"))?
    {
        axil_core::Dispatch::Handled(output) => {
            if !output.stdout.is_empty() {
                print!("{}", output.stdout);
                if !output.stdout.ends_with('\n') {
                    println!();
                }
            }
            if !output.stderr.is_empty() {
                eprint!("{}", output.stderr);
                if !output.stderr.ends_with('\n') {
                    eprintln!();
                }
            }
            Ok(output.exit_code)
        }
        axil_core::Dispatch::NotHandled => anyhow::bail!(
            "command `{command}` is declared by an Extension but it declined to handle \
             this invocation"
        ),
    }
}

/// Atomically replace the file at `old_path` with `new_bytes`, rolling back to
/// the original contents if `validate` rejects the result.
///
/// Pure filesystem logic, deliberately free of the WASM runtime so it is
/// unit-testable without `wasm-host`: the validate step is a closure the caller
/// supplies (in practice "re-inspect the upgraded `.wasm` and confirm it still
/// loads"). The replace is done via a same-directory temp file + `rename`, which
/// is atomic on the same filesystem — a reader never sees a half-written plugin.
/// The previous bytes are read up front so a failed validation can restore them
/// byte-for-byte, leaving the working plugin exactly as it was.
// Only the `wasm-host` upgrade handler calls this at runtime; the unit test
// exercises it in any build, so it's not dead — silence the default-build lint.
#[cfg_attr(not(feature = "wasm-host"), allow(dead_code))]
fn atomic_replace_with_rollback(
    old_path: &Path,
    new_bytes: &[u8],
    validate: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let old_bytes = std::fs::read(old_path)
        .with_context(|| format!("failed to read existing plugin `{}`", old_path.display()))?;

    let dir = old_path.parent().unwrap_or_else(|| Path::new("."));
    // Same-dir temp so the final rename is a same-filesystem (atomic) move.
    let tmp = dir.join(format!(
        ".{}.upgrade-tmp",
        old_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("plugin")
    ));
    std::fs::write(&tmp, new_bytes).with_context(|| {
        format!("failed to stage new plugin bytes at `{}`", tmp.display())
    })?;
    // Swap the staged file into place. POSIX `rename` atomically replaces an
    // existing destination, but on Windows `rename` errors when the destination
    // exists, so remove it first there. The in-memory `old_bytes` covers
    // rollback if either the swap or the validation below fails.
    #[cfg(windows)]
    let swapped = std::fs::remove_file(old_path).and_then(|()| std::fs::rename(&tmp, old_path));
    #[cfg(not(windows))]
    let swapped = std::fs::rename(&tmp, old_path);
    if let Err(e) = swapped {
        let _ = std::fs::remove_file(&tmp);
        // On Windows the original may already be gone (removed above); restore it
        // so a failed swap never leaves the plugin missing.
        if !old_path.exists() {
            let _ = std::fs::write(old_path, &old_bytes);
        }
        return Err(e).with_context(|| {
            format!("failed to swap new plugin into `{}`", old_path.display())
        });
    }

    // The new file is now live; prove it before declaring success. If it fails,
    // restore the original bytes so we never leave a broken plugin in place.
    match validate(old_path) {
        Ok(()) => Ok(()),
        Err(e) => {
            std::fs::write(old_path, &old_bytes).with_context(|| {
                format!(
                    "rollback failed: could not restore previous plugin at `{}` after a \
                     failed upgrade ({e})",
                    old_path.display()
                )
            })?;
            Err(e)
        }
    }
}

#[cfg(feature = "wasm-host")]
fn run_ext(cmd: ExtCommand, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    use std::sync::Arc;

    // `ext new` only writes files — it must work in a fresh checkout with no
    // database, so handle it before `require_db` opens anything.
    if let ExtCommand::New { name, path, caps } = &cmd {
        return scaffold_plugin(name, path.as_deref(), caps.as_deref(), out);
    }

    let db_path = require_db(db_opt)?;
    // Grants + disabled lists live in the db's own dir, not the process cwd.
    let config = plugin_config_for_db(&db_path);
    let dir = wasm_plugins::plugins_dir(&db_path);

    match cmd {
        ExtCommand::List => {
            let db = Arc::new(open_with_all_detected(&db_path)?);
            let records = wasm_plugins::load_and_register(&db, &dir, &config);
            let items: Vec<Value> = records
                .iter()
                .map(|r| {
                    json!({
                        "key": r.key,
                        "id": r.id,
                        "display_name": r.display_name,
                        "prefixes": r.prefixes,
                        "abi": r.abi,
                        "granted": r.granted,
                        "file": r.file.file_name().and_then(|f| f.to_str()),
                        "status": if r.error.is_some() { "failed" } else { "loaded" },
                        "error": r.error,
                    })
                })
                .collect();
            out.print_array(&items);
            Ok(EXIT_OK)
        }
        ExtCommand::Install { path } => {
            let db = Arc::new(open_with_all_detected(&db_path)?);
            // Gate: don't install a `.wasm` that won't load.
            let rec = wasm_plugins::inspect(&db, &path, &config)
                .with_context(|| format!("`{}` is not a loadable plugin", path.display()))?;
            std::fs::create_dir_all(&dir).context("failed to create plugins dir")?;
            let file_name = path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("invalid plugin path"))?;
            let dest = dir.join(file_name);
            std::fs::copy(&path, &dest).context("failed to copy plugin into plugins dir")?;
            out.print(&json!({
                "installed": rec.id,
                "prefixes": rec.prefixes,
                "file": dest.display().to_string(),
            }));
            if !out.quiet {
                out.status(&format!(
                    "installed plugin `{}` — its commands/tools are now available",
                    rec.id.as_deref().unwrap_or("?")
                ));
            }
            Ok(EXIT_OK)
        }
        ExtCommand::Remove { id } => {
            let db = Arc::new(open_with_all_detected(&db_path)?);
            let host = axil_runtime::WasmHost::new().context("WASM runtime init failed")?;
            let mut removed: Option<PathBuf> = None;
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) != Some("wasm") {
                        continue;
                    }
                    let rec = wasm_plugins::load_one(&db, &host, &p, &config, false, None);
                    if rec.id.as_deref() == Some(id.as_str()) {
                        std::fs::remove_file(&p).context("failed to delete plugin file")?;
                        // Sweep the orphaned compiled-module cache entry.
                        let cache_file =
                            dir.join(".cache").join(format!("{}.cwasm", wasm_plugins::plugin_key(&p)));
                        let _ = std::fs::remove_file(&cache_file);
                        removed = Some(p);
                        break;
                    }
                }
            }
            match removed {
                Some(p) => {
                    out.print(&json!({"removed": id, "file": p.display().to_string()}));
                    Ok(EXIT_OK)
                }
                None => anyhow::bail!("no installed plugin with id `{id}`"),
            }
        }
        ExtCommand::Upgrade { id, path } => {
            let db = Arc::new(open_with_all_detected(&db_path)?);
            let host = axil_runtime::WasmHost::new().context("WASM runtime init failed")?;

            // Resolve the existing plugin by id (inspect-only; don't register).
            let mut old_path: Option<PathBuf> = None;
            let mut old_abi: Option<String> = None;
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) != Some("wasm") {
                        continue;
                    }
                    let rec = wasm_plugins::load_one(&db, &host, &p, &config, false, None);
                    if rec.id.as_deref() == Some(id.as_str()) {
                        old_abi = rec.abi;
                        old_path = Some(p);
                        break;
                    }
                }
            }
            let old_path =
                old_path.ok_or_else(|| anyhow::anyhow!("no installed plugin with id `{id}`"))?;

            // Validate the NEW `.wasm` BEFORE touching the working plugin — never
            // replace a loadable plugin with a broken one.
            let new_rec = wasm_plugins::inspect(&db, &path, &config)
                .with_context(|| format!("`{}` is not a loadable plugin", path.display()))?;
            let new_id = new_rec.id.clone().unwrap_or_default();
            let new_abi = new_rec.abi.clone();

            // Same-logical-plugin only: a different id is a different plugin and
            // would silently inherit this one's grants + data.
            if new_id != id {
                anyhow::bail!(
                    "`{}` has id `{new_id}`, not `{id}` — that's a different plugin; use \
                     `ext remove {id}` + `ext install {}`",
                    path.display(),
                    path.display()
                );
            }

            let new_bytes = std::fs::read(&path)
                .with_context(|| format!("failed to read new plugin `{}`", path.display()))?;

            // Atomic replace over the SAME filename ⇒ same grant key, so existing
            // capability grants carry over and data (keyed by table-prefix/id, not
            // filename) is preserved. Re-inspect the swapped file to prove it loads;
            // on failure the helper restores the previous bytes (rollback).
            atomic_replace_with_rollback(&old_path, &new_bytes, |swapped| {
                wasm_plugins::inspect(&db, swapped, &config)
                    .map(|_| ())
                    .with_context(|| "upgraded plugin failed to load")
            })
            .with_context(|| {
                format!(
                    "upgrade failed, rolled back to previous version of `{id}` at `{}`",
                    old_path.display()
                )
            })?;

            // Invalidate the compiled-module cache so the next load recompiles
            // from the new bytes instead of deserializing the stale `.cwasm`.
            let cache_file = dir
                .join(".cache")
                .join(format!("{}.cwasm", wasm_plugins::plugin_key(&old_path)));
            let _ = std::fs::remove_file(&cache_file);

            // The grant key is the filename stem, which is unchanged, so grants
            // are preserved. The WIT ABI surfaces no "requested capabilities", so
            // we can't diff requested-vs-granted; instead we state that grants
            // carried over and any *new* capability need stays deny-by-default
            // until the operator runs `ext grant <key> <cap>` (no silent
            // privilege escalation across an upgrade).
            out.print(&json!({
                "upgraded": id,
                "file": old_path.display().to_string(),
                "abi": new_abi,
                "abi_from": old_abi,
                "prefixes": new_rec.prefixes,
                "granted": new_rec.granted,
                "note": "data and capability grants preserved (same filename ⇒ same grant key); \
                         any new capability the upgrade needs stays deny-by-default until granted \
                         via `ext grant`",
            }));
            if !out.quiet {
                out.status(&format!(
                    "upgraded plugin `{id}` — data and grants preserved"
                ));
            }
            Ok(EXIT_OK)
        }
        ExtCommand::Info { id } => {
            let db = Arc::new(open_with_all_detected(&db_path)?);
            let records = wasm_plugins::load_and_register(&db, &dir, &config);
            match records.iter().find(|r| r.id.as_deref() == Some(id.as_str())) {
                Some(r) => {
                    out.print(&json!({
                        "key": r.key,
                        "id": r.id,
                        "display_name": r.display_name,
                        "prefixes": r.prefixes,
                        "abi": r.abi,
                        "granted": r.granted,
                        "file": r.file.display().to_string(),
                        "status": if r.error.is_some() { "failed" } else { "loaded" },
                    }));
                    Ok(EXIT_OK)
                }
                None => anyhow::bail!("no installed plugin with id `{id}`"),
            }
        }

        ExtCommand::Grant { key, capability } => {
            set_plugin_cap(&key, &capability, true, &db_path, out)
        }
        ExtCommand::Revoke { key, capability } => {
            set_plugin_cap(&key, &capability, false, &db_path, out)
        }
        // Handled above `require_db` (needs no DB); unreachable here.
        ExtCommand::New { .. } => unreachable!("`ext new` is handled before require_db"),
    }
}

/// The canonical `axil:plugin` WIT contract, bundled into the binary so a
/// scaffolded plugin carries its own copy and builds detached from this repo.
#[cfg(feature = "wasm-host")]
const PLUGIN_WIT: &str = include_str!("../../../../wit/axil-plugin.wit");

/// The single canonical `sdk::Plugin` + `export_plugin!` source. The reference
/// guest, the conformance guest, and every `ext new` scaffold all consume this
/// one physical file — there is no second copy of the trait or the macro.
#[cfg(feature = "wasm-host")]
const PLUGIN_SDK_RS: &str = include_str!("../../axil-runtime/test-guest/src/sdk.rs");

/// Scaffold a buildable, detached WASM-plugin crate at `dest` (default
/// `./<name>`). Emits `Cargo.toml` (own `[workspace]`, cdylib, component target
/// pinned at the bundled WIT), `src/lib.rs` (a `sdk::Plugin` stub overriding
/// `handle_cli`), `src/sdk.rs` (the canonical authoring layer), `wit/`, a
/// `build.sh`, a `.gitignore`, and a `README.md` listing requested caps. Needs
/// no database. The result builds with `cargo component build --release`.
#[cfg(feature = "wasm-host")]
fn scaffold_plugin(
    name: &str,
    path: Option<&Path>,
    caps: Option<&str>,
    out: &Output,
) -> Result<i32> {
    // The plugin id is the kebab-case name; the crate name and prefix derive
    // from it deterministically so install + grant key are predictable.
    let id = name.trim();
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        || id.starts_with('-')
        || id.ends_with('-')
    {
        anyhow::bail!(
            "invalid plugin name `{name}` — use kebab-case (lowercase letters, \
             digits, single hyphens), e.g. `my-plugin`"
        );
    }
    // `_<name>_` with hyphens as underscores keeps the prefix a valid, reserved
    // table namespace the host's prefix jail accepts.
    let prefix = format!("_{}_", id.replace('-', "_"));
    let crate_name = id; // cargo accepts hyphens in package names

    let dest = match path {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from(id),
    };
    if dest.exists() {
        anyhow::bail!(
            "`{}` already exists — choose a fresh directory or pass `--path`",
            dest.display()
        );
    }

    let caps_list: Vec<&str> = caps
        .map(|c| {
            c.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    // Files.
    std::fs::create_dir_all(dest.join("src"))
        .with_context(|| format!("failed to create `{}`", dest.join("src").display()))?;
    std::fs::create_dir_all(dest.join("wit"))
        .with_context(|| format!("failed to create `{}`", dest.join("wit").display()))?;

    let cargo_toml = format!(
        r#"# {id} — an Axil WASM plugin (Tier-2 Extension).
#
# Detached from any parent workspace (own `[workspace]`) because it builds for
# wasm32-wasip1 via `cargo component`, not as a native member.
[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"
publish = false

[workspace]

[dependencies]
wit-bindgen-rt = {{ version = "0.41", features = ["bitflags"] }}

[lib]
crate-type = ["cdylib"]

[package.metadata.component]
package = "axil:{id}"

[package.metadata.component.target]
# The bundled copy of Axil's `axil:plugin` WIT — keeps the build self-contained.
path = "wit"
world = "plugin"

[package.metadata.component.dependencies]

[profile.release]
codegen-units = 1
opt-level = "s"
strip = true
"#
    );
    std::fs::write(dest.join("Cargo.toml"), cargo_toml)
        .with_context(|| format!("failed to write `{}`", dest.join("Cargo.toml").display()))?;

    let lib_rs = format!(
        r#"//! {id} — an Axil WASM plugin.
//!
//! Implement [`sdk::Plugin`] and override only the hooks you need; every other
//! method falls back to a "decline / empty / no-op" default. `export_plugin!`
//! then generates the real `axil:plugin` `Guest` impl and the component export.

#[allow(warnings)]
mod bindings;
mod sdk;

use bindings::axil::plugin::types::{{CliInvocation, CliOutput, CliSurface, DispatchCli, PluginError}};
use sdk::Plugin;

struct Component;

impl Plugin for Component {{
    fn id() -> String {{
        "{id}".to_string()
    }}

    fn table_prefixes() -> Vec<String> {{
        // This plugin owns tables under `{prefix}` (the host's prefix jail
        // rejects writes outside it).
        vec!["{prefix}".to_string()]
    }}

    fn cli_commands() -> Option<CliSurface> {{
        Some(CliSurface {{
            command: "{id}".to_string(),
            about: "an Axil WASM plugin".to_string(),
            subcommands: vec![],
        }})
    }}

    fn handle_cli(inv: CliInvocation) -> Result<DispatchCli, PluginError> {{
        Ok(DispatchCli::Handled(CliOutput {{
            exit_code: 0,
            stdout: format!("hello from {id}; args={{:?}}", inv.args),
            stderr: String::new(),
        }}))
    }}
}}

export_plugin!(Component);
"#
    );
    std::fs::write(dest.join("src").join("lib.rs"), lib_rs)
        .with_context(|| format!("failed to write `{}`", dest.join("src/lib.rs").display()))?;

    // The canonical SDK + WIT, copied verbatim from the bundled sources.
    std::fs::write(dest.join("src").join("sdk.rs"), PLUGIN_SDK_RS)
        .with_context(|| format!("failed to write `{}`", dest.join("src/sdk.rs").display()))?;
    std::fs::write(dest.join("wit").join("axil-plugin.wit"), PLUGIN_WIT)
        .with_context(|| format!("failed to write `{}`", dest.join("wit/axil-plugin.wit").display()))?;

    let build_sh = format!(
        "#!/usr/bin/env bash\n\
         # Build the {id} plugin into a `.wasm` component.\n\
         # Requires: cargo-component (it provisions the wasm32-wasip1 target itself).\n\
         set -euo pipefail\n\
         cd \"$(dirname \"$0\")\"\n\
         cargo component build --release\n\
         echo \"built: target/wasm32-wasip1/release/{wasm}.wasm\"\n\
         echo \"install with: axil ext install target/wasm32-wasip1/release/{wasm}.wasm\"\n",
        id = id,
        wasm = crate_name.replace('-', "_"),
    );
    std::fs::write(dest.join("build.sh"), build_sh)
        .with_context(|| format!("failed to write `{}`", dest.join("build.sh").display()))?;

    std::fs::write(dest.join(".gitignore"), "/target\n/src/bindings.rs\nCargo.lock\n")
        .with_context(|| format!("failed to write `{}`", dest.join(".gitignore").display()))?;

    let caps_section = if caps_list.is_empty() {
        "This plugin requests no host capabilities — it runs fully sandboxed.\n".to_string()
    } else {
        let mut s = String::from(
            "After install, grant the host capabilities this plugin needs \
             (deny-by-default):\n\n```sh\n",
        );
        for cap in &caps_list {
            s.push_str(&format!("axil ext grant {id} {cap}\n"));
        }
        s.push_str("```\n");
        s
    };
    let readme = format!(
        "# {id}\n\n\
         An Axil WASM plugin (Tier-2 Extension). Edit `src/lib.rs`, then:\n\n\
         ```sh\n\
         cargo component build --release\n\
         axil ext install target/wasm32-wasip1/release/{wasm}.wasm\n\
         axil {id} <args>\n\
         ```\n\n\
         ## Capabilities\n\n\
         {caps_section}\n\
         ## Authoring\n\n\
         Implement `sdk::Plugin` in `src/lib.rs` and override only the hooks you \
         need (`handle_cli`, `handle_mcp`, `boot_block`, `refresh`, \
         `recall_for_file`, `mcp_tools`). `src/sdk.rs` and `wit/` are the bundled \
         Axil contract — leave them as-is.\n",
        id = id,
        wasm = crate_name.replace('-', "_"),
        caps_section = caps_section,
    );
    std::fs::write(dest.join("README.md"), readme)
        .with_context(|| format!("failed to write `{}`", dest.join("README.md").display()))?;

    out.print(&json!({
        "created": id,
        "path": dest.display().to_string(),
        "prefix": prefix,
        "caps": caps_list,
        "build": "cargo component build --release",
    }));
    if !out.quiet {
        out.status(&format!(
            "scaffolded plugin `{id}` at `{}` — `cd` in and run `cargo component build --release`",
            dest.display()
        ));
    }
    Ok(EXIT_OK)
}

/// Grant or revoke a capability for a plugin by editing `[plugins.<key>]
/// capabilities` in axil.toml. Validates the capability name and is idempotent.
#[cfg(feature = "wasm-host")]
fn set_plugin_cap(
    key: &str,
    capability: &str,
    grant: bool,
    db_path: &Path,
    out: &Output,
) -> Result<i32> {
    if !axil_runtime::Capabilities::ALL_NAMES.contains(&capability) {
        anyhow::bail!(
            "unknown capability `{capability}` — valid: {}",
            axil_runtime::Capabilities::ALL_NAMES.join(", ")
        );
    }
    // Grants live next to the database (the same axil.toml the loader reads from
    // the db dir), not the process cwd — so a grant applies regardless of where
    // `axil` is run from.
    let dir = db_path
        .parent()
        .map(|d| d.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let config = axil_core::load_config_from(&dir).unwrap_or_default();
    let mut caps = config.plugin_capabilities(key);
    let changed = if grant {
        if caps.iter().any(|c| c == capability) {
            false
        } else {
            caps.push(capability.to_string());
            true
        }
    } else {
        let before = caps.len();
        caps.retain(|c| c != capability);
        before != caps.len()
    };

    let config_path = axil_core::find_config_file(&dir).unwrap_or_else(|| dir.join("axil.toml"));
    axil_core::set_config_string_array(&config_path, &format!("plugins.{key}.capabilities"), &caps)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    out.print(&json!({
        "key": key,
        "capability": capability,
        "granted": grant,
        "changed": changed,
        "capabilities": caps,
        "file": config_path.display().to_string(),
    }));
    if !out.quiet {
        let verb = if grant { "granted to" } else { "revoked from" };
        out.status(&format!(
            "capability `{capability}` {verb} plugin `{key}` — applies on the next \
             `axil` command (and in `axil boot` / `axil mcp`)"
        ));
    }
    Ok(EXIT_OK)
}

fn run_extensions(cmd: ExtensionsCommand, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("failed to get current directory")?;

    match cmd {
        ExtensionsCommand::List => {
            let config = axil_core::load_config_from(&cwd).unwrap_or_default();
            let items: Vec<Value> = axil_bundle::builtin_extensions_all()
                .iter()
                .map(|ext| {
                    let disabled = config.is_extension_disabled(ext.id());
                    json!({
                        "id": ext.id(),
                        "display_name": ext.display_name(),
                        "state": if disabled { "disabled" } else { "active" },
                        "table_prefixes": ext.table_prefixes(),
                    })
                })
                .collect();
            out.print_array(&items);
            Ok(EXIT_OK)
        }
        ExtensionsCommand::Enable { id } => set_extension_disabled(&cwd, &id, false, out),
        ExtensionsCommand::Disable { id } => set_extension_disabled(&cwd, &id, true, out),
    }
}

/// Add or remove `id` from `[extensions] disabled` in the resolved axil.toml.
///
/// Validates `id` against the compiled-in catalog so a typo can't silently
/// disable nothing. `changed` is `false` when the toggle was already in the
/// requested state (idempotent).
fn set_extension_disabled(cwd: &Path, id: &str, disabled: bool, out: &Output) -> Result<i32> {
    let known: Vec<String> = axil_bundle::builtin_extensions_all()
        .iter()
        .map(|e| e.id().to_string())
        .collect();
    if !known.iter().any(|k| k == id) {
        anyhow::bail!(
            "unknown built-in extension `{id}` — known: {}",
            if known.is_empty() {
                "(none compiled in)".to_string()
            } else {
                known.join(", ")
            }
        );
    }

    let config = axil_core::load_config_from(cwd).unwrap_or_default();
    let mut list = config.extensions.disabled.clone();
    let changed = if disabled {
        if list.iter().any(|d| d == id) {
            false
        } else {
            list.push(id.to_string());
            true
        }
    } else {
        let before = list.len();
        list.retain(|d| d != id);
        before != list.len()
    };

    let config_path = axil_core::find_config_file(cwd).unwrap_or_else(|| cwd.join("axil.toml"));
    axil_core::set_config_string_array(&config_path, "extensions.disabled", &list)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    out.print(&json!({
        "id": id,
        "disabled": disabled,
        "changed": changed,
        "file": config_path.display().to_string(),
    }));
    if !out.quiet {
        let verb = if disabled { "disabled" } else { "enabled" };
        let note = if changed {
            format!("extension `{id}` {verb} — reopen the database for it to take effect")
        } else {
            format!("extension `{id}` already {verb}")
        };
        out.status(&note);
    }
    Ok(EXIT_OK)
}

fn run_config(cmd: ConfigCommand, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("failed to get current directory")?;

    match cmd {
        ConfigCommand::Init => {
            let config_path = cwd.join("axil.toml");
            if config_path.exists() {
                anyhow::bail!("axil.toml already exists in {}", cwd.display());
            }
            std::fs::write(&config_path, axil_core::default_config_toml())
                .context("failed to write axil.toml")?;
            out.print(&json!({
                "created": true,
                "path": config_path.display().to_string(),
            }));
            Ok(EXIT_OK)
        }

        ConfigCommand::Show => {
            let config = axil_core::load_config_from(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&json!({
                "config": serde_json::to_value(&config).unwrap_or(json!(null)),
            }));
            if !out.quiet {
                if let Ok(toml_str) = toml::to_string_pretty(&config) {
                    out.status(&toml_str);
                }
            }
            Ok(EXIT_OK)
        }

        ConfigCommand::Get { key } => {
            let config = axil_core::load_config_from(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
            match axil_core::get_config_value(&config, &key) {
                Some(value) => {
                    out.print(&json!({"key": key, "value": value}));
                    Ok(EXIT_OK)
                }
                None => {
                    eprintln!("{{\"error\":\"key not found\",\"key\":{}}}", json!(key));
                    Ok(EXIT_NOT_FOUND)
                }
            }
        }

        ConfigCommand::Set { key, value } => {
            // Write to the resolved config file, or create in cwd if none exists
            let config_path =
                axil_core::find_config_file(&cwd).unwrap_or_else(|| cwd.join("axil.toml"));
            axil_core::set_config_value(&config_path, &key, &value)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.print(&json!({"set": true, "key": key, "value": value, "file": config_path.display().to_string()}));
            Ok(EXIT_OK)
        }
    }
}

// ─── Report commands ────────────────────────────────────────────────────────

fn run_report(cmd: ReportCommand, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("failed to get current directory")?;

    match cmd {
        ReportCommand::Generate => {
            let config = axil_core::load_config_from(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;

            let reports_dir = cwd.join(&config.dev.reports_dir);
            std::fs::create_dir_all(&reports_dir).context("failed to create reports directory")?;

            // Collect environment info
            let axil_version = env!("CARGO_PKG_VERSION").to_string();
            let os = std::env::consts::OS;
            let arch = std::env::consts::ARCH;

            let db_info = db_opt
                .as_ref()
                .and_then(|p| open_with_all_detected(p).ok())
                .and_then(|db| db.info().ok())
                .map(|info| {
                    let tables: serde_json::Map<String, Value> = info
                        .tables
                        .iter()
                        .map(|(name, count)| (name.clone(), json!(count)))
                        .collect();
                    json!({
                        "path": info.path.display().to_string(),
                        "size_bytes": info.total_size,
                        "record_count": info.total_records,
                        "tables": tables,
                    })
                })
                .unwrap_or(json!(null));

            // Collect enabled features
            #[allow(clippy::vec_init_then_push)]
            let features = {
                let mut v: Vec<&str> = Vec::new();
                #[cfg(feature = "vector")]
                v.push("vector");
                #[cfg(feature = "graph")]
                v.push("graph");
                #[cfg(feature = "fts")]
                v.push("fts");
                #[cfg(feature = "timeseries")]
                v.push("timeseries");
                #[cfg(feature = "embed")]
                v.push("embed");
                v
            };

            let now = chrono::Utc::now();
            let report = json!({
                "version": "1.0",
                "generated_at": now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                "axil_version": axil_version,
                "environment": {
                    "os": os,
                    "arch": arch,
                    "features": features,
                },
                "database": db_info,
                "problems": [],
                "usage_patterns": {},
                "config": serde_json::to_value(&config).unwrap_or(json!(null)),
            });

            let filename = format!("report-{}.json", now.format("%Y-%m-%d-%H%M%S"));
            let report_path = reports_dir.join(&filename);
            let report_str =
                serde_json::to_string_pretty(&report).context("failed to serialize report")?;
            std::fs::write(&report_path, &report_str).context("failed to write report")?;

            out.print(&json!({
                "generated": true,
                "path": report_path.display().to_string(),
                "size_bytes": report_str.len(),
            }));
            Ok(EXIT_OK)
        }

        ReportCommand::List => {
            let config = axil_core::load_config_from(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
            let reports_dir = cwd.join(&config.dev.reports_dir);

            if !reports_dir.exists() {
                out.print_array(&[]);
                return Ok(EXIT_OK);
            }

            let mut reports: Vec<Value> = Vec::new();
            for entry in
                std::fs::read_dir(&reports_dir).context("failed to read reports directory")?
            {
                let entry = entry?;
                let path = entry.path();
                if is_json_file(&path) {
                    let meta = entry.metadata()?;
                    reports.push(json!({
                        "path": path.display().to_string(),
                        "filename": path.file_name().unwrap().to_string_lossy(),
                        "size_bytes": meta.len(),
                    }));
                }
            }
            reports.sort_by(|a, b| a["filename"].as_str().cmp(&b["filename"].as_str()));

            out.print_array(&reports);
            Ok(EXIT_OK)
        }

        ReportCommand::Import { path, from } => {
            let config = axil_core::load_config_from(&cwd).map_err(|e| anyhow::anyhow!("{e}"))?;
            let incoming_dir = cwd.join(&config.diagnose.incoming_dir);
            std::fs::create_dir_all(&incoming_dir)
                .context("failed to create incoming directory")?;

            let source_path = if let Some(p) = path {
                p
            } else if let Some(project_dir) = from {
                // Find the latest report in the project's reports dir
                let project_config = axil_core::load_config_from(&project_dir)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let project_reports = project_dir.join(&project_config.dev.reports_dir);
                if !project_reports.exists() {
                    anyhow::bail!(
                        "no reports directory found at {}",
                        project_reports.display()
                    );
                }
                let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;
                for entry in std::fs::read_dir(&project_reports)? {
                    let entry = entry?;
                    let path = entry.path();
                    if is_json_file(&path) {
                        let modified = entry.metadata()?.modified()?;
                        if latest.as_ref().map(|(_, t)| modified > *t).unwrap_or(true) {
                            latest = Some((path, modified));
                        }
                    }
                }
                latest.map(|(p, _)| p).ok_or_else(|| {
                    anyhow::anyhow!("no reports found in {}", project_reports.display())
                })?
            } else {
                anyhow::bail!("provide a report path or use --from <project_dir>");
            };

            // Guard against accidentally importing huge files
            const MAX_REPORT_BYTES: u64 = 10 * 1024 * 1024; // 10 MB
            let meta = std::fs::metadata(&source_path)
                .with_context(|| format!("failed to stat report: {}", source_path.display()))?;
            if meta.len() > MAX_REPORT_BYTES {
                anyhow::bail!(
                    "report too large ({} bytes, max {} bytes): {}",
                    meta.len(),
                    MAX_REPORT_BYTES,
                    source_path.display()
                );
            }
            let contents = std::fs::read_to_string(&source_path)
                .with_context(|| format!("failed to read report: {}", source_path.display()))?;
            let _: Value = serde_json::from_str(&contents).context("report is not valid JSON")?;

            let filename = source_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("invalid report path"))?;
            let dest = incoming_dir.join(filename);
            std::fs::write(&dest, &contents).context("failed to write report")?;

            out.print(&json!({
                "imported": true,
                "source": source_path.display().to_string(),
                "destination": dest.display().to_string(),
            }));
            Ok(EXIT_OK)
        }
    }
}

// ─── Token-budgeted recall helpers ─────────────────────────────────

/// Format recall results according to the specified format, then apply budget.
fn format_recall_results(
    values: &[Value],
    format: &RecallFormat,
    budget: Option<usize>,
) -> Vec<Value> {
    // Fast path: Full format with no budget — return as-is without cloning
    if matches!(format, RecallFormat::Full) && budget.is_none() {
        return values.to_vec();
    }

    let formatted: Vec<Value> = match format {
        RecallFormat::Full | RecallFormat::ContextBlock => values.to_vec(),
        RecallFormat::Compact => values
            .iter()
            .map(|v| {
                let summary = pick_summary(v.get("data"))
                    .map(|s| truncate_str(s, 200))
                    .unwrap_or_default();
                json!({
                    "id": v.get("id"),
                    "score": v.get("score"),
                    "table": v.get("table"),
                    "summary": summary,
                })
            })
            .collect(),
        RecallFormat::Oneline => values
            .iter()
            .map(|v| {
                let score = v.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                let summary = pick_summary(v.get("data"))
                    .map(|s| truncate_str(s, 120))
                    .unwrap_or_else(|| "?".into());
                let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("?");
                json!(format!("{:.3} | {} | {}", score, summary, id))
            })
            .collect(),
    };

    // Apply token budget: estimate tokens as bytes/4, pack highest-scored first
    if let Some(max_tokens) = budget {
        let max_bytes = max_tokens * 4;
        let mut result = Vec::new();
        let mut used = 2; // opening/closing brackets
        for v in &formatted {
            let serialized = serde_json::to_string(v).unwrap_or_default();
            let entry_bytes = serialized.len() + 1; // comma
            if used + entry_bytes > max_bytes && !result.is_empty() {
                break;
            }
            used += entry_bytes;
            result.push(v.clone());
        }
        result
    } else {
        formatted
    }
}

/// Shared "best available human-readable string" extractor (12.6 cleanup).
///
/// Used by every surface that summarizes a record (`format_context_block`,
/// `format_recall_results`, `generate_brief::summarize`, `rerank_via_llm`)
/// so the priority order stays consistent and a new field (`content`,
/// `statement`, `fix`, …) only needs to be added once.
fn pick_summary(data: Option<&Value>) -> Option<&str> {
    let d = data?;
    for key in [
        "summary",
        "description",
        "statement",
        "error",
        "fix",
        "content",
        "path",
    ] {
        if let Some(s) = d.get(key).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

// ── Query expansion (12.5) ──────────────────────────────────────────────────

/// Expand a query with alias synonyms + one-hop graph-neighbor terms. Best-effort;
/// returns the original query on any failure so recall never breaks.
///
/// Strategy:
/// 1. Extract entities from the query via pattern-based extraction (no LLM).
/// 2. For each entity: resolve canonical name, harvest registered aliases.
/// 3. Look up a canonical entity record and take up to `neighbors * 3` one-hop outgoing neighbors.
/// 4. Append the novel terms to the original query, deduped.
///
/// Note: this only walks one hop — the `neighbors` budget caps fan-out, not depth.
fn expand_query(db: &axil_core::Axil, query: &str, neighbors: usize) -> String {
    let entities = axil_core::entity::extract_entities(query);
    if entities.is_empty() {
        return query.to_string();
    }

    let mut extra_terms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    #[cfg(feature = "memory")]
    {
        let mem = axil_memory::AgentMemory::new(db);
        let semantic = mem.semantic();
        for entity in &entities {
            // (1) alias set — resolve canonical, then list aliases for it.
            let canonical = semantic
                .resolve(&entity.name)
                .ok()
                .flatten()
                .unwrap_or_else(|| entity.name.clone());
            if canonical != entity.name {
                extra_terms.insert(canonical.clone());
            }
            if let Ok(aliases) = semantic.aliases(&canonical) {
                for a in aliases {
                    if a != entity.name && a.len() >= 2 {
                        extra_terms.insert(a);
                    }
                }
            }
        }
    }

    // (2) graph neighbors — build a single name→record index per query rather than
    // scanning the entities table once per entity (was O(entities × table_size)).
    // Also drop the old `facts`/`decisions` sweep: neither table stores records
    // keyed by a `name` field, so the match on line `data.name` never hit.
    #[cfg(all(feature = "graph", feature = "memory"))]
    {
        use axil_core::Direction;
        use std::collections::HashMap;
        let mem = axil_memory::AgentMemory::new(db);
        let semantic = mem.semantic();
        let by_name: HashMap<String, axil_core::Record> = db
            .list("_entities")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|r| {
                let n = r
                    .data
                    .get("name")
                    .and_then(|v| v.as_str())?
                    .to_ascii_lowercase();
                Some((n, r))
            })
            .collect();
        for entity in &entities {
            let canonical = semantic
                .resolve(&entity.name)
                .ok()
                .flatten()
                .unwrap_or_else(|| entity.name.clone());
            let Some(seed) = by_name.get(&canonical.to_ascii_lowercase()) else {
                continue;
            };
            let Ok(hop) = db.neighbors(&seed.id, None, Direction::Out) else {
                continue;
            };
            for n in hop.into_iter().take(neighbors.saturating_mul(3)) {
                if let Some(name) = n.data.get("name").and_then(|v| v.as_str()) {
                    if name.len() >= 2 && !name.eq_ignore_ascii_case(&entity.name) {
                        extra_terms.insert(name.to_string());
                    }
                }
            }
        }
    }

    if extra_terms.is_empty() {
        return query.to_string();
    }
    let extras: Vec<String> = extra_terms.into_iter().collect();
    eprintln!(
        "[expand] +{} terms: {}",
        extras.len(),
        extras
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    format!("{query} {}", extras.join(" "))
}

// ── LLM reranker (12.4) ─────────────────────────────────────────────────────

/// Rerank `results` in-place using the DB's configured LLM.
///
/// Strategy: send a single numbered list to the LLM, ask for a JSON array of
/// indices ordered by relevance. Cost-tracked and rate-limited by the DB's
/// `llm_call_guarded` path. On any failure (no LLM, bad JSON, limit hit)
/// returns `Err` — caller falls back to RRF order.
fn rerank_via_llm(db: &axil_core::Axil, query: &str, results: &mut Vec<Value>) -> Result<()> {
    if results.len() < 2 {
        return Ok(());
    }
    let rerank_count = results.len().min(20);
    let mut prompt = String::with_capacity(4096);
    prompt.push_str(
        "Rank the following passages by relevance to the QUERY.\n\
        Output ONLY a JSON array of 0-based indices in order, most relevant first.\n\
        Example: [3, 0, 1, 2]\n\n",
    );
    prompt.push_str(&format!("QUERY: {query}\n\nPASSAGES:\n"));
    for (i, v) in results.iter().take(rerank_count).enumerate() {
        let text = pick_summary(v.get("data"))
            .or_else(|| v.get("summary").and_then(|s| s.as_str()))
            .map(|s| truncate_str(s, 300))
            .unwrap_or_default();
        prompt.push_str(&format!("[{i}] {text}\n"));
    }
    let response = db
        .llm_complete(&prompt)
        .map_err(|e| anyhow::anyhow!("llm call: {e}"))?;

    // Extract the first JSON array from the response, tolerant of markdown fences / preamble.
    // `split_once` walks char boundaries safely — avoids the byte-indexing panic on multibyte preambles.
    let (_, after_open) = response
        .text
        .split_once('[')
        .ok_or_else(|| anyhow::anyhow!("no JSON array in LLM response"))?;
    let (inner, _) = after_open
        .split_once(']')
        .ok_or_else(|| anyhow::anyhow!("unterminated JSON array in LLM response"))?;
    let array_slice = format!("[{inner}]");
    let order: Vec<usize> =
        serde_json::from_str(&array_slice).map_err(|e| anyhow::anyhow!("parse order: {e}"))?;

    // Reorder: validated indices first, then anything left over in original order.
    let mut seen = vec![false; rerank_count];
    let mut reordered: Vec<Value> = Vec::with_capacity(results.len());
    for idx in order.iter().filter(|i| **i < rerank_count) {
        if !seen[*idx] {
            let mut item = results[*idx].clone();
            if let Some(obj) = item.as_object_mut() {
                obj.insert("rerank_source".to_string(), json!("llm"));
            }
            reordered.push(item);
            seen[*idx] = true;
        }
    }
    for (i, item) in results.iter().take(rerank_count).enumerate() {
        if !seen[i] {
            reordered.push(item.clone());
        }
    }
    for item in results.iter().skip(rerank_count) {
        reordered.push(item.clone());
    }
    *results = reordered;
    Ok(())
}

// ── Brief / Retro / Schedule helpers (12.3) ─────────────────────────────────

/// Resolve the window start for `--window <dur>` or `--after <iso>`.
/// Reuses `parse_duration` (s/m/h/d) for consistency with `axil since`.
fn resolve_window_start(
    window: &str,
    after: Option<&str>,
) -> Result<chrono::DateTime<chrono::Utc>> {
    if let Some(s) = after {
        let micros = parse_datetime_us(s)?;
        return chrono::DateTime::from_timestamp_micros(micros)
            .ok_or_else(|| anyhow::anyhow!("invalid --after value"));
    }
    // Pre-handle the week suffix since `parse_duration` stops at `d`.
    let normalized = if let Some(n) = window.trim().strip_suffix('w') {
        format!(
            "{}d",
            n.parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid --window: {window}"))?
                * 7
        )
    } else {
        window.trim().to_string()
    };
    let secs = parse_duration(&normalized)?;
    Ok(chrono::Utc::now() - chrono::Duration::seconds(secs as i64))
}

/// Build a structured brief/retro report from records since `since`.
fn generate_brief(
    db: &axil_core::Axil,
    since: chrono::DateTime<chrono::Utc>,
    window_label: &str,
    budget: Option<usize>,
) -> Result<Value> {
    // Push the time filter into storage via `db.since` (avoids full-table scans).
    let duration_secs = (chrono::Utc::now() - since).num_seconds().max(0) as u64;
    let fetch = |table: &str, limit: usize| -> Vec<axil_core::Record> {
        db.since(Some(table), duration_secs)
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .collect()
    };

    let sessions = fetch("_sessions", 50);
    let decisions = fetch("decisions", 50);
    let errors = fetch("errors", 50);
    let context_records = fetch("context", 30);

    // Open threads: errors in the window that have no corresponding fix note; supersedes check is overkill here.
    let open_errors: Vec<&axil_core::Record> = errors
        .iter()
        .filter(|r| {
            r.data
                .get("fix")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .is_empty()
        })
        .collect();

    // Top-of-mind: highest-importance records touched in the window.
    let mut important: Vec<(axil_core::Record, f32)> = decisions
        .iter()
        .chain(context_records.iter())
        .map(|r| (r.clone(), axil_core::importance::get_importance(&r.data)))
        .collect();
    important.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    important.truncate(5);

    // Narrative string used as a record summary if saved.
    let narrative = format!(
        "Window {window_label}: {} sessions, {} decisions, {} errors ({} open).",
        sessions.len(),
        decisions.len(),
        errors.len(),
        open_errors.len()
    );

    // Pattern recognition — leverage stored patterns if present, else skip gracefully.
    let patterns: Vec<Value> = fetch("_patterns", 5)
        .into_iter()
        .map(|r| {
            json!({
                "summary": r.data.get("summary").and_then(|v| v.as_str()).unwrap_or(""),
                "occurrences": r.data.get("occurrences").and_then(|v| v.as_u64()).unwrap_or(0),
            })
        })
        .collect();

    let summarize = |r: &axil_core::Record| -> Value {
        let summary = pick_summary(Some(&r.data))
            .map(|s| truncate_str(s, 200))
            .unwrap_or_default();
        json!({
            "id": r.id.to_string(),
            "table": r.table,
            "summary": summary,
            "created_at": r.created_at.to_rfc3339(),
        })
    };

    let report = json!({
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "window": window_label,
        "since": since.to_rfc3339(),
        "counts": {
            "sessions": sessions.len(),
            "decisions": decisions.len(),
            "errors": errors.len(),
            "open_errors": open_errors.len(),
            "context": context_records.len(),
        },
        "narrative": narrative,
        "recent_decisions": decisions.iter().take(8).map(summarize).collect::<Vec<_>>(),
        "open_threads": open_errors.iter().take(8).map(|r| summarize(r)).collect::<Vec<_>>(),
        "top_of_mind": important.iter().map(|(r, s)| {
            let mut v = summarize(r);
            v["importance"] = json!(s);
            v
        }).collect::<Vec<_>>(),
        "patterns": patterns,
    });

    if let Some(max_tokens) = budget {
        // Crude budget application: serialize, if over cap drop optional fields.
        let serialized = serde_json::to_string(&report)?;
        if serialized.len() / 4 > max_tokens {
            let trimmed = json!({
                "generated_at": report["generated_at"],
                "window": report["window"],
                "narrative": report["narrative"],
                "counts": report["counts"],
                "recent_decisions": report["recent_decisions"],
                "open_threads": report["open_threads"],
            });
            return Ok(trimmed);
        }
    }
    Ok(report)
}

/// Render a brief report to markdown for human reading.
fn render_brief_markdown(report: &Value, is_retro: bool) -> String {
    let mut out = String::new();
    let header = if is_retro {
        "# Retrospective"
    } else {
        "# Daily Brief"
    };
    out.push_str(header);
    out.push('\n');
    if let Some(w) = report.get("window").and_then(|v| v.as_str()) {
        out.push_str(&format!("\n*Window: {w}*\n"));
    }
    if let Some(n) = report.get("narrative").and_then(|v| v.as_str()) {
        out.push_str(&format!("\n{n}\n"));
    }
    let section = |out: &mut String, title: &str, key: &str| {
        if let Some(items) = report.get(key).and_then(|v| v.as_array()) {
            if !items.is_empty() {
                out.push_str(&format!("\n## {title}\n\n"));
                for item in items {
                    let s = item
                        .get("summary")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(no summary)");
                    let table = item.get("table").and_then(|v| v.as_str()).unwrap_or("");
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    out.push_str(&format!("- [{table}] {s} (id={id})\n"));
                }
            }
        }
    };
    section(&mut out, "Recent decisions", "recent_decisions");
    section(&mut out, "Open threads (un-fixed errors)", "open_threads");
    section(&mut out, "Top of mind", "top_of_mind");
    if let Some(patterns) = report.get("patterns").and_then(|v| v.as_array()) {
        if !patterns.is_empty() {
            out.push_str("\n## Patterns\n\n");
            for p in patterns {
                let s = p.get("summary").and_then(|v| v.as_str()).unwrap_or("");
                let occ = p.get("occurrences").and_then(|v| v.as_u64()).unwrap_or(0);
                out.push_str(&format!("- {s} (×{occ})\n"));
            }
        }
    }
    if is_retro {
        out.push_str("\n## Changes for next period\n\n_Fill in after review._\n");
    }
    out
}

/// Route to stdout in the requested format.
fn emit_brief(report: &Value, format: &BriefFormat, out: &Output) {
    match format {
        BriefFormat::Json => out.print(report),
        BriefFormat::Markdown => print!("{}", render_brief_markdown(report, false)),
    }
}

/// Schedule plan — materialized before any filesystem write so `--dry-run` can show it.
struct ScheduledTask {
    name: String,
    scheduler: String,
    install_path: PathBuf,
    command: String,
    content: String,
    /// systemd needs a companion .service unit alongside the .timer — populated only for that backend.
    companion_path: Option<PathBuf>,
    companion_content: Option<String>,
}

/// How often a scheduled task should fire. Drives the launchd plist and systemd `OnCalendar`.
#[derive(Clone, Copy)]
enum Cadence {
    Daily,
    /// Sunday.
    Weekly,
    /// 1st of the month.
    Monthly,
}

impl Cadence {
    fn for_name(name: &str) -> Self {
        match name {
            "weekly-retro" => Cadence::Weekly,
            "monthly-retro" => Cadence::Monthly,
            _ => Cadence::Daily,
        }
    }
}

/// Choose scheduler based on `--scheduler` flag + platform auto-detection.
fn resolve_scheduler(requested: &str) -> &'static str {
    match requested {
        "launchd" => "launchd",
        "systemd" => "systemd",
        "cron" => "cron",
        _ => {
            #[cfg(target_os = "macos")]
            {
                "launchd"
            }
            #[cfg(all(unix, not(target_os = "macos")))]
            {
                "systemd"
            }
            #[cfg(not(unix))]
            {
                "cron"
            }
        }
    }
}

/// Axil-owned directory for per-user scheduler artifacts.
fn axil_schedule_dir() -> Result<PathBuf> {
    let home =
        axil_core::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    Ok(home.join(".axil").join("schedule"))
}

fn plan_scheduled_task(
    name: &str,
    hour: u32,
    minute: u32,
    scheduler: &str,
    db_path: &Path,
) -> Result<ScheduledTask> {
    let valid = matches!(name, "daily-brief" | "weekly-retro" | "monthly-retro");
    if !valid {
        anyhow::bail!(
            "unknown schedule name `{name}`. Use daily-brief, weekly-retro, or monthly-retro."
        );
    }
    let sched = resolve_scheduler(scheduler).to_string();
    let cadence = Cadence::for_name(name);
    // Shell-quote the binary path: the executable's install location
    // may contain spaces or shell metacharacters (e.g. macOS
    // `/Applications/Some App.app/...`). `sh -lc '{command}'` would
    // otherwise break command parsing or, worst-case, allow
    // metacharacter injection from a hostile parent process.
    let axil_bin = shell_quote(
        &std::env::current_exe()
            .ok()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "axil".into()),
    );
    // Bind --db to the absolute project database path so the scheduler can run from $HOME / `/`
    // without needing AXIL_DB in the launchd/systemd/cron environment.
    let db_abs = db_path
        .canonicalize()
        .unwrap_or_else(|_| db_path.to_path_buf());
    let db_flag = format!("--db {}", shell_quote(&db_abs.display().to_string()));
    let command = match name {
        "daily-brief" => format!("{axil_bin} {db_flag} brief --window 24h --brief-format markdown"),
        "weekly-retro" => format!("{axil_bin} {db_flag} retro --window 7d --save"),
        "monthly-retro" => format!("{axil_bin} {db_flag} retro --window 30d --save"),
        _ => unreachable!(),
    };

    let dir = axil_schedule_dir()?;
    let (install_path, content, companion_path, companion_content) = match sched.as_str() {
        "launchd" => {
            let plist_path = dir.join(format!("com.axil.{name}.plist"));
            let content = render_launchd_plist(name, hour, minute, cadence, &command);
            (plist_path, content, None, None)
        }
        "systemd" => {
            let timer_path = dir.join(format!("axil-{name}.timer"));
            let service_path = dir.join(format!("axil-{name}.service"));
            let timer = render_systemd_timer(name, hour, minute, cadence);
            let service = render_systemd_service(name, &command);
            (timer_path, timer, Some(service_path), Some(service))
        }
        "cron" => {
            let cron_path = dir.join(format!("axil-{name}.cron"));
            let dow = match cadence {
                Cadence::Weekly => "0",
                _ => "*",
            };
            let dom = match cadence {
                Cadence::Monthly => "1",
                _ => "*",
            };
            let content = format!("{minute} {hour} {dom} * {dow} {command}\n");
            (cron_path, content, None, None)
        }
        _ => unreachable!(),
    };

    Ok(ScheduledTask {
        name: name.to_string(),
        scheduler: sched,
        install_path,
        command,
        content,
        companion_path,
        companion_content,
    })
}

/// POSIX shell single-quote escaping for paths that may contain spaces.
fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':'))
    {
        return s.to_string();
    }
    // Close-quote, escaped single-quote, reopen-quote — the classic sh idiom.
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn install_scheduled_task(task: &ScheduledTask) -> Result<()> {
    if let Some(parent) = task.install_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&task.install_path, &task.content)?;
    // systemd timers are dead-on-arrival without the matching service unit; write it next to the timer.
    if let (Some(cpath), Some(ccontent)) = (&task.companion_path, &task.companion_content) {
        std::fs::write(cpath, ccontent)?;
    }
    // For launchd we mirror to LaunchAgents so the OS picks it up; remain a template-only for systemd/cron
    // since those require sudo or user-bus activation the user should handle.
    if task.scheduler == "launchd" {
        if let Some(home) = axil_core::home_dir() {
            let la_dir = home.join("Library").join("LaunchAgents");
            if la_dir.exists() {
                let dest = la_dir.join(
                    task.install_path
                        .file_name()
                        .ok_or_else(|| anyhow::anyhow!("plist path missing file name"))?,
                );
                std::fs::copy(&task.install_path, &dest)?;
            }
        }
    }
    Ok(())
}

fn list_scheduled_tasks() -> Result<Vec<Value>> {
    let dir = axil_schedule_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let sched = if filename.ends_with(".plist") {
            "launchd"
        } else if filename.ends_with(".timer") {
            "systemd"
        } else if filename.ends_with(".cron") {
            "cron"
        } else {
            "?"
        };
        out.push(json!({
            "name": filename,
            "scheduler": sched,
            "path": path.display().to_string(),
        }));
    }
    Ok(out)
}

fn uninstall_scheduled_task(name: &str, dry_run: bool) -> Result<Vec<String>> {
    let dir = axil_schedule_dir()?;
    let mut removed: Vec<String> = Vec::new();
    for suffix in &[".plist", ".timer", ".service", ".cron"] {
        let p = dir.join(format!("axil-{name}{suffix}"));
        let p_alt = dir.join(format!("com.axil.{name}{suffix}"));
        for candidate in [p, p_alt] {
            if candidate.exists() {
                if !dry_run {
                    let _ = std::fs::remove_file(&candidate);
                }
                removed.push(candidate.display().to_string());
            }
        }
    }
    if let Some(home) = axil_core::home_dir() {
        let la = home
            .join("Library")
            .join("LaunchAgents")
            .join(format!("com.axil.{name}.plist"));
        if la.exists() {
            if !dry_run {
                let _ = std::fs::remove_file(&la);
            }
            removed.push(la.display().to_string());
        }
    }
    Ok(removed)
}

fn render_launchd_plist(
    name: &str,
    hour: u32,
    minute: u32,
    cadence: Cadence,
    command: &str,
) -> String {
    // launchd honors Weekday (0=Sunday) and Day (1..=31) keys. Daily sets neither so it fires every day.
    let cadence_keys = match cadence {
        Cadence::Daily => String::new(),
        Cadence::Weekly => "        <key>Weekday</key><integer>0</integer>\n".to_string(),
        Cadence::Monthly => "        <key>Day</key><integer>1</integer>\n".to_string(),
    };
    // Route stdout/stderr to a user-owned log directory rather than
    // predictable `/tmp/axil-<name>.{out,err}.log`. A local attacker
    // could pre-create symlinks at those `/tmp` paths and have the
    // scheduled task clobber arbitrary files on each run. The
    // schedule dir resolves to `~/Library/Logs/axil/schedule/` on
    // macOS — user-owned, attacker can't pre-create a symlink there
    // unless they already control the user account.
    let log_dir = launchd_log_dir_for_name(name);
    let stdout_path = log_dir
        .join(format!("axil-{name}.out.log"))
        .display()
        .to_string();
    let stderr_path = log_dir
        .join(format!("axil-{name}.err.log"))
        .display()
        .to_string();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyLists-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.axil.{name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-lc</string>
        <string>{command}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>{hour}</integer>
        <key>Minute</key><integer>{minute}</integer>
{cadence_keys}    </dict>
    <key>StandardOutPath</key><string>{stdout_path}</string>
    <key>StandardErrorPath</key><string>{stderr_path}</string>
</dict>
</plist>
"#
    )
}

/// Resolve the launchd log directory (user-owned). Creates the
/// directory if missing so the plist's `StandardOutPath` /
/// `StandardErrorPath` can write on first run. Falls back to
/// `<schedule_dir>/logs` if `axil_schedule_dir` resolves but the
/// user's `Library/Logs` doesn't exist; ultimate fallback to
/// `/tmp/axil-logs-<uid>` (still user-scoped, not the previous
/// shared `/tmp/axil-<name>.*` pattern).
fn launchd_log_dir_for_name(_name: &str) -> std::path::PathBuf {
    let home = axil_core::home_dir();
    let preferred = home
        .as_ref()
        .map(|h| h.join("Library").join("Logs").join("axil").join("schedule"));
    if let Some(dir) = preferred {
        let _ = std::fs::create_dir_all(&dir);
        if dir.exists() {
            return dir;
        }
    }
    // Last-resort fallback. Use uid in the path so concurrent users
    // on a shared box still can't symlink-attack each other.
    let uid = std::env::var("USER").unwrap_or_else(|_| "axil".to_string());
    let fallback = std::env::temp_dir().join(format!("axil-logs-{uid}"));
    let _ = std::fs::create_dir_all(&fallback);
    fallback
}

fn render_systemd_timer(name: &str, hour: u32, minute: u32, cadence: Cadence) -> String {
    let on_calendar = match cadence {
        Cadence::Daily => format!("*-*-* {hour:02}:{minute:02}:00"),
        Cadence::Weekly => format!("Sun *-*-* {hour:02}:{minute:02}:00"),
        Cadence::Monthly => format!("*-*-01 {hour:02}:{minute:02}:00"),
    };
    format!(
        r#"# Copy this file + axil-{name}.service into ~/.config/systemd/user/,
# then: systemctl --user daemon-reload && systemctl --user enable --now axil-{name}.timer

[Unit]
Description=Axil {name}

[Timer]
OnCalendar={on_calendar}
Persistent=true
Unit=axil-{name}.service

[Install]
WantedBy=timers.target
"#
    )
}

fn render_systemd_service(name: &str, command: &str) -> String {
    format!(
        r#"[Unit]
Description=Axil {name}

[Service]
Type=oneshot
ExecStart=/bin/sh -lc '{command}'
"#
    )
}

/// Collect the candidate files under `root_canonical` that match the
/// extension set and survive the exclude patterns. Shared by the dry-run
/// (`--stats`) plan and the real ingest pass so both see the same set.
#[cfg(feature = "indexer")]
fn ingest_collect_candidates(
    root_canonical: &Path,
    recursive: bool,
    exts: &std::collections::HashSet<String>,
    exclude_patterns: &[String],
) -> Vec<(PathBuf, u64)> {
    let is_excluded = |path: &Path| -> bool {
        let s = path.to_string_lossy();
        exclude_patterns.iter().any(|p| s.contains(p))
    };
    let walker = walkdir::WalkDir::new(root_canonical)
        .max_depth(if recursive { 32 } else { 1 })
        .into_iter()
        .filter_entry(|e| !is_excluded(e.path()));

    let mut candidates: Vec<(PathBuf, u64)> = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.into_path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if !exts.contains(&ext) {
            continue;
        }
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        candidates.push((path, size));
    }
    candidates
}

/// Run one incremental ingest pass and return the summary report.
///
/// This is the body shared by the one-shot `axil ingest` path and each tick
/// of `--watch`. Unchanged files are skipped via mtime then content-hash, so a
/// watch tick only re-ingests files that actually changed. State is reloaded
/// from `.axil/ingest.state.json` on every call, so each tick (and `--resume`)
/// builds on the prior run; checkpointing it back to disk is what makes the
/// next tick cheap.
#[cfg(feature = "indexer")]
#[allow(clippy::too_many_arguments)]
fn run_ingest_pass(
    db: &Axil,
    root_canonical: &Path,
    candidates: &[(PathBuf, u64)],
    state_path: &Path,
    table: &str,
    chunk_bytes: usize,
    resume: bool,
    out: &Output,
) -> Result<Value> {
    // Reload prior state every pass: this is what lets a watch tick (or
    // `--resume`) trust earlier hashes instead of re-ingesting everything.
    let mut state: serde_json::Map<String, Value> = if (resume) && state_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(state_path).unwrap_or_default())
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };

    let total = candidates.len();
    let started = std::time::Instant::now();
    let mut files_ingested: usize = 0;
    let mut files_skipped: usize = 0;
    let mut chunks_written: usize = 0;

    for (idx, (path, _size)) in candidates.iter().enumerate() {
        let key = path.display().to_string();
        let prev_entry = state.get(&key);

        // Fast skip: mtime unchanged → trust the prior hash and don't even read the file.
        // Saves one syscall + full file read per unchanged file (10k-file runs feel it).
        let mtime_now = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        if let (Some(prev), Some(now)) = (
            prev_entry
                .and_then(|v| v.get("mtime"))
                .and_then(|v| v.as_u64()),
            mtime_now,
        ) {
            if prev == now {
                files_skipped += 1;
                if (idx + 1) % 50 == 0 {
                    out.status(&format!(
                        "[ingest] {}/{} scanned ({} skipped)",
                        idx + 1,
                        total,
                        files_skipped
                    ));
                }
                continue;
            }
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                files_skipped += 1;
                continue;
            }
        };
        if content.trim().is_empty() {
            files_skipped += 1;
            continue;
        }

        // Shared with the indexer so one hash function catches all change-detection paths.
        let hash = axil_indexer::indexer::hash_content(&content);
        if let Some(prev_hash) = prev_entry
            .and_then(|v| v.get("hash"))
            .and_then(|v| v.as_str())
        {
            if prev_hash == hash {
                files_skipped += 1;
                if (idx + 1) % 50 == 0 {
                    out.status(&format!(
                        "[ingest] {}/{} scanned ({} skipped)",
                        idx + 1,
                        total,
                        files_skipped
                    ));
                }
                continue;
            }
        }

        // Purge any chunk records from a prior ingest of this exact path so the new
        // chunk set fully replaces the old one — otherwise edits accumulate duplicates.
        // We trust state first (O(1) id lookup), then fall back to a filtered list().
        let mut chunks_replaced: usize = 0;
        let prior_ids: Vec<String> = prev_entry
            .and_then(|v| v.get("record_ids"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if !prior_ids.is_empty() {
            for id_str in &prior_ids {
                if let Ok(rid) = axil_core::RecordId::from_string(id_str.as_str()) {
                    if db.delete(&rid).unwrap_or(false) {
                        chunks_replaced += 1;
                    }
                }
            }
        } else if prev_entry.is_some() {
            // Older ingest state without record_ids — scan the table once as a fallback.
            let abs = path.display().to_string();
            for r in db.list(table).unwrap_or_default() {
                if r.data.get("abs_path").and_then(|v| v.as_str()) == Some(abs.as_str()) {
                    if db.delete(&r.id).unwrap_or(false) {
                        chunks_replaced += 1;
                    }
                }
            }
        }

        // Chunk the file and insert fresh records, tracking their IDs for the next ingest.
        let chunks = chunk_text(&content, chunk_bytes);
        let rel = path.strip_prefix(root_canonical).unwrap_or(path);
        let mut new_ids: Vec<String> = Vec::with_capacity(chunks.len());
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let data = json!({
                "path": rel.display().to_string(),
                "abs_path": path.display().to_string(),
                "chunk_idx": chunk_idx,
                "chunk_count": chunks.len(),
                "content": chunk,
                "file_hash": hash,
                "source": "ingest",
            });
            if let Ok(rec) = db.insert(table, data) {
                new_ids.push(rec.id.to_string());
                chunks_written += 1;
            }
        }
        files_ingested += 1;
        if chunks_replaced > 0 {
            out.status(&format!(
                "[ingest] {} superseded {} prior chunk(s)",
                rel.display(),
                chunks_replaced
            ));
        }
        state.insert(
            key,
            json!({
                "hash": hash,
                "mtime": mtime_now,
                "chunks": chunks.len(),
                "record_ids": new_ids,
                "ingested_at": chrono::Utc::now().to_rfc3339(),
            }),
        );

        // Checkpoint state every 20 files so a crash loses little work.
        if files_ingested % 20 == 0 {
            let _ = std::fs::write(state_path, serde_json::to_string(&state)?);
        }

        if (idx + 1) % 10 == 0 || idx + 1 == total {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let rate = (idx + 1) as f64 / elapsed;
            out.status(&format!(
                "[ingest] {}/{} files | {} chunks | {:.1} files/s",
                idx + 1,
                total,
                chunks_written,
                rate
            ));
        }
    }

    // Final checkpoint.
    std::fs::write(state_path, serde_json::to_string_pretty(&state)?)?;

    Ok(json!({
        "root": root_canonical.display().to_string(),
        "files_total": total,
        "files_ingested": files_ingested,
        "files_skipped": files_skipped,
        "chunks_written": chunks_written,
        "elapsed_sec": started.elapsed().as_secs_f64(),
        "state_file": state_path.display().to_string(),
        "table": table,
    }))
}

/// Split text into chunks of at most `max_bytes`, splitting on paragraph boundaries (12.2).
/// Falls back to hard-wrap when a single paragraph exceeds the cap.
fn chunk_text(text: &str, max_bytes: usize) -> Vec<String> {
    // Guard against `max_bytes == 0` which would make the hard-wrap loop
    // never advance, and clamp tiny values so a single codepoint longer
    // than `max_bytes` can still emit. `min_chunk_bytes = 4` is enough
    // for the widest valid UTF-8 codepoint.
    let max_bytes = max_bytes.max(4);
    if text.len() <= max_bytes {
        return vec![text.trim().to_string()];
    }
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for para in paragraphs {
        if para.len() > max_bytes {
            // Flush current, then hard-wrap this long paragraph.
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
                current = String::new();
            }
            let bytes = para.as_bytes();
            let mut start = 0;
            while start < bytes.len() {
                let mut end = (start + max_bytes).min(bytes.len());
                // Back off to a UTF-8 boundary so we don't slice mid-codepoint.
                while end > start && !para.is_char_boundary(end) {
                    end -= 1;
                }
                // If backing off collapsed the window to zero (first
                // codepoint at `start` is longer than `max_bytes`), walk
                // forward past that codepoint so the loop always makes
                // progress.
                if end == start {
                    end = (start + 1).min(bytes.len());
                    while end < bytes.len() && !para.is_char_boundary(end) {
                        end += 1;
                    }
                }
                chunks.push(para[start..end].trim().to_string());
                start = end;
            }
            continue;
        }
        if current.len() + para.len() + 2 > max_bytes && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

/// Format recall results as an XML `<context>` block for UserPromptSubmit hook injection (12.1).
///
/// Output shape:
/// ```xml
/// <context source="axil">
///   [decisions] Short summary (id=01KP...)
///   [errors] Short summary (id=01KP...)
/// </context>
/// ```
/// Returns an empty string when there are no results so hooks can stay silent.
///
/// Escapes XML-special characters (`<`, `>`, `&`) in recalled text so a stored
/// memory containing `</context>` or XML-like instructions cannot break out of
/// the wrapper and act as injected instructions on the next turn.
fn format_context_block(values: &[Value], budget: Option<usize>) -> String {
    if values.is_empty() {
        return String::new();
    }
    // Split proxy hits into a "Relevant code" section so the pointer-first
    // format (`path:line symbol — why`) is visually distinct from memory hits.
    #[cfg_attr(not(feature = "indexer"), allow(unused_mut))]
    let mut code_lines: Vec<String> = Vec::new();
    let mut other_lines: Vec<String> = Vec::with_capacity(values.len());
    for v in values {
        let table = v.get("table").and_then(|s| s.as_str()).unwrap_or("?");
        let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("?");
        // Code-proxy hits only exist when the indexer feature populated
        // `_idx_code_proxies`; the branch is dead weight without it.
        #[cfg(feature = "indexer")]
        if table == axil_indexer::TABLE_CODE_PROXIES {
            code_lines.push(format_code_proxy_line(v));
            continue;
        }
        let summary = pick_summary(v.get("data"))
            .map(|s| truncate_str(s, 240))
            .unwrap_or_else(|| "(no summary)".into());
        other_lines.push(format!(
            "  [{}] {} (id={})",
            xml_escape_for_context(table),
            xml_escape_for_context(&summary),
            xml_escape_for_context(id),
        ));
    }
    // Build the combined `lines` list: code first, then memories — so the
    // first thing the agent sees is the actionable pointer.
    let mut lines: Vec<String> = Vec::with_capacity(code_lines.len() + other_lines.len() + 2);
    if !code_lines.is_empty() {
        lines.push("  # Relevant code".to_string());
        lines.extend(code_lines);
    }
    if !other_lines.is_empty() {
        if !lines.is_empty() {
            lines.push("  # Related memories".to_string());
        }
        lines.extend(other_lines);
    }

    // Apply token budget (bytes/4 ≈ tokens) packing highest-ranked first.
    if let Some(max_tokens) = budget {
        let max_bytes = max_tokens.saturating_mul(4);
        let opener = "<context source=\"axil\">\n";
        let closer = "</context>\n";
        let mut used = opener.len() + closer.len();
        let mut kept: Vec<String> = Vec::new();
        for line in &lines {
            let add = line.len() + 1;
            if used + add > max_bytes && !kept.is_empty() {
                break;
            }
            used += add;
            kept.push(line.clone());
        }
        lines = kept;
    }

    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::from("<context source=\"axil\">\n");
    for line in &lines {
        out.push_str(line);
        out.push('\n');
    }
    // Write-side reminder. Agents that only read the context block can
    // miss that Axil is bidirectional; surfacing the three core writes
    // here re-anchors them on every prompt without re-loading CLAUDE.md.
    // Square brackets (not angle brackets) so the wrapper stays valid XML.
    // Cost: ~25 tokens, negligible vs the memory entries above.
    out.push_str(
        "  # Tools (Bash): axil store [table] '{...}' | axil recall \"[query]\" | axil boot\n",
    );
    out.push_str("</context>\n");
    out
}

/// Resolve a `--code-ref` spec into a normalized `code_refs` entry.
///
/// Accepted forms:
/// - `proxy_id` — exact match against `_idx_code_proxies.proxy_id`.
/// - `canonical_id` — SCIP-style id, exact match against `_idx_code_proxies`.
/// - `path` or `path:line` — best-effort lookup; if multiple proxies share
///   the path, prefer the symbol whose `line_start` is closest to the
///   provided line. Falls back to the file proxy when no symbol matches.
///
/// Returns `Ok(None)` when nothing matched. Errors are reserved for
/// list-table failures.
#[cfg(feature = "indexer")]
fn resolve_code_ref(db: &axil_core::Axil, spec: &str) -> anyhow::Result<Option<Value>> {
    let proxies = db
        .list(axil_indexer::TABLE_CODE_PROXIES)
        .context("list proxies")?;
    if proxies.is_empty() {
        return Ok(None);
    }

    // 1) Exact proxy_id / canonical_id match.
    for r in &proxies {
        let pid = r.data.get("proxy_id").and_then(|v| v.as_str());
        let cid = r.data.get("canonical_id").and_then(|v| v.as_str());
        if pid == Some(spec) || cid == Some(spec) {
            return Ok(Some(proxy_to_code_ref(r)));
        }
    }

    // 2) path[:line] form.
    let (path_part, line_part) = match spec.rsplit_once(':') {
        Some((p, l)) => match l.parse::<u64>() {
            Ok(n) => (p, Some(n)),
            Err(_) => (spec, None),
        },
        None => (spec, None),
    };

    let mut path_matches: Vec<&axil_core::Record> = proxies
        .iter()
        .filter(|r| {
            r.data
                .get("path")
                .and_then(|v| v.as_str())
                .map(|p| p == path_part)
                .unwrap_or(false)
        })
        .collect();
    if path_matches.is_empty() {
        return Ok(None);
    }

    if let Some(target_line) = line_part {
        // Prefer the symbol proxy whose line_start is closest at-or-below
        // target_line. Falls back to the smallest absolute-distance match,
        // and finally to a file proxy.
        path_matches.sort_by_key(|r| {
            let line = r
                .data
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            (target_line as i64 - line as i64).abs()
        });
        return Ok(path_matches.first().map(|r| proxy_to_code_ref(r)));
    }

    // No line specified — prefer the file proxy.
    let file_proxy = path_matches.iter().find(|r| {
        r.data.get("kind").and_then(|v| v.as_str()) == Some(axil_indexer::ProxyKind::File.as_str())
    });
    Ok(file_proxy
        .map(|r| proxy_to_code_ref(r))
        .or_else(|| path_matches.first().map(|r| proxy_to_code_ref(r))))
}

/// Build a normalized `code_refs` entry from a proxy record.
#[cfg(feature = "indexer")]
fn proxy_to_code_ref(record: &axil_core::Record) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("source_record".to_string(), json!(record.id.to_string()));
    for key in [
        "proxy_id",
        "canonical_id",
        "path",
        "symbol",
        "line_start",
        "line_end",
    ] {
        if let Some(v) = record.data.get(key) {
            obj.insert(key.to_string(), v.clone());
        }
    }
    Value::Object(obj)
}

/// Format a `_idx_code_proxies` hit as `path:line symbol — why`.
///
/// All user-derived strings are XML-escaped — the same rules that protect
/// the existing `<context>` wrapper apply because a stored `proxy_text`
/// could in theory contain `</context>`-like fragments.
#[cfg(feature = "indexer")]
fn format_code_proxy_line(v: &Value) -> String {
    let data = v.get("data");
    let path = data
        .and_then(|d| d.get("path"))
        .and_then(|s| s.as_str())
        .unwrap_or("?");
    let line = data
        .and_then(|d| d.get("line_start"))
        .and_then(|n| n.as_u64());
    let symbol = data.and_then(|d| d.get("symbol")).and_then(|s| s.as_str());
    let breadcrumb = data
        .and_then(|d| d.get("breadcrumb"))
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let kind = data
        .and_then(|d| d.get("kind"))
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let why = data
        .and_then(|d| d.get("why"))
        .and_then(|s| s.as_str())
        .unwrap_or(if kind == axil_indexer::ProxyKind::Section.as_str() {
            "matched markdown section"
        } else {
            "matched code proxy"
        });
    let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("?");

    let pointer = match (line, symbol) {
        (Some(l), Some(s)) => format!("{path}:{l} {s}"),
        (Some(l), None) => format!("{path}:{l}"),
        (None, Some(s)) => format!("{path} {s}"),
        (None, None) => path.to_string(),
    };
    let bc = if !breadcrumb.is_empty() {
        format!(" [{}]", xml_escape_for_context(breadcrumb))
    } else {
        String::new()
    };
    format!(
        "  [code] {} — {}{} (id={})",
        xml_escape_for_context(&pointer),
        xml_escape_for_context(why),
        bc,
        xml_escape_for_context(id),
    )
}

/// Escape XML-special characters so untrusted recalled text cannot
/// break out of the enclosing `<context>...</context>` wrapper when the
/// hook output is injected into the next prompt. Only `<`, `>`, `&`
/// need escaping for this use — attributes are never derived from
/// stored data.
fn xml_escape_for_context(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

/// Truncate a string to max_len chars, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let mut end = max_len.saturating_sub(3);
    // Walk back to a char boundary to avoid UTF-8 slice panic
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}

/// Shortest token worth preserving as a high-signal span.
const MIN_SIGNAL_TOKEN_LEN: usize = 8;
/// Normalized-entropy bar above which a no-space token reads as an identifier
/// / hash / code rather than prose.
const SIGNAL_ENTROPY_THRESHOLD: f64 = 0.60;
/// Cap on the preserved-token tail appended after a truncation.
const SIGNAL_TAIL_CAP: usize = 128;

/// Normalized Shannon entropy of a string's character distribution, in `[0,1]`
/// (raw entropy ÷ `log2(len)`). High for UUIDs/hashes (diverse chars), low for
/// prose words (repeated letters). Cheap, no classifier.
fn entropy_norm(s: &str) -> f64 {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n <= 1 {
        return 0.0;
    }
    let mut freq: std::collections::HashMap<char, u32> = std::collections::HashMap::new();
    for c in &chars {
        *freq.entry(*c).or_insert(0) += 1;
    }
    let n_f = n as f64;
    let mut h = 0.0;
    for &count in freq.values() {
        let p = count as f64 / n_f;
        h -= p * p.log2();
    }
    h / n_f.log2()
}

/// A no-space token that carries unique signal worth preserving verbatim: an
/// identifier, UUID, hash, error code, or `path:line`. Gates on length + a
/// "not a plain word" shape (has a digit / separator / mixed case) so prose is
/// never preserved, then confirms with a normalized-entropy bar.
fn is_high_signal_token(t: &str) -> bool {
    if t.len() < MIN_SIGNAL_TOKEN_LEN {
        return false;
    }
    let has_digit = t.bytes().any(|b| b.is_ascii_digit());
    let has_sep = t
        .chars()
        .any(|c| matches!(c, '-' | '_' | ':' | '/' | '.' | '='));
    let mixed_case =
        t.chars().any(|c| c.is_ascii_uppercase()) && t.chars().any(|c| c.is_ascii_lowercase());
    if !(has_digit || has_sep || mixed_case) {
        return false;
    }
    entropy_norm(t) >= SIGNAL_ENTROPY_THRESHOLD
}

/// Truncate like [`truncate_str`], but rescue high-signal tokens (IDs, hashes,
/// error codes, `path:line`) from the dropped tail and append them, so a large
/// record keeps its identifiers instead of having them chopped mid-prose. The
/// entropy gate keys on no-space tokens, so prose is trimmed normally.
fn truncate_str_signal(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let mut cut = max_len.saturating_sub(3).min(s.len());
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    // Prefer a nearby whitespace boundary so we don't split a word/token.
    if let Some(ws) = s[..cut].rfind(char::is_whitespace) {
        if cut - ws < 24 {
            cut = ws;
        }
    }
    let head = s[..cut].trim_end();

    let mut preserved: Vec<&str> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for tok in s[cut..].split_whitespace() {
        let t = tok.trim_matches(|c: char| {
            !c.is_alphanumeric() && !matches!(c, '-' | '_' | ':' | '/' | '.' | '=')
        });
        if is_high_signal_token(t) && seen.insert(t) {
            preserved.push(t);
        }
    }
    if preserved.is_empty() {
        return format!("{head}...");
    }

    let mut tail = String::new();
    for t in preserved {
        if tail.len() + t.len() + 1 > SIGNAL_TAIL_CAP {
            break;
        }
        if !tail.is_empty() {
            tail.push(' ');
        }
        tail.push_str(t);
    }
    format!("{head}… [{tail}]")
}

/// Truncate a JSON Value's string fields.
fn truncate_value(v: &Value, max_len: usize) -> Value {
    match v {
        Value::String(s) => Value::String(truncate_str_signal(s, max_len)),
        Value::Object(obj) => {
            let truncated: serde_json::Map<String, Value> = obj
                .iter()
                .filter(|(k, _)| !k.starts_with('_')) // strip internal fields
                .map(|(k, v)| (k.clone(), truncate_value(v, max_len)))
                .collect();
            Value::Object(truncated)
        }
        other => other.clone(),
    }
}

/// Create a truncated JSON representation of a record.
fn truncate_record_json(record: &axil_core::Record, max_field_len: usize) -> Value {
    let mut v = json!({
        "id": record.id.to_string(),
        "table": record.table,
    });
    if let Some(obj) = record.data.as_object() {
        for (k, val) in obj {
            if k.starts_with('_') {
                continue;
            }
            v[k] = truncate_value(val, max_field_len);
        }
    }
    v
}

/// Apply a token budget to a JSON value by serializing and truncating.
fn apply_token_budget(v: &Value, budget: Option<usize>) -> Value {
    if let Some(max_tokens) = budget {
        let serialized = serde_json::to_string(v).unwrap_or_default();
        let max_bytes = max_tokens * 4;
        if serialized.len() > max_bytes {
            // Try to truncate array elements
            if let Some(obj) = v.as_object() {
                let mut result = serde_json::Map::new();
                let mut used = 2;
                for (k, val) in obj {
                    let entry = serde_json::to_string(val).unwrap_or_default();
                    if used + entry.len() + k.len() + 4 > max_bytes && !result.is_empty() {
                        break;
                    }
                    used += entry.len() + k.len() + 4;
                    result.insert(k.clone(), val.clone());
                }
                return Value::Object(result);
            }
        }
    }
    v.clone()
}

// ─── Boot context helpers ──────────────────────────────────────────

/// Convert boot JSON to a human-readable narrative format.
fn boot_to_narrative(data: &Value) -> String {
    let mut out = String::new();
    out.push_str("=== Axil Boot Context ===\n");
    out.push_str("INSTRUCTION: This database is your memory. Use `axil recall \"<topic>\"` for project questions. Do NOT read files or grep for status — query the DB first.\n\n");

    // Extension blocks render before rules/sessions/decisions. Each
    // block is a self-contained markdown fragment (Extensions own
    // their own headers); emit in registration order, blank-line separated.
    if let Some(blocks) = data.get("extension_blocks").and_then(|v| v.as_array()) {
        for entry in blocks {
            let text = entry.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let trimmed = text.trim_end();
            if !trimmed.is_empty() {
                out.push_str(trimmed);
                out.push_str("\n\n");
            }
        }
    }

    // Cross-agent delta from the semantic event log (present only when the
    // `event-log` feature is enabled and the log is on; absent otherwise, so
    // this renders nothing by default).
    if let Some(changes) = data.get("recent_changes").and_then(|v| v.as_array()) {
        if !changes.is_empty() {
            out.push_str("## Recent Changes (since last session)\n");
            for c in changes {
                let kind = c.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                let table = c.get("table").and_then(|v| v.as_str()).unwrap_or("");
                let rec = c.get("record_id").and_then(|v| v.as_str()).unwrap_or("");
                match c.get("agent_id").and_then(|v| v.as_str()) {
                    Some(agent) => {
                        out.push_str(&format!("- {kind} [{table}] {rec} (by {agent})\n"))
                    }
                    None => out.push_str(&format!("- {kind} [{table}] {rec}\n")),
                }
            }
            out.push('\n');
        }
    }

    if let Some(rules) = data.get("rules").and_then(|v| v.as_array()) {
        if !rules.is_empty() {
            out.push_str("## Rules (pinned — always apply)\n");
            for r in rules {
                let rule = r.get("rule").and_then(|v| v.as_str()).unwrap_or("?");
                out.push_str(&format!("- {}\n", rule));
            }
            out.push('\n');
        }
    }

    if let Some(sessions) = data.get("recent_sessions").and_then(|v| v.as_array()) {
        out.push_str("## Recent Sessions\n");
        for s in sessions {
            if let Some(summary) = s
                .get("summary")
                .or_else(|| s.get("files_changed"))
                .and_then(|v| v.as_str())
            {
                out.push_str(&format!("- {}\n", summary));
            } else {
                let id = s.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let file_count = s.get("file_count").and_then(|v| v.as_u64()).unwrap_or(0);
                out.push_str(&format!("- Session {} ({} files)\n", id, file_count));
            }
        }
        out.push('\n');
    }

    if let Some(decisions) = data.get("decisions").and_then(|v| v.as_array()) {
        out.push_str("## Decisions\n");
        for d in decisions {
            let summary = d.get("summary").and_then(|v| v.as_str()).unwrap_or("?");
            out.push_str(&format!("- {}\n", summary));
        }
        out.push('\n');
    }

    if let Some(errors) = data.get("errors").and_then(|v| v.as_array()) {
        out.push_str("## Known Issues\n");
        for e in errors {
            let error = e.get("error").and_then(|v| v.as_str()).unwrap_or("?");
            let fix = e.get("fix").and_then(|v| v.as_str()).unwrap_or("");
            if fix.is_empty() {
                out.push_str(&format!("- {}\n", error));
            } else {
                out.push_str(&format!("- {}: {}\n", error, fix));
            }
        }
        out.push('\n');
    }

    if let Some(arch) = data.get("architecture").and_then(|v| v.as_array()) {
        out.push_str("## Architecture Notes\n");
        for a in arch {
            let summary = a.get("summary").and_then(|v| v.as_str()).unwrap_or("?");
            out.push_str(&format!("- {}\n", summary));
        }
        out.push('\n');
    }

    if let Some(topic) = data.get("topic_recall").and_then(|v| v.as_array()) {
        out.push_str("## Related Context\n");
        for t in topic {
            let summary = t.get("summary").and_then(|v| v.as_str()).unwrap_or("?");
            let score = t.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            out.push_str(&format!("- [{:.2}] {}\n", score, summary));
        }
        out.push('\n');
    }

    out
}

/// Create a compact boot JSON (strip whitespace, abbreviate keys).
fn compact_boot_json(data: &Value) -> Value {
    if let Some(obj) = data.as_object() {
        let mut compact = serde_json::Map::new();
        for (k, v) in obj {
            if let Some(arr) = v.as_array() {
                let compacted: Vec<Value> = arr
                    .iter()
                    .map(|item| {
                        if let Some(obj) = item.as_object() {
                            let mut slim = serde_json::Map::new();
                            for (ik, iv) in obj {
                                if ik == "id"
                                    || ik == "score"
                                    || ik == "summary"
                                    || ik == "error"
                                    || ik == "fix"
                                    || ik == "type"
                                {
                                    slim.insert(ik.clone(), truncate_value(iv, 150));
                                }
                            }
                            Value::Object(slim)
                        } else {
                            item.clone()
                        }
                    })
                    .collect();
                compact.insert(k.clone(), json!(compacted));
            } else {
                compact.insert(k.clone(), v.clone());
            }
        }
        Value::Object(compact)
    } else {
        data.clone()
    }
}

// ─── Multi-agent instructions ──────────────────────────────────────

/// Core Axil commands shared across all agent instruction templates.
const AXIL_COMMANDS: &str = r#"- `axil boot` — load previous session context
- `axil recall "<query>" --top-k 5` — find relevant memories
- `axil code-search "<query>" --top-k 5` — find symbols/modules before `rg`/`grep`
- `axil fts "<term>" --limit 5` — exact text search before broad scans
- `axil store decisions '{"summary":"<what>","reason":"<why>"}'` — save decisions
- `axil store errors '{"error":"<what>","fix":"<how>"}'` — save gotchas
- `axil checkpoint '{"state":"<where things stand>","next_steps":["<remaining work>"]}'` — write a resume-able checkpoint at end of work"#;

fn agent_instructions_cursor(db_path: &Path) -> String {
    format!("# Axil Agent Memory\n\nThis project uses Axil for persistent agent memory at `{db}`.\n\n## Commands\n{cmds}\n\n## Search/Query Gate\nBefore `rg`, `grep`, `git grep`, `find`, `fd`, `ls`, `tree`, or any broad project query, run `axil recall`, `axil code-search`, or `axil fts` first. Then open the files Axil returns and verify current code.\n\n## Workflow\n1. Start: run `axil boot`\n2. Work: store decisions and errors as you go\n3. End: write a checkpoint (run `axil checkpoint`)\n",
        db = db_path.display(), cmds = AXIL_COMMANDS)
}

fn agent_instructions_windsurf(db_path: &Path) -> String {
    agent_instructions_cursor(db_path) // same markdown format
}

fn agent_instructions_codex(db_path: &Path) -> String {
    format!(
        r#"# Axil Agent Memory

This repo uses Axil as persistent agent memory at `{db}`.

## Axil-First Rules

Use Axil before broad repo exploration. Aim for almost every project-context lookup to start with Axil, then open the cited files directly.

- Search/query gate: before any repo-discovery command (`rg`, `grep`, `git grep`, `find`, `fd`, `ls`, `tree`) or any "where/how/what changed" question, run the most relevant Axil command first.
- Start work with `axil recall "<user request>" --top-k 5`; use `axil boot` for status/history questions.
- For symbols, APIs, modules, or "where is X?", run `axil code-search "<query>" --top-k 5` before `rg` or `grep`.
- For exact text in docs/configs, run `axil fts "<term>" --limit 5` before a broad text scan.
- For recent work, decisions, or gotchas, run `axil since 7d --limit 20` or `axil recall "<topic>" --top-k 5`.
- After Axil points to files or symbols, read/edit files normally to verify current code.

Bypass Axil only for a user-named exact file/line, a command/test output you just produced, or a tiny local edit where no project context is needed.

## Write-Back Rules

- Store design choices immediately: `axil store decisions '{{"summary":"<what>","reason":"<why>","files":["<path>"]}}'`
- Store bugs/gotchas immediately: `axil store errors '{{"error":"<what>","root_cause":"<why>","fix":"<how>"}}'`
- Store architecture learned while reading: `axil store context '{{"type":"architecture","summary":"<what you learned>","files":["<path>"]}}'`
- Before a final response after substantive work, write a checkpoint: `axil checkpoint '{{"state":"<where things stand>","next_steps":["<remaining work>"],"references":[{{"kind":"file","ref":"<path>"}}]}}'`
"#,
        db = db_path.display()
    )
}

/// Idempotently merge the Axil-managed `<!-- AXIL:BEGIN/END -->` block into
/// a markdown file: replace an existing block in place, otherwise append —
/// user content around the markers is never touched.
fn merge_axil_block(path: &Path, inner: &str) -> Result<bool> {
    const BEGIN: &str = "<!-- AXIL:BEGIN -->";
    const END: &str = "<!-- AXIL:END -->";

    let block = format!("{BEGIN}\n{}\n{END}\n", inner.trim());
    let existing = std::fs::read_to_string(path).unwrap_or_default();

    let next = if let Some(start) = existing.find(BEGIN) {
        if let Some(end_rel) = existing[start..].find(END) {
            let end = start + end_rel + END.len();
            format!("{}{}{}", &existing[..start], block, &existing[end..])
        } else if existing.trim().is_empty() {
            block
        } else {
            format!("{}\n\n{}", existing.trim_end(), block)
        }
    } else if existing.trim().is_empty() {
        block
    } else {
        format!("{}\n\n{}", existing.trim_end(), block)
    };

    let changed = existing != next;
    if changed {
        std::fs::write(path, next)?;
    }
    Ok(changed)
}

fn install_codex_agents_md(cwd: &Path, db_path: &Path) -> Result<bool> {
    merge_axil_block(&cwd.join("AGENTS.md"), &agent_instructions_codex(db_path))
}

/// Aider integration: the memory contract goes in `CONVENTIONS.md` (the
/// community-standard conventions file), loaded read-only via the `read:`
/// key in `.aider.conf.yml` — aider has no `conventions:` config option,
/// and earlier installs that wrote one could make aider reject its config.
fn install_aider_files(cwd: &Path, db_path: &Path) -> Result<()> {
    merge_axil_block(&cwd.join("CONVENTIONS.md"), &agent_instructions_cursor(db_path))?;

    let conf_path = cwd.join(".aider.conf.yml");
    if !conf_path.exists() {
        std::fs::write(
            &conf_path,
            "# Axil agent memory: load the memory contract read-only.\nread: [CONVENTIONS.md]\n",
        )?;
        return Ok(());
    }

    let existing = std::fs::read_to_string(&conf_path)?;
    // Self-heal: strip the invalid `conventions:` block earlier installs
    // appended (from the "# Axil agent memory" comment through its last
    // "  - Database path:" item).
    let mut lines: Vec<&str> = existing.lines().collect();
    if let Some(start) = lines
        .iter()
        .position(|l| l.trim() == "# Axil agent memory")
    {
        let end = lines
            .iter()
            .skip(start)
            .position(|l| l.trim_start().starts_with("- Database path:"))
            .map(|rel| start + rel);
        if let Some(end) = end {
            lines.drain(start..=end);
        }
    }
    let mut cleaned = lines.join("\n");

    if !cleaned.contains("CONVENTIONS.md") {
        if cleaned
            .lines()
            .any(|l| l.starts_with("read:") || l.starts_with("read :"))
        {
            // A read: key we didn't write — don't textually edit unknown
            // YAML; tell the user what to add instead.
            eprintln!(
                "[install] note: add CONVENTIONS.md to the `read:` list in {} so aider loads the Axil memory contract",
                conf_path.display()
            );
        } else {
            if !cleaned.is_empty() && !cleaned.ends_with('\n') {
                cleaned.push('\n');
            }
            cleaned.push_str(
                "# Axil agent memory: load the memory contract read-only.\nread: [CONVENTIONS.md]\n",
            );
        }
    }

    if cleaned != existing {
        std::fs::write(&conf_path, cleaned)?;
    }
    Ok(())
}

// ─── MCP registration ────────────────────────────────────────────────

/// Where one agent keeps its MCP config and how the entry is shaped.
#[cfg(feature = "mcp")]
struct McpTarget {
    /// Config file (project-relative or absolute in the home dir).
    config: PathBuf,
    /// True when the config lives in the user's home dir — the server
    /// entry then needs an absolute DB path instead of a project-relative
    /// one, and it points at *this* project's database.
    global: bool,
}

/// CLI wrapper around [`mcp_register`].
#[cfg(feature = "mcp")]
fn mcp_install(cwd: &Path, target: &str, dry_run: bool, out: &Output) -> Result<i32> {
    let result = mcp_register(cwd, target, dry_run)?;
    out.print(&result);
    Ok(EXIT_OK)
}

/// Register the Axil MCP server in an agent's config. Only the `axil`
/// entry is written; everything else in the file is preserved.
#[cfg(feature = "mcp")]
fn mcp_register(cwd: &Path, target: &str, dry_run: bool) -> Result<Value> {
    let db_rel = "./.axil/memory.axil";
    let db_abs = cwd.join(".axil").join("memory.axil");
    if !db_abs.is_file() {
        anyhow::bail!(
            "no database at {} — run `axil install` first",
            db_abs.display()
        );
    }
    let home = || -> Result<PathBuf> {
        axil_core::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))
    };

    let t = match target {
        "claude-code" => McpTarget {
            config: cwd.join(".mcp.json"),
            global: false,
        },
        "cursor" => McpTarget {
            config: cwd.join(".cursor").join("mcp.json"),
            global: false,
        },
        "windsurf" => McpTarget {
            config: home()?
                .join(".codeium")
                .join("windsurf")
                .join("mcp_config.json"),
            global: true,
        },
        // Project-scoped: Codex reads [mcp_servers.*] from a trusted
        // project's .codex/config.toml — the right scope for a per-project
        // memory DB.
        "codex" => McpTarget {
            config: cwd.join(".codex").join("config.toml"),
            global: false,
        },
        "copilot" => McpTarget {
            config: home()?.join(".copilot").join("mcp-config.json"),
            global: true,
        },
        "droid" => McpTarget {
            config: cwd.join(".factory").join("mcp.json"),
            global: false,
        },
        "qwen" => McpTarget {
            config: cwd.join(".qwen").join("settings.json"),
            global: false,
        },
        "antigravity" => McpTarget {
            config: cwd.join(".agents").join("mcp_config.json"),
            global: false,
        },
        // Prefer an existing opencode.jsonc — detection accepts it, and
        // writing a competing opencode.json would leave a config OpenCode
        // doesn't load. (A .jsonc with comments won't parse as JSON; the
        // merge below then surfaces a clear "fix it and rerun" error rather
        // than silently splitting the config.)
        "opencode" => McpTarget {
            config: if cwd.join("opencode.jsonc").is_file() {
                cwd.join("opencode.jsonc")
            } else {
                cwd.join("opencode.json")
            },
            global: false,
        },
        other => anyhow::bail!(
            "unknown MCP target '{other}'. Supported: claude-code, cursor, windsurf, codex, copilot, droid, qwen, antigravity, opencode"
        ),
    };

    let exe = resolved_axil_exe();
    // A per-user (global) config is shared across every project, so pinning
    // an absolute --db would make the next project's install overwrite the
    // single "axil" entry and rebind it to the wrong database. Instead pin
    // NO path: `axil mcp` auto-detects `.axil/memory.axil` by walking up from
    // the server's launch cwd, so one global entry serves every project.
    // Project-scoped configs live in the project, so a relative path is safe.
    let db_arg = db_rel.to_string();
    let args = if t.global {
        vec!["mcp".to_string()]
    } else {
        vec!["--db".to_string(), db_arg.clone(), "mcp".to_string()]
    };

    // Per-target entry shape.
    let entry_desc: Value;
    if target == "codex" {
        // Codex config is TOML: [mcp_servers.axil] command/args.
        entry_desc = json!({"table": "mcp_servers.axil", "command": exe, "args": args});
        if !dry_run {
            let existing = std::fs::read_to_string(&t.config).unwrap_or_default();
            let mut root: toml::Table = if existing.trim().is_empty() {
                toml::Table::new()
            } else {
                existing.parse().with_context(|| {
                    format!("{} is not valid TOML — fix it and rerun", t.config.display())
                })?
            };
            let servers = root
                .entry("mcp_servers".to_string())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            let servers = servers.as_table_mut().ok_or_else(|| {
                anyhow::anyhow!("mcp_servers in {} is not a table", t.config.display())
            })?;
            let mut axil = toml::Table::new();
            axil.insert("command".into(), toml::Value::String(exe.clone()));
            axil.insert(
                "args".into(),
                toml::Value::Array(args.iter().cloned().map(toml::Value::String).collect()),
            );
            servers.insert("axil".into(), toml::Value::Table(axil));
            if let Some(parent) = t.config.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&t.config, toml::to_string_pretty(&root)?)?;
        }
    } else {
        // JSON targets. OpenCode uses `mcp.<name>` with a command array;
        // everything else uses the standard `mcpServers.<name>` shape.
        let (top_key, entry) = if target == "opencode" {
            let mut command = vec![exe.clone()];
            command.extend(args.iter().cloned());
            (
                "mcp",
                json!({"type": "local", "command": command, "enabled": true}),
            )
        } else if target == "copilot" {
            // Copilot CLI requires a type and a tool allowlist per server.
            (
                "mcpServers",
                json!({"type": "local", "command": exe, "args": args, "tools": ["*"]}),
            )
        } else {
            ("mcpServers", json!({"command": exe, "args": args}))
        };
        entry_desc = json!({ "key": format!("{top_key}.axil"), "entry": entry });
        if !dry_run {
            let existing = std::fs::read_to_string(&t.config).unwrap_or_default();
            let mut root: serde_json::Map<String, Value> = if existing.trim().is_empty() {
                serde_json::Map::new()
            } else {
                serde_json::from_str(&existing).with_context(|| {
                    format!("{} is not valid JSON — fix it and rerun", t.config.display())
                })?
            };
            let servers = root.entry(top_key.to_string()).or_insert_with(|| json!({}));
            let servers = servers.as_object_mut().ok_or_else(|| {
                anyhow::anyhow!("{top_key} in {} is not an object", t.config.display())
            })?;
            servers.insert("axil".to_string(), entry);
            if let Some(parent) = t.config.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(
                &t.config,
                serde_json::to_string_pretty(&Value::Object(root))? + "\n",
            )?;
        }
    }

    let mut result = json!({
        "target": target,
        "config": t.config.display().to_string(),
        "written": !dry_run,
        "server": entry_desc,
    });
    if t.global {
        result["note"] = json!(format!(
            "{} is a per-user config shared across projects — the axil server pins no --db and auto-detects .axil/memory.axil from its launch directory, so it resolves to whichever project it runs in",
            t.config.display()
        ));
    }
    if target == "codex" {
        result["note"] =
            json!("Codex loads project-scoped .codex/config.toml only in trusted projects");
    }
    Ok(result)
}

// ─── Skill commands ─────────────────────────────────────────────────────────

/// Embedded skill file contents (compiled into the binary from the skills/ directory).
const SKILL_AXIL: &str = include_str!("skills/axil.md");
const SKILL_REPORT: &str = include_str!("skills/axil-report.md");
const SKILL_DIAGNOSE: &str = include_str!("skills/axil-diagnose.md");
const SKILL_OPTIMIZE: &str = include_str!("skills/axil-optimize.md");
const SKILL_AUTOAGENT: &str = include_str!("skills/axil-autoagent.md");
const SKILL_LEARN: &str = include_str!("skills/axil-learn.md");
const SKILL_RETRO: &str = include_str!("skills/axil-retro.md");
const SKILL_BRIEF: &str = include_str!("skills/axil-brief.md");
const SKILL_CHECKPOINT: &str = include_str!("skills/axil-checkpoint.md");
const CLAUDE_MD_TEMPLATE: &str = include_str!("templates/CLAUDE.md");
const OPENCODE_PLUGIN_TEMPLATE: &str = include_str!("templates/opencode-plugin.ts");

struct SkillInfo {
    name: &'static str,
    filename: &'static str,
    content: &'static str,
}

impl SkillInfo {
    /// Directory name under the skills root — the slash-command name.
    fn dir_name(&self) -> &'static str {
        self.filename.trim_end_matches(".md")
    }
}

/// Write one skill in the `skills/<name>/SKILL.md` directory layout (the
/// agentskills.io convention Claude Code, Codex, Antigravity, Qwen, Amp,
/// and Zed all load). Removes the legacy flat `<name>.md` file older
/// installs wrote, which current Claude Code no longer discovers.
fn write_skill(skills_root: &Path, skill: &SkillInfo) -> Result<PathBuf> {
    let dir = skills_root.join(skill.dir_name());
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join("SKILL.md");
    std::fs::write(&path, skill.content)
        .with_context(|| format!("failed to write {}", path.display()))?;
    let legacy = skills_root.join(skill.filename);
    if legacy.is_file() {
        let _ = std::fs::remove_file(legacy);
    }
    Ok(path)
}

const ALL_SKILLS: &[SkillInfo] = &[
    SkillInfo {
        name: "memory",
        filename: "axil.md",
        content: SKILL_AXIL,
    },
    SkillInfo {
        name: "report",
        filename: "axil-report.md",
        content: SKILL_REPORT,
    },
    SkillInfo {
        name: "diagnose",
        filename: "axil-diagnose.md",
        content: SKILL_DIAGNOSE,
    },
    SkillInfo {
        name: "optimize",
        filename: "axil-optimize.md",
        content: SKILL_OPTIMIZE,
    },
    SkillInfo {
        name: "autoagent",
        filename: "axil-autoagent.md",
        content: SKILL_AUTOAGENT,
    },
    SkillInfo {
        name: "learn",
        filename: "axil-learn.md",
        content: SKILL_LEARN,
    },
    SkillInfo {
        name: "retro",
        filename: "axil-retro.md",
        content: SKILL_RETRO,
    },
    SkillInfo {
        name: "brief",
        filename: "axil-brief.md",
        content: SKILL_BRIEF,
    },
    SkillInfo {
        name: "checkpoint",
        filename: "axil-checkpoint.md",
        content: SKILL_CHECKPOINT,
    },
];

fn skills_dir() -> Result<PathBuf> {
    axil_core::home_dir()
        .ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))
        .map(|h| h.join(".claude").join("skills"))
}

fn run_skill(cmd: SkillCommand, out: &Output) -> Result<i32> {
    match cmd {
        SkillCommand::Install { only } => {
            let dir = skills_dir()?;
            std::fs::create_dir_all(&dir).context("failed to create skills directory")?;

            let skills = ALL_SKILLS;
            let to_install: Vec<&SkillInfo> = if let Some(ref name) = only {
                let s = skills
                    .iter()
                    .find(|s| s.name == name.as_str())
                    .ok_or_else(|| {
                        let available: Vec<_> = skills.iter().map(|s| s.name).collect();
                        anyhow::anyhow!(
                            "unknown skill: {name}. Available: {}",
                            available.join(", ")
                        )
                    })?;
                vec![s]
            } else {
                skills.iter().collect()
            };

            let mut installed = Vec::new();
            for skill in &to_install {
                let path = write_skill(&dir, skill)?;
                installed.push(json!({
                    "name": skill.name,
                    "path": path.display().to_string(),
                }));
            }

            out.print(&json!({
                "installed": installed.len(),
                "skills": installed,
            }));
            Ok(EXIT_OK)
        }

        SkillCommand::List => {
            let dir = skills_dir()?;
            let skills = ALL_SKILLS;
            let mut results = Vec::new();

            for skill in skills {
                // Current layout first; fall back to the legacy flat file so
                // pre-migration installs still report as installed.
                let dir_path = dir.join(skill.dir_name()).join("SKILL.md");
                let legacy_path = dir.join(skill.filename);
                let path = if dir_path.exists() {
                    Some(dir_path)
                } else if legacy_path.exists() {
                    Some(legacy_path)
                } else {
                    None
                };
                let mut entry = json!({
                    "name": skill.name,
                    "filename": skill.filename,
                    "installed": path.is_some(),
                });
                if let Some(path) = path {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        entry["size_bytes"] = json!(meta.len());
                    }
                    entry["path"] = json!(path.display().to_string());
                }
                results.push(entry);
            }

            out.print_array(&results);
            Ok(EXIT_OK)
        }

        SkillCommand::Uninstall => {
            let dir = skills_dir()?;
            let skills = ALL_SKILLS;
            let mut removed = Vec::new();

            for skill in skills {
                let mut hit = false;
                let skill_dir = dir.join(skill.dir_name());
                if skill_dir.is_dir() {
                    std::fs::remove_dir_all(&skill_dir)
                        .with_context(|| format!("failed to remove {}", skill_dir.display()))?;
                    hit = true;
                }
                let legacy = dir.join(skill.filename);
                if legacy.is_file() {
                    std::fs::remove_file(&legacy)
                        .with_context(|| format!("failed to remove {}", legacy.display()))?;
                    hit = true;
                }
                if hit {
                    removed.push(skill.name);
                }
            }

            out.print(&json!({
                "removed": removed.len(),
                "skills": removed,
            }));
            Ok(EXIT_OK)
        }
    }
}

// ─── Utility ────────────────────────────────────────────────────────────────

/// Convert a scored search result to JSON.
#[allow(dead_code)]
fn scored_to_json(record: &axil_core::Record, score: f32) -> Value {
    json!({
        "id": record.id.to_string(),
        "score": round4(score),
        "data": record.data,
        "table": record.table,
        "created_at": format_dt(&record.created_at),
    })
}

/// Convert a Record to a JSON Value with consistent field names.
fn record_to_json(record: &axil_core::Record) -> Value {
    json!({
        "id": record.id.to_string(),
        "table": record.table,
        "data": record.data,
        "created_at": format_dt(&record.created_at),
        "updated_at": format_dt(&record.updated_at),
    })
}

/// Round a float to 4 decimal places.
#[allow(dead_code)]
fn round4(v: f32) -> f32 {
    (v * 10000.0).round() / 10000.0
}

/// Parse a comma-separated CLI argument into owned strings. Returns an
/// empty Vec when the option is absent or all entries are whitespace.
fn parse_comma_list(opt: &Option<String>) -> Vec<String> {
    opt.as_deref()
        .map(|s| {
            s.split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod agents_md_drift {
    use super::*;
    use std::path::Path;

    const BEGIN: &str = "<!-- AXIL:BEGIN -->";
    const END: &str = "<!-- AXIL:END -->";

    // Drop the one machine-specific line (the absolute db path) before
    // comparing — the rules body is what must not drift, the path is per-repo.
    fn normalize(block: &str) -> String {
        block
            .trim()
            .lines()
            .map(|l| {
                if l.starts_with("This repo uses Axil as persistent agent memory at") {
                    "This repo uses Axil as persistent agent memory at `<DB>`.".to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    // The committed root AGENTS.md is a *generated* artifact: its AXIL:BEGIN/END
    // block is emitted by agent_instructions_codex(). If the generator changes
    // and AGENTS.md isn't regenerated, Codex / VS Code agents read stale memory
    // rules with no error. This pins the committed copy to its generator (db
    // path aside) so the drift fails CI instead of silently shipping.
    #[test]
    fn committed_agents_md_matches_generator() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("workspace root above crates/adapters/axil-cli");
        let committed = std::fs::read_to_string(repo_root.join("AGENTS.md"))
            .expect("read committed AGENTS.md");

        let start = committed.find(BEGIN).expect("AGENTS.md missing AXIL:BEGIN");
        let end = committed.find(END).expect("AGENTS.md missing AXIL:END") + END.len();
        let committed_block = &committed[start..end];

        let generated = format!(
            "{BEGIN}\n{}\n{END}",
            agent_instructions_codex(Path::new("/PLACEHOLDER")).trim()
        );

        assert_eq!(
            normalize(committed_block),
            normalize(&generated),
            "AGENTS.md AXIL block drifted from agent_instructions_codex(); \
             regenerate AGENTS.md (re-run the Codex integration installer) and commit it."
        );
    }
}

#[cfg(test)]
mod extension_dispatch_tests {
    use super::*;
    use axil_core::{CliSubcommand, CliSurface};

    fn checkpoint_surface() -> CliSurface {
        CliSurface::new("checkpoint", "session checkpoint")
            .subcommand(CliSubcommand::new("write", "write a checkpoint"))
            .subcommand(CliSubcommand::new("show", "show the checkpoint"))
    }

    #[test]
    fn splits_declared_subcommand_into_command_path() {
        let tokens = vec!["checkpoint".to_string(), "show".to_string()];
        let inv = build_extension_invocation(&tokens, Some(&checkpoint_surface()), None);
        assert_eq!(inv.command_path, vec!["checkpoint", "show"]);
        assert!(inv.args.is_empty());
    }

    #[test]
    fn subcommand_plus_flags_keeps_flags_as_args() {
        let tokens = vec!["checkpoint".into(), "write".into(), "--final".into()];
        let inv = build_extension_invocation(&tokens, Some(&checkpoint_surface()), None);
        assert_eq!(inv.command_path, vec!["checkpoint", "write"]);
        assert_eq!(inv.args, vec!["--final"]);
    }

    #[test]
    fn non_subcommand_token_stays_an_arg() {
        // `checkpoint '{json}'` — the JSON is not a declared subcommand, so the
        // command path is just the top command and the JSON is a positional arg.
        let tokens = vec!["checkpoint".into(), "{\"goal\":\"x\"}".into()];
        let inv = build_extension_invocation(&tokens, Some(&checkpoint_surface()), None);
        assert_eq!(inv.command_path, vec!["checkpoint"]);
        assert_eq!(inv.args, vec!["{\"goal\":\"x\"}"]);
    }

    #[test]
    fn no_surface_keeps_everything_after_command_as_args() {
        let tokens = vec!["hello".into(), "world".into()];
        let inv = build_extension_invocation(&tokens, None, None);
        assert_eq!(inv.command_path, vec!["hello"]);
        assert_eq!(inv.args, vec!["world"]);
    }

    #[test]
    fn stdin_is_threaded_through() {
        let tokens = vec!["checkpoint".into()];
        let inv =
            build_extension_invocation(&tokens, Some(&checkpoint_surface()), Some("piped".into()));
        assert_eq!(inv.stdin.as_deref(), Some("piped"));
    }
}

#[cfg(test)]
mod adapter_tests {
    use super::*;
    use axil_core::Adapter;

    #[test]
    fn cli_adapter_identity() {
        let a = CliAdapter::new();
        assert_eq!(a.id(), "cli");
        assert_eq!(a.protocol(), axil_core::Protocol::Cli);
    }

    #[test]
    fn cli_adapter_bind_is_accepted() {
        // bind is a documented no-op for the CLI (it self-resolves its db), but
        // it must still honor the contract — accept the handle, return Ok.
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            axil_core::Axil::open(dir.path().join("c.axil"))
                .build()
                .unwrap(),
        );
        let mut a = CliAdapter::new();
        assert!(a.bind(db).is_ok());
    }
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    #[test]
    fn short_string_is_unchanged() {
        assert_eq!(truncate_str_signal("hello world", 100), "hello world");
    }

    #[test]
    fn prose_only_uses_plain_ellipsis() {
        let s = "the quick brown fox jumps over the lazy dog again and again and again";
        let out = truncate_str_signal(s, 30);
        assert!(out.ends_with("..."), "got: {out}");
        assert!(!out.contains('['), "no signal tail expected: {out}");
    }

    #[test]
    fn preserves_commit_sha_from_dropped_tail() {
        let s = "Fixed the auth timeout bug that was introduced in commit \
                 3404911abdfb4fd8f2c4c89533d5c56826303221 under heavy login load";
        let out = truncate_str_signal(s, 40);
        assert!(
            out.contains("3404911abdfb4fd8f2c4c89533d5c56826303221"),
            "full sha must survive: {out}"
        );
        assert!(out.contains('['), "expected a preserved-token tail: {out}");
    }

    #[test]
    fn preserves_uuid_and_pathline() {
        let s = "A long prose description that runs well past the budget so it must be cut \
                 referencing 01KQ2TBNYX24EXPXFQAVS0A67Q at crates/axil-core/src/db.rs:3297 ok";
        let out = truncate_str_signal(s, 30);
        assert!(out.contains("01KQ2TBNYX24EXPXFQAVS0A67Q"), "uuid: {out}");
        assert!(
            out.contains("crates/axil-core/src/db.rs:3297"),
            "path:line: {out}"
        );
    }

    #[test]
    fn high_signal_token_classification() {
        assert!(is_high_signal_token("01KQ2TBNYX24EXPXFQAVS0A67Q")); // uuid
        assert!(is_high_signal_token(
            "3404911abdfb4fd8f2c4c89533d5c56826303221"
        )); // sha
        assert!(is_high_signal_token("validate_token")); // snake_case identifier
        assert!(is_high_signal_token("path/to/file.rs:42")); // path:line
        assert!(!is_high_signal_token("responsibilities")); // plain prose word
        assert!(!is_high_signal_token("the")); // too short
        assert!(!is_high_signal_token("configuration")); // plain word, no signal shape
    }
}

#[cfg(test)]
mod upgrade_helper_tests {
    use super::atomic_replace_with_rollback;

    #[test]
    fn replace_succeeds_and_swaps_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("plugin.wasm");
        std::fs::write(&target, b"OLD").unwrap();

        let res = atomic_replace_with_rollback(&target, b"NEW", |p| {
            // Validate sees the new bytes already in place.
            assert_eq!(std::fs::read(p).unwrap(), b"NEW");
            Ok(())
        });

        assert!(res.is_ok(), "expected success, got {res:?}");
        assert_eq!(std::fs::read(&target).unwrap(), b"NEW");
        // No staging temp file is left behind.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("upgrade-tmp"))
            .collect();
        assert!(leftovers.is_empty(), "staging temp not cleaned up");
    }

    #[test]
    fn failed_validation_rolls_back_to_original_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("plugin.wasm");
        std::fs::write(&target, b"OLD").unwrap();

        let res = atomic_replace_with_rollback(&target, b"NEW", |_p| {
            anyhow::bail!("new version refused to load")
        });

        assert!(res.is_err(), "expected the validate error to propagate");
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("new version refused to load"));
        // The original plugin is restored byte-for-byte; the broken upgrade is gone.
        assert_eq!(std::fs::read(&target).unwrap(), b"OLD");
    }
}

#[cfg(test)]
mod installer_tests {
    use super::*;

    // A stale released binary on PATH predates `hook run`; wiring bare `axil`
    // for it would exit clap's usage code 2 on every hook fire, which agents
    // read as a blocking veto. So bare `axil` is chosen ONLY when the PATH
    // handshake passes — otherwise the absolute path is pinned.
    #[test]
    fn resolves_bare_axil_only_when_handshake_passes() {
        // On PATH and the handshake passes → portable bare name.
        assert_eq!(
            resolve_axil_exe_with(true, || true, || Some("/abs/axil".into())),
            "axil"
        );
        // On PATH but the handshake fails (stale binary) → pin the absolute
        // path; never bare `axil`.
        assert_eq!(
            resolve_axil_exe_with(true, || false, || Some("/abs/axil".into())),
            "/abs/axil"
        );
        // Not on PATH → absolute path regardless of what the probe would say.
        assert_eq!(
            resolve_axil_exe_with(false, || true, || Some("/abs/axil".into())),
            "/abs/axil"
        );
        // Absolute path unavailable → last-resort bare name.
        assert_eq!(resolve_axil_exe_with(false, || true, || None), "axil");
    }

    // The probe must not consult its expensive closure when nothing is on PATH.
    #[test]
    fn handshake_probe_is_skipped_when_not_on_path() {
        let mut probed = false;
        let exe = resolve_axil_exe_with(
            false,
            || {
                probed = true;
                true
            },
            || Some("/abs/axil".into()),
        );
        assert_eq!(exe, "/abs/axil");
        assert!(!probed, "probe must be short-circuited when off PATH");
    }

    // Antigravity's injection channel is PreInvocation, not SessionStart. The
    // installer must register the events parse_antigravity accepts, or context
    // injection is silently inert.
    #[test]
    fn antigravity_plugin_registers_preinvocation_not_sessionstart() {
        let dir = tempfile::tempdir().unwrap();
        let (plugin_dir, _registered) = install_antigravity_plugin(dir.path()).unwrap();

        let hooks: Value =
            serde_json::from_str(&std::fs::read_to_string(plugin_dir.join("hooks.json")).unwrap())
                .unwrap();
        let events = hooks["hooks"].as_object().expect("hooks object");
        let keys: Vec<&str> = events.keys().map(String::as_str).collect();

        assert!(
            keys.contains(&"PreInvocation"),
            "PreInvocation (the only injection channel) must be registered; got {keys:?}"
        );
        assert!(
            !keys.contains(&"SessionStart"),
            "SessionStart parses to nothing for Antigravity — must not be registered; got {keys:?}"
        );
        for e in ["PreToolUse", "PostToolUse", "Stop"] {
            assert!(keys.contains(&e), "missing {e}; got {keys:?}");
        }

        // Each hook passes its event through --event (payloads carry none).
        let cmd = events["PreInvocation"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(
            cmd.contains("hook run --dialect antigravity") && cmd.contains("--event PreInvocation"),
            "PreInvocation command wired wrong: {cmd}"
        );
    }

    // `--all --dry-run` must preview the same Claude Code files the real
    // `claude_code || all` install writes — the dry-run gate had drifted to a
    // bare `claude_code`, under-reporting the plan for `--all`.
    #[test]
    fn dry_run_all_includes_claude_code_files() {
        let dir = tempfile::tempdir().unwrap();
        let plan = dry_run_install_plan(
            dir.path(),
            false, // claude_code — deliberately off; `all` must still cover it
            false, false, false, false, false, false, false, false, false, false,
            true,  // all
            true,  // local — avoids depending on the global skills dir
        )
        .unwrap();
        let writes = plan["would_write"].as_array().unwrap();
        assert!(
            writes.iter().any(|w| w.as_str().unwrap_or("").contains("CLAUDE.md")),
            "--all dry-run must list .claude/CLAUDE.md; got {writes:?}"
        );

        // Control: with nothing selected, the Claude file set is absent.
        let empty = dry_run_install_plan(
            dir.path(),
            false, false, false, false, false, false, false, false, false, false, false,
            false, // all
            true,
        )
        .unwrap();
        assert!(
            !empty["would_write"]
                .as_array()
                .unwrap()
                .iter()
                .any(|w| w.as_str().unwrap_or("").contains("CLAUDE.md")),
            "no-selection dry-run must not list Claude files"
        );
    }
}

#[cfg(all(test, feature = "wasm-host"))]
mod scaffold_plugin_tests {
    use super::*;

    fn out() -> Output {
        Output {
            format: OutputFormat::Json,
            quiet: true,
            jsonl: false,
        }
    }

    #[test]
    fn emits_a_complete_buildable_crate() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("myplug");
        let code = scaffold_plugin("myplug", Some(&dest), Some("recall,records.write"), &out())
            .expect("scaffold should succeed");
        assert_eq!(code, EXIT_OK);

        // Every file cargo-component needs to build a detached guest is present.
        for rel in [
            "Cargo.toml",
            "src/lib.rs",
            "src/sdk.rs",
            "wit/axil-plugin.wit",
            "build.sh",
            ".gitignore",
            "README.md",
        ] {
            assert!(dest.join(rel).exists(), "missing scaffolded file: {rel}");
        }

        // The bundled SDK + WIT are the canonical copies, byte-for-byte.
        assert_eq!(
            std::fs::read_to_string(dest.join("src/sdk.rs")).unwrap(),
            PLUGIN_SDK_RS
        );
        assert_eq!(
            std::fs::read_to_string(dest.join("wit/axil-plugin.wit")).unwrap(),
            PLUGIN_WIT
        );

        let cargo = std::fs::read_to_string(dest.join("Cargo.toml")).unwrap();
        assert!(cargo.contains("name = \"myplug\""));
        assert!(cargo.contains("[workspace]"), "must be a detached workspace");
        assert!(cargo.contains("crate-type = [\"cdylib\"]"));
        assert!(cargo.contains("package = \"axil:myplug\""));
        // Component target points at the bundled WIT, not a repo-relative path.
        assert!(cargo.contains("path = \"wit\""));
        assert!(!cargo.contains("wasip2"), "no wasip2 lie");

        let lib = std::fs::read_to_string(dest.join("src/lib.rs")).unwrap();
        assert!(lib.contains("use sdk::Plugin;"));
        assert!(lib.contains("export_plugin!(Component);"));
        assert!(lib.contains("\"_myplug_\""), "prefix derived from name");

        // Requested caps land in the README as grant hints.
        let readme = std::fs::read_to_string(dest.join("README.md")).unwrap();
        assert!(readme.contains("axil ext grant myplug recall"));
        assert!(readme.contains("axil ext grant myplug records.write"));
    }

    #[test]
    fn hyphenated_name_underscores_the_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("my-plug");
        scaffold_plugin("my-plug", Some(&dest), None, &out()).unwrap();
        let lib = std::fs::read_to_string(dest.join("src/lib.rs")).unwrap();
        assert!(lib.contains("\"_my_plug_\""), "hyphens become underscores in the prefix");
        let cargo = std::fs::read_to_string(dest.join("Cargo.toml")).unwrap();
        assert!(cargo.contains("name = \"my-plug\""), "crate name keeps hyphens");
    }

    #[test]
    fn rejects_invalid_names() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["", "My-Plug", "-leading", "trailing-", "has space", "under_score"] {
            let dest = dir.path().join("x");
            let _ = std::fs::remove_dir_all(&dest);
            assert!(
                scaffold_plugin(bad, Some(&dest), None, &out()).is_err(),
                "name `{bad}` should be rejected"
            );
        }
    }

    #[test]
    fn refuses_to_overwrite_an_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("exists");
        std::fs::create_dir_all(&dest).unwrap();
        assert!(scaffold_plugin("exists", Some(&dest), None, &out()).is_err());
    }
}
