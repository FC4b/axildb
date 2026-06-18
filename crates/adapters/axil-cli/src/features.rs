//! `axil features` — binary build inspection + feature wizard.
//!
//! Axil's optional components (Engines, Extensions, Adapters — see
//! docs/src/extending/overview.md) are compile-time Cargo features, so a
//! shipped binary can't turn them on at runtime. What it *can* do is know
//! what it was built with (`cfg!`) and compose the exact `cargo install`
//! command for a different feature set. That's this module:
//!
//! - `catalog_json()` — status of every feature in this binary
//! - `run_wizard()`   — interactive picker that emits (and optionally runs)
//!   the matching `cargo install` command
//!
//! The catalog is kept honest by tests that parse this crate's Cargo.toml
//! `[features]` section — adding a feature there without updating the
//! catalog fails `cargo test -p axil-cli`.

use std::collections::BTreeSet;
use std::io::{self, IsTerminal, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

/// One selectable Cargo feature of the `axil-cli` crate.
pub struct FeatureInfo {
    pub name: &'static str,
    /// Extensibility tier: `core` / `engine` / `extension` / `adapter` / `opt-in`.
    pub tier: &'static str,
    pub description: &'static str,
    /// Other catalog features this one force-enables (Cargo feature deps).
    pub requires: &'static [&'static str],
    pub in_default: bool,
    pub in_full: bool,
}

/// Every feature in axil-cli's `[features]` section except the `default`
/// and `full` aggregates. Order = display order in the wizard.
pub const CATALOG: &[FeatureInfo] = &[
    FeatureInfo {
        name: "core",
        tier: "core",
        description: "Core storage (redb) — always present",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    // ── Engines (Tier 1) ────────────────────────────────────────────
    FeatureInfo {
        name: "vector",
        tier: "engine",
        description: "Vector search (HNSW) -> *.axil.vec",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "embed",
        tier: "engine",
        description: "Built-in ONNX embedder (BGE family)",
        requires: &["vector"],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "graph",
        tier: "engine",
        description: "Knowledge graph (edges, traversal) -> *.axil.graph",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "fts",
        tier: "engine",
        description: "Full-text search (Tantivy) -> *.axil.fts/",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "timeseries",
        tier: "engine",
        description: "Time-series queries -> *.axil.ts",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    // ── Extensions (Tier 2) ─────────────────────────────────────────
    FeatureInfo {
        name: "indexer",
        tier: "extension",
        description: "Structural code proxies (code-search / code-context)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "scip",
        tier: "extension",
        description: "SCIP code-graph ingest (precise call/ref edges)",
        requires: &["graph"],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "deps",
        tier: "extension",
        description: "Dependency doc memory (version-pinned library docs)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "checkpoint",
        tier: "extension",
        description: "Session checkpoints (structured resume state)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "memory",
        tier: "extension",
        description: "Agent memory patterns (TTL, superseding, sessions)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "rerank",
        tier: "extension",
        description: "Cross-encoder reranking for code recall",
        requires: &["indexer"],
        in_default: false,
        in_full: true,
    },
    // ── Adapters (Tier 3) ───────────────────────────────────────────
    FeatureInfo {
        name: "mcp",
        tier: "adapter",
        description: "MCP server (stdio) for agent integration",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "ql",
        tier: "adapter",
        description: "AxilQL query language + REPL",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "http",
        tier: "adapter",
        description: "HTTP API server (axum)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    // ── Deliberate opt-ins (excluded from `full` on purpose) ───────
    FeatureInfo {
        name: "llm-http",
        tier: "opt-in",
        description: "OpenAI-compatible LlmProvider (Path B intelligence)",
        requires: &[],
        in_default: true,
        in_full: true,
    },
    FeatureInfo {
        name: "web-docs",
        tier: "opt-in",
        description: "HTTP doc fetcher for deps (offline-first => off)",
        requires: &["deps"],
        in_default: false,
        in_full: false,
    },
    FeatureInfo {
        name: "otel",
        tier: "opt-in",
        description: "OpenTelemetry instrumentation",
        requires: &[],
        in_default: false,
        in_full: false,
    },
];

/// Was `name` compiled into this binary? `cfg!` needs literals, hence the match.
fn is_compiled(name: &str) -> bool {
    match name {
        "core" => true,
        "vector" => cfg!(feature = "vector"),
        "embed" => cfg!(feature = "embed"),
        "graph" => cfg!(feature = "graph"),
        "fts" => cfg!(feature = "fts"),
        "timeseries" => cfg!(feature = "timeseries"),
        "indexer" => cfg!(feature = "indexer"),
        "scip" => cfg!(feature = "scip"),
        "deps" => cfg!(feature = "deps"),
        "checkpoint" => cfg!(feature = "checkpoint"),
        "memory" => cfg!(feature = "memory"),
        "rerank" => cfg!(feature = "rerank"),
        "mcp" => cfg!(feature = "mcp"),
        "ql" => cfg!(feature = "ql"),
        "http" => cfg!(feature = "http"),
        "llm-http" => cfg!(feature = "llm-http"),
        "web-docs" => cfg!(feature = "web-docs"),
        "otel" => cfg!(feature = "otel"),
        _ => false,
    }
}

fn find(name: &str) -> Option<&'static FeatureInfo> {
    CATALOG.iter().find(|f| f.name == name)
}

/// `axil features` payload: one object per feature, build status included.
pub fn catalog_json() -> Vec<Value> {
    CATALOG
        .iter()
        .map(|f| {
            json!({
                "name": f.name,
                "tier": f.tier,
                "compiled": is_compiled(f.name),
                "default": f.in_default,
                "full": f.in_full,
                "requires": f.requires,
                "description": f.description,
            })
        })
        .collect()
}

fn compiled_set() -> BTreeSet<&'static str> {
    CATALOG
        .iter()
        .filter(|f| is_compiled(f.name))
        .map(|f| f.name)
        .collect()
}

fn default_set() -> BTreeSet<&'static str> {
    CATALOG
        .iter()
        .filter(|f| f.in_default)
        .map(|f| f.name)
        .collect()
}

fn full_set() -> BTreeSet<&'static str> {
    CATALOG
        .iter()
        .filter(|f| f.in_full)
        .map(|f| f.name)
        .collect()
}

/// Enable `name` plus everything it requires (transitively).
fn enable(selected: &mut BTreeSet<&'static str>, name: &'static str) {
    if selected.insert(name) {
        if let Some(f) = find(name) {
            for r in f.requires {
                if let Some(dep) = find(r) {
                    enable(selected, dep.name);
                }
            }
        }
    }
}

/// Disable `name` plus everything that requires it (transitively) —
/// dropping `graph` must also drop `scip`, or the build breaks.
fn disable(selected: &mut BTreeSet<&'static str>, name: &'static str) {
    if selected.remove(name) {
        let dependents: Vec<&'static str> = CATALOG
            .iter()
            .filter(|f| f.requires.contains(&name))
            .map(|f| f.name)
            .collect();
        for d in dependents {
            disable(selected, d);
        }
    }
}

/// Build the `cargo` argv (without the leading `cargo`) that produces a
/// binary with exactly `selected`. `local_path` is the in-repo crate path
/// when installing from a source checkout, `None` for crates.io.
fn install_argv(selected: &BTreeSet<&'static str>, local_path: Option<&str>) -> Vec<String> {
    let mut argv = vec!["install".to_string()];
    match local_path {
        Some(p) => {
            argv.push("--path".into());
            argv.push(p.into());
        }
        None => argv.push("axil-cli".into()),
    }
    // Replacing an already-installed same-version binary needs --force.
    argv.push("--force".into());

    if *selected == default_set() {
        return argv; // the default build needs no feature flags at all
    }
    if *selected == full_set() {
        // default ⊆ full, so layering `full` on top of default is exact.
        argv.push("--features".into());
        argv.push("full".into());
        return argv;
    }
    argv.push("--no-default-features".into());
    argv.push("--features".into());
    let mut names: Vec<&str> = selected.iter().copied().collect();
    if !names.contains(&"core") {
        names.insert(0, "core");
    }
    argv.push(names.join(","));
    argv
}

/// Detect whether the cwd is an axildb source checkout we can install from.
fn detect_local_path() -> Option<&'static str> {
    const IN_REPO: &str = "crates/adapters/axil-cli";
    Path::new(IN_REPO)
        .join("Cargo.toml")
        .exists()
        .then_some(IN_REPO)
}

fn tier_heading(tier: &str) -> &'static str {
    match tier {
        "engine" => "Engines (Tier 1 — storage substrate)",
        "extension" => "Extensions (Tier 2 — capabilities)",
        "adapter" => "Adapters (Tier 3 — protocol surfaces)",
        "opt-in" => "Opt-ins (deliberately excluded from `full`)",
        _ => "Core",
    }
}

/// Numbered, tier-grouped view of the current selection. Returns the
/// number → feature mapping used by the toggle loop (`core` is fixed and
/// gets no number).
fn render(selected: &BTreeSet<&'static str>) -> Vec<&'static str> {
    let mut order: Vec<&'static str> = Vec::new();
    let mut last_tier = "";
    println!();
    for f in CATALOG {
        if f.tier == "core" {
            continue;
        }
        if f.tier != last_tier {
            println!("  {}", tier_heading(f.tier));
            last_tier = f.tier;
        }
        order.push(f.name);
        let mark = if selected.contains(f.name) { "x" } else { " " };
        println!(
            "    {:>2} [{}] {:<12} {}",
            order.len(),
            mark,
            f.name,
            f.description
        );
    }
    println!();
    println!("  toggle: <number>   presets: a=all(full)  d=default  m=minimal");
    println!("  finish: <enter>    quit: q");
    order
}

fn prompt(msg: &str) -> Result<String> {
    print!("{msg}");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .context("failed to read from stdin")?;
    Ok(line.trim().to_string())
}

/// Interactive feature picker. Seeds the selection from what this binary
/// was compiled with, lets the user toggle components (dependency closure
/// enforced both directions), then prints — and optionally runs — the
/// `cargo install` command that produces that build.
pub fn run_wizard(quiet: bool) -> Result<i32> {
    if !io::stdin().is_terminal() {
        bail!(
            "`axil features --wizard` is interactive and needs a terminal.\n\
             For scripted builds compose the command yourself, e.g.:\n  \
             cargo install axil-cli --force --no-default-features --features \"core,vector,graph\""
        );
    }

    let mut selected = compiled_set();
    if !quiet {
        println!("Axil feature wizard — compose your build");
        println!("(initial selection = what this binary was compiled with)");
    }

    loop {
        let order = render(&selected);
        let input = prompt("> ")?;
        match input.as_str() {
            "" | "done" => break,
            "q" | "quit" => {
                println!("aborted — nothing changed");
                return Ok(crate::EXIT_OK);
            }
            "a" | "all" => selected = full_set(),
            "d" | "default" => selected = default_set(),
            "m" | "minimal" => {
                selected = BTreeSet::new();
                selected.insert("core");
            }
            other => match other.parse::<usize>() {
                Ok(n) if (1..=order.len()).contains(&n) => {
                    let name = order[n - 1];
                    if selected.contains(name) {
                        disable(&mut selected, name);
                    } else {
                        enable(&mut selected, name);
                    }
                }
                _ => println!("  ? unrecognized: {other}"),
            },
        }
        // core is the always-on baseline; presets and closures keep it.
        selected.insert("core");
    }

    let argv = install_argv(&selected, detect_local_path());
    let rendered = format!("cargo {}", argv.join(" "));
    println!();
    println!("Selected: {}", selected.iter().copied().collect::<Vec<_>>().join(", "));
    println!("Install command:");
    println!("  {rendered}");

    if selected == compiled_set() {
        println!("(identical to the current binary — running it would only rebuild)");
    }

    let answer = prompt("Run it now? [y/N] ")?;
    if answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes") {
        let status = std::process::Command::new("cargo")
            .args(&argv)
            .status()
            .context("failed to launch cargo — is it on PATH?")?;
        return Ok(status.code().unwrap_or(crate::EXIT_ERROR));
    }
    println!("Not run — copy the command above when ready.");
    Ok(crate::EXIT_OK)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse this crate's Cargo.toml `[features]` section: returns
    /// (all feature names, default list, full list).
    fn cargo_toml_features() -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let text = std::fs::read_to_string(manifest).expect("read Cargo.toml");
        let section = text
            .split("[features]")
            .nth(1)
            .expect("[features] section")
            .split("\n[")
            .next()
            .unwrap();

        let mut names = BTreeSet::new();
        let mut default = BTreeSet::new();
        let mut full = BTreeSet::new();
        for line in section.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim().to_string();
            let listed: BTreeSet<String> = value
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
                .collect();
            match key.as_str() {
                "default" => default = listed,
                "full" => full = listed,
                _ => {
                    names.insert(key);
                }
            }
        }
        (names, default, full)
    }

    #[test]
    fn catalog_covers_every_cargo_feature() {
        let (names, _, _) = cargo_toml_features();
        let catalog: BTreeSet<String> = CATALOG.iter().map(|f| f.name.to_string()).collect();
        assert_eq!(
            catalog, names,
            "features.rs CATALOG out of sync with Cargo.toml [features]"
        );
    }

    #[test]
    fn catalog_default_flags_match_cargo_toml() {
        let (_, default, _) = cargo_toml_features();
        for f in CATALOG {
            assert_eq!(
                f.in_default,
                default.contains(f.name),
                "in_default for `{}` disagrees with Cargo.toml default list",
                f.name
            );
        }
    }

    #[test]
    fn catalog_full_flags_match_cargo_toml() {
        let (_, _, full) = cargo_toml_features();
        for f in CATALOG {
            if f.name == "core" {
                continue; // core is the baseline; `full` doesn't list it (it's empty anyway)
            }
            assert_eq!(
                f.in_full,
                full.contains(f.name),
                "in_full for `{}` disagrees with Cargo.toml full list",
                f.name
            );
        }
    }

    #[test]
    fn enabling_pulls_required_features() {
        let mut s = BTreeSet::new();
        enable(&mut s, "embed");
        assert!(s.contains("vector"), "embed implies vector");
        enable(&mut s, "web-docs");
        assert!(s.contains("deps"), "web-docs implies deps");
        enable(&mut s, "scip");
        assert!(s.contains("graph"), "scip implies graph");
    }

    #[test]
    fn disabling_drops_dependent_features() {
        let mut s = full_set();
        disable(&mut s, "vector");
        assert!(!s.contains("embed"), "dropping vector must drop embed");
        disable(&mut s, "indexer");
        assert!(!s.contains("rerank"), "dropping indexer must drop rerank");
        disable(&mut s, "deps");
        assert!(!s.contains("web-docs"), "dropping deps must drop web-docs");
    }

    #[test]
    fn install_argv_default_set_needs_no_flags() {
        let argv = install_argv(&default_set(), None);
        assert_eq!(argv, vec!["install", "axil-cli", "--force"]);
    }

    #[test]
    fn install_argv_full_set_uses_full_shorthand() {
        let argv = install_argv(&full_set(), None);
        assert_eq!(argv, vec!["install", "axil-cli", "--force", "--features", "full"]);
    }

    #[test]
    fn install_argv_custom_set_pins_no_default_features() {
        let mut s = BTreeSet::new();
        enable(&mut s, "vector");
        enable(&mut s, "graph");
        let argv = install_argv(&s, Some("crates/adapters/axil-cli"));
        assert_eq!(
            argv,
            vec![
                "install",
                "--path",
                "crates/adapters/axil-cli",
                "--force",
                "--no-default-features",
                "--features",
                "core,graph,vector"
            ]
        );
    }
}
