//! Production pattern arms on the U5 and U3 sets, without the model crates.
//!
//! * `--entities <u5.jsonl> [--project-dir <dir>]` prints what `heuristic
//!   --entities` prints: the entities `axil_core::extract_entities` finds in
//!   each text, capped at the number `Axil::auto_link` keeps. With
//!   `--project-dir`, extraction uses the gazetteer `Axil::auto_link` builds
//!   for a database under `<dir>/.axil/` (`Gazetteer::for_project_dir`; a
//!   database with `_deps` rows adds those names too).
//! * `--capture <u3.jsonl> [--all-bash]` prints what `heuristic --capture`
//!   prints: every capture `auto_capture::analyze` returns, with the table and
//!   importance of the record it would store. Each text is analyzed with the
//!   source it would arrive with: shell output (`bash_failure`,
//!   `bash_output`) as the hook sends it (`bash`), anything else as `axil
//!   auto-capture` labels it by default (`auto`). `--all-bash` analyzes every
//!   text as `bash`, the call `heuristic --capture` makes.
//!
//! * `--records <export.jsonl> [--project-dir <dir>]` reads an `axil export`
//!   and prints, per record, the importance insert would compute for its
//!   data (`_`-prefixed fields removed) and how many entities auto-linking
//!   would keep, so two builds can be compared on a real database.
//!
//! Output is one JSON line per text, the shape `run_u5_bakeoff.py` and
//! `run_u3_bakeoff.py` read, so `run_regex_arm.py` can score it with their
//! code.

use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use axil_core::auto_capture::{analyze, capture_to_record};
use axil_core::importance::compute_importance;
use axil_core::{extract_entities, extract_entities_with, EntityType, Gazetteer, Record};
use serde_json::{json, Value};

/// `Axil::auto_link` keeps at most this many entities per record.
const MAX_AUTO_LINK_ENTITIES: usize = 50;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: typed-decisions-regex-arm \
                 ((--entities <u5.jsonl> | --records <export.jsonl>) [--project-dir <dir>] \
                 | --capture <u3.jsonl> [--all-bash])";
    let (Some(mode), Some(path)) = (args.first(), args.get(1)) else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let all_bash = mode == "--capture" && args.get(2).map(String::as_str) == Some("--all-bash");
    let project_dir = match (args.get(2).map(String::as_str), args.get(3)) {
        (Some("--project-dir"), Some(dir)) => Some(std::path::PathBuf::from(dir)),
        (None, _) => None,
        (Some("--all-bash"), None) if all_bash => None,
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    match mode.as_str() {
        "--entities" => entities(path, project_dir.as_deref()),
        "--capture" => capture(path, all_bash),
        "--records" => records(path, project_dir.as_deref()),
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
}

/// Read a JSONL file into rows, skipping blank lines.
fn read_rows(path: &str) -> Vec<Value> {
    BufReader::new(std::fs::File::open(path).expect("open set"))
        .lines()
        .map(|l| l.expect("read line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("parse row"))
        .collect()
}

fn entities(path: &str, project_dir: Option<&std::path::Path>) {
    let rows = read_rows(path);
    let gazetteer = project_dir.map(Gazetteer::for_project_dir);
    if let Some(g) = &gazetteer {
        eprintln!("{}", json!({ "gazetteer_names": g.len() }));
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for row in &rows {
        let record = Record::new("decisions", json!({ "summary": row["text"] }));
        let t = Instant::now();
        let text = axil_core::util::record_text(&record);
        let mut found = match &gazetteer {
            Some(g) => extract_entities_with(&text, g),
            None => extract_entities(&text),
        };
        found.truncate(MAX_AUTO_LINK_ENTITIES);
        let extract_us = t.elapsed().as_secs_f64() * 1e6;
        let entities: Vec<Value> = found
            .iter()
            .map(|e| {
                let (axil_type, lang_hint) = match &e.entity_type {
                    EntityType::File => ("file", None),
                    EntityType::Code => ("code", None),
                    EntityType::Project => ("project", None),
                    EntityType::Reference => ("reference", None),
                    EntityType::CodeSymbol { lang_hint } => ("code_symbol", lang_hint.clone()),
                };
                json!({
                    "text": e.source_text,
                    "name": e.name,
                    "axil_type": axil_type,
                    "lang_hint": lang_hint,
                })
            })
            .collect();
        writeln!(
            out,
            "{}",
            json!({ "id": row["id"], "entities": entities, "extract_us": extract_us })
        )
        .expect("write");
    }
}

fn capture(path: &str, all_bash: bool) {
    let rows = read_rows(path);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for row in &rows {
        let text = row["text"].as_str().unwrap_or_default();
        let shell_output = matches!(row["source"].as_str(), Some("bash_failure" | "bash_output"));
        let source = if all_bash || shell_output {
            "bash"
        } else {
            "auto"
        };
        let t = Instant::now();
        let captures = analyze(text, source);
        let analyze_us = t.elapsed().as_secs_f64() * 1e6;
        let captures: Vec<Value> = captures
            .iter()
            .map(|c| {
                let (table, data) = capture_to_record(c);
                json!({
                    "type": c.capture_type,
                    "confidence": c.confidence,
                    "summary": c.summary,
                    "table": table,
                    // Insert keeps an `_importance` the record carries.
                    "importance": data["_importance"]
                        .as_f64()
                        .map_or_else(|| compute_importance(&data), |v| v as f32),
                })
            })
            .collect();
        writeln!(
            out,
            "{}",
            json!({
                "id": row["id"],
                "captures": captures,
                "plain_importance": compute_importance(&json!({ "summary": text })),
                "analyze_us": analyze_us,
            })
        )
        .expect("write");
    }
}

fn records(path: &str, project_dir: Option<&std::path::Path>) {
    let gazetteer = project_dir.map(Gazetteer::for_project_dir);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for row in read_rows(path) {
        if row["kind"] != "record" {
            continue;
        }
        let table = row["table"].as_str().unwrap_or_default();
        let mut data = row["data"].clone();
        if let Some(obj) = data.as_object_mut() {
            obj.retain(|k, _| !k.starts_with('_'));
        }
        let record = Record::new(table, data.clone());
        let text = axil_core::util::record_text(&record);
        let mut found = match &gazetteer {
            Some(g) => extract_entities_with(&text, g),
            None => extract_entities(&text),
        };
        found.truncate(MAX_AUTO_LINK_ENTITIES);
        writeln!(
            out,
            "{}",
            json!({
                "id": row["id"],
                "table": table,
                "importance": compute_importance(&data),
                "entities": found.len(),
                "pure_entities": extract_entities(&axil_core::util::value_text(&data)).len(),
            })
        )
        .expect("write");
    }
}
