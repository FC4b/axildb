//! `axil contradictions`: the pairs a local model flagged as possible
//! contradictions, for a reviewer to accept or dismiss, and (in a build with
//! the `decide` feature) the judge that flags them.
//!
//! Listing, dismissing and accepting read and write only the review rows and
//! edges, so they work in every build. Judging loads a model (load time and
//! memory: `benchmarks/results/typed-decisions-u1-where-it-runs-*.json`), so
//! it runs only in `axil contradictions run` and in the pass `axil worker run`
//! adds when `[decide] review = true`: the hook drainer runs that worker in a
//! detached process at session close. A hook never loads the model, and
//! neither does `axil store`.

use std::path::Path;

use anyhow::{Context, Result};
#[cfg(feature = "decide")]
use axil_core::Axil;
use axil_core::{RecordId, ReviewOptions};
use clap::Subcommand;
use serde_json::{json, Value};

use crate::{Output, EXIT_OK};

/// `axil contradictions …`.
#[derive(Subcommand)]
pub(crate) enum ContradictionsCommand {
    /// List the open flags (the default): newer/older pairs a local model
    /// scored at or above its threshold. Nothing has been changed.
    List,
    /// Every judged pair with its score, flagged or not, most recent first.
    Log {
        /// Only the pairs judged for this (newer) record.
        #[arg(long)]
        record: Option<String>,
        /// Most review rows to show.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Both records hold: close the flag and remove its edge.
    Dismiss {
        /// The newer record of the flagged pair.
        newer: String,
        /// The older record of the flagged pair.
        older: String,
    },
    /// The newer record replaced the older one: mark the older superseded.
    /// Read both first; the flag is a model's suggestion, not a verdict.
    Accept {
        /// The newer record of the flagged pair.
        newer: String,
        /// The older record, which it replaced.
        older: String,
    },
    /// Judge new records now: the pass `axil worker run` runs when
    /// `[decide] review = true`, or one record with `--record`.
    #[cfg(feature = "decide")]
    Run {
        /// Judge this record (again) instead of the new ones.
        #[arg(long)]
        record: Option<String>,
        /// Most records to judge in this pass.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Show whether the judge's model is downloaded, or download it.
    #[cfg(feature = "decide")]
    Model {
        /// Download (or repair) the pinned model files.
        #[arg(long)]
        download: bool,
    },
}

/// Run `axil contradictions`.
pub(crate) fn handle(
    command: Option<ContradictionsCommand>,
    db_opt: &Option<std::path::PathBuf>,
    out: &Output,
) -> Result<i32> {
    #[cfg(feature = "decide")]
    if let Some(ContradictionsCommand::Model { download }) = &command {
        return model(*download, out);
    }
    let db_path = crate::require_db(db_opt)?;
    match command.unwrap_or(ContradictionsCommand::List) {
        ContradictionsCommand::List => {
            let db = crate::open_with_all_detected(&db_path)?;
            let open = db
                .possible_contradictions()
                .context("failed to read possible contradictions")?;
            let rows: Vec<Value> = open.iter().map(flag_json).collect();
            out.print_array(&rows);
            if !rows.is_empty() {
                out.status(
                    "Suggestions from a local model; nothing was changed. Read both records, then: \
                     `axil contradictions accept <newer> <older>` if the newer replaced the older, \
                     `axil contradictions dismiss <newer> <older>` if both hold.",
                );
            }
            Ok(EXIT_OK)
        }
        ContradictionsCommand::Log { record, limit } => {
            let db = crate::open_with_all_detected(&db_path)?;
            let rows: Vec<Value> = db
                .contradiction_reviews()?
                .into_iter()
                .filter(|r| {
                    record
                        .as_deref()
                        .is_none_or(|id| r.data.get("record").and_then(Value::as_str) == Some(id))
                })
                .take(limit)
                .map(|r| {
                    let mut v = r.data;
                    v["judged_at"] = json!(crate::format_dt(&r.updated_at));
                    v
                })
                .collect();
            out.print_array(&rows);
            Ok(EXIT_OK)
        }
        ContradictionsCommand::Dismiss { newer, older } => {
            let db = crate::open_with_all_detected(&db_path)?;
            let (newer, older) = (parse_id(&newer)?, parse_id(&older)?);
            if !db.dismiss_possible_contradiction(&newer, &older)? {
                anyhow::bail!("no open possible contradiction from {newer} to {older}");
            }
            out.print(&json!({"dismissed": true, "newer": newer, "older": older}));
            Ok(EXIT_OK)
        }
        ContradictionsCommand::Accept { newer, older } => {
            let db = crate::open_with_all_detected(&db_path)?;
            let (newer, older) = (parse_id(&newer)?, parse_id(&older)?);
            let flag = db
                .possible_contradictions()?
                .into_iter()
                .find(|f| f.newer.id == newer && f.older.id == older)
                .with_context(|| {
                    format!("no open possible contradiction from {newer} to {older}")
                })?;
            let superseded = db.mark_superseded(&older, &flag.newer, Some(flag.similarity))?;
            if !superseded {
                anyhow::bail!(
                    "{older} was not marked superseded: its table's lifecycle policy or a pin \
                     protects it (`axil contradictions dismiss {newer} {older}` closes the flag)"
                );
            }
            out.print(&json!({"superseded": older, "by": newer}));
            Ok(EXIT_OK)
        }
        #[cfg(feature = "decide")]
        ContradictionsCommand::Run { record, limit } => {
            let db = crate::open_with_embedder(&db_path)?;
            let mut opts = DecideConfig::load(&db_path, out).options;
            if let Some(limit) = limit {
                opts.max_records = limit;
            }
            match record {
                Some(id) => {
                    let id = parse_id(&id)?;
                    let judge = load_judge()?;
                    let pairs =
                        db.review_possible_contradictions(&id, judge.as_ref(), opts.candidates)?;
                    out.print(&json!({"record": id, "model": judge.name(), "pairs": pairs}));
                }
                None => {
                    let report = db.review_new_records(&opts, load_judge)?;
                    out.print(&serde_json::to_value(report)?);
                }
            }
            Ok(EXIT_OK)
        }
        #[cfg(feature = "decide")]
        ContradictionsCommand::Model { .. } => unreachable!("handled above"),
    }
}

fn parse_id(id: &str) -> Result<RecordId> {
    RecordId::from_string(id).with_context(|| format!("invalid record ID {id:?}"))
}

/// One open flag as an agent reads it: both records' headlines, the score
/// and where it came from.
fn flag_json(flag: &axil_core::PossibleContradiction) -> Value {
    let side = |r: &axil_core::Record| {
        json!({
            "id": r.id,
            "table": r.table,
            "text": axil_core::boot::clip_one_line(&axil_core::boot::record_headline(r), 200),
        })
    };
    json!({
        "newer": side(&flag.newer),
        "older": side(&flag.older),
        "p": (flag.probability * 1000.0).round() / 1000.0,
        "threshold": flag.threshold,
        "similarity": (f64::from(flag.similarity) * 1000.0).round() / 1000.0,
        "model": flag.model,
        "calibration": flag.calibration,
        "judged_at": crate::format_dt(&flag.judged_at),
    })
}

/// The `[decide]` section of the `axil.toml` nearest the database.
///
/// ```toml
/// [decide]
/// review = true          # judge new records in `axil worker run`
/// tables = ["decisions", "context", "rules"]
/// max_records = 20       # per pass, newest first
/// window_days = 7        # only records this recent
/// candidates = 5         # older records judged per new one
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(feature = "decide"), allow(dead_code))]
pub(crate) struct DecideConfig {
    /// Whether `axil worker run` runs the review pass. Off unless set.
    pub review: bool,
    /// What a pass judges.
    pub options: ReviewOptions,
}

#[cfg_attr(not(feature = "decide"), allow(dead_code))]
impl DecideConfig {
    /// Read the section for the database at `db_path`, printing a warning
    /// for each problem. Fails safe: a malformed section turns review off.
    pub fn load(db_path: &Path, out: &Output) -> Self {
        let dir = db_path.parent().unwrap_or_else(|| Path::new("."));
        let Some(file) = axil_core::find_config_file(dir) else {
            return Self::from_toml("").0;
        };
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        let (config, warnings) = Self::from_toml(&text);
        for w in warnings {
            out.status(&format!("warning: {}: [decide] {w}", file.display()));
        }
        config
    }

    /// Parse the `[decide]` section of an `axil.toml` document; also returns
    /// one warning per problem. Any problem turns review off.
    pub fn from_toml(text: &str) -> (Self, Vec<String>) {
        let mut config = Self {
            review: false,
            options: ReviewOptions::default(),
        };
        let mut warnings = Vec::new();
        let doc: toml::Table = match text.parse() {
            Ok(doc) => doc,
            Err(e) => {
                warnings.push(format!(
                    "not read, the file is not valid TOML ({e}); review is off"
                ));
                return (config, warnings);
            }
        };
        let Some(section) = doc.get("decide") else {
            return (config, warnings);
        };
        let Some(section) = section.as_table() else {
            warnings.push("must be a table; review is off".into());
            return (config, warnings);
        };
        let positive = |key: &str, v: &toml::Value| match v.as_integer() {
            Some(n) if n > 0 => Ok(n as usize),
            _ => Err(format!("{key} must be a positive integer")),
        };
        let mut review = false;
        for (key, value) in section {
            let result = match key.as_str() {
                "review" => value
                    .as_bool()
                    .map(|b| review = b)
                    .ok_or_else(|| "review must be true or false".to_string()),
                "tables" => value
                    .as_array()
                    .and_then(|a| {
                        a.iter()
                            .map(|t| t.as_str().map(str::to_string))
                            .collect::<Option<Vec<_>>>()
                    })
                    .map(|t| config.options.tables = t)
                    .ok_or_else(|| "tables must be a list of table names".to_string()),
                "max_records" => positive(key, value).map(|n| config.options.max_records = n),
                "window_days" => positive(key, value)
                    .map(|n| config.options.window = chrono::Duration::days(n as i64)),
                "candidates" => positive(key, value).map(|n| config.options.candidates = n),
                other => Err(format!("unknown key {other:?}")),
            };
            if let Err(e) = result {
                warnings.push(e);
            }
        }
        config.review = review && warnings.is_empty();
        if review && !warnings.is_empty() {
            warnings.push("review is off until the section is fixed".into());
        }
        (config, warnings)
    }
}

/// Load the U1 judge from the pinned model directory. Never downloads: a
/// background pass must not start a ~96 MB transfer on its own.
#[cfg(feature = "decide")]
fn load_judge() -> axil_core::Result<Box<dyn axil_core::PairJudge>> {
    use axil_decide::{Calibration, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
    let dir = U1
        .default_dir()
        .ok_or_else(|| axil_core::AxilError::plugin("no home directory for the judge's model"))?;
    if !U1.is_available(&dir) {
        return Err(axil_core::AxilError::plugin(format!(
            "the judge's model is not in {}: run `axil contradictions model --download`",
            dir.display()
        )));
    }
    let judge = PairClassifier::load(&U1, &dir, Calibration::u1())
        .map_err(|e| axil_core::AxilError::plugin(e.to_string()))?;
    Ok(Box::new(judge))
}

/// `axil contradictions model [--download]`.
#[cfg(feature = "decide")]
fn model(download: bool, out: &Output) -> Result<i32> {
    use axil_decide::NLI_DEBERTA_V3_XSMALL_QINT8 as U1;
    let dir = U1
        .default_dir()
        .context("no home directory for the judge's model")?;
    if download {
        out.status(&format!("fetching {} into {} ...", U1.name, dir.display()));
        U1.download(&dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let verified = U1.is_available(&dir) && U1.verify(&dir).is_ok();
    out.print(&json!({
        "model": format!("{}@{}", U1.name, U1.revision),
        "dir": dir.display().to_string(),
        "bytes": U1.files().iter().map(|f| f.size).sum::<u64>(),
        "verified": verified,
    }));
    if !verified {
        out.status("not downloaded (or a file fails its pinned sha256): run `axil contradictions model --download`");
    }
    Ok(EXIT_OK)
}

/// The review pass `axil worker run` adds when `[decide] review = true`:
/// its report, or the error that stopped it. `None` when review is off.
#[cfg(feature = "decide")]
pub(crate) fn worker_review(db: &Axil, db_path: &Path, out: &Output) -> Option<Value> {
    let config = DecideConfig::load(db_path, out);
    if !config.review {
        return None;
    }
    Some(match db.review_new_records(&config.options, load_judge) {
        Ok(report) => serde_json::to_value(report).unwrap_or(Value::Null),
        Err(e) => {
            out.status(&format!("warning: possible-contradiction review stopped: {e}"));
            json!({"error": e.to_string()})
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_is_off_unless_the_section_turns_it_on() {
        for text in [
            "",
            "[database]\npath = \"x\"\n",
            "[decide]\n",
            "[decide]\nreview = false\n",
        ] {
            let (config, warnings) = DecideConfig::from_toml(text);
            assert!(!config.review, "{text:?}");
            assert!(warnings.is_empty(), "{text:?}: {warnings:?}");
            assert_eq!(config.options, ReviewOptions::default());
        }
    }

    #[test]
    fn the_section_sets_every_knob() {
        let (config, warnings) = DecideConfig::from_toml(
            "[decide]\nreview = true\ntables = [\"decisions\"]\nmax_records = 3\n\
             window_days = 2\ncandidates = 4\n",
        );
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(config.review);
        assert_eq!(config.options.tables, ["decisions"]);
        assert_eq!(config.options.max_records, 3);
        assert_eq!(config.options.window, chrono::Duration::days(2));
        assert_eq!(config.options.candidates, 4);
    }

    #[test]
    fn a_malformed_section_turns_review_off_and_says_why() {
        for text in [
            "[decide]\nreview = \"yes\"\n",
            "[decide]\nreview = true\nmax_records = 0\n",
            "[decide]\nreview = true\ntables = \"decisions\"\n",
            "[decide]\nreview = true\nwindow = 7\n",
            "decide = 1\n",
            "[decide\nreview = true\n",
        ] {
            let (config, warnings) = DecideConfig::from_toml(text);
            assert!(!config.review, "{text:?}");
            assert!(!warnings.is_empty(), "{text:?}");
        }
    }

    /// Hooks run under kill timeouts, and a model load takes about a second
    /// (more on a busy machine): the judge must never load in one. It loads
    /// only through `load_judge` in this module, which the hook module never
    /// reaches, and `main.rs` reaches only through `worker_review` (the
    /// worker the drainer runs in its own process) and `handle`.
    #[test]
    fn hooks_never_load_the_judge() {
        let hook = include_str!("hook_brain.rs");
        for name in [
            "axil_decide",
            "PairClassifier",
            "load_judge",
            "review_new_records",
            "contradictions::",
        ] {
            assert!(!hook.contains(name), "hook_brain.rs names {name}");
        }
        let main = include_str!("main.rs");
        for name in [
            "axil_decide",
            "PairClassifier",
            "load_judge",
            "review_new_records",
        ] {
            assert!(!main.contains(name), "main.rs names {name}");
        }
    }
}
