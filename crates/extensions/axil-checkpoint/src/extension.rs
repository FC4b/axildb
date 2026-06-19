//! `axil-checkpoint` as a Tier-2 [`Extension`].
//!
//! Surfaces:
//! - `axil checkpoint <json|-> [--session <id>] [--final]` — write a snapshot
//!   (default) or a final checkpoint for the most-recent active session.
//! - `axil checkpoint show` — print the current checkpoint (stored, else derived).
//! - MCP tools: `checkpoint` (write) and `checkpoint_show` (read).
//! - `boot_block` — emits the "Resume Here" section when a checkpoint is
//!   available (stored or derived).

use axil_core::error::Result as CoreResult;
use axil_core::{
    Axil, AxilError, CliArg, CliInvocation, CliOutput, CliSubcommand, CliSurface, Dispatch,
    Extension, McpCall, McpSurface, McpTool, RecordId,
};
use serde_json::{json, Value};

use crate::{
    current_checkpoint, render_resume_block, write_for_session, write_with_active_session, Checkpoint,
    CheckpointError, CheckpointKind, Source,
};

/// `axil-checkpoint` as an Axil Extension.
///
/// Construct with [`CheckpointExtension::default`] and register through
/// [`axil_core::AxilBuilder::with_extension`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CheckpointExtension;

impl Extension for CheckpointExtension {
    fn id(&self) -> &str {
        "checkpoint"
    }

    fn display_name(&self) -> &str {
        "Session Checkpoint"
    }

    fn table_prefixes(&self) -> &[&str] {
        &["_checkpoint_"]
    }

    fn cli_commands(&self) -> Option<CliSurface> {
        Some(
            CliSurface::new(
                "checkpoint",
                "Write or read a structured session checkpoint so a fresh agent can resume.",
            )
            .subcommand(
                CliSubcommand::new(
                    "write",
                    "Write a checkpoint JSON (positional or - for stdin) to the most-recent active session.",
                )
                .arg(
                    CliArg::new("json", "Inline JSON object or `-` to read from stdin.")
                        .takes_value(true),
                )
                .arg(
                    CliArg::new(
                        "session",
                        "Attach to a specific session id instead of the latest active.",
                    )
                    .takes_value(true),
                )
                .arg(CliArg::new(
                    "final",
                    "Mark this checkpoint as the final one for its session (does not end the session by itself).",
                )),
            )
            .subcommand(CliSubcommand::new(
                "show",
                "Print the current checkpoint (stored if present, otherwise derived).",
            )),
        )
    }

    fn mcp_tools(&self) -> Option<McpSurface> {
        Some(McpSurface::new(vec![
            McpTool::new(
                "checkpoint",
                "Write a structured session checkpoint so a fresh agent can resume. Fields: goal, state, next_steps[], open_questions[], references[], summary. All optional but at least one required.",
                json!({
                        "type": "object",
                        "properties": {
                            "goal":           { "type": "string", "description": "The user's north-star intent for the session." },
                            "state":          { "type": "string", "description": "One or two sentences on where things stand right now." },
                            "next_steps":     { "type": "array", "items": { "type": "string" }, "description": "Ordered actionable steps to resume." },
                            "open_questions": { "type": "array", "items": { "type": "string" }, "description": "Unresolved blockers or decisions." },
                            "references": {
                                "type": "array",
                                "description": "Typed pointers, not copies. kind ∈ {commit,pr,file,plan,record,…}; record kinds resolve live at boot.",
                                "items": {
                                    "type": "object",
                                    "required": ["kind", "ref"],
                                    "properties": {
                                        "kind": { "type": "string" },
                                        "ref":  { "type": "string" },
                                        "note": { "type": "string" }
                                    }
                                }
                            },
                            "summary": { "type": "string", "description": "Optional one-line mirror; embedded for semantic recall." },
                            "session": { "type": "string", "description": "Attach to a specific session id instead of the latest active." },
                            "final":   { "type": "boolean", "description": "Mark as the final checkpoint for its session. Default false." }
                        }
                    }),
            ),
            McpTool::new(
                "checkpoint_show",
                "Return the current checkpoint (stored if present, otherwise derived from the latest session).",
                json!({ "type": "object", "properties": {} }),
            ),
        ]))
    }

    /// "Resume Here" boot block. Emits the stored checkpoint when one
    /// exists; otherwise falls back to a derived checkpoint assembled
    /// from the latest session record. Returns `None` only when the
    /// database has nothing meaningful to surface.
    fn boot_block(&self, db: &Axil) -> Option<String> {
        let (checkpoint, source) = current_checkpoint(db)?;
        let body = render_resume_block(db, &checkpoint)?;
        // Tag the source so the agent knows whether the prior session
        // actually wrote a checkpoint or we're guessing from leftovers.
        Some(if source == Source::Derived {
            format!("{body}- _(source: derived — no explicit checkpoint stored)_\n")
        } else {
            body
        })
    }

    fn handle_cli(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        // We only claim the `checkpoint` top-level command.
        let Some(top) = invocation.command_path.first() else {
            return Ok(Dispatch::NotHandled);
        };
        if top != "checkpoint" {
            return Ok(Dispatch::NotHandled);
        }
        // Two shapes: `checkpoint <subcommand> …` and `checkpoint <json>` (the
        // common-case shorthand — no subcommand needed for a write).
        let sub = invocation.command_path.get(1).map(String::as_str);
        match sub {
            Some("show") => Ok(Dispatch::Handled(handle_show(db))),
            Some("write") | None => handle_write_cli(db, invocation),
            _ => Ok(Dispatch::NotHandled),
        }
    }

    fn handle_mcp(
        &self,
        db: &Axil,
        call: &McpCall,
    ) -> CoreResult<Dispatch<Value>> {
        match call.tool.as_str() {
            "checkpoint" => handle_write_mcp(db, &call.params).map(Dispatch::Handled),
            "checkpoint_show" => Ok(Dispatch::Handled(handle_show_mcp(db))),
            _ => Ok(Dispatch::NotHandled),
        }
    }
}

/// Resolve the JSON payload from CLI args: a positional `<json>`, `-`
/// for stdin, or — when neither is provided and stdin was captured —
/// fall through to stdin. Mirrors `axil deps ingest`'s contract.
fn read_payload(invocation: &CliInvocation) -> Result<Value, AxilError> {
    // Find the first positional that isn't a known flag or its value.
    // Known flags (must match `cli_commands()` — adding one here without
    // declaring it there will silently swallow the user's argument):
    //   --session <id>  (takes a value)
    //   --final         (bare flag)
    let mut positional: Option<&String> = None;
    let mut iter = invocation.args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--session" {
            iter.next(); // consume the value too
            continue;
        }
        if arg == "--final" || arg.starts_with("--") {
            continue;
        }
        positional = Some(arg);
        break;
    }
    let raw = match positional {
        Some(s) if s == "-" => invocation
            .stdin
            .clone()
            .ok_or_else(|| AxilError::InvalidQuery("checkpoint: `-` requested stdin but none was captured".into()))?,
        Some(s) => s.clone(),
        None => invocation
            .stdin
            .clone()
            .ok_or_else(|| AxilError::InvalidQuery("checkpoint: provide a JSON object as a positional arg or via stdin".into()))?,
    };
    serde_json::from_str(&raw)
        .map_err(|e| AxilError::InvalidQuery(format!("checkpoint: payload is not valid JSON: {e}")))
}

fn flag_set(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn named_arg(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    let prefix = format!("{name}=");
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
        if let Some(rest) = arg.strip_prefix(prefix.as_str()) {
            return Some(rest.to_string());
        }
    }
    None
}

fn handle_write_cli(
    db: &Axil,
    invocation: &CliInvocation,
) -> CoreResult<Dispatch<CliOutput>> {
    let value = read_payload(invocation)?;
    let checkpoint = Checkpoint::from_value(value).map_err(checkpoint_err_to_axil)?;
    let session_override = named_arg(&invocation.args, "--session");
    let is_final = flag_set(&invocation.args, "--final");
    let record = write_resolved(db, &checkpoint, session_override.as_deref(), is_final)?;
    Ok(Dispatch::Handled(json_stdout(write_response(&record, &checkpoint))))
}

fn handle_write_mcp(db: &Axil, params: &Value) -> CoreResult<Value> {
    // Pull side-channel fields out of the params object before parsing
    // the checkpoint so they don't accidentally land on the record.
    let mut payload = params
        .as_object()
        .cloned()
        .ok_or_else(|| AxilError::InvalidQuery("checkpoint: params must be a JSON object".into()))?;
    let session_override = payload
        .remove("session")
        .and_then(|v| v.as_str().map(String::from));
    let is_final = payload
        .remove("final")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let checkpoint = Checkpoint::from_value(Value::Object(payload)).map_err(checkpoint_err_to_axil)?;
    let record = write_resolved(db, &checkpoint, session_override.as_deref(), is_final)?;
    Ok(write_response(&record, &checkpoint))
}

fn write_resolved(
    db: &Axil,
    checkpoint: &Checkpoint,
    session_override: Option<&str>,
    is_final: bool,
) -> CoreResult<axil_core::Record> {
    let kind = if is_final {
        CheckpointKind::Final
    } else {
        CheckpointKind::Snapshot
    };
    match session_override {
        Some(id) => {
            let sid = RecordId::from_string(id)
                .map_err(|e| AxilError::InvalidQuery(format!("checkpoint: invalid session id `{id}`: {e}")))?;
            write_for_session(db, &sid, checkpoint, kind).map_err(checkpoint_err_to_axil)
        }
        None => write_with_active_session(db, checkpoint, kind).map_err(checkpoint_err_to_axil),
    }
}

fn write_response(record: &axil_core::Record, checkpoint: &Checkpoint) -> Value {
    json!({
        "id": record.id.to_string(),
        "session_id": record.data.get("session_id"),
        "kind": record.data.get("kind"),
        "fields": {
            "goal": checkpoint.goal.is_some(),
            "state": checkpoint.state.is_some(),
            "next_steps": checkpoint.next_steps.len(),
            "open_questions": checkpoint.open_questions.len(),
            "references": checkpoint.references.len(),
            "summary": checkpoint.summary.is_some(),
        }
    })
}

fn handle_show(db: &Axil) -> CliOutput {
    json_stdout(handle_show_mcp(db))
}

fn handle_show_mcp(db: &Axil) -> Value {
    match current_checkpoint(db) {
        Some((checkpoint, source)) => {
            let block = render_resume_block(db, &checkpoint).unwrap_or_default();
            json!({
                "source": source.as_str(),
                "checkpoint": checkpoint,
                "rendered": block,
            })
        }
        None => json!({ "source": null, "checkpoint": null, "rendered": "" }),
    }
}

fn json_stdout(value: Value) -> CliOutput {
    CliOutput {
        exit_code: 0,
        stdout: serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        stderr: String::new(),
    }
}

fn checkpoint_err_to_axil(e: CheckpointError) -> AxilError {
    match e {
        CheckpointError::Axil(a) => a,
        other => AxilError::InvalidQuery(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TABLE_CHECKPOINTS, TABLE_SESSIONS as SESSIONS};

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn id_and_table_prefix() {
        let ext = CheckpointExtension;
        assert_eq!(ext.id(), "checkpoint");
        assert_eq!(ext.display_name(), "Session Checkpoint");
        let p = ext.table_prefixes();
        assert_eq!(p, &["_checkpoint_"]);
        assert!(TABLE_CHECKPOINTS.starts_with(p[0]));
    }

    #[test]
    fn cli_surface_advertises_write_and_show() {
        let ext = CheckpointExtension;
        let surface = ext.cli_commands().unwrap();
        assert_eq!(surface.command, "checkpoint");
        let names: Vec<&str> = surface.subcommands.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"write"));
        assert!(names.contains(&"show"));
    }

    #[test]
    fn mcp_surface_exposes_checkpoint_tools() {
        let ext = CheckpointExtension;
        let surface = ext.mcp_tools().unwrap();
        let names: Vec<&str> = surface.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"checkpoint"));
        assert!(names.contains(&"checkpoint_show"));
    }

    #[test]
    fn boot_block_quiet_on_fresh_db() {
        let (db, _dir) = temp_db();
        assert!(CheckpointExtension.boot_block(&db).is_none());
    }

    #[test]
    fn boot_block_renders_stored_checkpoint_resume_block() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["checkpoint".into()],
            args: vec![r#"{"goal":"ship axil-checkpoint","next_steps":["wire boot block"]}"#.into()],
            stdin: None,
        };
        let out = CheckpointExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("checkpoint write should be handled");
        assert_eq!(out.exit_code, 0);

        let block = CheckpointExtension.boot_block(&db).unwrap();
        assert!(block.starts_with("## Resume Here"));
        assert!(block.contains("ship axil-checkpoint"));
        assert!(block.contains("wire boot block"));
        // Stored, not derived — should NOT carry the derived-source tag.
        assert!(!block.contains("derived"));
    }

    #[test]
    fn boot_block_falls_back_to_derived_when_no_checkpoint_stored() {
        let (db, _dir) = temp_db();
        db.insert(
            SESSIONS,
            json!({
                "status": "ended",
                "summary": "wired checkpoint scaffold",
                "decisions_made": ["tier-2 extension"]
            }),
        )
        .unwrap();
        let block = CheckpointExtension.boot_block(&db).unwrap();
        assert!(block.contains("wired checkpoint scaffold"));
        assert!(block.contains("derived"));
    }

    #[test]
    fn handle_cli_write_via_positional_json() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["checkpoint".into(), "write".into()],
            args: vec![r#"{"state":"halfway","next_steps":["a","b"]}"#.into()],
            stdin: None,
        };
        let out = CheckpointExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["fields"]["next_steps"], 2);
        assert_eq!(v["kind"], CheckpointKind::Snapshot.as_str());
        assert_eq!(db.list(TABLE_CHECKPOINTS).unwrap().len(), 1);
    }

    #[test]
    fn handle_cli_write_via_stdin() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["checkpoint".into()],
            args: vec!["-".into()],
            stdin: Some(r#"{"goal":"ship"}"#.into()),
        };
        let out = CheckpointExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["fields"]["goal"], true);
    }

    #[test]
    fn handle_cli_write_final_stamps_kind_final() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["checkpoint".into(), "write".into()],
            args: vec!["--final".into(), r#"{"goal":"ship"}"#.into()],
            stdin: None,
        };
        let out = CheckpointExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["kind"], CheckpointKind::Final.as_str());
    }

    #[test]
    fn handle_cli_show_returns_derived_when_no_stored() {
        let (db, _dir) = temp_db();
        db.insert(SESSIONS, json!({"status": "ended", "summary": "x"}))
            .unwrap();
        let out = CheckpointExtension
            .handle_cli(
                &db,
                &CliInvocation {
                    command_path: vec!["checkpoint".into(), "show".into()],
                    args: vec![],
                    stdin: None,
                },
            )
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["source"], "derived");
        assert!(v["rendered"].as_str().unwrap().contains("Resume Here"));
    }

    #[test]
    fn handle_cli_unknown_subcommand_declines() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["checkpoint".into(), "definitely-not-real".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            CheckpointExtension.handle_cli(&db, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn handle_cli_non_checkpoint_top_command_declines() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["recall".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            CheckpointExtension.handle_cli(&db, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn handle_mcp_write_then_show() {
        let (db, _dir) = temp_db();
        let call = McpCall {
            tool: "checkpoint".into(),
            params: json!({"goal":"mcp goal","next_steps":["one"]}),
        };
        let v = CheckpointExtension
            .handle_mcp(&db, &call)
            .unwrap()
            .handled()
            .expect("handled");
        assert_eq!(v["fields"]["next_steps"], 1);

        let show = McpCall {
            tool: "checkpoint_show".into(),
            params: json!({}),
        };
        let shown = CheckpointExtension
            .handle_mcp(&db, &show)
            .unwrap()
            .handled()
            .expect("handled");
        assert_eq!(shown["source"], "stored");
        assert_eq!(shown["checkpoint"]["goal"], "mcp goal");
    }

    #[test]
    fn handle_mcp_unknown_tool_declines() {
        let (db, _dir) = temp_db();
        let call = McpCall {
            tool: "not-a-checkpoint-tool".into(),
            params: Value::Null,
        };
        assert!(matches!(
            CheckpointExtension.handle_mcp(&db, &call).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn registers_in_axil_builder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path)
            .with_extension(CheckpointExtension)
            .build()
            .unwrap();
        assert_eq!(db.extensions().len(), 1);
        assert_eq!(db.extensions()[0].id(), "checkpoint");
    }
}
