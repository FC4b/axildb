//! Interactive wizard for bare `axil install`.
//!
//! Flags stay the source of truth for scripts and CI: the wizard only runs
//! when `axil install` is invoked with no selection flags, on a real
//! terminal, without `--quiet`. It detects which agent tooling already
//! exists in the project (`.claude/`, `.cursor/`, …), pre-checks those
//! integrations, and lets the user toggle integrations + bootstrap/local
//! before the normal install path runs. Selecting nothing and pressing
//! Enter reproduces today's bare install (DB only) exactly.

use std::io::{self, IsTerminal, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// What the user picked in the wizard — maps 1:1 onto the `axil install`
/// flags it replaces.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InstallChoices {
    pub claude_code: bool,
    pub codex: bool,
    pub copilot: bool,
    pub droid: bool,
    pub antigravity: bool,
    pub qwen: bool,
    pub cursor: bool,
    pub windsurf: bool,
    pub aider: bool,
    pub agents_md: bool,
    pub bootstrap: bool,
    pub local: bool,
}

pub enum WizardOutcome {
    /// No TTY (or `--quiet`) — caller proceeds with the flags it was given.
    NotInteractive,
    /// User quit — caller exits without writing anything.
    Aborted,
    Choices(InstallChoices),
}

/// Agent tooling already present in the project — used to pre-check the
/// wizard selection, not to gate anything.
pub fn detect_agents(cwd: &Path) -> InstallChoices {
    InstallChoices {
        claude_code: cwd.join(".claude").is_dir() || cwd.join("CLAUDE.md").is_file(),
        codex: cwd.join(".codex").is_dir(),
        copilot: cwd.join(".github").join("copilot-instructions.md").is_file()
            || cwd.join(".github").join("hooks").is_dir(),
        droid: cwd.join(".factory").is_dir(),
        antigravity: cwd.join(".agents").is_dir() || cwd.join(".antigravity.md").is_file(),
        qwen: cwd.join(".qwen").is_dir(),
        cursor: cwd.join(".cursor").is_dir(),
        windsurf: cwd.join(".windsurfrules").is_file() || cwd.join(".windsurf").is_dir(),
        aider: cwd.join(".aider.conf.yml").is_file(),
        agents_md: cwd.join("AGENTS.md").is_file(),
        // Defaults for the non-agent toggles: bootstrap is almost always
        // what you want on a code repo; repo-local skills stay opt-in.
        bootstrap: true,
        local: false,
    }
}

/// Number of agent-integration items at the head of the wizard list —
/// everything after them renders under the "Options" header and is
/// excluded from the "all agents" toggle.
const AGENT_ITEM_COUNT: usize = 10;

struct Item {
    label: &'static str,
    detail: &'static str,
    detected: bool,
    checked: bool,
}

fn items_from(choices: &InstallChoices, detected: &InstallChoices) -> Vec<Item> {
    vec![
        Item {
            label: "claude-code",
            detail: "Claude Code (skills + brain hook + CLAUDE.md)",
            detected: detected.claude_code,
            checked: choices.claude_code,
        },
        Item {
            label: "codex",
            detail: "OpenAI Codex (hooks + MCP + skills)",
            detected: detected.codex,
            checked: choices.codex,
        },
        Item {
            label: "copilot",
            detail: "GitHub Copilot CLI (hooks + MCP)",
            detected: detected.copilot,
            checked: choices.copilot,
        },
        Item {
            label: "droid",
            detail: "Factory Droid (hooks + MCP)",
            detected: detected.droid,
            checked: choices.droid,
        },
        Item {
            label: "antigravity",
            detail: "Google Antigravity (hooks + MCP + rules + skills)",
            detected: detected.antigravity,
            checked: choices.antigravity,
        },
        Item {
            label: "qwen",
            detail: "Qwen Code (hooks + MCP)",
            detected: detected.qwen,
            checked: choices.qwen,
        },
        Item {
            label: "cursor",
            detail: "Cursor (.cursor/rules)",
            detected: detected.cursor,
            checked: choices.cursor,
        },
        Item {
            label: "windsurf",
            detail: "Windsurf (.windsurfrules)",
            detected: detected.windsurf,
            checked: choices.windsurf,
        },
        Item {
            label: "aider",
            detail: "Aider (CONVENTIONS.md + read list)",
            detected: detected.aider,
            checked: choices.aider,
        },
        Item {
            label: "agents-md",
            detail: "AGENTS.md contract (read by most agents)",
            detected: detected.agents_md,
            checked: choices.agents_md,
        },
        Item {
            label: "bootstrap",
            detail: "Index the codebase now (code-search works immediately)",
            detected: false,
            checked: choices.bootstrap,
        },
        Item {
            label: "local",
            detail: "Repo-local skills (.claude/skills/ here, not ~/.claude)",
            detected: false,
            checked: choices.local,
        },
    ]
}

fn choices_from(items: &[Item]) -> InstallChoices {
    let on = |label: &str| items.iter().any(|i| i.label == label && i.checked);
    InstallChoices {
        claude_code: on("claude-code"),
        codex: on("codex"),
        copilot: on("copilot"),
        droid: on("droid"),
        antigravity: on("antigravity"),
        qwen: on("qwen"),
        cursor: on("cursor"),
        windsurf: on("windsurf"),
        aider: on("aider"),
        agents_md: on("agents-md"),
        bootstrap: on("bootstrap"),
        local: on("local"),
    }
}

fn render(items: &[Item]) {
    println!();
    println!("  Agent integrations");
    for (n, item) in items.iter().enumerate() {
        if n == AGENT_ITEM_COUNT {
            println!("  Options");
        }
        let mark = if item.checked { "x" } else { " " };
        let tag = if item.detected { "  [detected]" } else { "" };
        println!(
            "    {:>2} [{}] {:<12} {}{}",
            n + 1,
            mark,
            item.label,
            item.detail,
            tag
        );
    }
    println!();
    println!("  toggle: <number>   all agents: a   none: n   install: <enter>   quit: q");
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

/// Run the wizard when interactive; see module docs for the gate.
pub fn maybe_run(cwd: &Path, quiet: bool) -> Result<WizardOutcome> {
    if quiet || !io::stdin().is_terminal() {
        return Ok(WizardOutcome::NotInteractive);
    }

    let detected = detect_agents(cwd);
    // AGENTS.md is the cross-tool contract — pre-check its toggle even when
    // no AGENTS.md exists yet, matching the flag path's default-on behavior.
    let mut initial = detected;
    initial.agents_md = true;
    let mut items = items_from(&initial, &detected);

    println!("Axil project install — set up agent memory in {}", cwd.display());
    if cwd.join(".axil").join("version").exists() {
        println!("note: .axil/ already exists here — `axil sync` updates an existing install; continuing will re-install");
    }
    println!("Detected agent tooling is pre-selected. Toggle, then Enter to install.");

    loop {
        render(&items);
        let input = prompt("> ")?;
        match input.as_str() {
            "" | "install" => break,
            "q" | "quit" => return Ok(WizardOutcome::Aborted),
            "a" | "all" => {
                for item in items.iter_mut().take(AGENT_ITEM_COUNT) {
                    item.checked = true;
                }
            }
            "n" | "none" => {
                for item in items.iter_mut() {
                    item.checked = false;
                }
            }
            other => match other.parse::<usize>() {
                Ok(k) if (1..=items.len()).contains(&k) => {
                    items[k - 1].checked = !items[k - 1].checked;
                }
                _ => println!("  ? unrecognized: {other}"),
            },
        }
    }

    let choices = choices_from(&items);
    let picked: Vec<&str> = items
        .iter()
        .filter(|i| i.checked)
        .map(|i| i.label)
        .collect();
    if picked.is_empty() {
        println!("Installing database only (no agent integration).");
    } else {
        println!("Installing: {}", picked.join(", "));
    }
    Ok(WizardOutcome::Choices(choices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Unique scratch dir per test — no tempfile dependency.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axil-install-wizard-{}-{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn detects_nothing_in_empty_project() {
        let dir = scratch("empty");
        let d = detect_agents(&dir);
        assert!(
            !d.claude_code
                && !d.codex
                && !d.copilot
                && !d.droid
                && !d.cursor
                && !d.windsurf
                && !d.aider
                && !d.agents_md
        );
        assert!(d.bootstrap, "bootstrap defaults on");
        assert!(!d.local, "local defaults off");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_present_agent_tooling() {
        let dir = scratch("detect");
        fs::create_dir_all(dir.join(".claude")).unwrap();
        fs::create_dir_all(dir.join(".cursor")).unwrap();
        fs::create_dir_all(dir.join(".codex")).unwrap();
        fs::create_dir_all(dir.join(".factory")).unwrap();
        fs::create_dir_all(dir.join(".github").join("hooks")).unwrap();
        fs::write(dir.join(".windsurfrules"), "").unwrap();
        fs::write(dir.join("AGENTS.md"), "").unwrap();
        let d = detect_agents(&dir);
        assert!(d.claude_code && d.codex && d.copilot && d.droid);
        assert!(d.cursor && d.windsurf && d.agents_md);
        assert!(!d.aider);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn claude_md_alone_counts_as_claude_code() {
        let dir = scratch("claudemd");
        fs::write(dir.join("CLAUDE.md"), "# hi").unwrap();
        assert!(detect_agents(&dir).claude_code);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn items_roundtrip_to_choices() {
        let dir = scratch("roundtrip");
        fs::create_dir_all(dir.join(".cursor")).unwrap();
        let detected = detect_agents(&dir);
        let items = items_from(&detected, &detected);
        let choices = choices_from(&items);
        assert_eq!(choices, detected, "render/parse round-trip must be lossless");
        let _ = fs::remove_dir_all(&dir);
    }
}
