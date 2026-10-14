//! Auto-capture.
//!
//! Analyzes text from agent actions and auto-extracts knowledge:
//! errors, decisions, and context. The agent doesn't need to explicitly store.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::LazyLock;

/// Classification of captured knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureType {
    /// An error or failure detected in output.
    Error,
    /// A decision or choice made by the agent.
    Decision,
    /// General context or learning.
    Context,
    /// A standing constraint on later work ("never ...", "every X must ...").
    Rule,
    /// Not significant enough to capture.
    Skip,
}

/// A piece of auto-captured knowledge.
#[derive(Debug, Clone, Serialize)]
pub struct CapturedKnowledge {
    pub capture_type: CaptureType,
    pub summary: String,
    pub confidence: f32,
    pub source: String,
}

/// Error detection patterns. Each is matched against lowercased input.
///
/// Patterns that are specific (rustc error codes, `panicked at`) need less
/// surrounding context to count. Patterns that are weak standalone
/// (`timeout`, `not found`, `failed`) are only scored as errors when
/// surrounded by error-like context or stack-trace markers; see
/// `error_confidence` for the context checks and
/// `is_pattern_false_positive` for the negation guards.
const ERROR_PATTERNS: &[&str] = &[
    "error[e",            // rustc E0308 etc — very specific
    "error:",             // generic "error: ..."
    "failed",             // build/test failure (gated by context)
    "panicked at",        // Rust panics (with location)
    "panic!",             // explicit panic invocation
    "exception:",         // typed exception header line
    "traceback",          // python tracebacks
    "fatal",              // fatal errors
    "segfault",           // segfaults
    "cannot find",        // missing files/modules
    "not found",          // 404s, missing deps (gated)
    "permission denied",  // auth/fs
    "connection refused", // network
    "timed out",          // network/op timeout (stronger than "timeout")
    "out of memory",
];

/// Decision detection patterns. Matched against lowercased input.
///
/// Future-tense phrases ("will use", "should use") are intentionally NOT
/// here — they're speculation, not decisions. Imperative-of-intent ("let's")
/// stays because it signals commitment in agent text.
const DECISION_PATTERNS: &[&str] = &[
    "decided to",
    "switching to",
    "adopted ",
    "going with",
    "let's use",
    "chose to ", // "chose " alone matched "he chose vanilla"
    "decision:", // explicit decision marker
    "we picked ",
    "we settled on",
    "moved from ", // migration: "moved from X to Y"
];

/// Phrases that, if present anywhere in the text, suppress a capture even
/// when a pattern matches. These are common false-positive contexts that
/// are too weak to special-case per pattern.
const CAPTURE_NEGATIONS: &[&str] = &[
    "no matches found",
    "0 failed",
    "0 failures",
    "test result: ok",
    "all tests passed",
    "tests passed",
    "don't panic",
    "do not panic",
    "with the exception of",
    "exception handling",
    "no errors",
    "no error found",
];

/// Shell prefixes of a message about the command line itself (a glob that
/// matched nothing, zsh reading `===` as a command, a quoting error).
const SHELL_PREFIXES: &[&str] = &["(eval):", "zsh:", "bash:", "-bash:", "sh:"];

/// Shell messages that do report a failure of the program that ran.
const SHELL_FAILURES: &[&str] = &[
    "abort",
    "bus error",
    "killed",
    "out of memory",
    "segmentation fault",
    "terminated",
];

/// What rg (after its `regex parse error:` header) and grep say about a
/// regex the agent wrote; nothing in the project failed.
const REGEX_SYNTAX_ERRORS: &[&str] = &[
    "backreferences are not supported",
    "empty character class",
    "incomplete escape sequence",
    "invalid character class range",
    "invalid character range",
    "invalid regular expression",
    "look-around",
    "repetition operator missing expression",
    "trailing backslash",
    "unclosed character class",
    "unclosed counted repetition",
    "unclosed group",
    "unmatched (",
    "unmatched [",
    "unopened group",
    "unrecognized escape sequence",
];

/// git refusing a command as typed: a guessed ref or path, a branch that
/// exists, a worktree in the way. Nothing in the project failed.
const GIT_REFUSALS: &[&str] = &[
    "already checked out",
    "already exists",
    "already used by worktree",
    "ambiguous argument",
    "bad revision",
    "cannot delete branch",
    "couldn't find remote ref",
    "did not match any",
    "does not exist in",
    "exists on disk, but not in",
    "invalid reference",
    "is not fully merged",
    "needed a single revision",
    "no upstream",
    "not a git repository",
    "not a valid object name",
    "not something we can merge",
    "refusing to",
    "unknown revision",
    "would be overwritten",
    "you have unstaged changes",
];

/// A program rejecting its own command line (clap, cargo, argparse, BSD
/// tools): a typo'd flag or subcommand, a missing argument.
const USAGE_ERRORS: &[&str] = &[
    "a value is required for",
    "cannot be used with",
    "illegal option",
    "invalid choice",
    "invalid option",
    "no bin target named",
    "no example target named",
    "no such command",
    "no such subcommand",
    "no test target named",
    "not found in workspace",
    "required arguments were not provided",
    "the following arguments are required",
    "unexpected argument",
    "unknown option",
    "unrecognized arguments",
    "unrecognized option",
    "unrecognized subcommand",
];

/// A search hit or numbered listing (`path:12:` / `path-12-` / `12:`)
/// followed by source text rather than by a compiler-style ` message`:
/// right after the prefix, or after an indent. `file.c:10:5: error: ...`
/// (one space, then the message) is not matched.
static PRINTED_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[^\s:]+\.[A-Za-z0-9]+[:-])?\d+[:-](?:\d+[:-])?(?:[^\s\d]|\s{2,}|$)").unwrap()
});

/// The start of a source line printed for reading.
const CODE_LINE_STARTS: &[&str] = &[
    "#![", "#[", "/*", "//", "def ", "fn ", "impl ", "let ", "match ", "pub ", "raise ", "return ",
    "use ",
];

/// Phrases with which agent prose reports a failure or a defect that
/// happened or exists (not one being looked for). Matched lowercased.
const FAILURE_REPORTS: &[&str] = &[
    "broke because",
    "broken since",
    "can never fire",
    "crashed because",
    "dead since",
    "does not work",
    "doesn't work",
    "failed because",
    "failed once",
    "fails because",
    "fails intermittently",
    "has been broken",
    "has been dead",
    "is broken",
    "never fires",
    "panicked because",
    "panics",
    "ran out of memory",
    "runs out of memory",
    "silently",
    "stopped working",
    "was broken",
];

/// A stated root cause ("Root cause: ...", "the root cause is ...", "Root
/// cause of X ..."). One the agent is still looking for is narration, and
/// "root cause in one line" names a report field.
static ROOT_CAUSE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:\b(find|finding|look for|looking for|investigate|investigating|identify|determine|chase|chasing|hunt|locate|likely|possible|probable|potential)\s+(?:the\s+|a\s+)?)?\broot cause\b(?:\*\*)?\s*(?::|—|is\b|was\b|of\b|here\b|turned\b|lies\b|appears\b)")
        .unwrap()
});

/// How narration opens: what the agent is about to do or doing, with no
/// finding ("Let me check ...", "Now the tests.").
const NARRATION_STARTS: &[&str] = &[
    "checking",
    "i'll ",
    "i will ",
    "i'm going to ",
    "investigating",
    "let me ",
    "let's ",
    "looking ",
    "next,",
    "now ",
    "time to ",
];

/// Words of a `fix:` commit subject that names no defect ("address review",
/// "misc cleanup").
const NO_DEFECT_FIXES: &[&str] = &[
    "cleanup", "comments", "feedback", "misc", "nit", "nits", "review", "reviews", "tidy",
];

/// A conventional-commit fix subject: `fix(scope): what was wrong`.
static FIX_SUBJECT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:fix|bugfix|hotfix)(?:\([^)]*\))?!?:\s+\S").unwrap());

/// A list of standing rules: `CONVENTIONS:`, `**Rules:**`, `- Invariant:`.
static RULE_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:[-*]\s+|\d+\.\s+)?(?:\*\*)?(?:conventions?|rules?|ground rules|house rules|constraints?|invariants?)(?:\*\*)?\s*:",
    )
    .unwrap()
});

/// A sentence that opens with a standing imperative. Capitalized, so a
/// wrapped line that happens to start with "never" does not count.
static RULE_IMPERATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:^|[.;!]\s+)(?:[-*]\s+|\d+\.\s+)?(?:\*\*)?(?:Never|NEVER|Always|ALWAYS|Do not|Do NOT|DO NOT|Don't|DON'T)\s",
    )
    .unwrap()
});

/// "Every new table name must ...": a rule over a whole kind of thing.
static RULE_EVERY_MUST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bevery\s+(?:[\w-]+\s+){1,4}must\b").unwrap());

/// A finding the agent labeled as one: `Gotcha: ...`, `**Lesson learned:**`,
/// a list item `- TIL: ...`.
static FINDING_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:[-*]\s+)?(?:\*\*)?(?:gotcha|lessons?(?: learned)?|learned|til|key (?:insight|finding)|takeaway|note to self|for future reference)(?:\*\*)?\s*:",
    )
    .unwrap()
});

/// "It turns out ..." / "turned out that ..." reports a finding, unless the
/// sentence is conditional ("if it turns out ...").
static TURNS_OUT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:(if|whether|unless|case)\s+)?(?:it\s+|this\s+|that\s+)?(?:turns out|turned out that)\b")
        .unwrap()
});

/// Minimum line length for a capture summary. Below this, a match is
/// almost certainly too short to be a useful knowledge entry.
const MIN_SUMMARY_LEN: usize = 20;

/// Minimum total-input length. Tiny strings get skipped entirely.
const MIN_INPUT_LEN: usize = 30;

/// The source the brain hook gives a failed shell command's output.
///
/// Command output is read for failures only. A decision phrase in it is text
/// the command printed (a task file, a log, a commit list), not a choice the
/// agent made.
pub const COMMAND_OUTPUT_SOURCE: &str = "bash";

/// Analyze text and extract capturable knowledge.
///
/// Returns a list of captures with confidence scores, the most specific
/// class first (an error, a decision, a rule, then context). Text from
/// [`COMMAND_OUTPUT_SOURCE`] only yields errors.
/// Callers (e.g. `axil auto-capture`) should only auto-store captures with
/// confidence >= 0.7.
pub fn analyze(text: &str, source: &str) -> Vec<CapturedKnowledge> {
    if text.len() < MIN_INPUT_LEN {
        return Vec::new();
    }
    let text_lower = text.to_lowercase();

    let lines: Vec<&str> = text.lines().collect();
    let test_failure = has_test_failure(&lines, &text_lower);

    // Global negation: any of these phrases in the whole text kills the
    // capture. Cheaper than per-pattern handling and cuts the dominant
    // false-positive class (successful test output misclassified as error).
    // A failing test outranks them: a run of several test binaries prints
    // `test result: ok. ... 0 failed` for the ones that passed.
    if !test_failure && CAPTURE_NEGATIONS.iter().any(|neg| text_lower.contains(neg)) {
        return Vec::new();
    }

    let guards = TextGuards::of(&lines, &text_lower);
    let mut captures = Vec::new();
    let mut claimed_line: Option<usize> = None;

    // Detect errors. We take the first matching pattern to avoid multi-capture
    // for the same event — errors tend to produce several patterns in one
    // block (e.g. "panic at" + "traceback"), but they describe one failure.
    for pattern in ERROR_PATTERNS {
        if !text_lower.contains(pattern) {
            continue;
        }
        // The first line where the pattern reads as a failure: a passing
        // suite's `0 failed` line does not hide the failing test below it.
        let Some((idx, line)) = lines.iter().enumerate().find(|(_, l)| {
            let summary = l.trim();
            l.to_lowercase().contains(pattern)
                && !guards.is_guarded(l)
                && summary.len() >= MIN_SUMMARY_LEN
                && !is_pattern_false_positive(pattern, summary)
        }) else {
            continue;
        };
        let summary = line.trim();
        captures.push(CapturedKnowledge {
            capture_type: CaptureType::Error,
            summary: truncate(summary, 200),
            confidence: error_confidence(&text_lower, pattern, test_failure),
            source: source.to_string(),
        });
        claimed_line = Some(idx);
        break;
    }

    if source == COMMAND_OUTPUT_SOURCE {
        return captures;
    }

    // Agent prose reporting a failure, or a fix commit subject naming the
    // defect. It replaces a weaker pattern capture of the same text.
    let weak_error = captures.first().is_some_and(|c| c.confidence < 0.8);
    if captures.is_empty() || weak_error {
        if let Some((idx, line)) = failure_report(text, &lines, &guards) {
            captures.clear();
            captures.push(CapturedKnowledge {
                capture_type: CaptureType::Error,
                summary: truncate(line.trim(), 200),
                confidence: 0.8,
                source: source.to_string(),
            });
            claimed_line = Some(idx);
        }
    }

    // Detect decisions. Skip the line already claimed by the error capture
    // so "decided to abort because of the error" doesn't double-count.
    for pattern in DECISION_PATTERNS {
        if !text_lower.contains(pattern) {
            continue;
        }
        let Some((_idx, line)) = lines.iter().enumerate().find(|(i, l)| {
            Some(*i) != claimed_line && l.to_lowercase().contains(pattern) && !guards.is_guarded(l)
        }) else {
            continue;
        };
        let summary = line.trim();
        if summary.len() < MIN_SUMMARY_LEN {
            continue;
        }
        captures.push(CapturedKnowledge {
            capture_type: CaptureType::Decision,
            summary: truncate(summary, 200),
            confidence: decision_confidence(&text_lower, pattern),
            source: source.to_string(),
        });
        break;
    }

    // Detect standing rules.
    let rule = lines
        .iter()
        .enumerate()
        .find(|(i, l)| Some(*i) != claimed_line && !guards.is_guarded(l) && is_rule(l));
    if let Some((_, line)) = rule {
        let summary = line.trim();
        if summary.len() >= MIN_SUMMARY_LEN {
            captures.push(CapturedKnowledge {
                capture_type: CaptureType::Rule,
                summary: truncate(summary, 200),
                confidence: 0.8,
                source: source.to_string(),
            });
        }
    }

    // Detect findings worth keeping as context.
    let finding = lines
        .iter()
        .enumerate()
        .find(|(i, l)| Some(*i) != claimed_line && !guards.is_guarded(l) && is_finding(l));
    if let Some((_, line)) = finding {
        let summary = line.trim();
        if summary.len() >= MIN_SUMMARY_LEN {
            captures.push(CapturedKnowledge {
                capture_type: CaptureType::Context,
                summary: truncate(summary, 200),
                confidence: 0.8,
                source: source.to_string(),
            });
        }
    }

    captures
}

/// The line of agent prose that reports a failure: a fix commit subject
/// that names its defect, a stated root cause, or a failure phrase, outside
/// narration and outside a report that the problem is resolved.
fn failure_report<'a>(
    text: &str,
    lines: &[&'a str],
    guards: &TextGuards,
) -> Option<(usize, &'a str)> {
    let subject = text.trim();
    if !subject.contains('\n') && FIX_SUBJECT.is_match(subject) {
        let lower = subject.to_lowercase();
        let names_no_defect = lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| NO_DEFECT_FIXES.contains(&w))
            || lower.contains("clean up");
        if names_no_defect {
            return None;
        }
        let idx = lines.iter().position(|l| l.trim() == subject)?;
        return Some((idx, lines[idx]));
    }
    lines
        .iter()
        .enumerate()
        .map(|(i, l)| (i, *l))
        .find(|(_, l)| {
            if guards.is_guarded(l) || l.trim().len() < MIN_SUMMARY_LEN {
                return false;
            }
            let lower = l.to_lowercase();
            let opening = lower
                .trim_start()
                .trim_start_matches(['-', '*', ' '])
                .trim_start();
            if NARRATION_STARTS.iter().any(|n| opening.starts_with(n)) || lower.contains("resolved")
            {
                return false;
            }
            ROOT_CAUSE.captures_iter(l).any(|c| c.get(1).is_none())
                || FAILURE_REPORTS.iter().any(|p| lower.contains(p))
        })
}

/// A line that states a standing rule: a rule list's label, a sentence
/// opening with "Never" / "Always" / "Do not" / "Don't", or "every X must".
fn is_rule(line: &str) -> bool {
    RULE_LABEL.is_match(line) || RULE_IMPERATIVE.is_match(line) || RULE_EVERY_MUST.is_match(line)
}

/// A line the agent wrote to record a finding: a labeled one, or "it turns
/// out ..." outside a condition.
fn is_finding(line: &str) -> bool {
    if FINDING_LABEL.is_match(line) {
        return true;
    }
    TURNS_OUT.captures_iter(line).any(|c| c.get(1).is_none())
}

/// A cargo or pytest run that reports a failed test: `test x ... FAILED`,
/// `test result: FAILED`, cargo's `error: test failed`, pytest's
/// `FAILED tests/x.py::name`.
fn has_test_failure(lines: &[&str], text_lower: &str) -> bool {
    text_lower.contains("test result: failed")
        || text_lower.contains("error: test failed")
        || lines.iter().any(|l| {
            let l = l.trim();
            (l.starts_with("test ") && l.ends_with(" ... FAILED"))
                || (l.starts_with("FAILED ") && l.contains("::"))
        })
}

/// What a text says about all of its lines: whether it prints a diff, and
/// whether rg reported a regex of the agent's as unparseable.
struct TextGuards {
    in_diff: bool,
    regex_error: bool,
}

impl TextGuards {
    fn of(lines: &[&str], text_lower: &str) -> Self {
        Self {
            in_diff: lines.iter().any(|l| {
                l.starts_with("diff --git ") || l.starts_with("@@ -") || l.starts_with("Diff in ")
            }),
            regex_error: text_lower.contains("regex parse error"),
        }
    }

    /// A line whose failure words are not a failure of the project: the
    /// command line was wrong, or the command printed text for reading.
    fn is_guarded(&self, line: &str) -> bool {
        let lower = line.trim().to_lowercase();
        self.is_command_line_mistake(&lower) || self.is_printed_text(line)
    }

    fn is_command_line_mistake(&self, lower: &str) -> bool {
        if SHELL_PREFIXES.iter().any(|p| lower.starts_with(p)) {
            return !SHELL_FAILURES.iter().any(|f| lower.contains(f));
        }
        if lower.contains("regex parse error") {
            return true;
        }
        let regex_message = REGEX_SYNTAX_ERRORS.iter().any(|m| lower.contains(m));
        if regex_message
            && (lower.starts_with("grep: ")
                || lower.starts_with("rg: ")
                || (self.regex_error && lower.starts_with("error: ")))
        {
            return true;
        }
        let reported = lower.starts_with("error: ")
            || lower.starts_with("fatal: ")
            || lower.contains(": error: ")
            || lower.contains(": illegal option");
        reported
            && (GIT_REFUSALS.iter().any(|m| lower.contains(m))
                || USAGE_ERRORS.iter().any(|m| lower.contains(m)))
    }

    /// Source code, a search hit or a diff line, printed for reading. Code
    /// that handles errors (`eprintln!("... failed")`) is not a failure.
    fn is_printed_text(&self, line: &str) -> bool {
        if PRINTED_LINE.is_match(line) {
            return true;
        }
        if self.in_diff && (line.starts_with('+') || line.starts_with('-')) {
            return true;
        }
        let trimmed = line.trim();
        // Prose quotes code in backticks (`assert!(x)`); that is not a
        // source line.
        let unquoted = without_code_spans(trimmed);
        CODE_LINE_STARTS.iter().any(|p| trimmed.starts_with(p))
            || unquoted.contains("!(")
            || unquoted.contains("=>")
            || trimmed.ends_with(';')
            || trimmed.ends_with('{')
    }
}

/// `line` without its backticked spans.
fn without_code_spans(line: &str) -> String {
    line.split('`').step_by(2).collect()
}

/// Returns `true` when a pattern match looks like a known false positive
/// given its surrounding line. Per-pattern guards; gets the short/weak
/// signals without adding a full regex engine pass.
fn is_pattern_false_positive(pattern: &str, line: &str) -> bool {
    let lower = line.to_lowercase();
    match pattern {
        "failed" => {
            // Test-result summary line: "0 failed" is accounting, not an
            // error.
            if lower.starts_with("0 failed")
                || lower.contains(" 0 failed")
                || lower.contains("tests passed")
            {
                return true;
            }
            // Without a technical noun in the line, "failed" in free-form
            // prose (UX writing, docs) is noise.
            let has_technical_noun = [
                "test",
                "build",
                "compile",
                "exit",
                "assertion",
                "check",
                "job",
                "step",
                "pipeline",
                "deploy",
                "migration",
                "request",
            ]
            .iter()
            .any(|k| lower.contains(k));
            !has_technical_noun
        }
        "not found" => {
            // Documentation like "if file not found, create it" is not an
            // error being reported.
            lower.contains("if ") && lower.contains("not found")
        }
        "cannot find" => {
            // UX copy ("users cannot find the button") vs compiler / linker
            // error ("cannot find crate `x`").
            !(lower.contains("crate")
                || lower.contains("module")
                || lower.contains("function")
                || lower.contains("method")
                || lower.contains("file")
                || lower.contains("package")
                || lower.contains("library")
                || lower.contains("dependency")
                || lower.contains("symbol"))
        }
        "out of memory" => {
            // "keep the code-index vectors out of memory recall" is about
            // where rows live. An OOM is something that runs or is out of
            // memory, or a message that ends there ("error: out of memory").
            let Some(at) = lower.find(pattern) else {
                return false;
            };
            let prev = lower[..at]
                .trim_end()
                .rsplit(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or_default();
            let ran_out = matches!(
                prev,
                "" | "run"
                    | "runs"
                    | "ran"
                    | "running"
                    | "go"
                    | "goes"
                    | "went"
                    | "is"
                    | "was"
                    | "are"
                    | "were"
                    | "been"
            );
            let next = lower[at + pattern.len()..].trim_start();
            let ends_there = !next.starts_with(|c: char| c.is_alphabetic());
            !(ran_out || ends_there)
        }
        "fatal" => {
            // "fatal error" and "fatal:" are real; "fatal flaw" / "fatal
            // attraction" in free-form text are not.
            !lower.contains("fatal error")
                && !lower.contains("fatal:")
                && !lower.contains("fatal exception")
                && !lower.contains("fatal signal")
        }
        _ => false,
    }
}

fn decision_confidence(text: &str, pattern: &str) -> f32 {
    let mut conf: f32 = 0.6;
    // Stronger signals: explicit marker, migration verbs.
    if pattern == "decision:" {
        conf += 0.2;
    }
    if pattern == "moved from " {
        conf += 0.1;
    }
    if pattern == "we settled on" {
        conf += 0.15;
    }
    // Weaker signals: reasoning-without-commitment.
    if text.contains("maybe") || text.contains("perhaps") || text.contains("might") {
        conf -= 0.2;
    }
    conf.clamp(0.1, 1.0)
}

/// Build a storable JSON record from a capture.
pub fn capture_to_record(capture: &CapturedKnowledge) -> (String, Value) {
    let table = match capture.capture_type {
        CaptureType::Error => "errors",
        CaptureType::Decision => "decisions",
        CaptureType::Context => "context",
        CaptureType::Rule => "rules",
        CaptureType::Skip => "context",
    };

    let data = match capture.capture_type {
        CaptureType::Error => json!({
            "error": capture.summary,
            "_auto_captured": true,
            "_capture_source": capture.source,
            "_capture_confidence": capture.confidence,
        }),
        CaptureType::Decision => json!({
            "summary": capture.summary,
            "_auto_captured": true,
            "_capture_source": capture.source,
            "_capture_confidence": capture.confidence,
        }),
        CaptureType::Rule => json!({
            "rule": capture.summary,
            "_auto_captured": true,
            "_capture_source": capture.source,
            "_capture_confidence": capture.confidence,
        }),
        _ => json!({
            "summary": capture.summary,
            "type": "auto_captured",
            "_auto_captured": true,
            "_capture_source": capture.source,
            "_capture_confidence": capture.confidence,
        }),
    };

    (table.to_string(), data)
}

/// Confidence scoring for error detection.
fn error_confidence(text: &str, pattern: &str, test_failure: bool) -> f32 {
    let mut conf = 0.6f32;

    // A test runner reporting a failed test is as specific as a rustc code.
    if test_failure {
        conf += 0.2;
    }

    // Higher confidence for specific error formats
    if text.contains("error[e") {
        conf += 0.2;
    } // rustc with error code
    if text.contains("panic at") || text.contains("panicked at") {
        conf += 0.2;
    }
    if text.contains("stack trace") || text.contains("traceback") {
        conf += 0.1;
    }
    if text.contains("exit code") || text.contains("exit status") {
        conf += 0.1;
    }

    // Lower confidence for common false positives
    if pattern == "not found" && text.contains("no matches found") {
        conf -= 0.3;
    }
    if pattern == "failed" && text.contains("0 failed") && !test_failure {
        conf -= 0.4;
    }

    conf.clamp(0.1, 1.0)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..s.floor_char_boundary(max.saturating_sub(3))])
    }
}

/// Trait extension to find char boundary.
#[allow(dead_code)]
trait FloorCharBoundary {
    fn floor_char_boundary(&self, index: usize) -> usize;
}

impl FloorCharBoundary for str {
    fn floor_char_boundary(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.len();
        }
        let mut i = index;
        while i > 0 && !self.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_rust_error() {
        let output = "error[E0308]: mismatched types\n  --> src/main.rs:10:5";
        let captures = analyze(output, "bash");
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].capture_type, CaptureType::Error);
        assert!(captures[0].confidence >= 0.7);
    }

    #[test]
    fn detect_generic_error() {
        let output = "error: could not compile `axil-core`\nsome other output";
        let captures = analyze(output, "bash");
        assert!(!captures.is_empty());
        assert_eq!(captures[0].capture_type, CaptureType::Error);
    }

    #[test]
    fn skip_passing_tests() {
        let output = "test result: ok. 219 passed; 0 failed; 0 ignored";
        let captures = analyze(output, "bash");
        // "0 failed" should not trigger error capture (or low confidence)
        let errors: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Error && c.confidence >= 0.7)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn detect_decision() {
        let output = "After considering the options, decided to use JWT instead of sessions";
        let captures = analyze(output, "agent");
        let decisions: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Decision)
            .collect();
        assert_eq!(decisions.len(), 1);
    }

    #[test]
    fn no_capture_for_normal_output() {
        let output = "Compiling axil-core v0.2.0\nFinished in 2.3s";
        let captures = analyze(output, "bash");
        assert!(captures.is_empty());
    }

    #[test]
    fn capture_to_record_error() {
        let cap = CapturedKnowledge {
            capture_type: CaptureType::Error,
            summary: "connection refused by remote host".into(),
            confidence: 0.8,
            source: "bash".into(),
        };
        let (table, data) = capture_to_record(&cap);
        assert_eq!(table, "errors");
        assert_eq!(data["_auto_captured"], true);
    }

    // ── Precision guards (new) ─────────────────────────────────────────

    #[test]
    fn reject_short_input() {
        // Shorter than MIN_INPUT_LEN — skip without even checking.
        let captures = analyze("error: x", "bash");
        assert!(captures.is_empty());
    }

    #[test]
    fn reject_test_success_noise() {
        // Real cargo output: "test result: ok" short-circuits the whole
        // analyzer via the global negation list.
        let captures = analyze(
            "test result: ok. 42 passed; 0 failed; 0 ignored; 2 filtered out",
            "bash",
        );
        assert!(captures.is_empty(), "captures: {:?}", captures);
    }

    #[test]
    fn reject_ux_cannot_find() {
        // Free-form UX discussion — "cannot find" without a technical noun
        // should not fire.
        let captures = analyze(
            "Our users cannot find the checkout button on mobile — UX pass next week.",
            "notes",
        );
        let errs: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Error)
            .collect();
        assert!(errs.is_empty(), "errs: {:?}", errs);
    }

    #[test]
    fn accept_compiler_cannot_find() {
        // But same pattern with a technical noun IS an error.
        let captures = analyze(
            "error[E0463]: cannot find crate `tokio` in the dependency tree",
            "bash",
        );
        assert!(captures
            .iter()
            .any(|c| c.capture_type == CaptureType::Error));
    }

    #[test]
    fn reject_fatal_non_error() {
        // "fatal flaw" in writing should not register as an error.
        let captures = analyze(
            "The proposal has a fatal flaw that we need to discuss at next standup.",
            "notes",
        );
        let errs: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Error)
            .collect();
        assert!(errs.is_empty(), "errs: {:?}", errs);
    }

    #[test]
    fn accept_panicked_at() {
        let captures = analyze(
            "thread 'main' panicked at src/lib.rs:42:9: index out of range",
            "bash",
        );
        assert_eq!(captures[0].capture_type, CaptureType::Error);
        assert!(captures[0].confidence >= 0.7);
    }

    #[test]
    fn reject_future_tense_decision() {
        // "will use" is speculation, not a decision — removed from the
        // pattern list, so this text should NOT produce a decision capture.
        let captures = analyze(
            "At some point we will use Redis, but not this sprint — need to evaluate first.",
            "notes",
        );
        let decisions: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Decision)
            .collect();
        assert!(decisions.is_empty(), "captured: {:?}", decisions);
    }

    #[test]
    fn accept_explicit_decision_marker() {
        let captures = analyze(
            "Decision: we're keeping the existing redis cluster and sharding manually",
            "notes",
        );
        let decisions: Vec<_> = captures
            .iter()
            .filter(|c| c.capture_type == CaptureType::Decision)
            .collect();
        assert_eq!(decisions.len(), 1);
        // Explicit marker → extra confidence boost.
        assert!(decisions[0].confidence >= 0.75);
    }

    #[test]
    fn error_and_decision_dedup_same_line() {
        // One line contains both an error signature and "decided to": the
        // error claims the line, the decision must find a different line
        // or be dropped.
        let captures = analyze(
            "Saw: error: connection refused, decided to retry after backoff.\nUnrelated context line.",
            "agent",
        );
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].capture_type, CaptureType::Error);
    }

    // ── Guards: a failure of the command line, or text printed for reading ──

    fn errors_in(text: &str) -> Vec<CapturedKnowledge> {
        analyze(text, "bash")
            .into_iter()
            .filter(|c| c.capture_type == CaptureType::Error)
            .collect()
    }

    #[test]
    fn zsh_reading_equals_signs_as_a_command_is_not_an_error() {
        let text = "Exit code 1\nThe import finished and the table is ready for review.\n\
                    (eval):1: === not found";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_shell_kill_is_still_a_failure_line() {
        let guards = TextGuards::of(&[], "");
        assert!(!guards.is_guarded("zsh: killed     cargo test --release -p axil-core"));
        assert!(guards.is_guarded("zsh: no such file or directory: ./target/release/axil"));
    }

    #[test]
    fn an_unparseable_rg_regex_is_not_an_error() {
        let text = "Exit code 2\nrg: regex parse error:\n    (?:fn add(|fn remove)\n    ^\n\
                    error: unclosed group";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_git_refusal_is_not_an_error() {
        let text = "Exit code 128\nfatal: ambiguous argument 'feat/x..dev': unknown revision or \
                    path not in the working tree.";
        assert!(errors_in(text).is_empty());
        let text = "Exit code 1\nerror: pathspec 'crates/axil-core/src/old.rs' did not match \
                    any file(s) known to git";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_usage_error_is_not_an_error() {
        let text = "Exit code 2\nerror: unexpected argument '--fast' found\n\n\
                    Usage: axil recall [OPTIONS] <QUERY>";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_usage_error_does_not_hide_a_failing_test_in_the_same_output() {
        let text = "error: unexpected argument 'hits_with_nothing' found\n\
                    test db::tests::a_failed_batch_rescores_the_rest ... FAILED";
        let errors = errors_in(text);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].summary.starts_with("test db::tests::"),
            "{errors:?}"
        );
    }

    #[test]
    fn search_hits_and_source_lines_are_not_errors() {
        let text = "crates/axil-core/src/query.rs:851:                        eprintln!(\n\
                    crates/axil-core/src/worker.rs:743:      eprintln!(\"maintenance worker error: {e}\");\n\
                    crates/axil-core/src/db.rs:1702:     .map_err(|e| format!(\"open failed: {e}\"))?;";
        assert!(errors_in(text).is_empty());
        let text = "19652:fn format_failed_line(v: &Value) -> String {\n\
                    19653-    // error: the proxy row has no path";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_compiler_message_after_a_location_is_still_an_error() {
        let errors = errors_in("src/main.c:10:5: error: expected ';' before 'return'");
        assert_eq!(errors.len(), 1, "{errors:?}");
        let errors = errors_in("./main.go:12:2: fatal error: undefined reference to foo");
        assert_eq!(errors.len(), 1, "{errors:?}");
    }

    #[test]
    fn diff_lines_are_not_errors() {
        let text = "diff --git a/src/db.rs b/src/db.rs\n@@ -612,6 +690,9 @@ impl Db {\n\
                    -        return Err(\"connection refused by peer\".into())\n\
                    +        warn!(\"retrying after connection refused\")";
        assert!(errors_in(text).is_empty());
    }

    #[test]
    fn a_failing_test_beside_passing_suites_is_an_error() {
        let text = "test result: ok. 9 passed; 0 failed; 0 ignored; finished in 27.87s\n\
                    test db::tests::a_failed_batch_rescores_the_rest ... FAILED\n\
                    test result: FAILED. 41 passed; 1 failed; 0 ignored";
        let errors = errors_in(text);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].summary.starts_with("test db::tests::"),
            "{errors:?}"
        );
        assert!(errors[0].confidence >= 0.8, "{errors:?}");
    }

    #[test]
    fn prose_reporting_a_failure_is_an_error() {
        for text in [
            "**Root cause of the broken vector store:** every open wrote to the file.",
            "The first attempt failed because `axil store` gives up on a held writer lock.",
            "- The vector store in this DB has been dead since June 18, and tools say healthy.",
            "Any parse error in `[lifecycle]` becomes the default config silently.",
            "Large vectors overflow to NaN and the index panics on every later open.",
            "`DistCosine` runs `assert!(dist >= -0.00002)`, which panics on NaN distances.",
            "fix(vector): never let a broken vector store go silent",
            "fix(cli): run the unit tests on init",
        ] {
            let errors = errors_in_prose(text);
            assert_eq!(errors.len(), 1, "{text}: {errors:?}");
            assert!(errors[0].confidence >= 0.8, "{text}");
        }
    }

    #[test]
    fn narration_and_fixes_that_name_no_defect_are_not_errors() {
        for text in [
            "Let me find the root cause before I touch the lock code.",
            "Now checking why the import failed because of the schema change:",
            "The likely root cause is the lock, but I have not confirmed it yet.",
            "fix: address Codex review (Windows replace, struct break)",
            "fix(wasm): clear remaining review issues + cleanup",
            "fix: misc tidy-ups in the init path",
            "REPORT: per item: status (fixed / not-a-bug), root cause in one line, what changed.",
            "**Deadlock: resolved.** It was broken by the lock order; 200 runs are green.",
        ] {
            let errors = errors_in_prose(text);
            assert!(
                errors.iter().all(|c| c.confidence < 0.8),
                "{text}: {errors:?}"
            );
        }
    }

    #[test]
    fn a_failure_report_in_command_output_needs_failure_output() {
        let text = "Exit code 1\nThe first attempt failed because the writer lock was held.";
        assert!(errors_in(text).iter().all(|c| c.confidence < 0.8));
    }

    fn errors_in_prose(text: &str) -> Vec<CapturedKnowledge> {
        analyze(text, "agent")
            .into_iter()
            .filter(|c| c.capture_type == CaptureType::Error)
            .collect()
    }

    #[test]
    fn a_standing_constraint_is_a_rule() {
        for text in [
            "CONVENTIONS: match surrounding style. Public APIs need doc comments.",
            "Tests pass on dev. Never edit the generated fixtures by hand; regenerate them.",
            "- Do not run `cargo fmt` on whole crates; the tree is not fmt-clean.",
            "Every new internal table name must start with an underscore.",
        ] {
            let captures = analyze(text, "agent");
            assert_eq!(captures.len(), 1, "{text}: {captures:?}");
            assert_eq!(captures[0].capture_type, CaptureType::Rule, "{text}");
            let (table, data) = capture_to_record(&captures[0]);
            assert_eq!(table, "rules");
            assert!(data["rule"].is_string());
        }
    }

    #[test]
    fn a_wrapped_never_or_a_rule_in_command_output_is_not_a_rule() {
        let text = "The vector store on this machine has\nnever been repaired since June.";
        assert!(analyze(text, "agent").is_empty());
        let text = "Exit code 1\n- Never edit the generated fixtures by hand; regenerate them.";
        assert!(analyze(text, COMMAND_OUTPUT_SOURCE).is_empty());
    }

    #[test]
    fn a_labeled_finding_is_context() {
        for text in [
            "Gotcha: the hook runs in a fresh process, so the model loads on every call.",
            "- **Lesson learned:** redb repairs the file on the first writable open.",
            "It turns out the CI cache key ignores the lockfile, so bumps reuse stale builds.",
        ] {
            let captures = analyze(text, "agent");
            assert_eq!(captures.len(), 1, "{text}: {captures:?}");
            assert_eq!(captures[0].capture_type, CaptureType::Context, "{text}");
            assert!(captures[0].confidence >= 0.8, "{text}");
        }
    }

    #[test]
    fn a_conditional_or_command_output_finding_is_not_context() {
        let text = "If it turns out the cache is stale, I'll clear it and rerun the suite.";
        assert!(analyze(text, "agent").is_empty());
        let text = "Exit code 1\nGotcha: the hook runs in a fresh process every time it fires";
        assert!(analyze(text, COMMAND_OUTPUT_SOURCE).is_empty());
    }

    #[test]
    fn command_output_yields_failures_only() {
        let text = "Exit code 1\n### Decision: we settled on redb for the core store\n\
                    error: could not compile `axil-core` due to 2 previous errors";
        let from_shell = analyze(text, COMMAND_OUTPUT_SOURCE);
        assert_eq!(from_shell.len(), 1, "{from_shell:?}");
        assert_eq!(from_shell[0].capture_type, CaptureType::Error);
        let from_agent = analyze(text, "agent");
        assert_eq!(from_agent.len(), 2, "{from_agent:?}");
        assert_eq!(from_agent[1].capture_type, CaptureType::Decision);
    }

    #[test]
    fn out_of_memory_is_an_oom_only_when_something_runs_out() {
        assert!(
            errors_in("The new item keeps the code-index vectors out of memory recall.").is_empty()
        );
        let errors = errors_in("- Batch embedding runs out of memory with 8,192-token models.");
        assert_eq!(errors.len(), 1, "{errors:?}");
        let errors = errors_in("memory allocation failed; error: out of memory (os error 12)");
        assert_eq!(errors.len(), 1, "{errors:?}");
    }
}
