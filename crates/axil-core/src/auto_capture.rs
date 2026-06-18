//! Auto-capture — Phase 10.4
//!
//! Analyzes text from agent actions and auto-extracts knowledge:
//! errors, decisions, and context. The agent doesn't need to explicitly store.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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

/// Minimum line length for a capture summary. Below this, a match is
/// almost certainly too short to be a useful knowledge entry.
const MIN_SUMMARY_LEN: usize = 20;

/// Minimum total-input length. Tiny strings get skipped entirely.
const MIN_INPUT_LEN: usize = 30;

/// Analyze text and extract capturable knowledge.
///
/// Returns a list of captures with confidence scores.
/// Callers (e.g. `axil auto-capture`) should only auto-store captures with
/// confidence >= 0.7.
pub fn analyze(text: &str, source: &str) -> Vec<CapturedKnowledge> {
    if text.len() < MIN_INPUT_LEN {
        return Vec::new();
    }
    let text_lower = text.to_lowercase();

    // Global negation: any of these phrases in the whole text kills the
    // capture. Cheaper than per-pattern handling and cuts the dominant
    // false-positive class (successful test output misclassified as error).
    if CAPTURE_NEGATIONS.iter().any(|neg| text_lower.contains(neg)) {
        return Vec::new();
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut captures = Vec::new();
    let mut claimed_line: Option<usize> = None;

    // Detect errors. We take the first matching pattern to avoid multi-capture
    // for the same event — errors tend to produce several patterns in one
    // block (e.g. "panic at" + "traceback"), but they describe one failure.
    for pattern in ERROR_PATTERNS {
        if !text_lower.contains(pattern) {
            continue;
        }
        let Some((idx, line)) = lines
            .iter()
            .enumerate()
            .find(|(_, l)| l.to_lowercase().contains(pattern))
        else {
            continue;
        };
        let _ = idx;
        let summary = line.trim();
        if summary.len() < MIN_SUMMARY_LEN {
            continue;
        }
        if is_pattern_false_positive(pattern, summary) {
            continue;
        }
        captures.push(CapturedKnowledge {
            capture_type: CaptureType::Error,
            summary: truncate(summary, 200),
            confidence: error_confidence(&text_lower, pattern),
            source: source.to_string(),
        });
        claimed_line = Some(idx);
        break;
    }

    // Detect decisions. Skip the line already claimed by the error capture
    // so "decided to abort because of the error" doesn't double-count.
    for pattern in DECISION_PATTERNS {
        if !text_lower.contains(pattern) {
            continue;
        }
        let Some((idx, line)) = lines
            .iter()
            .enumerate()
            .find(|(i, l)| Some(*i) != claimed_line && l.to_lowercase().contains(pattern))
        else {
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

    captures
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
fn error_confidence(text: &str, pattern: &str) -> f32 {
    let mut conf = 0.6f32;

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
    if pattern == "failed" && text.contains("0 failed") {
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
}
