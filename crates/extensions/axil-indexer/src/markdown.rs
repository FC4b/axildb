//! Markdown heading splitter for structural recall.
//!
//! Walks a markdown source by ATX headings (`#`, `##`, ...) and produces
//! one `ParsedSection` per heading. Each section carries:
//! - the full heading path (e.g. `[" ", "Data Model", "Identity"]`),
//! - line start/end for navigation,
//! - the heading text as the section "symbol",
//! - the section body trimmed and squeezed for proxy text use.
//!
//! Files with no ATX headings produce a single synthetic top-level section
//! covering the whole file so downstream code always sees at least one
//! section per markdown source.

/// A markdown section ready to be promoted to a `CodeProxy` of kind
/// `Section`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSection {
    /// Heading depth (1-based, matches `#` count). `0` for the synthetic
    /// whole-file section produced when no headings exist.
    pub depth: usize,
    /// Full heading path from the document root to this heading.
    pub heading_path: Vec<String>,
    /// Heading text — the leaf of `heading_path`, or "(document)" for the
    /// synthetic whole-file section.
    pub heading: String,
    /// 1-based line where the heading appears.
    pub line_start: usize,
    /// 1-based line of the last line in this section (inclusive).
    pub line_end: usize,
    /// Section body (text below the heading until the next sibling/parent
    /// heading), with leading/trailing whitespace trimmed.
    pub body: String,
}

/// Split markdown source into structural sections by ATX headings.
///
/// Behavior:
/// - ATX-only (setext `===` / `---` underlines are not split, by design —
///   keeps the splitter cheap and deterministic).
/// - Code fences are respected: heading-looking lines inside ``` blocks are
///   ignored.
/// - The path of each section is the chain of ancestor headings whose depth
///   is strictly less than the current heading's depth.
pub fn split_sections(source: &str) -> Vec<ParsedSection> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // First pass: find all heading positions.
    //
    // Fence handling: CommonMark says a closing fence must use the same
    // character as the opener and at least as many of them. Treating any
    // triple-backtick line as a toggle (the previous behavior) trips on
    // nested code blocks inside doc examples. We track the open count
    // and character so a `\`\`\`` inside a `\`\`\`\`-fenced block stays
    // inside the block.
    let mut headings: Vec<(usize, usize, String)> = Vec::new(); // (line_idx, depth, text)
    let mut fence_state: Option<(char, usize)> = None; // (fence_char, open_count)
    for (idx, raw) in lines.iter().enumerate() {
        let trimmed = raw.trim_start();
        if let Some((ch, count, clean)) = fence_run(trimmed) {
            match fence_state {
                None => {
                    // Opening fence — info strings are allowed here.
                    fence_state = Some((ch, count));
                    continue;
                }
                Some((open_ch, open_count)) => {
                    // Closing fence — CommonMark requires same char,
                    // count >= opener, AND no info string. A line like
                    // `\`\`\`python` inside a `\`\`\`` block is NOT a
                    // closer; it's just text inside the block.
                    if clean && ch == open_ch && count >= open_count {
                        fence_state = None;
                    }
                    continue;
                }
            }
        }
        if fence_state.is_some() {
            continue;
        }
        if let Some((depth, text)) = parse_atx_heading(trimmed) {
            headings.push((idx, depth, text));
        }
    }

    if headings.is_empty() {
        // No headings — emit one synthetic whole-file section.
        let body = source.trim().to_string();
        return vec![ParsedSection {
            depth: 0,
            heading_path: Vec::new(),
            heading: "(document)".into(),
            line_start: 1,
            line_end: lines.len().max(1),
            body,
        }];
    }

    // Second pass: build sections, tracking the active heading-path stack.
    let mut sections = Vec::with_capacity(headings.len());
    let mut stack: Vec<(usize, String)> = Vec::new(); // (depth, heading text)
    for (i, (line_idx, depth, text)) in headings.iter().enumerate() {
        // Pop any deeper-or-equal-depth headings off the stack.
        while let Some(&(top_depth, _)) = stack.last() {
            if top_depth >= *depth {
                stack.pop();
            } else {
                break;
            }
        }
        let heading_path: Vec<String> = stack
            .iter()
            .map(|(_, h)| h.clone())
            .chain(std::iter::once(text.clone()))
            .collect();

        let body_start = line_idx + 1;
        let body_end_exclusive = if i + 1 < headings.len() {
            headings[i + 1].0
        } else {
            lines.len()
        };
        let body_lines = if body_start < body_end_exclusive {
            &lines[body_start..body_end_exclusive]
        } else {
            &[][..]
        };
        let body = body_lines
            .iter()
            .copied()
            .collect::<Vec<&str>>()
            .join("\n")
            .trim()
            .to_string();

        sections.push(ParsedSection {
            depth: *depth,
            heading_path,
            heading: text.clone(),
            line_start: line_idx + 1,
            line_end: body_end_exclusive.max(line_idx + 1),
            body,
        });

        stack.push((*depth, text.clone()));
    }

    sections
}

/// If `line` (already left-trimmed) is a fence — a run of `\`` or `~`
/// of length ≥3 — returns `(fence_char, count, has_only_fence_then_ws)`.
/// `has_only_fence_then_ws` is `true` when the line is just the fence
/// run plus optional whitespace. Callers use it to enforce CommonMark's
/// "closing fence may not have an info string" rule. Returns `None`
/// for non-fence lines.
fn fence_run(line: &str) -> Option<(char, usize, bool)> {
    let mut chars = line.chars();
    let first = chars.next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let mut count = 1usize;
    for ch in line.chars().skip(1) {
        if ch == first {
            count += 1;
        } else {
            break;
        }
    }
    if count < 3 {
        return None;
    }
    let rest = &line[count..];
    let clean = rest.chars().all(|c| c.is_whitespace());
    Some((first, count, clean))
}

fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let mut chars = line.chars();
    let mut depth = 0usize;
    while let Some('#') = chars.clone().next() {
        chars.next();
        depth += 1;
        if depth > 6 {
            return None;
        }
    }
    if depth == 0 {
        return None;
    }
    let rest = chars.as_str();
    if !rest.starts_with(' ') && !rest.starts_with('\t') && !rest.is_empty() {
        // `#word` is not a heading per CommonMark. Be conservative.
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some((depth, text))
}

/// SCIP-style canonical id for a markdown section. Uses the doc path plus
/// the joined heading path so two sections with the same heading text in
/// different files do not collide.
pub fn section_canonical_id(path: &str, heading_path: &[String]) -> String {
    if heading_path.is_empty() {
        return format!("{path}#(document)");
    }
    format!("{path}#{}", heading_path.join(">"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_headings() {
        let src = "# Top\nintro\n\n## Sub A\nbody A\n\n## Sub B\nbody B\n";
        let sections = split_sections(src);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].heading, "Top");
        assert_eq!(sections[0].heading_path, vec!["Top".to_string()]);
        assert_eq!(
            sections[1].heading_path,
            vec!["Top".to_string(), "Sub A".to_string()]
        );
        assert_eq!(
            sections[2].heading_path,
            vec!["Top".to_string(), "Sub B".to_string()]
        );
    }

    #[test]
    fn fence_with_info_string_on_inner_line_does_not_close_block() {
        // CommonMark: a closing fence cannot carry an info string. A
        // line like ```python inside an opened ``` block is just code,
        // not a close.
        let src = "# Top\n```\n```python\n# pretend heading\nprint('x')\n```\n## After\nbody\n";
        let sections = split_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(names, vec!["Top", "After"]);
    }

    #[test]
    fn nested_triple_backtick_inside_quad_fence_stays_inside_block() {
        // Outer fence is 4 backticks; an inner triple-backtick line must
        // not pop the fence state. A `# Real` line *after* the close
        // should still register as a heading.
        let src = "# Top\n````md\n```\n# not a heading\n```\n````\n## Real\nbody\n";
        let sections = split_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(names, vec!["Top", "Real"]);
    }

    #[test]
    fn ignores_headings_inside_code_fence() {
        let src = "# Top\n```\n# not a heading\n```\n## Real\nbody\n";
        let sections = split_sections(src);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, "Top");
        assert_eq!(sections[1].heading, "Real");
    }

    #[test]
    fn no_headings_emits_synthetic_section() {
        let src = "just some text\nand more text\n";
        let sections = split_sections(src);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].depth, 0);
        assert_eq!(sections[0].heading, "(document)");
        assert!(sections[0].body.contains("just some text"));
    }

    #[test]
    fn line_numbers_are_one_based() {
        let src = "intro\n# H\nbody\n";
        let sections = split_sections(src);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].line_start, 2);
    }

    #[test]
    fn handles_skipped_levels() {
        // `### Deep` with no ## parent — depth-2 stack is empty, so the
        // path is `Top > Deep`.
        let src = "# Top\n### Deep\nbody\n";
        let sections = split_sections(src);
        assert_eq!(
            sections[1].heading_path,
            vec!["Top".to_string(), "Deep".to_string()]
        );
    }

    #[test]
    fn canonical_id_includes_heading_path() {
        let id = section_canonical_id(
            "tasks/phase-13b.md",
            &["Data Model".into(), "Identity".into()],
        );
        assert_eq!(id, "tasks/phase-13b.md#Data Model>Identity");
    }
}
