//! Temporal expressions in queries.
//!
//! A query can point at a time ("3 days ago", "March 15"): recall boosts
//! records near it. It can also name a span ("in March", "last 3 days",
//! "between 2026-03-01 and 2026-03-15", "before 2026-04-01"): recall then
//! also keeps only the candidates inside the span, unless none are. Both
//! differ from recency decay, which always prefers newer records: they
//! prefer the time the query is about, even if that time is months ago.

use std::sync::LazyLock;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use regex::Regex;

use crate::record::Record;

/// When what a record describes happened: its `event_time` field (RFC 3339
/// or `YYYY-MM-DD`) when it has a readable one, else when it was stored. A
/// note written today about last month's release says when the release was
/// in `event_time`, so a question about last month finds it.
pub fn record_time(record: &Record) -> DateTime<Utc> {
    record
        .data
        .get("event_time")
        .and_then(|v| v.as_str())
        .and_then(|t| {
            DateTime::parse_from_rfc3339(t)
                .map(|t| t.with_timezone(&Utc))
                .ok()
                .or_else(|| {
                    let date = NaiveDate::parse_from_str(t, "%Y-%m-%d").ok()?;
                    Some(date.and_hms_opt(12, 0, 0)?.and_utc())
                })
        })
        .unwrap_or(record.created_at)
}

/// A span of time a query names, `start` inclusive and `end` exclusive. A
/// side left `None` is unbounded ("before March" has no start).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// First instant inside the span.
    pub start: Option<DateTime<Utc>>,
    /// First instant after the span.
    pub end: Option<DateTime<Utc>>,
}

impl TimeRange {
    /// Whether `time` falls inside the span.
    pub fn contains(&self, time: &DateTime<Utc>) -> bool {
        self.start.is_none_or(|start| *time >= start) && self.end.is_none_or(|end| *time < end)
    }

    /// Days from `time` to the nearest edge of the span, 0 inside it.
    fn days_outside(&self, time: &DateTime<Utc>) -> f64 {
        let secs = match (self.start, self.end) {
            (Some(start), _) if *time < start => (start - *time).num_seconds(),
            (_, Some(end)) if *time >= end => (*time - end).num_seconds(),
            _ => 0,
        };
        secs as f64 / 86400.0
    }
}

/// Result of parsing a temporal expression from a query.
#[derive(Debug, Clone)]
pub struct TemporalTarget {
    /// The target timestamp extracted from the query.
    pub target: DateTime<Utc>,
    /// Window in days around the target (or around `range`) for boosting.
    pub window_days: f64,
    /// The span the query names, if it names one. Recall keeps only the
    /// candidates inside it, unless none are. A point in time ("3 days ago",
    /// "March 15") has none and only boosts.
    pub range: Option<TimeRange>,
}

impl TemporalTarget {
    /// A point in time, boosted within `window_days` of it, with no span.
    pub fn around(target: DateTime<Utc>, window_days: f64) -> Self {
        Self {
            target,
            window_days,
            range: None,
        }
    }

    /// A span, boosted across its whole length and for half its length (at
    /// least three days) outside it. An open side is closed at `now`, or a
    /// year before the other side, to find the middle.
    fn spanning(range: TimeRange, now: &DateTime<Utc>) -> Self {
        let end = range.end.unwrap_or(*now);
        let start = range.start.unwrap_or(end - Duration::days(365));
        let half_days = (end - start).num_seconds() as f64 / 86400.0 / 2.0;
        Self {
            target: start + (end - start) / 2,
            window_days: half_days.max(3.0),
            range: Some(range),
        }
    }
}

/// Compute temporal proximity boost for a record relative to a target.
///
/// `temporal_boost = max(0.0, 0.40 * (1.0 - days_diff / window_days))`, where
/// `days_diff` is the distance to the target, or to the nearest edge of the
/// target's span (0 inside it).
pub fn temporal_boost(record_time: &DateTime<Utc>, target: &TemporalTarget) -> f32 {
    let diff_days = match target.range {
        Some(range) => range.days_outside(record_time),
        None => {
            (record_time.timestamp() - target.target.timestamp()).unsigned_abs() as f64 / 86400.0
        }
    };
    let boost = 0.40 * (1.0 - diff_days / target.window_days);
    boost.max(0.0) as f32
}

/// Month names, in calendar order.
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// A count of units: digits, a number word up to twelve, "a couple (of)" or
/// "a few".
const COUNT: &str = r"\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a\s+)?couple(?:\s+of)?|(?:a\s+)?few";

/// A unit of time, singular.
const UNIT: &str = r"minute|hour|day|week|month|year";

fn months_alternation() -> String {
    MONTHS.join("|")
}

/// A date a span starts or ends at, captured whole: an ISO date, a month
/// with an optional day and year, or a year.
fn date_pattern() -> String {
    format!(
        r"(\d{{4}}-\d{{2}}-\d{{2}}|(?:{m})(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?(?:,?\s+\d{{4}})?|(?:19|20)\d{{2}})",
        m = months_alternation()
    )
}

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("temporal pattern compiles")
}

/// "between <date> and <date>"
static BETWEEN: LazyLock<Regex> = LazyLock::new(|| {
    let d = date_pattern();
    regex(&format!(r"\bbetween\s+{d}\s+and\s+{d}\b"))
});

/// "from <date> to <date>"
static FROM_TO: LazyLock<Regex> = LazyLock::new(|| {
    let d = date_pattern();
    regex(&format!(r"\bfrom\s+{d}\s+(?:to|until|till|through)\s+{d}\b"))
});

/// "before <date>"
static BEFORE: LazyLock<Regex> = LazyLock::new(|| {
    regex(&format!(r"\b(?:before|until|prior\s+to)\s+{}\b", date_pattern()))
});

/// "after <date>", "since <date>"
static AFTER: LazyLock<Regex> =
    LazyLock::new(|| regex(&format!(r"\b(after|since)\s+{}\b", date_pattern())));

/// "in March", "during March 2026", "in 2025"
static IN_PERIOD: LazyLock<Regex> = LazyLock::new(|| {
    regex(&format!(
        r"\b(?:in|during|throughout)\s+((?:{m})(?:,?\s+\d{{4}})?|(?:19|20)\d{{2}})\b",
        m = months_alternation()
    ))
});

/// "March 2026"
static MONTH_YEAR: LazyLock<Regex> =
    LazyLock::new(|| regex(&format!(r"\b((?:{}),?\s+\d{{4}})\b", months_alternation())));

/// "last 3 days", "in the past two weeks", "previous few months"
static LAST_N: LazyLock<Regex> = LazyLock::new(|| {
    regex(&format!(
        r"\b(?:(?:in|over|during|within)\s+)?(?:the\s+)?(?:last|past|previous)\s+({COUNT})\s+({UNIT})s?\b"
    ))
});

/// "in the past month", "over the last week"
static WITHIN_LAST: LazyLock<Regex> = LazyLock::new(|| {
    regex(&format!(
        r"\b(?:in|over|during|within)\s+the\s+(?:last|past|previous)\s+({UNIT})\b"
    ))
});

/// "last hour", "past week": one unit, for [`lookback`].
static LAST_ONE: LazyLock<Regex> =
    LazyLock::new(|| regex(&format!(r"\b(?:last|past)\s+({UNIT})s?\b")));

/// "3 days ago", "two weeks ago", "a couple of days ago"
static AGO: LazyLock<Regex> =
    LazyLock::new(|| regex(&format!(r"\b({COUNT})\s+({UNIT})s?\s+ago\b")));

/// "March 15", "March 15th", "March 15, 2026"
static MONTH_DAY: LazyLock<Regex> = LazyLock::new(|| {
    regex(&format!(
        r"\b(?:{})\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?",
        months_alternation()
    ))
});

/// The number a [`COUNT`] match stands for.
fn count_value(text: &str) -> Option<f64> {
    let text = text.trim();
    if let Ok(n) = text.parse::<u32>() {
        return Some(f64::from(n));
    }
    let words: Vec<&str> = text.split_whitespace().filter(|w| *w != "of").collect();
    let word = match words.as_slice() {
        ["a"] => "one",
        [.., last] => *last,
        [] => return None,
    };
    let n = match word {
        "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        _ => return None,
    };
    Some(f64::from(n))
}

/// Length of one [`UNIT`] in days.
fn unit_days(unit: &str) -> f64 {
    match unit {
        "minute" => 1.0 / 1440.0,
        "hour" => 1.0 / 24.0,
        "day" => 1.0,
        "week" => 7.0,
        "month" => 30.0,
        _ => 365.0,
    }
}

fn days(n: f64) -> Duration {
    Duration::seconds((n * 86400.0).round() as i64)
}

fn midnight(date: NaiveDate) -> Option<DateTime<Utc>> {
    Some(date.and_hms_opt(0, 0, 0)?.and_utc())
}

/// The span a captured date covers, `[start, end)`: its day, month or year.
/// A date without a year is the latest one that has begun by `now`; a year
/// still to come names no span.
fn date_span(text: &str, now: &DateTime<Utc>) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let text = text.trim();
    if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return Some((midnight(date)?, midnight(date.succ_opt()?)?));
    }
    let words: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty())
        .collect();
    let first = *words.first()?;
    let Some(month) = MONTHS.iter().position(|m| *m == first) else {
        let year: i32 = first.parse().ok()?;
        if year > now.year() {
            return None;
        }
        let start = midnight(NaiveDate::from_ymd_opt(year, 1, 1)?)?;
        return Some((start, midnight(NaiveDate::from_ymd_opt(year + 1, 1, 1)?)?));
    };
    let month = month as u32 + 1;
    let mut day = None;
    let mut year = None;
    for word in &words[1..] {
        let digits = word.trim_end_matches(|c: char| c.is_ascii_alphabetic());
        match digits.len() {
            4 => year = digits.parse::<i32>().ok(),
            1 | 2 => day = digits.parse::<u32>().ok(),
            _ => {}
        }
    }
    let span_in = |year: i32| -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        match day {
            Some(day) => {
                let date = NaiveDate::from_ymd_opt(year, month, day)?;
                Some((midnight(date)?, midnight(date.succ_opt()?)?))
            }
            None => {
                let start = NaiveDate::from_ymd_opt(year, month, 1)?;
                let next = if month == 12 {
                    NaiveDate::from_ymd_opt(year + 1, 1, 1)?
                } else {
                    NaiveDate::from_ymd_opt(year, month + 1, 1)?
                };
                Some((midnight(start)?, midnight(next)?))
            }
        }
    };
    match year {
        Some(year) => span_in(year),
        None => {
            let this_year = span_in(now.year())?;
            if this_year.0 > *now {
                span_in(now.year() - 1)
            } else {
                Some(this_year)
            }
        }
    }
}

/// `original` with the bytes of `phrase` cut out and the spaces around the
/// cut collapsed. `phrase` indexes an ASCII-lowercased copy of `original`,
/// whose byte offsets are the same.
fn without(original: &str, phrase: std::ops::Range<usize>) -> String {
    let before = original[..phrase.start].trim();
    let after = original[phrase.end..].trim();
    match (before.is_empty(), after.is_empty()) {
        (true, _) => after.to_string(),
        (_, true) => before.to_string(),
        _ => format!("{before} {after}"),
    }
}

/// A span the query names, with the byte range of the phrase naming it.
fn parse_span(lower: &str, now: &DateTime<Utc>) -> Option<(TimeRange, std::ops::Range<usize>)> {
    if let Some(c) = BETWEEN.captures(lower).or_else(|| FROM_TO.captures(lower)) {
        let first = date_span(&c[1], now)?;
        let second = date_span(&c[2], now)?;
        let (start, end) = if first.0 < second.1 {
            (first.0, second.1)
        } else {
            (second.0, first.1)
        };
        let range = TimeRange {
            start: Some(start),
            end: Some(end),
        };
        return Some((range, c.get(0)?.range()));
    }
    if let Some(c) = BEFORE.captures(lower) {
        let (start, _) = date_span(&c[1], now)?;
        let range = TimeRange {
            start: None,
            end: Some(start),
        };
        return Some((range, c.get(0)?.range()));
    }
    if let Some(c) = AFTER.captures(lower) {
        let (start, end) = date_span(&c[2], now)?;
        let from = if &c[1] == "since" { start } else { end };
        let range = TimeRange {
            start: Some(from),
            end: None,
        };
        return Some((range, c.get(0)?.range()));
    }
    // "in March 15" names a day: leave it to the point forms.
    let in_period = IN_PERIOD.captures(lower).filter(|c| {
        let rest = &lower[c.get(0).map_or(0, |m| m.end())..];
        !rest.trim_start().starts_with(|ch: char| ch.is_ascii_digit())
    });
    if let Some(c) = in_period.or_else(|| MONTH_YEAR.captures(lower)) {
        let (start, end) = date_span(&c[1], now)?;
        let range = TimeRange {
            start: Some(start),
            end: Some(end),
        };
        return Some((range, c.get(0)?.range()));
    }
    let lookback = LAST_N
        .captures(lower)
        .and_then(|c| Some((count_value(&c[1])? * unit_days(&c[2]), c.get(0)?.range())))
        .or_else(|| {
            let c = WITHIN_LAST.captures(lower)?;
            Some((unit_days(&c[1]), c.get(0)?.range()))
        });
    if let Some((span_days, phrase)) = lookback {
        let range = TimeRange {
            start: Some(*now - days(span_days)),
            end: None,
        };
        return Some((range, phrase));
    }
    None
}

/// Try to parse a temporal expression from a query string.
///
/// Returns the parsed target and the query with the temporal expression removed.
/// If no temporal expression is found, returns `None`.
///
/// Spans (the target carries a [`TimeRange`]):
/// - `"between <date> and <date>"`, `"from <date> to <date>"`
/// - `"before <date>"`, `"after <date>"`, `"since <date>"`
/// - `"in March"`, `"during March 2026"`, `"March 2026"`, `"in 2025"`
/// - `"last 3 days"`, `"past two weeks"`, `"in the past month"`
///
/// where a date is `"2026-03-15"`, `"March 15"`, `"March"` or `"2026"`.
/// "after"/"before" followed by anything else ("after the release") name no
/// span.
///
/// Points in time:
/// - `"N days ago"`, `"two weeks ago"`, `"a couple of days ago"` (minutes to years)
/// - `"last week"`, `"last month"`, `"recently"` (7 days)
/// - `"yesterday"`, `"today"`
/// - Absolute dates: `"March 15"`, `"March 15th"`, `"2026-03-15"`
pub fn parse_temporal(query: &str, now: &DateTime<Utc>) -> Option<(TemporalTarget, String)> {
    // ASCII lowercasing keeps byte offsets, so a phrase found here can be
    // cut out of the original query.
    let ascii = query.to_ascii_lowercase();
    if let Some((range, phrase)) = parse_span(&ascii, now) {
        return Some((TemporalTarget::spanning(range, now), without(query, phrase)));
    }

    // "N days/weeks/months ago"
    if let Some(c) = AGO.captures(&ascii) {
        if let Some(n) = count_value(&c[1]) {
            let unit = unit_days(&c[2]);
            let target = *now - days(n * unit);
            let window = (unit * n * 0.5).max(3.0);
            let phrase = c.get(0)?.range();
            return Some((TemporalTarget::around(target, window), without(query, phrase)));
        }
    }

    let lower = query.to_lowercase();

    // "last week", "last month"
    if let Some(target) = parse_last_period(&lower, now) {
        let cleaned = remove_temporal_phrase(query, &lower);
        return Some((target, cleaned));
    }

    // "yesterday", "today", "recently"
    if let Some(target) = parse_simple_relative(&lower, now) {
        let cleaned = remove_temporal_phrase(query, &lower);
        return Some((target, cleaned));
    }

    // ISO date: "2026-03-15"
    if let Some(target) = parse_iso_date(&lower, now) {
        let cleaned = remove_temporal_phrase(query, &lower);
        return Some((target, cleaned));
    }

    // "March 15", "March 15th", "March 15, 2026"
    if let Some(m) = MONTH_DAY.find(&ascii) {
        let (start, _) = date_span(m.as_str(), now)?;
        let target = TemporalTarget::around(start + Duration::hours(12), 3.0);
        return Some((target, without(query, m.range())));
    }

    None
}

/// How far back a query looks, for callers that keep only records newer
/// than a cutoff: `"last 3 days"`, `"past two weeks"`, `"last hour"`,
/// `"since yesterday"` and `"today"` (a day), `"recent"` (a week). `None`
/// when the query names no lookback.
pub fn lookback(query: &str) -> Option<Duration> {
    let lower = query.to_ascii_lowercase();
    if let Some(c) = LAST_N.captures(&lower) {
        return Some(days(count_value(&c[1])? * unit_days(&c[2])));
    }
    if let Some(c) = LAST_ONE.captures(&lower) {
        return Some(days(unit_days(&c[1])));
    }
    if lower.contains("yesterday") || lower.contains("today") {
        return Some(Duration::days(1));
    }
    if lower.contains("recent") {
        return Some(Duration::days(7));
    }
    None
}

fn parse_last_period(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    if lower.contains("last week") {
        let target = *now - chrono::Duration::days(7);
        return Some(TemporalTarget::around(target, 7.0));
    }
    if lower.contains("last month") {
        let target = *now - chrono::Duration::days(30);
        return Some(TemporalTarget::around(target, 15.0));
    }
    None
}

fn parse_simple_relative(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    if lower.contains("yesterday") {
        let target = *now - chrono::Duration::days(1);
        return Some(TemporalTarget::around(target, 1.5));
    }
    if lower.contains("today") {
        return Some(TemporalTarget::around(*now, 1.0));
    }
    if lower.contains("recently") {
        let target = *now - chrono::Duration::days(3);
        return Some(TemporalTarget::around(target, 7.0));
    }
    None
}

fn parse_iso_date(lower: &str, _now: &DateTime<Utc>) -> Option<TemporalTarget> {
    // Find yyyy-mm-dd pattern
    for word in lower.split_whitespace() {
        if let Ok(date) = NaiveDate::parse_from_str(word, "%Y-%m-%d") {
            let dt = date.and_hms_opt(12, 0, 0)?.and_utc();
            return Some(TemporalTarget::around(dt, 3.0));
        }
    }
    None
}

/// Remove the temporal phrase from the query, returning the cleaned query.
fn remove_temporal_phrase(original: &str, lower: &str) -> String {
    let patterns = ["last week", "last month", "yesterday", "today", "recently"];

    for pat in &patterns {
        if let Some(pos) = lower.find(pat) {
            let end = pos + pat.len();
            let mut result = String::new();
            result.push_str(original[..pos].trim());
            if !result.is_empty() && end < original.len() {
                result.push(' ');
            }
            result.push_str(original[end..].trim());
            return result.trim().to_string();
        }
    }

    // ISO dates
    let words: Vec<&str> = original.split_whitespace().collect();
    let lower_words: Vec<&str> = lower.split_whitespace().collect();
    words
        .iter()
        .zip(lower_words.iter())
        .filter(|(_, lw)| NaiveDate::parse_from_str(lw, "%Y-%m-%d").is_err())
        .map(|(w, _)| *w)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap()
    }

    fn day(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    /// The span `query` names, and the query left once it is cut out.
    fn span(query: &str) -> (TimeRange, String) {
        let (target, cleaned) = parse_temporal(query, &fixed_now()).expect(query);
        (target.range.expect(query), cleaned)
    }

    #[test]
    fn parse_days_ago() {
        let now = fixed_now();
        let (target, cleaned) = parse_temporal("auth error 3 days ago", &now).unwrap();
        assert!(
            (target.target - (now - chrono::Duration::days(3)))
                .num_seconds()
                .abs()
                < 2
        );
        assert_eq!(cleaned, "auth error");
        assert!(target.range.is_none());
    }

    #[test]
    fn parse_weeks_ago() {
        let now = fixed_now();
        let (target, cleaned) = parse_temporal("what happened 2 weeks ago", &now).unwrap();
        assert!(
            (target.target - (now - chrono::Duration::days(14)))
                .num_seconds()
                .abs()
                < 2
        );
        assert_eq!(cleaned, "what happened");
    }

    #[test]
    fn counts_in_words_ago() {
        let now = fixed_now();
        for (query, days_back, cleaned) in [
            ("I cooked something a couple of days ago", 2, "I cooked something"),
            ("the deploy two weeks ago failed", 14, "the deploy failed"),
            ("a week ago we changed it", 7, "we changed it"),
            ("a few days ago", 3, ""),
        ] {
            let (target, rest) = parse_temporal(query, &now).expect(query);
            let expected = now - chrono::Duration::days(days_back);
            assert!((target.target - expected).num_seconds().abs() < 2, "{query}");
            assert_eq!(rest, cleaned, "{query}");
        }
        // "how many days ago" counts nothing.
        assert!(parse_temporal("how many days ago did I read it", &now).is_none());
    }

    #[test]
    fn parse_last_week() {
        let now = fixed_now();
        let (target, _) = parse_temporal("deployments last week", &now).unwrap();
        assert!(
            (target.target - (now - chrono::Duration::days(7)))
                .num_seconds()
                .abs()
                < 2
        );
        assert_eq!(target.window_days, 7.0);
        assert!(target.range.is_none());
    }

    #[test]
    fn parse_yesterday() {
        let now = fixed_now();
        let (target, cleaned) = parse_temporal("yesterday auth fix", &now).unwrap();
        assert!(
            (target.target - (now - chrono::Duration::days(1)))
                .num_seconds()
                .abs()
                < 2
        );
        assert_eq!(cleaned, "auth fix");
    }

    #[test]
    fn parse_iso_date() {
        let now = fixed_now();
        let (target, cleaned) = parse_temporal("bugs from 2026-03-15", &now).unwrap();
        assert_eq!(
            target.target.date_naive(),
            NaiveDate::from_ymd_opt(2026, 3, 15).unwrap()
        );
        assert_eq!(cleaned, "bugs from");
    }

    #[test]
    fn parse_month_day() {
        let now = fixed_now();
        let (target, cleaned) = parse_temporal("deployment on March 15", &now).unwrap();
        assert_eq!(
            target.target.date_naive(),
            NaiveDate::from_ymd_opt(2026, 3, 15).unwrap()
        );
        assert_eq!(cleaned, "deployment on");
        assert!(target.range.is_none());

        let (target, cleaned) = parse_temporal("the March 15th issue", &now).unwrap();
        assert_eq!(target.target.date_naive(), day(2026, 3, 15).date_naive());
        assert_eq!(cleaned, "the issue");
        let (target, _) = parse_temporal("released on March 15, 2025", &now).unwrap();
        assert_eq!(target.target.date_naive(), day(2025, 3, 15).date_naive());
        // A day still to come this year is last year's.
        let (target, _) = parse_temporal("the May 2 outage", &now).unwrap();
        assert_eq!(target.target.date_naive(), day(2025, 5, 2).date_naive());
    }

    #[test]
    fn a_month_is_a_span() {
        let (range, cleaned) = span("what did we decide in March about caching");
        assert_eq!(range.start, Some(day(2026, 3, 1)));
        assert_eq!(range.end, Some(day(2026, 4, 1)));
        assert_eq!(cleaned, "what did we decide about caching");

        // The latest May that has begun is last year's.
        let (range, _) = span("during May");
        assert_eq!(range.start, Some(day(2025, 5, 1)));
        assert_eq!(range.end, Some(day(2025, 6, 1)));

        let (range, cleaned) = span("the March 2025 outage");
        assert_eq!(range.start, Some(day(2025, 3, 1)));
        assert_eq!(range.end, Some(day(2025, 4, 1)));
        assert_eq!(cleaned, "the outage");

        let (range, _) = span("December 2025 release notes");
        assert_eq!(range.end, Some(day(2026, 1, 1)));
    }

    #[test]
    fn a_year_is_a_span_unless_it_is_still_to_come() {
        let (range, cleaned) = span("what broke in 2025");
        assert_eq!(range.start, Some(day(2025, 1, 1)));
        assert_eq!(range.end, Some(day(2026, 1, 1)));
        assert_eq!(cleaned, "what broke");
        assert!(parse_temporal("buffers in 2048 bytes", &fixed_now()).is_none());
    }

    #[test]
    fn between_and_from_to_are_spans() {
        let (range, cleaned) = span("errors between 2026-03-01 and 2026-03-15");
        assert_eq!(range.start, Some(day(2026, 3, 1)));
        assert_eq!(range.end, Some(day(2026, 3, 16)));
        assert_eq!(cleaned, "errors");

        let (range, cleaned) = span("from March 3 to March 9 what changed");
        assert_eq!(range.start, Some(day(2026, 3, 3)));
        assert_eq!(range.end, Some(day(2026, 3, 10)));
        assert_eq!(cleaned, "what changed");

        // Dates given in the wrong order still name the span between them.
        let (range, _) = span("between March 9 and March 3");
        assert_eq!(range.start, Some(day(2026, 3, 3)));
        assert_eq!(range.end, Some(day(2026, 3, 10)));
    }

    #[test]
    fn before_after_and_since_are_open_spans() {
        let (range, cleaned) = span("decisions before 2026-03-01");
        assert_eq!(range, TimeRange { start: None, end: Some(day(2026, 3, 1)) });
        assert_eq!(cleaned, "decisions");

        let (range, _) = span("what changed after March");
        assert_eq!(range, TimeRange { start: Some(day(2026, 4, 1)), end: None });
        let (range, _) = span("what changed since March");
        assert_eq!(range, TimeRange { start: Some(day(2026, 3, 1)), end: None });
        let (range, _) = span("errors since 2026-04-01");
        assert_eq!(range.start, Some(day(2026, 4, 1)));
    }

    #[test]
    fn lookbacks_are_spans_ending_now() {
        let now = fixed_now();
        for (query, days_back, cleaned) in [
            ("errors in the last 3 days", 3, "errors"),
            ("past two weeks of commits", 14, "of commits"),
            ("grocery spend in the past month", 30, "grocery spend"),
            ("over the last week what broke", 7, "what broke"),
        ] {
            let (range, rest) = span(query);
            assert_eq!(range.end, None, "{query}");
            let start = range.start.expect(query);
            assert!((start - (now - Duration::days(days_back))).num_seconds().abs() < 2, "{query}");
            assert_eq!(rest, cleaned, "{query}");
        }
    }

    #[test]
    fn before_and_after_an_event_name_no_span() {
        let now = fixed_now();
        for query in [
            "How are compiled WASM plugin modules cached between runs?",
            "After comparing with CodeGraph, did we decide to build a code explore tool?",
            "What should I do before grepping the repository for context?",
            "How do release archives get built after release-plz publishes?",
            "How much faster did I finish compared to my previous year's time?",
            "How many months before my anniversary did Rachel get engaged?",
            "How many postcards have I added since I started collecting again?",
        ] {
            let range = parse_temporal(query, &now).and_then(|(t, _)| t.range);
            assert!(range.is_none(), "{query}: {range:?}");
        }
    }

    #[test]
    fn lookback_reads_the_window_back_from_now() {
        let d = |n: i64| Some(Duration::days(n));
        assert_eq!(lookback("last 3 days"), d(3));
        assert_eq!(lookback("show me the last 7 days of activity"), d(7));
        assert_eq!(lookback("last 2 hours"), Some(Duration::hours(2)));
        assert_eq!(lookback("past 1 hour"), Some(Duration::hours(1)));
        assert_eq!(lookback("last two weeks"), d(14));
        assert_eq!(lookback("last hour"), Some(Duration::hours(1)));
        assert_eq!(lookback("last day"), d(1));
        assert_eq!(lookback("past week"), d(7));
        assert_eq!(lookback("since yesterday"), d(1));
        assert_eq!(lookback("anything new today"), d(1));
        assert_eq!(lookback("what changed recently"), d(7));
        assert_eq!(lookback("last 30 minutes"), Some(Duration::minutes(30)));
        assert_eq!(lookback("last month"), d(30));
        assert_eq!(lookback("how does auth work"), None);
    }

    #[test]
    fn no_temporal_returns_none() {
        let now = fixed_now();
        assert!(parse_temporal("auth error fix", &now).is_none());
    }

    #[test]
    fn temporal_boost_at_target() {
        let now = fixed_now();
        let target = TemporalTarget::around(now, 7.0);
        let boost = temporal_boost(&now, &target);
        assert!((boost - 0.40).abs() < 0.01);
    }

    #[test]
    fn temporal_boost_outside_window() {
        let now = fixed_now();
        let target = TemporalTarget::around(now, 7.0);
        let old = now - chrono::Duration::days(10);
        let boost = temporal_boost(&old, &target);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn temporal_boost_half_window() {
        let now = fixed_now();
        let target = TemporalTarget::around(now, 10.0);
        let half = now - chrono::Duration::days(5);
        let boost = temporal_boost(&half, &target);
        assert!((boost - 0.20).abs() < 0.02);
    }

    #[test]
    fn record_time_reads_event_time_before_the_store_time() {
        let mut record = Record::new("context", serde_json::json!({"summary": "x"}));
        record.created_at = fixed_now();
        assert_eq!(record_time(&record), fixed_now());
        record.data["event_time"] = serde_json::json!("2026-03-12");
        assert_eq!(record_time(&record), day(2026, 3, 12) + Duration::hours(12));
        record.data["event_time"] = serde_json::json!("2026-03-12T08:30:00+02:00");
        assert_eq!(
            record_time(&record),
            Utc.with_ymd_and_hms(2026, 3, 12, 6, 30, 0).unwrap()
        );
        record.data["event_time"] = serde_json::json!("sometime in March");
        assert_eq!(record_time(&record), fixed_now(), "unreadable: the store time");
    }

    #[test]
    fn a_span_boosts_all_of_itself_and_fades_outside() {
        let (target, _) = parse_temporal("in March", &fixed_now()).unwrap();
        assert!((temporal_boost(&day(2026, 3, 2), &target) - 0.40).abs() < 1e-6);
        assert!((temporal_boost(&day(2026, 3, 31), &target) - 0.40).abs() < 1e-6);
        let after = temporal_boost(&day(2026, 4, 5), &target);
        assert!(after > 0.0 && after < 0.40, "{after}");
        assert_eq!(temporal_boost(&day(2026, 1, 1), &target), 0.0);
    }
}
