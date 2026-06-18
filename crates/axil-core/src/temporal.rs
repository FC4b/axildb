//! Temporal date parsing for query-time boost.
//!
//! Parses relative time expressions in queries and computes a proximity
//! boost for records near the target date. This differs from recency decay
//! (which always prefers newer) — temporal proximity boosts records *near
//! the mentioned time*, even if that time is months ago.

use chrono::{DateTime, Datelike, NaiveDate, Utc};

/// Result of parsing a temporal expression from a query.
#[derive(Debug, Clone)]
pub struct TemporalTarget {
    /// The target timestamp extracted from the query.
    pub target: DateTime<Utc>,
    /// Window in days around the target for boosting.
    pub window_days: f64,
}

/// Compute temporal proximity boost for a record relative to a target.
///
/// `temporal_boost = max(0.0, 0.40 * (1.0 - days_diff / window_days))`
pub fn temporal_boost(record_time: &DateTime<Utc>, target: &TemporalTarget) -> f32 {
    let diff_secs = (record_time.timestamp() - target.target.timestamp()).unsigned_abs() as f64;
    let diff_days = diff_secs / 86400.0;
    let boost = 0.40 * (1.0 - diff_days / target.window_days);
    boost.max(0.0) as f32
}

/// Try to parse a temporal expression from a query string.
///
/// Returns the parsed target and the query with the temporal expression removed.
/// If no temporal expression is found, returns `None`.
///
/// Supported patterns:
/// - `"N days ago"`, `"N weeks ago"`, `"N months ago"`
/// - `"last week"`, `"last month"`, `"recently"` (7 days)
/// - `"yesterday"`, `"today"`
/// - Absolute dates: `"March 15"`, `"2026-03-15"`
pub fn parse_temporal(query: &str, now: &DateTime<Utc>) -> Option<(TemporalTarget, String)> {
    let lower = query.to_lowercase();

    // "N days/weeks/months ago"
    if let Some(target) = parse_n_units_ago(&lower, now) {
        let cleaned = remove_temporal_phrase(query, &lower);
        return Some((target, cleaned));
    }

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

    // "March 15" or "march 15"
    if let Some(target) = parse_month_day(&lower, now) {
        let cleaned = remove_temporal_phrase(query, &lower);
        return Some((target, cleaned));
    }

    None
}

fn parse_n_units_ago(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    // Match patterns like "3 days ago", "2 weeks ago", "1 month ago"
    let re_patterns = [
        ("days ago", 1.0),
        ("day ago", 1.0),
        ("weeks ago", 7.0),
        ("week ago", 7.0),
        ("months ago", 30.0),
        ("month ago", 30.0),
    ];

    for (suffix, unit_days) in &re_patterns {
        if let Some(pos) = lower.find(suffix) {
            // Look for number before the suffix
            let prefix = lower[..pos].trim();
            if let Some(n) = parse_trailing_number(prefix) {
                let days_back = n as f64 * unit_days;
                let target = *now - chrono::Duration::seconds((days_back * 86400.0) as i64);
                let window = (unit_days * n as f64 * 0.5).max(3.0);
                return Some(TemporalTarget {
                    target,
                    window_days: window,
                });
            }
        }
    }
    None
}

fn parse_trailing_number(s: &str) -> Option<u32> {
    // Find the last word and try to parse it as a number
    let trimmed = s.trim();
    let last_word = trimmed.rsplit_once(' ').map(|(_, w)| w).unwrap_or(trimmed);
    last_word.parse::<u32>().ok()
}

fn parse_last_period(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    if lower.contains("last week") {
        let target = *now - chrono::Duration::days(7);
        return Some(TemporalTarget {
            target,
            window_days: 7.0,
        });
    }
    if lower.contains("last month") {
        let target = *now - chrono::Duration::days(30);
        return Some(TemporalTarget {
            target,
            window_days: 15.0,
        });
    }
    None
}

fn parse_simple_relative(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    if lower.contains("yesterday") {
        let target = *now - chrono::Duration::days(1);
        return Some(TemporalTarget {
            target,
            window_days: 1.5,
        });
    }
    if lower.contains("today") {
        return Some(TemporalTarget {
            target: *now,
            window_days: 1.0,
        });
    }
    if lower.contains("recently") {
        let target = *now - chrono::Duration::days(3);
        return Some(TemporalTarget {
            target,
            window_days: 7.0,
        });
    }
    None
}

fn parse_iso_date(lower: &str, _now: &DateTime<Utc>) -> Option<TemporalTarget> {
    // Find yyyy-mm-dd pattern
    for word in lower.split_whitespace() {
        if let Ok(date) = NaiveDate::parse_from_str(word, "%Y-%m-%d") {
            let dt = date.and_hms_opt(12, 0, 0)?.and_utc();
            return Some(TemporalTarget {
                target: dt,
                window_days: 3.0,
            });
        }
    }
    None
}

fn parse_month_day(lower: &str, now: &DateTime<Utc>) -> Option<TemporalTarget> {
    let months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];

    for (name, month) in &months {
        if let Some(pos) = lower.find(name) {
            let after = lower[pos + name.len()..].trim();
            if let Some(day) = after
                .split_whitespace()
                .next()
                .and_then(|w| w.trim_end_matches(',').parse::<u32>().ok())
            {
                // Try current year first; if that date is in the future, use previous year
                let year = now.year();
                if let Some(date) = NaiveDate::from_ymd_opt(year, *month, day) {
                    let dt = date.and_hms_opt(12, 0, 0)?.and_utc();
                    let target_dt = if dt > *now {
                        // Date is in the future — assume previous year
                        NaiveDate::from_ymd_opt(year - 1, *month, day)?
                            .and_hms_opt(12, 0, 0)?
                            .and_utc()
                    } else {
                        dt
                    };
                    return Some(TemporalTarget {
                        target: target_dt,
                        window_days: 3.0,
                    });
                }
            }
        }
    }
    None
}

/// Remove the temporal phrase from the query, returning the cleaned query.
fn remove_temporal_phrase(original: &str, lower: &str) -> String {
    let patterns = [
        // N units ago
        "days ago",
        "day ago",
        "weeks ago",
        "week ago",
        "months ago",
        "month ago",
        // Relative
        "last week",
        "last month",
        "yesterday",
        "today",
        "recently",
    ];

    for pat in &patterns {
        if let Some(pos) = lower.find(pat) {
            // For "N units ago", also remove the preceding number
            let start = if pat.ends_with("ago") {
                let prefix = lower[..pos].trim();
                let num_start = prefix.rfind(' ').map(|p| p + 1).unwrap_or(0);
                // Check if it's a number
                if prefix[num_start..].trim().parse::<u32>().is_ok() {
                    num_start
                } else {
                    pos
                }
            } else {
                pos
            };
            let end = pos + pat.len();
            let mut result = String::new();
            result.push_str(original[..start].trim());
            if !result.is_empty() && end < original.len() {
                result.push(' ');
            }
            result.push_str(original[end..].trim());
            return result.trim().to_string();
        }
    }

    // ISO date or month day — find and remove
    let words: Vec<&str> = original.split_whitespace().collect();
    let lower_words: Vec<&str> = lower.split_whitespace().collect();

    // Remove ISO dates
    let filtered: Vec<&str> = words
        .iter()
        .zip(lower_words.iter())
        .filter(|(_, lw)| NaiveDate::parse_from_str(lw, "%Y-%m-%d").is_err())
        .map(|(w, _)| *w)
        .collect();
    if filtered.len() != words.len() {
        return filtered.join(" ");
    }

    // Remove "Month Day" patterns
    let months = [
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
    let mut skip_next = false;
    let mut result = Vec::new();
    for (w, lw) in words.iter().zip(lower_words.iter()) {
        if skip_next {
            // Check if this is a day number
            if lw.trim_end_matches(',').parse::<u32>().is_ok() {
                skip_next = false;
                continue;
            }
            skip_next = false;
        }
        if months.iter().any(|m| lw.starts_with(m)) {
            skip_next = true;
            continue;
        }
        result.push(*w);
    }
    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap()
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
    }

    #[test]
    fn no_temporal_returns_none() {
        let now = fixed_now();
        assert!(parse_temporal("auth error fix", &now).is_none());
    }

    #[test]
    fn temporal_boost_at_target() {
        let now = fixed_now();
        let target = TemporalTarget {
            target: now,
            window_days: 7.0,
        };
        let boost = temporal_boost(&now, &target);
        assert!((boost - 0.40).abs() < 0.01);
    }

    #[test]
    fn temporal_boost_outside_window() {
        let now = fixed_now();
        let target = TemporalTarget {
            target: now,
            window_days: 7.0,
        };
        let old = now - chrono::Duration::days(10);
        let boost = temporal_boost(&old, &target);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn temporal_boost_half_window() {
        let now = fixed_now();
        let target = TemporalTarget {
            target: now,
            window_days: 10.0,
        };
        let half = now - chrono::Duration::days(5);
        let boost = temporal_boost(&half, &target);
        assert!((boost - 0.20).abs() < 0.02);
    }
}
