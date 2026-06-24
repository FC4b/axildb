//! Multi-signal score fusion for intelligent recall.
//!
//! Combines vector similarity, recency decay, graph proximity, keyword overlap,
//! temporal proximity, preference matching, and relevance feedback into a single
//! ranked score. All signals are computed without an LLM.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::record::Record;
use crate::temporal::TemporalTarget;

/// Weights for each scoring signal in the fusion formula.
///
/// ```text
/// final_score = w_vector * vector_sim
///             + w_recency * recency_decay(age)
///             + w_graph * graph_proximity
///             + w_keyword * bm25_score
///             + w_feedback * feedback_boost
///             + w_temporal * temporal_proximity
///             + w_preference * preference_match
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub vector: f32,
    pub recency: f32,
    pub graph: f32,
    pub keyword: f32,
    pub feedback: f32,
    pub temporal: f32,
    pub preference: f32,
    /// Weight for activation-level scoring.
    #[serde(default = "default_activation_weight")]
    pub activation: f32,
    /// Weight for importance scoring.
    #[serde(default = "default_importance_weight")]
    pub importance: f32,
    /// Weight for reciprocal rank fusion (combines vector + FTS rank agreement).
    #[serde(default = "default_rrf_weight")]
    pub rrf: f32,
}

fn default_activation_weight() -> f32 {
    0.0
}

fn default_importance_weight() -> f32 {
    0.0
}

fn default_rrf_weight() -> f32 {
    0.0
}

impl Default for ScoreWeights {
    fn default() -> Self {
        // RRF weight intentionally 0 at default — RRF values live on
        // 1/(60+rank) scale, which is too small to contribute meaningfully
        // alongside similarity/recency (both on [0,1]) unless rescaled.
        // Adaptive weight renormalization is the primary ranking improvement;
        // RRF stays plumbed for future tuning but off by default.
        Self {
            vector: 0.40,
            recency: 0.15,
            graph: 0.15,
            keyword: 0.10,
            feedback: 0.05,
            temporal: 0.10,
            preference: 0.05,
            activation: 0.0,
            importance: 0.0,
            rrf: 0.0,
        }
    }
}

/// Breakdown of individual scoring signals for a single result.
///
/// `#[non_exhaustive]`: construct it with [`ScoreExplanation::new`] rather than a
/// struct literal so out-of-crate callers stay source-compatible as the struct
/// gains fields (e.g. `query_class` was added without a major bump).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScoreExplanation {
    /// Individual signal scores that were combined: `(signal_name, raw_value)`.
    pub signals: Vec<(String, f32)>,
    /// Human-readable summary of why this result ranked where it did.
    pub summary: String,
    /// How the query was classified during recall, e.g. `"identifier:uuid"` or
    /// `"natural-language"`. `Some` only on the multi-signal recall path; the
    /// identifier classes mean an FTS rank tilt was applied to exact-identifier
    /// lookups (visible as the `fts_identifier_tilt` signal). Skipped from
    /// serialization when absent so existing JSON output is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_class: Option<String>,
}

impl ScoreExplanation {
    /// Build an explanation from the combined `signals` + `summary`.
    /// `query_class` defaults to `None` (recall fills it in once the query is
    /// classified). Prefer this over a struct literal: the struct is
    /// `#[non_exhaustive]`, so this is the forward-compatible constructor.
    pub fn new(signals: Vec<(String, f32)>, summary: String) -> Self {
        Self {
            signals,
            summary,
            query_class: None,
        }
    }
}

/// A single recall result with scoring breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// The matching record.
    pub record: Record,
    /// Final fused score (0.0–1.0).
    pub score: f32,
    /// Scoring breakdown showing each signal's contribution.
    pub explanation: ScoreExplanation,
}

/// Configuration for the recall scoring engine.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// Signal weights.
    pub weights: ScoreWeights,
    /// Recency half-life in hours (default: 168 = 1 week).
    pub recency_half_life_hours: f64,
    /// Current time (for testing).
    pub now: DateTime<Utc>,
    /// Optional temporal target parsed from the query.
    pub temporal_target: Option<TemporalTarget>,
    /// Keywords extracted from the query (lowercased, stop words removed).
    pub query_keywords: Vec<String>,
    /// Activation-level scoring configuration.
    pub activation_config: crate::activation::ActivationConfig,
    /// Optional scope filter. If set, only return records matching this scope.
    /// Multiple scopes can be provided for "widen" queries.
    pub scope_filter: Vec<String>,
    /// Minimum confidence threshold (0.0–1.0). Records below this are excluded.
    pub min_confidence: Option<f32>,
    /// Minimum importance threshold (0.0–1.0). Records below this are excluded.
    pub min_importance: Option<f32>,
    /// Query-Time Chunk reranking. When `Some(top_k)`, after session-level
    /// candidates are scored the top-K are re-scored by embedding their text
    /// in overlapping windows at query time and blending the best-chunk
    /// cosine with the fused score. Lifts recall on long documents where the
    /// answer sits beyond the indexed embedding window, without the shared-
    /// timestamp pitfall of index-time chunking. `None` disables QTC.
    pub qtc: Option<QtcConfig>,
    /// Near-duplicate collapse before truncation to `top_k`.
    pub dedup: DedupConfig,
}

/// Near-duplicate collapse configuration for recall.
///
/// When enabled, recall collapses near-identical, same-table results (by lexical
/// SimHash) into a single highest-scored representative *before* truncating to
/// `top_k`, so the scarce result slots aren't spent on duplicates. At the
/// conservative default threshold this catches only **near-exact** redundancy —
/// the same text re-stored, or case/whitespace/punctuation/tiny-edit variants —
/// not semantic paraphrases (a one-word synonym swap is already several SimHash
/// bits away). The collapse is silent: at that threshold the kept representative
/// carries essentially the same content as the ones dropped.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Collapse near-duplicates when true. Defaults to `false` so existing
    /// callers (boot, MCP siblings, tests) keep their exact result sets; the
    /// agent-facing recall paths opt in explicitly.
    pub enabled: bool,
    /// Max Hamming distance (of 64 SimHash bits) for two results to count as
    /// near-duplicates. Lower is stricter. Default 3 — deliberately tight so it
    /// only collapses near-exact text; raise it to also merge looser variants
    /// at the risk of collapsing genuinely distinct records.
    pub hamming_threshold: u32,
    /// Skip dedup for normalized text shorter than this many chars — short
    /// strings collide too readily under SimHash. Default 24.
    pub min_text_len: usize,
    /// Completeness k-widening: after the top-k cut, compare how well the kept
    /// subset compresses (DEFLATE level 1) against the full post-dedup
    /// candidate pool. If the kept set is *materially more compressible* than
    /// the pool — i.e. the pool holds diverse content that the cut dropped —
    /// widen k by a bounded amount and re-trim once. Off by default so the raw
    /// library result set is unchanged; the agent-facing recall paths opt in.
    pub completeness_widen: bool,
    /// Compression-ratio gap (pool ratio − kept ratio) above which a dropped
    /// diverse cluster is inferred and k is widened. A higher kept-vs-pool gap
    /// means the kept set is far more redundant than the pool it was cut from.
    /// Default 0.15 (15 percentage points), per the spec heuristic.
    pub widen_threshold: f32,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hamming_threshold: 3,
            min_text_len: 24,
            completeness_widen: false,
            widen_threshold: 0.15,
        }
    }
}

/// Parameters for Query-Time Chunk reranking.
#[derive(Debug, Clone)]
pub struct QtcConfig {
    /// Number of top candidates to rescore with chunk-level embeddings.
    pub top_k: usize,
    /// Max characters per chunk window.
    pub chunk_chars: usize,
    /// Stride between chunk windows (`chunk_chars - stride_chars` = overlap).
    pub stride_chars: usize,
    /// Blend weight: `new_score = alpha * best_chunk + (1-alpha) * fused_score`.
    pub alpha: f32,
}

impl Default for QtcConfig {
    fn default() -> Self {
        // Tuned on LongMemEval-S 150-Q: top_k=20, chunk_chars=1200, stride=900
        // (25% overlap), alpha=0.7. Hits 97.3% hit rate / 94.0% recall at the
        // oracle ceiling. See README investigation notes.
        Self {
            top_k: 20,
            chunk_chars: 1200,
            stride_chars: 900,
            alpha: 0.7,
        }
    }
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            weights: ScoreWeights::default(),
            recency_half_life_hours: 168.0,
            now: Utc::now(),
            temporal_target: None,
            query_keywords: Vec::new(),
            activation_config: crate::activation::ActivationConfig::default(),
            scope_filter: Vec::new(),
            min_confidence: None,
            min_importance: None,
            qtc: None,
            dedup: DedupConfig::default(),
        }
    }
}

/// Raw signal values for a single record before fusion.
#[derive(Debug, Clone, Default)]
pub struct SignalValues {
    /// Cosine similarity from vector search (0.0–1.0).
    pub vector_similarity: f32,
    /// BM25 score from full-text search (0.0–1.0).
    pub keyword_match: f32,
    /// Graph proximity (0.0–1.0), based on hops from active context.
    pub graph_proximity: f32,
    /// Feedback boost from past relevance signals (0.0–1.0).
    pub feedback_boost: f32,
    /// Preference match score (0.0–1.0).
    pub preference_match: f32,
    /// Activation-level score (0.0–1.0), based on access frequency and decay.
    pub activation: f32,
    /// Reciprocal Rank Fusion of vector + FTS ranks (0.0–1.0 after normalization).
    /// Captures rank-agreement across retrievers and is robust to score-scale drift.
    pub rrf: f32,
}

/// Compute recency decay using a true half-life curve.
///
/// Returns 1.0 at age=0 and 0.5 at `age = half_life_hours`.
pub fn recency_decay(
    record_time: &DateTime<Utc>,
    now: &DateTime<Utc>,
    half_life_hours: f64,
) -> f32 {
    let age_secs = (*now - *record_time).num_seconds().max(0) as f64;
    let age_hours = age_secs / 3600.0;
    let safe_half_life = half_life_hours.max(f64::EPSILON);
    let decay = (-(std::f64::consts::LN_2) * age_hours / safe_half_life).exp();
    decay as f32
}

/// Compute keyword overlap as fraction of query keywords found in text.
///
/// Uses the multiplicative fusion validated by MemPalace (+1.2% on LongMemEval):
/// `keyword_overlap = count(matching_keywords) / count(query_keywords)`
pub fn keyword_overlap(query_keywords: &[String], record_text: &str) -> f32 {
    if query_keywords.is_empty() {
        return 0.0;
    }
    crate::util::overlapping_chunks(record_text, 1600, 400)
        .into_iter()
        .map(|chunk| {
            let lower = chunk.to_lowercase();
            let matching = query_keywords
                .iter()
                .filter(|kw| lower.contains(kw.as_str()))
                .count();
            matching as f32 / query_keywords.len() as f32
        })
        .fold(0.0, f32::max)
}

/// Extract non-stop-word keywords from a query string.
pub fn extract_keywords(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "must", "of", "in", "to", "for", "with", "on", "at", "by", "from", "as", "into",
        "about", "between", "through", "during", "before", "after", "above", "below", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "but", "and", "or", "if", "because", "until", "while", "that", "which", "who",
        "whom", "this", "these", "those", "what", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "him", "his", "she", "her", "they", "them", "their",
    ];

    query
        .to_lowercase()
        .split_whitespace()
        // Strip punctuation BEFORE stop word check so "the," matches "the"
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 1 && !STOP_WORDS.contains(w))
        .map(|w| w.to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Fuse all signals into a final score and produce an explanation.
pub fn fuse_signals(
    record: &Record,
    signals: &SignalValues,
    config: &RecallConfig,
) -> (f32, ScoreExplanation) {
    let w = &config.weights;

    // Compute recency
    let recency = recency_decay(
        &record.created_at,
        &config.now,
        config.recency_half_life_hours,
    );

    // Compute temporal proximity boost
    let temporal = config
        .temporal_target
        .as_ref()
        .map(|t| crate::temporal::temporal_boost(&record.created_at, t))
        .unwrap_or(0.0);

    // Compute keyword overlap from record text
    let keyword_from_overlap = if !config.query_keywords.is_empty() {
        let text = crate::util::searchable_text(&record.data);
        keyword_overlap(&config.query_keywords, &text)
    } else {
        0.0
    };

    // Use the max of BM25 keyword score and keyword overlap
    let keyword_score = signals.keyword_match.max(keyword_from_overlap);

    // Compute activation score
    let activation_score = if w.activation > 0.0 {
        crate::activation::activation_boost(record, &config.now, &config.activation_config)
    } else {
        0.0
    };

    // Compute importance score (uses effective importance if available, else base)
    let importance_score = if w.importance > 0.0 {
        record
            .data
            .get("_effective_importance")
            .or_else(|| record.data.get("_importance"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5) // default for records without importance
    } else {
        0.0
    };

    // Per-record adaptive reweighting was tried but hurt recall:
    // it penalized records with richer signal profiles because more active
    // signals → smaller scale factor per signal. On LongMemEval-s this flipped
    // recency-dominated distractors past answer records. Keeping the static
    // weighted sum; signal-dead-across-corpus redistribution should be a
    // one-shot pre-computation if ever re-attempted, not per-record.
    let final_score = w.vector * signals.vector_similarity
        + w.recency * recency
        + w.graph * signals.graph_proximity
        + w.keyword * keyword_score
        + w.feedback * signals.feedback_boost
        + w.temporal * temporal
        + w.preference * signals.preference_match
        + w.activation * activation_score
        + w.importance * importance_score
        + w.rrf * signals.rrf;

    // Build explanation
    let mut signal_list = Vec::new();
    signal_list.push(("vector_similarity".to_string(), signals.vector_similarity));
    signal_list.push(("recency".to_string(), recency));
    if signals.graph_proximity > 0.0 {
        signal_list.push(("graph_proximity".to_string(), signals.graph_proximity));
    }
    if keyword_score > 0.0 {
        signal_list.push(("keyword_match".to_string(), keyword_score));
    }
    if signals.feedback_boost > 0.0 {
        signal_list.push(("feedback_boost".to_string(), signals.feedback_boost));
    }
    if temporal > 0.0 {
        signal_list.push(("temporal_proximity".to_string(), temporal));
    }
    if signals.preference_match > 0.0 {
        signal_list.push(("preference_match".to_string(), signals.preference_match));
    }
    if activation_score > 0.0 {
        signal_list.push(("activation".to_string(), activation_score));
    }
    if importance_score > 0.0 && w.importance > 0.0 {
        signal_list.push(("importance".to_string(), importance_score));
    }
    if signals.rrf > 0.0 {
        signal_list.push(("rrf".to_string(), signals.rrf));
    }

    // Build human-readable summary
    let top_signals: Vec<String> = {
        let mut weighted: Vec<(String, f32)> = vec![
            ("vector".into(), w.vector * signals.vector_similarity),
            ("recency".into(), w.recency * recency),
            ("graph".into(), w.graph * signals.graph_proximity),
            ("keyword".into(), w.keyword * keyword_score),
            ("feedback".into(), w.feedback * signals.feedback_boost),
            ("temporal".into(), w.temporal * temporal),
            ("preference".into(), w.preference * signals.preference_match),
            ("activation".into(), w.activation * activation_score),
            ("rrf".into(), w.rrf * signals.rrf),
        ];
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        weighted
            .into_iter()
            .filter(|(_, v)| *v > 0.001)
            .take(3)
            .map(|(name, val)| format!("{name}: {val:.2}"))
            .collect()
    };

    let summary = if top_signals.is_empty() {
        "no scoring signals".to_string()
    } else {
        format!("score {final_score:.2} ({})", top_signals.join(", "))
    };

    let explanation = ScoreExplanation {
        signals: signal_list,
        summary,
        // Set by the caller (recall) once the query is classified; fuse_signals
        // itself is query-class agnostic.
        query_class: None,
    };

    (final_score, explanation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_weights_sum_to_one() {
        let w = ScoreWeights::default();
        let sum =
            w.vector + w.recency + w.graph + w.keyword + w.feedback + w.temporal + w.preference;
        // rrf defaults to 0 and is excluded from the normalization invariant.
        assert!((sum - 1.0).abs() < 0.001, "weights sum to {sum}");
    }

    #[test]
    fn recency_decay_at_zero() {
        let now = Utc::now();
        let decay = recency_decay(&now, &now, 168.0);
        assert!((decay - 1.0).abs() < 0.01);
    }

    #[test]
    fn recency_decay_at_half_life() {
        let now = Utc::now();
        let week_ago = now - chrono::Duration::hours(168);
        let decay = recency_decay(&week_ago, &now, 168.0);
        assert!((decay - 0.5).abs() < 0.01, "decay = {decay}");
    }

    #[test]
    fn keyword_overlap_full_match() {
        let keywords = vec!["auth".into(), "error".into()];
        let text = "Fixed auth error in login flow";
        assert!((keyword_overlap(&keywords, text) - 1.0).abs() < 0.01);
    }

    #[test]
    fn keyword_overlap_partial() {
        let keywords = vec!["auth".into(), "error".into(), "deploy".into()];
        let text = "Fixed auth timeout";
        let overlap = keyword_overlap(&keywords, text);
        assert!((overlap - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn keyword_overlap_uses_matching_chunk_for_long_text() {
        let keywords = vec!["auth".into(), "timeout".into()];
        let long_prefix = "noise ".repeat(500);
        let text = format!("{long_prefix}fixed auth timeout in pool tuning");
        let overlap = keyword_overlap(&keywords, &text);
        assert!((overlap - 1.0).abs() < 0.01, "overlap = {overlap}");
    }

    #[test]
    fn extract_keywords_filters_stop_words() {
        let kw = extract_keywords("the auth error in the login flow");
        assert!(kw.contains(&"auth".to_string()));
        assert!(kw.contains(&"error".to_string()));
        assert!(kw.contains(&"login".to_string()));
        assert!(kw.contains(&"flow".to_string()));
        assert!(!kw.contains(&"the".to_string()));
        assert!(!kw.contains(&"in".to_string()));
    }

    #[test]
    fn fuse_signals_produces_score() {
        let record = Record::new("sessions", json!({"summary": "Fixed auth bug"}));
        let signals = SignalValues {
            vector_similarity: 0.9,
            keyword_match: 0.5,
            ..Default::default()
        };
        let config = RecallConfig::default();
        let (score, explanation) = fuse_signals(&record, &signals, &config);
        assert!(score > 0.0);
        assert!(!explanation.signals.is_empty());
        assert!(!explanation.summary.is_empty());
    }

    #[test]
    fn fuse_signals_all_zero() {
        let record = Record::new("test", json!({}));
        let signals = SignalValues::default();
        let mut config = RecallConfig::default();
        config.now = record.created_at; // same time = recency 1.0
        let (score, _) = fuse_signals(&record, &signals, &config);
        // Only recency contributes at weight 0.15, value 1.0 → score ≈ 0.15.
        assert!(score > 0.0);
        assert!(score < 0.20);
    }

    #[test]
    fn qtc_config_default_is_sane() {
        // Guards the tuned QTC parameters — changing any of these should
        // be a conscious decision, since they were validated on
        // LongMemEval-S 150-Q at 97.3% hit / 94.0% recall.
        let qtc = QtcConfig::default();
        assert_eq!(qtc.top_k, 20);
        assert!(qtc.chunk_chars > 0 && qtc.chunk_chars <= 2000);
        assert!(qtc.stride_chars > 0 && qtc.stride_chars <= qtc.chunk_chars);
        // alpha must be a valid blending weight
        assert!((0.0..=1.0).contains(&qtc.alpha));
    }

    #[test]
    fn recall_config_default_has_qtc_disabled() {
        // QTC is opt-in so existing library users aren't silently pushed
        // into a slower (but higher-quality) code path on upgrade.
        let cfg = RecallConfig::default();
        assert!(cfg.qtc.is_none());
    }
}
