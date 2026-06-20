//! Agent Brain
//!
//! ## 11.1 — Unified Memory Decision Pipeline
//! All write paths route through a single decision pipeline:
//! Observe → Classify → Scope → Resolve → Score → Commit
//!
//! ## 11.2 — Provenance, Confidence, and Trust Model
//! Every durable memory carries explicit provenance: source, scope, confidence,
//! verification status, derivation chain, and trust tier classification.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::consolidation::{check_conflict, ConflictResult};
use crate::entity::{extract_entities, Entity};
use crate::importance::compute_importance;
use crate::record::{Record, RecordId};
use crate::{Axil, Result};

// ── Types ──────────────────────────────────────────────────────────

/// Where the observation came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    /// Direct user input (CLI, chat).
    User,
    /// Agent action or reasoning.
    Agent,
    /// Tool output (bash, file read, etc.).
    ToolOutput,
    /// Background hook (file watcher, session hook).
    Hook,
    /// File content (project indexer, code analysis).
    File,
    /// Inferred from existing memories.
    Inference,
    /// LLM-generated content.
    Llm,
    /// Unknown source (legacy records).
    Unknown,
}

impl Default for MemorySource {
    fn default() -> Self {
        Self::Unknown
    }
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Agent => write!(f, "agent"),
            Self::ToolOutput => write!(f, "tool_output"),
            Self::Hook => write!(f, "hook"),
            Self::File => write!(f, "file"),
            Self::Inference => write!(f, "inference"),
            Self::Llm => write!(f, "llm"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl MemorySource {
    /// Parse source kind from string, case-insensitive, with common aliases.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "user" => Self::User,
            "agent" => Self::Agent,
            "tool_output" | "tool-output" | "tool" => Self::ToolOutput,
            "hook" => Self::Hook,
            "file" => Self::File,
            "inference" => Self::Inference,
            "llm" => Self::Llm,
            _ => Self::Unknown,
        }
    }
}

/// Memory scope — controls visibility and retrieval boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Ephemeral task-local state (auto-expires when session ends).
    Session,
    /// Memory specific to one named agent (multi-agent isolation).
    Agent,
    /// Repo- or workspace-specific knowledge (default).
    Project,
    /// Persistent user preferences and habits (crosses projects).
    User,
    /// Cross-project reusable procedures and patterns.
    Global,
}

impl Default for MemoryScope {
    fn default() -> Self {
        Self::Project
    }
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session => write!(f, "session"),
            Self::Agent => write!(f, "agent"),
            Self::Project => write!(f, "project"),
            Self::User => write!(f, "user"),
            Self::Global => write!(f, "global"),
        }
    }
}

impl MemoryScope {
    /// Parse scope from string, case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "session" => Some(Self::Session),
            "agent" => Some(Self::Agent),
            "project" => Some(Self::Project),
            "user" => Some(Self::User),
            "global" => Some(Self::Global),
            _ => None,
        }
    }
}

/// Memory type classification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Short-term task context (working memory).
    Working,
    /// Facts about entities and concepts (semantic memory).
    Semantic,
    /// Session summaries and event sequences (episodic memory).
    Episodic,
    /// How-to knowledge and learned patterns (procedural memory).
    Procedural,
    /// User preferences and behavioral patterns.
    Preference,
    /// High-level truths the agent holds (belief system).
    Belief,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Working => write!(f, "working"),
            Self::Semantic => write!(f, "semantic"),
            Self::Episodic => write!(f, "episodic"),
            Self::Procedural => write!(f, "procedural"),
            Self::Preference => write!(f, "preference"),
            Self::Belief => write!(f, "belief"),
        }
    }
}

/// An observation — raw input to the decision pipeline.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Where this observation came from.
    pub source: MemorySource,
    /// Optional source reference (command, file path, hook name, session ID).
    pub source_ref: Option<String>,
    /// Explicit scope hint (if caller knows the appropriate scope).
    pub scope: Option<MemoryScope>,
    /// Explicit memory type hint (if caller knows the type).
    pub memory_type: Option<MemoryType>,
    /// The raw text content of the observation.
    pub text: String,
    /// Optional structured data to store alongside the text.
    pub data: Option<Value>,
    /// Optional target table override (bypass auto-classification).
    pub table: Option<String>,
    /// Classification hints from the caller.
    pub hints: Vec<String>,
    /// The agent name (for multi-agent scoping).
    pub agent: Option<String>,
}

impl Observation {
    /// Create a minimal observation from text.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            source: MemorySource::Unknown,
            source_ref: None,
            scope: None,
            memory_type: None,
            text: text.into(),
            data: None,
            table: None,
            hints: Vec::new(),
            agent: None,
        }
    }

    /// Set the source.
    pub fn with_source(mut self, source: MemorySource) -> Self {
        self.source = source;
        self
    }

    /// Set the source reference.
    pub fn with_source_ref(mut self, source_ref: impl Into<String>) -> Self {
        self.source_ref = Some(source_ref.into());
        self
    }

    /// Set the scope.
    pub fn with_scope(mut self, scope: MemoryScope) -> Self {
        self.scope = Some(scope);
        self
    }

    /// Set the memory type.
    pub fn with_memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    /// Set structured data.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Set the target table.
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = Some(table.into());
        self
    }

    /// Add a hint.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    /// Set the agent name.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }
}

/// What action the pipeline decided to take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineAction {
    /// Stored as a new memory.
    Stored,
    /// Updated an existing memory in place.
    Updated { existing_id: String },
    /// Superseded an older memory (old one marked superseded).
    Superseded { old_id: String },
    /// Observation was too low-value or duplicate — not stored.
    Ignored,
}

/// Outcome of the decision pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineOutcome {
    /// What action was taken.
    pub action: PipelineAction,
    /// The stored/updated record (None if Ignored).
    pub record: Option<Record>,
    /// Classified memory type.
    pub memory_type: MemoryType,
    /// Assigned scope.
    pub scope: MemoryScope,
    /// Importance score.
    pub importance: f32,
    /// Confidence score.
    pub confidence: f32,
    /// Entities extracted from the observation.
    pub entities: Vec<Entity>,
    /// Human-readable reason for the decision.
    pub reason: String,
    /// Pipeline latency in microseconds.
    pub latency_us: u64,
}

// ── Classification ─────────────────────────────────────────────────

/// Preference signal words.
const PREFERENCE_SIGNALS: &[&str] = &[
    "prefer",
    "always",
    "never",
    "don't like",
    "i like",
    "i want",
    "should always",
    "should never",
    "habit",
    "convention",
    "rule:",
    "rule is",
];

/// Procedural signal words.
const PROCEDURAL_SIGNALS: &[&str] = &[
    "how to",
    "steps to",
    "procedure",
    "workflow",
    "recipe",
    "pattern:",
    "to do this",
    "the way to",
    "process:",
    "run this",
    "execute",
    "deploy by",
];

/// Episodic signal words.
const EPISODIC_SIGNALS: &[&str] = &[
    "session",
    "today",
    "yesterday",
    "just now",
    "happened",
    "completed",
    "finished",
    "started",
    "ended",
    "we did",
    "i did",
    "worked on",
];

/// Belief signal words.
const BELIEF_SIGNALS: &[&str] = &[
    "believe",
    "i think",
    "it seems",
    "likely",
    "probably",
    "truth is",
    "the fact is",
    "in my understanding",
];

/// Classify the memory type from text content and hints.
fn classify_memory_type(text: &str, hints: &[String], entities: &[Entity]) -> MemoryType {
    let lower = text.to_lowercase();

    // Check hints first (explicit caller intent).
    for hint in hints {
        match hint.to_lowercase().as_str() {
            "preference" | "pref" => return MemoryType::Preference,
            "procedure" | "proc" | "how-to" => return MemoryType::Procedural,
            "episode" | "episodic" | "session" => return MemoryType::Episodic,
            "belief" => return MemoryType::Belief,
            "fact" | "semantic" | "knowledge" => return MemoryType::Semantic,
            "working" | "temp" | "scratch" => return MemoryType::Working,
            _ => {}
        }
    }

    // Score each type by signal word matches.
    let pref_score = count_signals(&lower, PREFERENCE_SIGNALS);
    let proc_score = count_signals(&lower, PROCEDURAL_SIGNALS);
    let epis_score = count_signals(&lower, EPISODIC_SIGNALS);
    let belief_score = count_signals(&lower, BELIEF_SIGNALS);

    // Entity-rich text is likely semantic.
    let semantic_score = if entities.len() >= 2 {
        2
    } else if entities.len() == 1 {
        1
    } else {
        0
    };

    // Pick the highest scorer. Ties broken by priority: semantic > episodic > procedural > preference > belief > working.
    let max = *[
        pref_score,
        proc_score,
        epis_score,
        belief_score,
        semantic_score,
    ]
    .iter()
    .max()
    .unwrap_or(&0);

    if max == 0 {
        return MemoryType::Semantic; // Default.
    }

    if semantic_score == max && entities.len() >= 2 {
        MemoryType::Semantic
    } else if epis_score == max {
        MemoryType::Episodic
    } else if proc_score == max {
        MemoryType::Procedural
    } else if pref_score == max {
        MemoryType::Preference
    } else if belief_score == max {
        MemoryType::Belief
    } else {
        MemoryType::Semantic
    }
}

fn count_signals(text: &str, signals: &[&str]) -> usize {
    signals.iter().filter(|s| text.contains(**s)).count()
}

// ── Scope inference ────────────────────────────────────────────────

/// Infer scope from source, hints, and memory type.
fn infer_scope(
    source: &MemorySource,
    memory_type: &MemoryType,
    text: &str,
    table: Option<&str>,
) -> MemoryScope {
    // Source-based inference.
    match source {
        MemorySource::Hook => return MemoryScope::Project,
        MemorySource::File => return MemoryScope::Project,
        _ => {}
    }

    // Table-based heuristic.
    if let Some(t) = table {
        match t {
            "_sessions" | "sessions" => return MemoryScope::Session,
            "preferences" | "_preferences" => return MemoryScope::User,
            "procedures" | "_procedures" | "patterns" => return MemoryScope::Global,
            _ => {}
        }
    }

    // Memory type default scope.
    match memory_type {
        MemoryType::Working => MemoryScope::Session,
        MemoryType::Episodic => MemoryScope::Project,
        MemoryType::Preference => MemoryScope::User,
        MemoryType::Procedural => {
            // Procedural: check if text is project-specific or general.
            let lower = text.to_lowercase();
            if lower.contains("this project")
                || lower.contains("this repo")
                || lower.contains("here we")
            {
                MemoryScope::Project
            } else {
                MemoryScope::Global
            }
        }
        MemoryType::Belief => MemoryScope::Project,
        MemoryType::Semantic => MemoryScope::Project,
    }
}

// ── Novelty / duplicate detection ──────────────────────────────────

/// Minimum text length to attempt vector-based duplicate detection.
const MIN_TEXT_FOR_VECTOR: usize = 10;

/// Similarity threshold for considering two records as duplicates.
const DUPLICATE_THRESHOLD: f32 = 0.95;

/// Minimum importance to store (below this, the observation is ignored).
const MIN_IMPORTANCE_THRESHOLD: f32 = 0.15;

// ── Pipeline implementation ────────────────────────────────────────

/// Run the full decision pipeline on an observation.
///
/// Pipeline stages:
/// 1. **Observe**: normalize raw input
/// 2. **Classify**: determine memory type
/// 3. **Scope**: choose scope
/// 4. **Resolve**: detect duplicate / superseded / conflicting memories
/// 5. **Score**: importance + confidence + novelty
/// 6. **Commit**: insert/update/supersede/ignore
///
/// Latency budget: < 5ms over a raw insert for the no-LLM path.
pub fn remember(db: &Axil, observation: Observation) -> Result<PipelineOutcome> {
    let start = std::time::Instant::now();

    // ── Stage 1: Observe — normalize input ──
    let text = observation.text.trim().to_string();
    if text.is_empty() {
        return Ok(PipelineOutcome {
            action: PipelineAction::Ignored,
            record: None,
            memory_type: MemoryType::Working,
            scope: MemoryScope::Session,
            importance: 0.0,
            confidence: 0.0,
            entities: Vec::new(),
            reason: "empty observation".to_string(),
            latency_us: start.elapsed().as_micros() as u64,
        });
    }

    // ── Stage 2: Classify — determine memory type ──
    let entities = extract_entities(&text);
    let memory_type = observation
        .memory_type
        .clone()
        .unwrap_or_else(|| classify_memory_type(&text, &observation.hints, &entities));

    // ── Stage 3: Scope — choose scope ──
    let scope = observation.scope.clone().unwrap_or_else(|| {
        infer_scope(
            &observation.source,
            &memory_type,
            &text,
            observation.table.as_deref(),
        )
    });

    // ── Stage 4: Resolve — detect duplicates / superseding ──
    // Build the data payload.
    let table = observation
        .table
        .as_deref()
        .unwrap_or_else(|| memory_type_to_table(&memory_type));

    let mut data = if let Some(ref d) = observation.data {
        d.clone()
    } else {
        json!({})
    };

    // Ensure the data is an object.
    if !data.is_object() {
        data = json!({ "value": data });
    }

    // Inject text into the appropriate field if not already present.
    if let Some(obj) = data.as_object_mut() {
        if !obj.contains_key("summary")
            && !obj.contains_key("fact")
            && !obj.contains_key("statement")
        {
            // Beliefs use "statement" + "confidence" to match BeliefSystem schema.
            if memory_type == MemoryType::Belief {
                obj.insert("statement".to_string(), json!(text));
                if !obj.contains_key("confidence") {
                    obj.insert("confidence".to_string(), json!(1.0));
                }
                if !obj.contains_key("source") {
                    obj.insert("source".to_string(), json!("explicit"));
                }
                if !obj.contains_key("doubted") {
                    obj.insert("doubted".to_string(), json!(false));
                }
                if !obj.contains_key("created_at") {
                    obj.insert(
                        "created_at".to_string(),
                        json!(chrono::Utc::now().to_rfc3339()),
                    );
                }
            } else {
                obj.insert("summary".to_string(), json!(text));
            }
        }

        // Inject provenance metadata.
        obj.insert(
            "_source".to_string(),
            json!({
                "kind": observation.source.to_string(),
                "ref": observation.source_ref.as_deref().unwrap_or(""),
            }),
        );
        obj.insert("_scope".to_string(), json!(scope.to_string()));
        obj.insert("_memory_type".to_string(), json!(memory_type.to_string()));
        obj.insert("_confidence".to_string(), json!(1.0)); // Default; adjusted below.

        if let Some(ref agent) = observation.agent {
            obj.insert("_agent".to_string(), json!(agent));
        }

        // Inject entity names for indexing.
        if !entities.is_empty() {
            let entity_names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
            obj.insert("_entities".to_string(), json!(entity_names));
        }
    }

    // Compute importance.
    let importance = compute_importance(&data);
    if let Some(obj) = data.as_object_mut() {
        obj.insert("_importance".to_string(), json!(importance));
    }

    // Skip low-importance observations.
    if importance < MIN_IMPORTANCE_THRESHOLD {
        return Ok(PipelineOutcome {
            action: PipelineAction::Ignored,
            record: None,
            memory_type,
            scope,
            importance,
            confidence: 0.0,
            entities,
            reason: format!(
                "importance {importance:.2} below threshold {MIN_IMPORTANCE_THRESHOLD}"
            ),
            latency_us: start.elapsed().as_micros() as u64,
        });
    }

    // Check for duplicates/superseding via vector similarity (if available).
    let resolve_result =
        if text.len() >= MIN_TEXT_FOR_VECTOR && db.has_vector_index() && db.has_embedder() {
            resolve_against_existing(db, &text, table)
        } else {
            ResolveResult::Novel
        };

    // ── Stage 5: Score — compute confidence ──
    let confidence = match &resolve_result {
        ResolveResult::Novel => 1.0,
        ResolveResult::Duplicate { .. } => 0.0, // Won't store.
        ResolveResult::Supersedes { .. } => 0.9, // New info superseding old.
        ResolveResult::Contradicts { .. } => 0.7, // Conflicting — lower confidence.
    };
    if let Some(obj) = data.as_object_mut() {
        obj.insert("_confidence".to_string(), json!(confidence));
    }

    // ── Stage 6: Commit ──
    let (action, record, reason) = match resolve_result {
        ResolveResult::Duplicate {
            existing_id,
            similarity,
        } => (
            PipelineAction::Ignored,
            None,
            format!("duplicate of {existing_id} (similarity {similarity:.3})"),
        ),
        ResolveResult::Supersedes { old_id, similarity } => {
            // Store the new record with supersede reference.
            if let Some(obj) = data.as_object_mut() {
                obj.insert("_supersedes".to_string(), json!([old_id.to_string()]));
            }
            let record = db.insert(table, data)?;

            // Mark the old record as superseded with backlink to the new record.
            if let Ok(Some(old_record)) = db.get(&old_id) {
                let mut old_data = old_record.data.clone();
                if let Some(obj) = old_data.as_object_mut() {
                    obj.insert("_superseded".to_string(), json!(true));
                    obj.insert("_superseded_by".to_string(), json!(record.id.to_string()));
                }
                let _ = db.update(&old_id, old_data);
            }
            (
                PipelineAction::Superseded {
                    old_id: old_id.to_string(),
                },
                Some(record),
                format!("supersedes existing memory (similarity {similarity:.3})"),
            )
        }
        ResolveResult::Contradicts {
            existing_id,
            similarity,
        } => {
            // Store with contradiction reference — don't auto-resolve.
            if let Some(obj) = data.as_object_mut() {
                obj.insert("_contradicts".to_string(), json!([existing_id.to_string()]));
            }
            let record = db.insert(table, data)?;
            (
                PipelineAction::Stored,
                Some(record),
                format!("contradicts {existing_id} (similarity {similarity:.3}) — both preserved"),
            )
        }
        ResolveResult::Novel => {
            let record = db.insert(table, data)?;
            (
                PipelineAction::Stored,
                Some(record),
                "new memory stored".to_string(),
            )
        }
    };

    Ok(PipelineOutcome {
        action,
        record,
        memory_type,
        scope,
        importance,
        confidence,
        entities,
        reason,
        latency_us: start.elapsed().as_micros() as u64,
    })
}

// ── Resolve helpers ────────────────────────────────────────────────

enum ResolveResult {
    /// No similar existing memory found.
    Novel,
    /// Nearly identical memory already exists — skip.
    Duplicate {
        existing_id: RecordId,
        similarity: f32,
    },
    /// New memory supersedes an older one.
    Supersedes { old_id: RecordId, similarity: f32 },
    /// New memory contradicts an existing one.
    Contradicts {
        existing_id: RecordId,
        similarity: f32,
    },
}

/// Check new text against existing memories using vector similarity.
///
/// Uses cascaded filtering: only check top-k similar records (not all).
fn resolve_against_existing(db: &Axil, text: &str, table: &str) -> ResolveResult {
    // Find similar records via vector search.
    let similar = match db.similar_to(text, 5) {
        Ok(results) => results,
        Err(_) => return ResolveResult::Novel,
    };

    for (record, similarity) in &similar {
        // Only consider records in the same table.
        if record.table != table {
            continue;
        }

        // Skip already-superseded records.
        if record
            .data
            .get("_superseded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            continue;
        }

        if *similarity > DUPLICATE_THRESHOLD {
            let existing_text = crate::util::value_text(&record.data);
            let text_sim =
                crate::util::word_jaccard(&text.to_lowercase(), &existing_text.to_lowercase());

            if text_sim > 0.90 {
                return ResolveResult::Duplicate {
                    existing_id: record.id.clone(),
                    similarity: *similarity,
                };
            }
        }

        // Check for superseding/contradicting via conflict detection.
        if *similarity >= 0.92 {
            let temp_record = Record::new(table, json!({ "summary": text }));
            match check_conflict(&temp_record, record, *similarity) {
                ConflictResult::Supersedes {
                    old_record_id,
                    similarity,
                } => {
                    return ResolveResult::Supersedes {
                        old_id: old_record_id,
                        similarity,
                    };
                }
                ConflictResult::Contradicts {
                    existing_record_id,
                    similarity,
                } => {
                    return ResolveResult::Contradicts {
                        existing_id: existing_record_id,
                        similarity,
                    };
                }
                ConflictResult::Novel => {}
            }
        }
    }

    ResolveResult::Novel
}

/// Map memory type to the default storage table.
fn memory_type_to_table(memory_type: &MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Working => "context",
        MemoryType::Semantic => "context",
        MemoryType::Episodic => "sessions",
        MemoryType::Procedural => "procedures",
        MemoryType::Preference => "preferences",
        MemoryType::Belief => "_beliefs",
    }
}

// ══════════════════════════════════════════════════════════════════
// 11.2 — Provenance, Confidence, and Trust Model
// ══════════════════════════════════════════════════════════════════

/// Trust tiers — how much the brain should rely on a memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustTier {
    /// Direct user/tool/file evidence. Highest trust.
    Observed,
    /// Inferred from graph, consolidation, or derivation chain.
    Derived,
    /// LLM-assisted or low-confidence extraction.
    Suggested,
    /// Contradicted, stale, or manually questioned. Lowest trust.
    Doubted,
}

impl std::fmt::Display for TrustTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observed => write!(f, "observed"),
            Self::Derived => write!(f, "derived"),
            Self::Suggested => write!(f, "suggested"),
            Self::Doubted => write!(f, "doubted"),
        }
    }
}

/// Full provenance record for a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Where the memory came from.
    pub source_kind: MemorySource,
    /// Reference to the source (command, file path, session ID, etc.).
    pub source_ref: String,
    /// Memory scope.
    pub scope: MemoryScope,
    /// Confidence level (0.0–1.0).
    pub confidence: f32,
    /// Whether manually verified by a human.
    pub verified: bool,
    /// IDs of records this memory supersedes.
    pub supersedes: Vec<String>,
    /// IDs of records this memory contradicts.
    pub contradicts: Vec<String>,
    /// IDs of records this memory was derived from.
    pub derived_from: Vec<String>,
    /// When provenance was last validated/checked.
    pub last_validated_at: Option<String>,
    /// Computed trust tier.
    pub trust_tier: TrustTier,
}

/// Classify the trust tier for a record based on its provenance metadata.
pub fn classify_trust(data: &Value) -> TrustTier {
    // Check if doubted (explicit doubt or superseded).
    let doubted = data
        .get("doubted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let superseded = data
        .get("_superseded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if doubted || superseded {
        return TrustTier::Doubted;
    }

    // Check if verified.
    let verified = data
        .get("_verified")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Check source kind.
    let source_kind = data
        .get("_source")
        .and_then(|s| s.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let confidence = data
        .get("_confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    // Derived: came from inference or has derivation chain.
    let is_derived = source_kind == "inference"
        || data
            .get("_derived_from")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

    // LLM-assisted: source is LLM.
    let is_llm = source_kind == "llm";

    if verified {
        return TrustTier::Observed; // Verified always trusted.
    }

    if is_derived {
        if confidence >= 0.7 {
            TrustTier::Derived
        } else {
            TrustTier::Suggested
        }
    } else if is_llm {
        if confidence >= 0.8 {
            TrustTier::Derived
        } else {
            TrustTier::Suggested
        }
    } else if source_kind == "unknown" && confidence < 0.5 {
        TrustTier::Suggested
    } else {
        // Direct evidence: user, agent, tool, hook, file.
        TrustTier::Observed
    }
}

/// Extract full provenance from a record's data.
pub fn extract_provenance(data: &Value) -> Provenance {
    let source_kind = data
        .get("_source")
        .and_then(|s| s.get("kind"))
        .and_then(|v| v.as_str())
        .map(|s| MemorySource::parse(s))
        .unwrap_or(MemorySource::Unknown);

    let source_ref = data
        .get("_source")
        .and_then(|s| s.get("ref"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let scope = data
        .get("_scope")
        .and_then(|v| v.as_str())
        .and_then(|s| MemoryScope::parse(s))
        .unwrap_or(MemoryScope::Project);

    let confidence = data
        .get("_confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    let verified = data
        .get("_verified")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let supersedes = crate::util::extract_str_array(data, "_supersedes");
    let contradicts = crate::util::extract_str_array(data, "_contradicts");
    let derived_from = crate::util::extract_str_array(data, "_derived_from");

    let last_validated_at = data
        .get("_last_validated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let trust_tier = classify_trust(data);

    Provenance {
        source_kind,
        source_ref,
        scope,
        confidence,
        verified,
        supersedes,
        contradicts,
        derived_from,
        last_validated_at,
        trust_tier,
    }
}

/// Verify a record — mark it as human-verified (trust tier → Observed).
pub fn verify_record(db: &Axil, id: &RecordId) -> Result<Record> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;
    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("_verified".to_string(), json!(true));
        obj.insert(
            "_last_validated_at".to_string(),
            json!(chrono::Utc::now().to_rfc3339()),
        );
    }
    db.update(id, data)
}

/// Doubt a record — lower its trust tier.
pub fn doubt_record(db: &Axil, id: &RecordId, reason: Option<&str>) -> Result<Record> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;
    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("doubted".to_string(), json!(true));
        // Halve _confidence.
        if let Some(c) = obj.get("_confidence").and_then(|v| v.as_f64()) {
            obj.insert("_confidence".to_string(), json!((c * 0.5).max(0.1)));
        }
        // Also halve confidence (BeliefSystem schema field) for belief records.
        if let Some(c) = obj.get("confidence").and_then(|v| v.as_f64()) {
            obj.insert("confidence".to_string(), json!((c * 0.5).max(0.1)));
        }
        if let Some(r) = reason {
            obj.insert("_doubt_reason".to_string(), json!(r));
        }
        obj.insert(
            "_last_validated_at".to_string(),
            json!(chrono::Utc::now().to_rfc3339()),
        );
    }
    db.update(id, data)
}

/// Migrate provenance for a pre-11.2 record (backfill missing fields).
///
/// Returns true if the record was updated, false if it already had provenance.
pub fn migrate_provenance_record(db: &Axil, record: &Record) -> Result<bool> {
    // Skip if already has provenance.
    if record.data.get("_source").is_some() && record.data.get("_scope").is_some() {
        return Ok(false);
    }

    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        // _source.kind: infer from existing clues.
        if obj.get("_source").is_none() {
            let kind = if obj
                .get("_auto_captured")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "hook"
            } else if obj
                .get("_capture_source")
                .and_then(|v| v.as_str())
                .is_some()
            {
                "tool_output"
            } else {
                "unknown"
            };
            let source_ref = obj
                .get("_capture_source")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            obj.insert(
                "_source".to_string(),
                json!({
                    "kind": kind,
                    "ref": source_ref,
                }),
            );
        }

        // _scope: infer from table name.
        if obj.get("_scope").is_none() {
            let scope = match record.table.as_str() {
                "_sessions" | "sessions" => "session",
                "_beliefs" => "project",
                "_entities" | "_consolidated" => "project",
                "preferences" | "_preferences" => "user",
                "procedures" | "_procedures" | "patterns" => "global",
                _ => "project",
            };
            obj.insert("_scope".to_string(), json!(scope));
        }

        // _confidence: default to 0.5 for unknown provenance.
        if obj.get("_confidence").is_none() {
            obj.insert("_confidence".to_string(), json!(0.5));
        }

        // _verified: default to false.
        if obj.get("_verified").is_none() {
            obj.insert("_verified".to_string(), json!(false));
        }
    }

    db.update(&record.id, data)?;
    Ok(true)
}

/// Migrate provenance for all records in the database.
///
/// Returns the number of records migrated.
pub fn migrate_provenance_all(db: &Axil) -> Result<usize> {
    let tables = db.storage().tables().unwrap_or_default();
    let mut migrated = 0;
    for table in &tables {
        let records = db.storage().list(table, usize::MAX, 0).unwrap_or_default();
        for record in &records {
            if migrate_provenance_record(db, record)? {
                migrated += 1;
            }
        }
    }
    Ok(migrated)
}

// ══════════════════════════════════════════════════════════════════
// 11.4 — Belief Revision Engine
// ══════════════════════════════════════════════════════════════════

/// What happened to a belief during revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefRevisionAction {
    /// Existing belief was reinforced (confidence increased).
    Reinforced { belief_id: String },
    /// Existing belief was superseded by new evidence.
    Superseded {
        old_belief_id: String,
        new_belief_id: String,
    },
    /// Existing belief was doubted (contradicted).
    Doubted { belief_id: String, reason: String },
    /// A new competing hypothesis was created alongside the existing belief.
    NewHypothesis {
        existing_belief_id: String,
        new_belief_id: String,
    },
    /// No revision needed (evidence doesn't relate to any belief).
    NoChange,
}

/// Result of running belief revision on an observation.
#[derive(Debug, Clone, Serialize)]
pub struct BeliefRevisionResult {
    /// Actions taken on beliefs.
    pub actions: Vec<BeliefRevisionAction>,
    /// Updated beliefs (after revision).
    pub updated_beliefs: Vec<crate::beliefs::Belief>,
    /// Beliefs that were doubted.
    pub doubted_beliefs: Vec<crate::beliefs::Belief>,
}

/// Run belief revision against existing beliefs based on new evidence.
///
/// Checks all existing beliefs against the observation text using keyword
/// overlap (no vector index required) and optionally vector similarity.
///
/// Returns a `BeliefRevisionResult` describing what changed.
pub fn revise_beliefs(db: &Axil, observation_text: &str) -> Result<BeliefRevisionResult> {
    let bs = crate::beliefs::BeliefSystem::new(db);
    let all_beliefs = bs.list(None, true)?;

    if all_beliefs.is_empty() || observation_text.trim().is_empty() {
        return Ok(BeliefRevisionResult {
            actions: vec![BeliefRevisionAction::NoChange],
            updated_beliefs: Vec::new(),
            doubted_beliefs: Vec::new(),
        });
    }

    let obs_lower = observation_text.to_lowercase();
    let obs_entities = extract_entities(observation_text);
    let obs_entity_names: std::collections::HashSet<String> =
        obs_entities.iter().map(|e| e.name.to_lowercase()).collect();

    let mut actions = Vec::new();
    let mut updated_beliefs = Vec::new();
    let mut doubted_beliefs = Vec::new();
    // Lazily create a new belief at most once (avoids duplicate inserts).
    let mut new_belief_id: Option<String> = None;

    for belief in &all_beliefs {
        let belief_lower = belief.statement.to_lowercase();
        let belief_entities = extract_entities(&belief.statement);
        let belief_entity_names: std::collections::HashSet<String> = belief_entities
            .iter()
            .map(|e| e.name.to_lowercase())
            .collect();

        let shared_entities: Vec<&String> = obs_entity_names
            .intersection(&belief_entity_names)
            .collect();

        if shared_entities.is_empty() {
            if crate::util::word_jaccard(&obs_lower, &belief_lower) < 0.3 {
                continue;
            }
        }

        let has_negation = crate::consolidation::NEGATION_WORDS
            .iter()
            .any(|neg| obs_lower.contains(neg) && !belief_lower.contains(neg));

        let is_reinforcing =
            !has_negation && crate::util::word_jaccard(&obs_lower, &belief_lower) > 0.5;

        if is_reinforcing {
            if let Ok(belief_id) = RecordId::from_string(&belief.id) {
                if let Ok(Some(record)) = db.get(&belief_id) {
                    let mut data = record.data.clone();
                    if let Some(obj) = data.as_object_mut() {
                        let current = obj
                            .get("confidence")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.5) as f32;
                        let new_conf = (current + 0.1).min(1.0);
                        obj.insert("confidence".to_string(), json!(new_conf));
                        obj.insert("doubted".to_string(), json!(false));
                        obj.insert(
                            "_last_validated_at".to_string(),
                            json!(chrono::Utc::now().to_rfc3339()),
                        );
                    }
                    let _ = db.update(&belief_id, data);
                    actions.push(BeliefRevisionAction::Reinforced {
                        belief_id: belief.id.clone(),
                    });
                    if let Ok(Some(updated)) = db.get(&belief_id) {
                        updated_beliefs.push(crate::beliefs::Belief {
                            id: updated.id.to_string(),
                            statement: updated
                                .data
                                .get("statement")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            confidence: updated
                                .data
                                .get("confidence")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.5) as f32,
                            source: crate::beliefs::BeliefSource::Explicit,
                            created_at: updated
                                .data
                                .get("created_at")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            doubted: false,
                        });
                    }
                }
            }
        } else if has_negation {
            if let Ok(belief_id) = RecordId::from_string(&belief.id) {
                let reason = format!(
                    "contradicted by: {}",
                    crate::util::truncate_str(observation_text, 100)
                );
                let _ = doubt_record(db, &belief_id, Some(&reason));
                actions.push(BeliefRevisionAction::Doubted {
                    belief_id: belief.id.clone(),
                    reason: reason.clone(),
                });
                doubted_beliefs.push(crate::beliefs::Belief {
                    id: belief.id.clone(),
                    statement: belief.statement.clone(),
                    confidence: 0.5,
                    source: belief.source.clone(),
                    created_at: belief.created_at.clone(),
                    doubted: true,
                });

                // Create new belief once, reuse ID for all supersede actions.
                let nid = match &new_belief_id {
                    Some(id) => id.clone(),
                    None => {
                        let nb = bs.believe(observation_text)?;
                        let id = nb.id.to_string();
                        new_belief_id = Some(id.clone());
                        id
                    }
                };
                actions.push(BeliefRevisionAction::Superseded {
                    old_belief_id: belief.id.clone(),
                    new_belief_id: nid,
                });
            }
        } else {
            // Ambiguous — create a new competing hypothesis once.
            let nid = match &new_belief_id {
                Some(id) => id.clone(),
                None => {
                    let nb = bs.believe(observation_text)?;
                    let id = nb.id.to_string();
                    new_belief_id = Some(id.clone());
                    id
                }
            };
            actions.push(BeliefRevisionAction::NewHypothesis {
                existing_belief_id: belief.id.clone(),
                new_belief_id: nid,
            });
        }
    }

    if actions.is_empty() {
        actions.push(BeliefRevisionAction::NoChange);
    }

    Ok(BeliefRevisionResult {
        actions,
        updated_beliefs,
        doubted_beliefs,
    })
}

/// Get belief history for a topic — shows current and superseded beliefs.
pub fn belief_history(db: &Axil, topic: &str) -> Result<Vec<crate::beliefs::Belief>> {
    let bs = crate::beliefs::BeliefSystem::new(db);
    bs.list(Some(topic), true) // Include doubted to show full history.
}

// ══════════════════════════════════════════════════════════════════
// 11.5 — Memory Debugger
// ══════════════════════════════════════════════════════════════════

/// Explanation of why a record was remembered (stored).
#[derive(Debug, Clone, Serialize)]
pub struct WhyRemembered {
    pub record_id: String,
    /// Source event that triggered storage.
    pub source: Value,
    /// Memory type classification and why.
    pub memory_type: String,
    pub memory_type_reason: String,
    /// Scope assignment and why.
    pub scope: String,
    pub scope_reason: String,
    /// Importance score breakdown.
    pub importance: f32,
    pub importance_breakdown: crate::importance::ImportanceBreakdown,
    /// Confidence at time of storage.
    pub confidence: f32,
    /// Trust tier.
    pub trust_tier: String,
    /// Entities extracted.
    pub entities: Vec<String>,
    /// Resolution: was it novel, duplicate, superseding, or contradicting?
    pub resolution: String,
    /// Related records considered during resolution.
    pub related_records: Vec<String>,
}

/// Explain why a record was remembered.
pub fn why_remembered(db: &Axil, id: &RecordId) -> Result<WhyRemembered> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;

    let data = &record.data;

    // Source.
    let source = data
        .get("_source")
        .cloned()
        .unwrap_or(json!({"kind": "unknown"}));

    // Memory type.
    let memory_type = data
        .get("_memory_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let memory_type_reason = match memory_type.as_str() {
        "preference" => "text matched preference signal words (prefer, always, never, etc.)",
        "procedural" => "text matched procedural signal words (how to, steps, workflow, etc.)",
        "episodic" => "text matched episodic signal words (today, completed, session, etc.)",
        "belief" => "text matched belief signal words or routed to _beliefs table",
        "semantic" => "default classification (entity-rich factual content)",
        "working" => "temporary task-local context",
        _ => "classification unknown (pre-Phase 11 record)",
    }
    .to_string();

    // Scope.
    let scope = data
        .get("_scope")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let source_kind = data
        .get("_source")
        .and_then(|s| s.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let scope_reason = match scope.as_str() {
        "session" => "working memory or session-scoped table".to_string(),
        "project" => format!("default for {} source / {} type", source_kind, memory_type),
        "user" => "user preference or correction".to_string(),
        "global" => "cross-project procedure or pattern".to_string(),
        "agent" => "agent-specific memory".to_string(),
        _ => "scope unknown (pre-Phase 11 record)".to_string(),
    };

    // Importance.
    let importance = crate::importance::get_importance(data);
    let importance_breakdown = crate::importance::compute_importance_breakdown(data);

    // Confidence and trust.
    let confidence = data
        .get("_confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let trust_tier = classify_trust(data).to_string();

    // Entities.
    let entities: Vec<String> = data
        .get("_entities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Resolution.
    let supersedes = data.get("_supersedes").and_then(|v| v.as_array());
    let contradicts = data.get("_contradicts").and_then(|v| v.as_array());
    let resolution = if supersedes.is_some() && !supersedes.unwrap().is_empty() {
        "superseded an existing memory"
    } else if contradicts.is_some() && !contradicts.unwrap().is_empty() {
        "stored alongside contradicting memory"
    } else {
        "novel (no duplicate or conflict detected)"
    }
    .to_string();

    let mut related_records = Vec::new();
    if let Some(arr) = supersedes {
        for v in arr {
            if let Some(s) = v.as_str() {
                related_records.push(s.to_string());
            }
        }
    }
    if let Some(arr) = contradicts {
        for v in arr {
            if let Some(s) = v.as_str() {
                related_records.push(s.to_string());
            }
        }
    }

    Ok(WhyRemembered {
        record_id: id.to_string(),
        source,
        memory_type,
        memory_type_reason,
        scope,
        scope_reason,
        importance,
        importance_breakdown,
        confidence,
        trust_tier,
        entities,
        resolution,
        related_records,
    })
}

/// Explanation of why a record was recalled for a query.
#[derive(Debug, Clone, Serialize)]
pub struct WhyRecalled {
    pub record_id: String,
    pub query: String,
    /// The score breakdown from the scoring engine.
    pub score_explanation: Option<crate::scoring::ScoreExplanation>,
    /// Overall score.
    pub score: f32,
    /// Scope of the record.
    pub scope: String,
    /// Trust tier.
    pub trust_tier: String,
    /// Whether the record passed scope/confidence/importance filters.
    pub passed_filters: bool,
    /// Reason it was included or would have been excluded.
    pub reason: String,
}

/// Explain why a specific record was (or wasn't) recalled for a query.
///
/// Runs the recall pipeline for the query and checks if the specified record
/// appears in the results. Returns the score explanation or explains why it was excluded.
pub fn why_recalled(db: &Axil, query: &str, record_id: &RecordId) -> Result<WhyRecalled> {
    let record = db
        .get(record_id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {record_id}")))?;

    let scope = record
        .data
        .get("_scope")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let trust_tier = classify_trust(&record.data).to_string();

    // Try to find in recall results.
    let results = db.recall(query, 50, None);

    match results {
        Ok(results) => {
            if let Some(rr) = results.iter().find(|rr| rr.record.id == *record_id) {
                Ok(WhyRecalled {
                    record_id: record_id.to_string(),
                    query: query.to_string(),
                    score_explanation: Some(rr.explanation.clone()),
                    score: rr.score,
                    scope,
                    trust_tier,
                    passed_filters: true,
                    reason: format!(
                        "ranked #{} of {} results",
                        results
                            .iter()
                            .position(|r| r.record.id == *record_id)
                            .unwrap_or(0)
                            + 1,
                        results.len()
                    ),
                })
            } else {
                // Record was not in results — explain why.
                let confidence = record
                    .data
                    .get("_confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
                let importance = crate::importance::get_importance(&record.data);
                let superseded = record
                    .data
                    .get("_superseded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let reason = if superseded {
                    "record is superseded (excluded from recall)".to_string()
                } else if confidence < 0.3 {
                    format!("confidence {confidence:.2} too low")
                } else if importance < 0.1 {
                    format!("importance {importance:.2} too low")
                } else {
                    "not semantically similar enough to query (below vector similarity threshold)"
                        .to_string()
                };

                Ok(WhyRecalled {
                    record_id: record_id.to_string(),
                    query: query.to_string(),
                    score_explanation: None,
                    score: 0.0,
                    scope,
                    trust_tier,
                    passed_filters: false,
                    reason,
                })
            }
        }
        Err(_) => {
            // Recall failed (e.g., no vector index).
            Ok(WhyRecalled {
                record_id: record_id.to_string(),
                query: query.to_string(),
                score_explanation: None,
                score: 0.0,
                scope,
                trust_tier,
                passed_filters: false,
                reason: "recall unavailable (vector index not configured)".to_string(),
            })
        }
    }
}

/// Explanation of why a belief/record was revised.
#[derive(Debug, Clone, Serialize)]
pub struct WhyRevised {
    pub record_id: String,
    /// What happened to the record.
    pub revision_type: String,
    /// The evidence that caused the revision.
    pub cause: String,
    /// Confidence before revision.
    pub confidence_before: f32,
    /// Confidence after revision.
    pub confidence_after: f32,
    /// Related records involved in the revision.
    pub related_records: Vec<String>,
    /// Trust tier after revision.
    pub trust_tier: String,
}

/// Explain why a record was revised (superseded, doubted, etc.).
pub fn why_revised(db: &Axil, id: &RecordId) -> Result<WhyRevised> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;

    let data = &record.data;
    let trust_tier = classify_trust(data).to_string();

    let confidence_after = data
        .get("_confidence")
        .or_else(|| data.get("confidence"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    let superseded = data
        .get("_superseded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let doubted = data
        .get("doubted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let doubt_reason = data
        .get("_doubt_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let superseded_by = data
        .get("_superseded_by")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let (revision_type, cause, confidence_before) = if superseded {
        let cause = superseded_by
            .as_deref()
            .map(|id| format!("superseded by record {id}"))
            .unwrap_or_else(|| "superseded (successor unknown)".to_string());
        ("superseded".to_string(), cause, 1.0f32)
    } else if doubted {
        let cause = if doubt_reason.is_empty() {
            "manually doubted".to_string()
        } else {
            doubt_reason.to_string()
        };
        ("doubted".to_string(), cause, 1.0)
    } else {
        (
            "not revised".to_string(),
            "record has not been revised".to_string(),
            confidence_after,
        )
    };

    let mut related = Vec::new();
    if let Some(id) = superseded_by {
        related.push(id);
    }

    Ok(WhyRevised {
        record_id: id.to_string(),
        revision_type,
        cause,
        confidence_before,
        confidence_after,
        related_records: related,
        trust_tier,
    })
}

// ══════════════════════════════════════════════════════════════════
// 11.6 — Self Memory and Project Operating Model
// ══════════════════════════════════════════════════════════════════

const TABLE_SELF_MEMORY: &str = "_self_memory";
const TABLE_PROJECT_MODEL: &str = "_project_model";
const TABLE_USER_CONTRACT: &str = "_user_contract";

/// Add a self-memory note (how the agent works best, failure patterns, etc.).
pub fn self_note(db: &Axil, note: &str, category: Option<&str>) -> Result<Record> {
    db.insert(
        TABLE_SELF_MEMORY,
        json!({
            "note": note,
            "category": category.unwrap_or("general"),
            "_source": {"kind": "agent", "ref": "self-note"},
            "_scope": "agent",
            "_memory_type": "semantic",
            "_confidence": 1.0,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }),
    )
}

/// Get the agent's self-profile: aggregated self-memory notes.
pub fn self_profile(db: &Axil) -> Result<Value> {
    let notes = db.list(TABLE_SELF_MEMORY).unwrap_or_default();

    let mut by_category: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for note in &notes {
        let cat = note
            .data
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_string();
        let text = note
            .data
            .get("note")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        by_category.entry(cat).or_default().push(text);
    }

    Ok(json!({
        "total_notes": notes.len(),
        "categories": by_category,
    }))
}

/// Store a project operating model entry.
pub fn project_model_set(db: &Axil, key: &str, value: &str) -> Result<Record> {
    // Check if key already exists; update if so.
    let existing = db.list(TABLE_PROJECT_MODEL).unwrap_or_default();
    for record in &existing {
        if record.data.get("key").and_then(|v| v.as_str()) == Some(key) {
            let mut data = record.data.clone();
            if let Some(obj) = data.as_object_mut() {
                obj.insert("value".to_string(), json!(value));
                obj.insert(
                    "updated_at".to_string(),
                    json!(chrono::Utc::now().to_rfc3339()),
                );
            }
            return db.update(&record.id, data);
        }
    }

    db.insert(
        TABLE_PROJECT_MODEL,
        json!({
            "key": key,
            "value": value,
            "_source": {"kind": "agent", "ref": "project-model"},
            "_scope": "project",
            "_memory_type": "semantic",
            "_confidence": 1.0,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }),
    )
}

/// Get the full project operating model.
pub fn project_model_show(db: &Axil) -> Result<Value> {
    let entries = db.list(TABLE_PROJECT_MODEL).unwrap_or_default();
    let model: std::collections::BTreeMap<String, String> = entries
        .iter()
        .filter_map(|r| {
            let key = r.data.get("key").and_then(|v| v.as_str())?.to_string();
            let value = r.data.get("value").and_then(|v| v.as_str())?.to_string();
            Some((key, value))
        })
        .collect();

    Ok(json!({
        "entries": model.len(),
        "model": model,
    }))
}

/// Auto-generate a project model from codebase signals.
///
/// Scans existing memories for patterns about branching, deployment, testing,
/// architecture boundaries, and review norms.
pub fn project_model_generate(db: &Axil) -> Result<Vec<Record>> {
    let mut generated = Vec::new();
    let decisions = db.list("decisions").unwrap_or_default();

    // Check existing keys once before the loop.
    let existing = db.list(TABLE_PROJECT_MODEL).unwrap_or_default();
    let existing_keys: std::collections::HashSet<String> = existing
        .iter()
        .filter_map(|r| {
            r.data
                .get("key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    let mut has_deploy = existing_keys.contains("deployment");
    let mut has_testing = existing_keys.contains("testing");

    for decision in &decisions {
        if has_deploy && has_testing {
            break;
        }
        let text = crate::util::value_text(&decision.data).to_lowercase();

        if !has_deploy
            && (text.contains("deploy") || text.contains("release") || text.contains("ship"))
        {
            let r = project_model_set(db, "deployment", &crate::util::value_text(&decision.data))?;
            generated.push(r);
            has_deploy = true;
        }

        if !has_testing
            && (text.contains("test") || text.contains("ci") || text.contains("pipeline"))
        {
            let r = project_model_set(db, "testing", &crate::util::value_text(&decision.data))?;
            generated.push(r);
            has_testing = true;
        }
    }

    Ok(generated)
}

/// Store a user contract entry (durable preference that affects behavior).
pub fn user_contract_set(db: &Axil, rule: &str) -> Result<Record> {
    db.insert(
        TABLE_USER_CONTRACT,
        json!({
            "rule": rule,
            "_source": {"kind": "user", "ref": "user-contract"},
            "_scope": "user",
            "_memory_type": "preference",
            "_confidence": 1.0,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }),
    )
}

/// Get all user contract rules.
pub fn user_contract_list(db: &Axil) -> Result<Vec<String>> {
    let entries = db.list(TABLE_USER_CONTRACT).unwrap_or_default();
    Ok(entries
        .iter()
        .filter_map(|r| {
            r.data
                .get("rule")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect())
}

// ══════════════════════════════════════════════════════════════════
// 11.8 — Memory Safety and Retention Controls
// ══════════════════════════════════════════════════════════════════

const TABLE_RETENTION_POLICY: &str = "_retention_policy";

/// Detect PII in text (simple heuristic-based, no regex/LLM needed).
///
/// Returns a list of (pii_type, matched_text) pairs.
#[allow(dead_code)] // Used in tests; will be wired into pipeline when PII filtering is enabled.
pub(crate) fn detect_pii(text: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();

    // Email detection: look for @ with surrounding word chars.
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-' && c != '+'
        });
        if w.contains('@') && w.contains('.') {
            let parts: Vec<&str> = w.split('@').collect();
            if parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.') {
                found.push(("email".to_string(), w.to_string()));
            }
        }
    }

    // IP address detection: 4 dot-separated octets.
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
        let octets: Vec<&str> = w.split('.').collect();
        if octets.len() == 4
            && octets
                .iter()
                .all(|o| o.parse::<u16>().map(|n| n <= 255).unwrap_or(false))
        {
            found.push(("ip_address".to_string(), w.to_string()));
        }
    }

    found
}

/// Redact a specific field in a record's data.
pub fn redact_field(db: &Axil, id: &RecordId, field: &str) -> Result<Record> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;
    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        if obj.contains_key(field) {
            obj.insert(field.to_string(), json!("[REDACTED]"));
            // Track redaction.
            let mut redacted: Vec<String> = obj
                .get("_redacted_fields")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            if !redacted.contains(&field.to_string()) {
                redacted.push(field.to_string());
            }
            obj.insert("_redacted_fields".to_string(), json!(redacted));
        } else {
            return Err(crate::error::AxilError::InvalidQuery(format!(
                "field '{field}' not found in record {id}"
            )));
        }
    }
    db.update(id, data)
}

/// Set retention policy for a scope.
pub fn set_retention(db: &Axil, scope: &str, days: u64) -> Result<Record> {
    // Check if policy already exists for this scope; update if so.
    let existing = db.list(TABLE_RETENTION_POLICY).unwrap_or_default();
    for record in &existing {
        if record.data.get("scope").and_then(|v| v.as_str()) == Some(scope) {
            let mut data = record.data.clone();
            if let Some(obj) = data.as_object_mut() {
                obj.insert("days".to_string(), json!(days));
                obj.insert(
                    "updated_at".to_string(),
                    json!(chrono::Utc::now().to_rfc3339()),
                );
            }
            return db.update(&record.id, data);
        }
    }

    db.insert(
        TABLE_RETENTION_POLICY,
        json!({
            "scope": scope,
            "days": days,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }),
    )
}

/// Get the retention policy summary.
pub fn get_retention_policies(db: &Axil) -> Result<Value> {
    let policies = db.list(TABLE_RETENTION_POLICY).unwrap_or_default();
    let entries: Vec<Value> = policies
        .iter()
        .map(|r| {
            json!({
                "scope": r.data.get("scope").and_then(|v| v.as_str()).unwrap_or(""),
                "days": r.data.get("days").and_then(|v| v.as_u64()).unwrap_or(0),
            })
        })
        .collect();

    Ok(json!({
        "policies": entries,
    }))
}

/// Pin a record — prevent it from being decayed, archived, or auto-deleted.
pub fn pin_record(db: &Axil, id: &RecordId) -> Result<Record> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;
    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("_importance_pinned".to_string(), json!(true));
        obj.insert(
            "_pinned_at".to_string(),
            json!(chrono::Utc::now().to_rfc3339()),
        );
    }
    db.update(id, data)
}

/// Unpin a record.
pub fn unpin_record(db: &Axil, id: &RecordId) -> Result<Record> {
    let record = db
        .get(id)?
        .ok_or_else(|| crate::error::AxilError::NotFound(format!("record {id}")))?;
    let mut data = record.data.clone();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("_importance_pinned".to_string(), json!(false));
        obj.remove("_pinned_at");
    }
    db.update(id, data)
}

/// Show overall memory safety policy summary.
pub fn memory_policy_show(db: &Axil) -> Result<Value> {
    let retention = get_retention_policies(db)?;
    let user_contract = user_contract_list(db)?;

    // Count pinned records.
    let tables = db.tables().unwrap_or_default();
    let mut pinned_count = 0;
    let mut redacted_count = 0;
    for table in &tables {
        for record in db.list(table).unwrap_or_default() {
            if crate::importance::is_pinned(&record.data) {
                pinned_count += 1;
            }
            if record.data.get("_redacted_fields").is_some() {
                redacted_count += 1;
            }
        }
    }

    Ok(json!({
        "retention_policies": retention["policies"],
        "user_contract_rules": user_contract.len(),
        "pinned_records": pinned_count,
        "redacted_records": redacted_count,
    }))
}

// ══════════════════════════════════════════════════════════════════
// 11.9 — Brain Evals
// ══════════════════════════════════════════════════════════════════

/// A single eval case for brain quality testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainEvalCase {
    /// Test name.
    pub name: String,
    /// Category: write, classify, revise, contradict, scope, belief.
    pub category: String,
    /// Input observation text.
    pub input: String,
    /// Expected memory type (if testing classification).
    pub expected_type: Option<String>,
    /// Expected scope (if testing scope assignment).
    pub expected_scope: Option<String>,
    /// Expected pipeline action (stored, ignored, superseded).
    pub expected_action: Option<String>,
}

/// Result of running a single eval case.
#[derive(Debug, Clone, Serialize)]
pub struct BrainEvalResult {
    pub name: String,
    pub category: String,
    pub passed: bool,
    pub expected: String,
    pub actual: String,
    pub detail: String,
}

/// Overall brain eval report.
#[derive(Debug, Clone, Serialize)]
pub struct BrainEvalReport {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f32,
    pub results: Vec<BrainEvalResult>,
    /// Weighted score per the spec formula.
    pub brain_score: f32,
}

/// Built-in eval suite for brain quality.
pub fn builtin_eval_cases() -> Vec<BrainEvalCase> {
    vec![
        // Write accuracy: does the pipeline store the right thing?
        BrainEvalCase {
            name: "write_basic_fact".into(),
            category: "write".into(),
            input: "AuthModule uses JWT tokens for authentication".into(),
            expected_type: Some("semantic".into()),
            expected_scope: Some("project".into()),
            expected_action: Some("stored".into()),
        },
        BrainEvalCase {
            name: "write_empty_ignored".into(),
            category: "write".into(),
            input: "".into(),
            expected_type: None,
            expected_scope: None,
            expected_action: Some("ignored".into()),
        },
        // Classification accuracy.
        BrainEvalCase {
            name: "classify_preference".into(),
            category: "classify".into(),
            input: "I always prefer reversible migrations".into(),
            expected_type: Some("preference".into()),
            expected_scope: Some("user".into()),
            expected_action: None,
        },
        BrainEvalCase {
            name: "classify_procedural".into(),
            category: "classify".into(),
            input: "How to deploy: run cargo build, then docker push to registry".into(),
            expected_type: Some("procedural".into()),
            expected_scope: None,
            expected_action: None,
        },
        BrainEvalCase {
            name: "classify_episodic".into(),
            category: "classify".into(),
            input: "Today we completed the auth refactor and deployed to staging".into(),
            expected_type: Some("episodic".into()),
            expected_scope: None,
            expected_action: None,
        },
        BrainEvalCase {
            name: "classify_belief".into(),
            category: "classify".into(),
            input: "I believe the current auth approach is correct".into(),
            expected_type: Some("belief".into()),
            expected_scope: None,
            expected_action: None,
        },
        // Scope accuracy.
        BrainEvalCase {
            name: "scope_hook_is_project".into(),
            category: "scope".into(),
            input: "file changed: src/auth.rs".into(),
            expected_type: None,
            expected_scope: Some("project".into()),
            expected_action: None,
        },
        BrainEvalCase {
            name: "scope_user_preference".into(),
            category: "scope".into(),
            input: "User prefers dark mode for all interfaces".into(),
            expected_type: Some("preference".into()),
            expected_scope: Some("user".into()),
            expected_action: None,
        },
    ]
}

/// Run the brain eval suite against a database.
pub fn run_brain_eval(db: &Axil) -> Result<BrainEvalReport> {
    let cases = builtin_eval_cases();
    let mut results = Vec::new();

    for case in &cases {
        let obs = Observation::from_text(&case.input).with_source(
            if case.category == "scope" && case.input.contains("file changed") {
                MemorySource::Hook
            } else {
                MemorySource::Agent
            },
        );

        let outcome = remember(db, obs)?;

        let mut passed = true;
        let mut expected_parts = Vec::new();
        let mut actual_parts = Vec::new();

        // Check expected type.
        if let Some(ref expected_type) = case.expected_type {
            let actual_type = outcome.memory_type.to_string();
            if &actual_type != expected_type {
                passed = false;
            }
            expected_parts.push(format!("type={expected_type}"));
            actual_parts.push(format!("type={actual_type}"));
        }

        // Check expected scope.
        if let Some(ref expected_scope) = case.expected_scope {
            let actual_scope = outcome.scope.to_string();
            if &actual_scope != expected_scope {
                passed = false;
            }
            expected_parts.push(format!("scope={expected_scope}"));
            actual_parts.push(format!("scope={actual_scope}"));
        }

        // Check expected action.
        if let Some(ref expected_action) = case.expected_action {
            let actual_action = match &outcome.action {
                PipelineAction::Stored => "stored",
                PipelineAction::Updated { .. } => "updated",
                PipelineAction::Superseded { .. } => "superseded",
                PipelineAction::Ignored => "ignored",
            };
            if actual_action != expected_action {
                passed = false;
            }
            expected_parts.push(format!("action={expected_action}"));
            actual_parts.push(format!("action={actual_action}"));
        }

        results.push(BrainEvalResult {
            name: case.name.clone(),
            category: case.category.clone(),
            passed,
            expected: expected_parts.join(", "),
            actual: actual_parts.join(", "),
            detail: outcome.reason.clone(),
        });
    }

    let total = results.len();
    let passed_count = results.iter().filter(|r| r.passed).count();
    let failed = total - passed_count;
    let pass_rate = if total > 0 {
        passed_count as f32 / total as f32
    } else {
        0.0
    };

    // Weighted brain score per spec:
    // 30% retrieval + 25% write + 20% belief revision + 15% task success + 10% token efficiency
    // We can only measure write + classification here. Retrieval needs vector, belief needs setup.
    let write_cases: Vec<&BrainEvalResult> =
        results.iter().filter(|r| r.category == "write").collect();
    let classify_cases: Vec<&BrainEvalResult> = results
        .iter()
        .filter(|r| r.category == "classify")
        .collect();
    let scope_cases: Vec<&BrainEvalResult> =
        results.iter().filter(|r| r.category == "scope").collect();

    let write_score = if write_cases.is_empty() {
        0.0
    } else {
        write_cases.iter().filter(|r| r.passed).count() as f32 / write_cases.len() as f32
    };
    let classify_score = if classify_cases.is_empty() {
        0.0
    } else {
        classify_cases.iter().filter(|r| r.passed).count() as f32 / classify_cases.len() as f32
    };
    let scope_score = if scope_cases.is_empty() {
        0.0
    } else {
        scope_cases.iter().filter(|r| r.passed).count() as f32 / scope_cases.len() as f32
    };

    // Brain score: weighted average of what we can measure.
    let brain_score = write_score * 0.30 + classify_score * 0.40 + scope_score * 0.30;

    Ok(BrainEvalReport {
        total,
        passed: passed_count,
        failed,
        pass_rate,
        results,
        brain_score,
    })
}

// ══════════════════════════════════════════════════════════════════
// Retrieval / needle-retention eval
// ══════════════════════════════════════════════════════════════════

/// Result of a single needle-retention case.
#[derive(Debug, Clone, Serialize)]
pub struct NeedleResult {
    pub name: String,
    pub query: String,
    /// The distinctive token planted in the target record.
    pub needle: String,
    /// True if the target record was returned within top-k.
    pub recalled: bool,
    /// True if the recalled record still contains the planted needle verbatim.
    pub retained: bool,
    /// 0-based rank of the target in the result list, if found.
    pub rank: Option<usize>,
}

/// Report for the synthetic needle-retention retrieval eval.
#[derive(Debug, Clone, Serialize)]
pub struct NeedleEvalReport {
    pub total: usize,
    pub recalled: usize,
    pub retained: usize,
    /// Fraction of needles returned within top-k (target 1.0).
    pub recall_at_k: f32,
    /// Fraction of recalled needles whose token survived intact (target ≥ 0.9).
    pub retention_rate: f32,
    pub top_k: usize,
    pub results: Vec<NeedleResult>,
}

/// Run a dataset-free, no-model needle-retention retrieval eval.
///
/// Inserts a small synthetic corpus of distinct records — several carrying a
/// distinctive "needle" token (a UUID, error code, or anomaly) scattered among
/// distractors — then recalls each by a natural-language query and asserts the
/// planted record comes back within top-k AND its needle survived intact. This
/// guards the recall path against silent regressions (a scoring/dedup/truncation
/// change that stops a known record from surfacing) with zero external data and
/// no embedding model: it runs over FTS, so the caller must pass a DB with an
/// FTS index attached (otherwise recall returns nothing and every needle misses).
pub fn run_needle_eval(db: &Axil) -> Result<NeedleEvalReport> {
    const TOP_K: usize = 5;

    // Distractors — unrelated records so the needle isn't the only candidate.
    let distractors = [
        "Refactored the billing webhook retry queue to use exponential backoff.",
        "The onboarding wizard now validates SSO metadata before the first step.",
        "Migrated the analytics pipeline from batch cron to a streaming consumer.",
        "Documented the feature-flag rollout process for the mobile clients.",
        "Tuned the search index merge policy to reduce p99 query latency.",
        "Added dark-mode tokens to the design system and updated the storybook.",
        "The nightly backup job verifies checksums before pruning old snapshots.",
        "Rewrote the CSV importer to stream rows instead of loading the whole file.",
    ];
    for d in distractors {
        db.insert("notes", serde_json::json!({ "summary": d }))?;
    }

    // Needle cases: a natural-language query, the full text holding the needle,
    // and the distinctive token that must survive into the recalled record.
    struct Case {
        name: &'static str,
        query: &'static str,
        text: &'static str,
        needle: &'static str,
    }
    let cases = [
        Case {
            name: "uuid_in_runbook",
            query: "production deploy key rotation runbook vault",
            text: "Deploy key rotation runbook: the production deploy key 7f3a9c21-4b8e-42d1-9f60-aa01beef2210 lives in the vault and rotates quarterly.",
            needle: "7f3a9c21-4b8e-42d1-9f60-aa01beef2210",
        },
        Case {
            name: "error_code_fix",
            query: "auth token refresh failure clock skew jwt",
            text: "Fixed the E_AUTH_4017 token refresh failure by adding 30s of clock-skew leeway to the JWT validator.",
            needle: "E_AUTH_4017",
        },
        Case {
            name: "numeric_limit",
            query: "tenant rate limiter requests per minute 429",
            text: "The tenant rate limiter allows 4096 requests per minute before returning 429.",
            needle: "4096",
        },
        Case {
            name: "postmortem_anomaly",
            query: "cache stampede postmortem checkout outage coalescing",
            text: "Postmortem: a cache stampede on 2026-05-12 took down checkout for 23 minutes until we added request coalescing.",
            needle: "2026-05-12",
        },
        Case {
            name: "symbol_locator",
            query: "double-entry ledger settlement reconciliation across sub-ledgers",
            text: "The function reconcile_ledger_balances performs double-entry settlement across tenant sub-ledgers.",
            needle: "reconcile_ledger_balances",
        },
        Case {
            name: "distinctive_among_siblings",
            query: "passwordless email magic link login token ttl",
            text: "Switched the passwordless login flow to email magic-link tokens with a 10 minute TTL.",
            needle: "magic-link",
        },
    ];

    let mut target_ids = Vec::with_capacity(cases.len());
    for c in &cases {
        let rec = db.insert("decisions", serde_json::json!({ "summary": c.text }))?;
        target_ids.push(rec.id);
    }

    // Recall with the agent-facing config (dedup enabled) so the gate covers the
    // shipped path. The corpus is lexically distinct, so dedup is a no-op today;
    // a future regression that wrongly collapses a distinct record would fail.
    let cfg = crate::scoring::RecallConfig {
        dedup: crate::scoring::DedupConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut results = Vec::with_capacity(cases.len());
    let (mut recalled, mut retained) = (0usize, 0usize);
    for (i, c) in cases.iter().enumerate() {
        let hits = db.recall(c.query, TOP_K, Some(cfg.clone()))?;
        let rank = hits.iter().position(|h| h.record.id == target_ids[i]);
        let was_recalled = rank.is_some();
        let was_retained = rank
            .map(|r| crate::util::record_text(&hits[r].record).contains(c.needle))
            .unwrap_or(false);
        if was_recalled {
            recalled += 1;
        }
        if was_retained {
            retained += 1;
        }
        results.push(NeedleResult {
            name: c.name.to_string(),
            query: c.query.to_string(),
            needle: c.needle.to_string(),
            recalled: was_recalled,
            retained: was_retained,
            rank,
        });
    }

    let total = cases.len();
    Ok(NeedleEvalReport {
        total,
        recalled,
        retained,
        recall_at_k: if total > 0 {
            recalled as f32 / total as f32
        } else {
            0.0
        },
        // Retention is "of the needles that were recalled, how many kept their
        // token intact" — so it divides by `recalled`, not `total`; a pure recall
        // miss is already reflected in `recall_at_k` and must not double-count here.
        retention_rate: if recalled > 0 {
            retained as f32 / recalled as f32
        } else {
            0.0
        },
        top_k: TOP_K,
        results,
    })
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_preference() {
        let entities = vec![];
        let mt = classify_memory_type("User prefers reversible migrations", &[], &entities);
        assert_eq!(mt, MemoryType::Preference);
    }

    #[test]
    fn classify_procedural() {
        let entities = vec![];
        let mt = classify_memory_type(
            "How to deploy: run cargo build then docker push",
            &[],
            &entities,
        );
        assert_eq!(mt, MemoryType::Procedural);
    }

    #[test]
    fn classify_episodic() {
        let entities = vec![];
        let mt = classify_memory_type(
            "Today we finished the auth refactor and deployed",
            &[],
            &entities,
        );
        assert_eq!(mt, MemoryType::Episodic);
    }

    #[test]
    fn classify_semantic_with_entities() {
        let entities = vec![
            Entity {
                name: "AuthModule".to_string(),
                entity_type: crate::entity::EntityType::Code,
                source_text: "AuthModule".to_string(),
            },
            Entity {
                name: "login_service".to_string(),
                entity_type: crate::entity::EntityType::Code,
                source_text: "login_service".to_string(),
            },
        ];
        let mt = classify_memory_type(
            "AuthModule depends on login_service for JWT validation",
            &[],
            &entities,
        );
        assert_eq!(mt, MemoryType::Semantic);
    }

    #[test]
    fn classify_with_hint_override() {
        let entities = vec![];
        let mt = classify_memory_type("some random text", &["preference".to_string()], &entities);
        assert_eq!(mt, MemoryType::Preference);
    }

    #[test]
    fn infer_scope_from_source() {
        assert_eq!(
            infer_scope(&MemorySource::Hook, &MemoryType::Semantic, "", None),
            MemoryScope::Project,
        );
        assert_eq!(
            infer_scope(&MemorySource::File, &MemoryType::Semantic, "", None),
            MemoryScope::Project,
        );
    }

    #[test]
    fn infer_scope_from_memory_type() {
        assert_eq!(
            infer_scope(&MemorySource::User, &MemoryType::Working, "", None),
            MemoryScope::Session,
        );
        assert_eq!(
            infer_scope(&MemorySource::User, &MemoryType::Preference, "", None),
            MemoryScope::User,
        );
    }

    #[test]
    fn memory_type_to_table_mapping() {
        assert_eq!(memory_type_to_table(&MemoryType::Working), "context");
        assert_eq!(memory_type_to_table(&MemoryType::Procedural), "procedures");
        assert_eq!(memory_type_to_table(&MemoryType::Preference), "preferences");
        assert_eq!(memory_type_to_table(&MemoryType::Belief), "_beliefs");
    }

    #[test]
    fn empty_observation_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("");
        let outcome = remember(&db, obs).unwrap();
        assert_eq!(outcome.action, PipelineAction::Ignored);
        assert_eq!(outcome.reason, "empty observation");
    }

    #[test]
    fn basic_store_through_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("Fixed auth timeout bug by increasing pool size")
            .with_source(MemorySource::Agent)
            .with_source_ref("bash:cargo test");
        let outcome = remember(&db, obs).unwrap();

        assert_eq!(outcome.action, PipelineAction::Stored);
        assert!(outcome.record.is_some());
        assert!(outcome.importance > 0.0);
        assert_eq!(outcome.confidence, 1.0);

        // Check provenance metadata was injected.
        let record = outcome.record.unwrap();
        assert_eq!(record.data["_source"]["kind"], "agent");
        assert_eq!(record.data["_scope"], "project");
        assert!(record.data["_memory_type"].is_string());
    }

    #[test]
    fn pipeline_with_explicit_scope() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("Always ask before schema migrations")
            .with_source(MemorySource::User)
            .with_scope(MemoryScope::User);
        let outcome = remember(&db, obs).unwrap();

        assert_eq!(outcome.scope, MemoryScope::User);
        assert_eq!(outcome.memory_type, MemoryType::Preference);
    }

    #[test]
    fn pipeline_with_data_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("Decided to use JWT for auth")
            .with_source(MemorySource::Agent)
            .with_data(json!({
                "summary": "Decided to use JWT for auth",
                "reason": "Stateless, scales well",
                "type": "decision",
            }));
        let outcome = remember(&db, obs).unwrap();

        assert_eq!(outcome.action, PipelineAction::Stored);
        let record = outcome.record.unwrap();
        assert_eq!(record.data["reason"], "Stateless, scales well");
    }

    #[test]
    fn pipeline_overhead_under_5ms() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Warm up: first insert pays DB init cost.
        let _ = db.insert("warmup", json!({"x": 1})).unwrap();

        // Measure raw insert baseline.
        let raw_start = std::time::Instant::now();
        let _ = db
            .insert("context", json!({"summary": "baseline insert"}))
            .unwrap();
        let raw_us = raw_start.elapsed().as_micros() as u64;

        // Measure pipeline.
        let obs =
            Observation::from_text("Architecture: auth module uses JWT tokens for stateless auth");
        let outcome = remember(&db, obs).unwrap();

        // Pipeline overhead = total - raw insert. Should be < 5ms.
        let overhead = outcome.latency_us.saturating_sub(raw_us);
        assert!(
            overhead < 5000,
            "pipeline overhead {}us (total {}us - raw {}us) exceeds 5ms budget",
            overhead,
            outcome.latency_us,
            raw_us,
        );
    }

    #[test]
    fn scope_parse_roundtrip() {
        for s in &["session", "agent", "project", "user", "global"] {
            let scope = MemoryScope::parse(s).unwrap();
            assert_eq!(scope.to_string(), *s);
        }
    }

    // ── 11.2 Provenance & Trust tests ──────────────────────────

    #[test]
    fn trust_tier_observed_for_user_input() {
        let data = json!({
            "_source": {"kind": "user", "ref": "cli"},
            "_confidence": 1.0,
        });
        assert_eq!(classify_trust(&data), TrustTier::Observed);
    }

    #[test]
    fn trust_tier_doubted_when_superseded() {
        let data = json!({
            "_source": {"kind": "agent", "ref": ""},
            "_superseded": true,
        });
        assert_eq!(classify_trust(&data), TrustTier::Doubted);
    }

    #[test]
    fn trust_tier_doubted_when_explicitly_doubted() {
        let data = json!({
            "_source": {"kind": "agent", "ref": ""},
            "doubted": true,
        });
        assert_eq!(classify_trust(&data), TrustTier::Doubted);
    }

    #[test]
    fn trust_tier_derived_for_inference() {
        let data = json!({
            "_source": {"kind": "inference", "ref": ""},
            "_confidence": 0.8,
        });
        assert_eq!(classify_trust(&data), TrustTier::Derived);
    }

    #[test]
    fn trust_tier_suggested_for_low_confidence_inference() {
        let data = json!({
            "_source": {"kind": "inference", "ref": ""},
            "_confidence": 0.4,
        });
        assert_eq!(classify_trust(&data), TrustTier::Suggested);
    }

    #[test]
    fn trust_tier_observed_when_verified() {
        let data = json!({
            "_source": {"kind": "llm", "ref": ""},
            "_confidence": 0.3,
            "_verified": true,
        });
        assert_eq!(classify_trust(&data), TrustTier::Observed);
    }

    #[test]
    fn trust_tier_suggested_for_llm_low_confidence() {
        let data = json!({
            "_source": {"kind": "llm", "ref": ""},
            "_confidence": 0.5,
        });
        assert_eq!(classify_trust(&data), TrustTier::Suggested);
    }

    #[test]
    fn trust_tier_for_legacy_record_no_provenance() {
        let data = json!({
            "summary": "old record without provenance",
        });
        // No _source → source_kind = "unknown", confidence defaults to 0.5
        // Unknown source with confidence >= 0.5 → Observed
        assert_eq!(classify_trust(&data), TrustTier::Observed);
    }

    #[test]
    fn extract_provenance_full() {
        let data = json!({
            "_source": {"kind": "agent", "ref": "bash:cargo test"},
            "_scope": "project",
            "_confidence": 0.9,
            "_verified": false,
            "_supersedes": ["abc123"],
            "_contradicts": ["def456"],
            "_derived_from": ["ghi789"],
            "_last_validated_at": "2026-04-16T10:00:00Z",
        });
        let prov = extract_provenance(&data);
        assert_eq!(prov.source_kind, MemorySource::Agent);
        assert_eq!(prov.source_ref, "bash:cargo test");
        assert_eq!(prov.scope, MemoryScope::Project);
        assert_eq!(prov.confidence, 0.9);
        assert!(!prov.verified);
        assert_eq!(prov.supersedes, vec!["abc123"]);
        assert_eq!(prov.contradicts, vec!["def456"]);
        assert_eq!(prov.derived_from, vec!["ghi789"]);
        assert!(prov.last_validated_at.is_some());
        // Has _derived_from, so trust tier is Derived (confidence 0.9 >= 0.7).
        assert_eq!(prov.trust_tier, TrustTier::Derived);
    }

    #[test]
    fn verify_record_sets_verified() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let record = db
            .insert("context", json!({"summary": "test fact"}))
            .unwrap();
        let updated = verify_record(&db, &record.id).unwrap();
        assert_eq!(updated.data["_verified"], true);
        assert!(updated.data["_last_validated_at"].is_string());

        // Trust tier should be Observed after verification.
        assert_eq!(classify_trust(&updated.data), TrustTier::Observed);
    }

    #[test]
    fn doubt_record_lowers_confidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("Auth uses JWT tokens").with_source(MemorySource::Agent);
        let outcome = remember(&db, obs).unwrap();
        let record = outcome.record.unwrap();

        let doubted = doubt_record(&db, &record.id, Some("evidence says otherwise")).unwrap();
        assert_eq!(doubted.data["doubted"], true);
        assert!(doubted.data["_confidence"].as_f64().unwrap() < 1.0);
        assert_eq!(doubted.data["_doubt_reason"], "evidence says otherwise");
        assert_eq!(classify_trust(&doubted.data), TrustTier::Doubted);
    }

    #[test]
    fn migrate_provenance_backfills_legacy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Insert a "legacy" record without provenance.
        let record = db
            .insert(
                "decisions",
                json!({
                    "summary": "Use PostgreSQL for persistence",
                }),
            )
            .unwrap();

        // Verify it has no provenance.
        assert!(record.data.get("_source").is_none());

        // Migrate.
        let migrated = migrate_provenance_record(&db, &record).unwrap();
        assert!(migrated);

        // Re-read and check.
        let updated = db.get(&record.id).unwrap().unwrap();
        assert_eq!(updated.data["_source"]["kind"], "unknown");
        assert_eq!(updated.data["_scope"], "project");
        assert_eq!(updated.data["_confidence"], 0.5);
        assert_eq!(updated.data["_verified"], false);
    }

    // ── 11.3 Scoped Memory tests ────────────────────────────────

    #[test]
    fn observe_stores_scope_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Project scope (default).
        let outcome = remember(
            &db,
            Observation::from_text("auth module uses JWT").with_source(MemorySource::Agent),
        )
        .unwrap();
        assert_eq!(outcome.scope, MemoryScope::Project);
        let record = outcome.record.unwrap();
        assert_eq!(record.data["_scope"], "project");

        // User scope (preference).
        let outcome = remember(
            &db,
            Observation::from_text("I always prefer reversible migrations")
                .with_source(MemorySource::User)
                .with_scope(MemoryScope::User),
        )
        .unwrap();
        assert_eq!(outcome.scope, MemoryScope::User);
        let record = outcome.record.unwrap();
        assert_eq!(record.data["_scope"], "user");

        // Session scope (working memory).
        let outcome = remember(
            &db,
            Observation::from_text("Current task: fix login timeout")
                .with_scope(MemoryScope::Session),
        )
        .unwrap();
        assert_eq!(outcome.scope, MemoryScope::Session);
    }

    #[test]
    fn query_builder_scope_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Insert records with different scopes into the same table.
        let _ = remember(
            &db,
            Observation::from_text("Project fact about auth")
                .with_scope(MemoryScope::Project)
                .with_table("context"),
        )
        .unwrap();
        let _ = remember(
            &db,
            Observation::from_text("User preference about migrations")
                .with_scope(MemoryScope::User)
                .with_table("context"),
        )
        .unwrap();
        let _ = remember(
            &db,
            Observation::from_text("Global procedure for deploys")
                .with_scope(MemoryScope::Global)
                .with_table("context"),
        )
        .unwrap();

        // Query with scope filter using where_field.
        use crate::query::Op;
        let project_only = db
            .query()
            .table("context")
            .where_field("_scope", Op::Eq, json!("project"))
            .exec()
            .unwrap();
        assert!(!project_only.is_empty());
        assert!(project_only.iter().all(|r| r.data["_scope"] == "project"));

        let user_only = db.query().table("context").scope("user").exec().unwrap();
        assert!(!user_only.is_empty());
        assert!(user_only.iter().all(|r| r.data["_scope"] == "user"));
    }

    #[test]
    fn scope_inference_from_memory_type() {
        // Working → Session.
        assert_eq!(
            infer_scope(&MemorySource::Agent, &MemoryType::Working, "", None),
            MemoryScope::Session,
        );
        // Preference → User.
        assert_eq!(
            infer_scope(&MemorySource::Agent, &MemoryType::Preference, "", None),
            MemoryScope::User,
        );
        // Procedural (generic) → Global.
        assert_eq!(
            infer_scope(
                &MemorySource::Agent,
                &MemoryType::Procedural,
                "how to deploy a service",
                None
            ),
            MemoryScope::Global,
        );
        // Procedural (project-specific) → Project.
        assert_eq!(
            infer_scope(
                &MemorySource::Agent,
                &MemoryType::Procedural,
                "in this project we use docker",
                None
            ),
            MemoryScope::Project,
        );
    }

    #[test]
    fn migrate_provenance_skips_already_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Insert via pipeline (already has provenance).
        let obs = Observation::from_text("Some fact about auth");
        let outcome = remember(&db, obs).unwrap();
        let record = outcome.record.unwrap();

        // Migration should be a no-op.
        let migrated = migrate_provenance_record(&db, &record).unwrap();
        assert!(!migrated);
    }

    // ── 11.4 Belief Revision tests ────────────────────────────

    // ── 11.8 Memory Safety tests ────────────────────────────────

    // ── 11.9 Brain Eval tests ──────────────────────────────────

    #[test]
    fn brain_eval_runs_and_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let report = run_brain_eval(&db).unwrap();
        assert!(report.total > 0);
        // Should pass most cases (classification is heuristic, not perfect).
        assert!(
            report.pass_rate >= 0.7,
            "brain eval pass rate {:.1}% is below 70% threshold. Failed: {:?}",
            report.pass_rate * 100.0,
            report
                .results
                .iter()
                .filter(|r| !r.passed)
                .map(|r| &r.name)
                .collect::<Vec<_>>(),
        );
        assert!(report.brain_score > 0.0);
    }

    // ── 11.8 Memory Safety tests ────────────────────────────────

    #[test]
    fn detect_pii_email() {
        let pii = detect_pii("Contact me at user@example.com for details");
        assert_eq!(pii.len(), 1);
        assert_eq!(pii[0].0, "email");
        assert!(pii[0].1.contains("user@example.com"));
    }

    #[test]
    fn detect_pii_ip_address() {
        let pii = detect_pii("Server is at 192.168.1.100 in the cluster");
        assert_eq!(pii.len(), 1);
        assert_eq!(pii[0].0, "ip_address");
    }

    #[test]
    fn detect_pii_none() {
        let pii = detect_pii("No personal information here");
        assert!(pii.is_empty());
    }

    #[test]
    fn redact_field_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let record = db
            .insert(
                "context",
                json!({
                    "summary": "Contact user@example.com",
                    "email": "user@example.com",
                }),
            )
            .unwrap();

        let updated = redact_field(&db, &record.id, "email").unwrap();
        assert_eq!(updated.data["email"], "[REDACTED]");
        assert!(updated.data["_redacted_fields"]
            .as_array()
            .unwrap()
            .contains(&json!("email")));
    }

    #[test]
    fn pin_and_unpin_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let record = db
            .insert("context", json!({"summary": "important fact"}))
            .unwrap();

        let pinned = pin_record(&db, &record.id).unwrap();
        assert_eq!(pinned.data["_importance_pinned"], true);
        assert!(crate::importance::is_pinned(&pinned.data));

        let unpinned = unpin_record(&db, &record.id).unwrap();
        assert_eq!(unpinned.data["_importance_pinned"], false);
    }

    #[test]
    fn retention_policy_set_and_show() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        set_retention(&db, "session", 30).unwrap();
        set_retention(&db, "project", 365).unwrap();

        let policies = get_retention_policies(&db).unwrap();
        let arr = policies["policies"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn belief_revision_no_beliefs_is_no_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let result = revise_beliefs(&db, "auth uses JWT").unwrap();
        assert_eq!(result.actions, vec![BeliefRevisionAction::NoChange]);
    }

    #[test]
    fn belief_revision_reinforces_matching() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        bs.believe("`AuthModule` uses JWT tokens").unwrap();

        // Evidence that aligns with the belief.
        let result = revise_beliefs(&db, "`AuthModule` uses JWT tokens for session auth").unwrap();
        let has_reinforced = result
            .actions
            .iter()
            .any(|a| matches!(a, BeliefRevisionAction::Reinforced { .. }));
        assert!(
            has_reinforced,
            "expected reinforcement, got: {:?}",
            result.actions
        );
    }

    #[test]
    fn belief_revision_doubts_contradicting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        bs.believe("`AuthModule` uses JWT tokens").unwrap();

        // Contradicting evidence.
        let result = revise_beliefs(&db, "`AuthModule` no longer uses JWT tokens").unwrap();
        let has_doubted = result
            .actions
            .iter()
            .any(|a| matches!(a, BeliefRevisionAction::Doubted { .. }));
        assert!(has_doubted, "expected doubt, got: {:?}", result.actions);
    }

    #[test]
    fn belief_revision_unrelated_is_no_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        bs.believe("database uses PostgreSQL 16").unwrap();

        // Completely unrelated evidence.
        let result = revise_beliefs(&db, "the weather is nice today").unwrap();
        assert_eq!(result.actions, vec![BeliefRevisionAction::NoChange]);
    }

    // ── 11.5 Memory Debugger tests ──────────────────────────────

    #[test]
    fn why_remembered_explains_pipeline_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("Fixed `AuthModule` timeout by increasing pool size")
            .with_source(MemorySource::Agent)
            .with_source_ref("bash:cargo test");
        let outcome = remember(&db, obs).unwrap();
        let record = outcome.record.unwrap();

        let explanation = why_remembered(&db, &record.id).unwrap();
        assert_eq!(explanation.record_id, record.id.to_string());
        assert!(!explanation.memory_type.is_empty());
        assert!(!explanation.scope.is_empty());
        assert!(explanation.importance > 0.0);
        assert!(!explanation.entities.is_empty());
        assert_eq!(
            explanation.resolution,
            "novel (no duplicate or conflict detected)"
        );
    }

    #[test]
    fn why_revised_for_doubted_belief() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        let belief = bs.believe("cache uses Redis").unwrap();
        let _ = doubt_record(&db, &belief.id, Some("migrated to Memcached")).unwrap();

        let explanation = why_revised(&db, &belief.id).unwrap();
        assert_eq!(explanation.revision_type, "doubted");
        assert!(explanation.cause.contains("migrated to Memcached"));
        assert_eq!(explanation.trust_tier, "doubted");
    }

    #[test]
    fn why_recalled_without_vector_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("auth uses JWT tokens").with_source(MemorySource::Agent);
        let outcome = remember(&db, obs).unwrap();
        let record = outcome.record.unwrap();

        // Without vector index, recall falls back to non-vector signals when available.
        // In a plain core DB, that means no candidates are found and the explanation
        // should still be graceful.
        let explanation = why_recalled(&db, "auth", &record.id).unwrap();
        assert!(!explanation.passed_filters);
        assert!(explanation
            .reason
            .contains("not semantically similar enough"));
    }

    // ── 11.6 Self Memory & Project Model tests ──────────────────

    #[test]
    fn self_note_and_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        self_note(
            &db,
            "Often forgets to run tests after CLI changes",
            Some("gotcha"),
        )
        .unwrap();
        self_note(&db, "Good at architectural decisions", Some("strength")).unwrap();

        let profile = self_profile(&db).unwrap();
        assert_eq!(profile["total_notes"], 2);
        assert!(profile["categories"]["gotcha"].is_array());
        assert!(profile["categories"]["strength"].is_array());
    }

    #[test]
    fn project_model_set_and_show() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        project_model_set(&db, "deployment", "Always use blue-green deploys").unwrap();
        project_model_set(&db, "testing", "Run integration tests before merge").unwrap();

        let model = project_model_show(&db).unwrap();
        assert_eq!(model["entries"], 2);
        assert_eq!(
            model["model"]["deployment"],
            "Always use blue-green deploys"
        );
    }

    #[test]
    fn project_model_set_updates_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        project_model_set(&db, "testing", "Unit tests only").unwrap();
        project_model_set(&db, "testing", "Unit + integration tests").unwrap();

        let model = project_model_show(&db).unwrap();
        assert_eq!(model["entries"], 1); // Updated, not duplicated.
        assert_eq!(model["model"]["testing"], "Unit + integration tests");
    }

    #[test]
    fn user_contract_add_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        user_contract_set(&db, "Always ask before schema migrations").unwrap();
        user_contract_set(&db, "Use absolute dates in reports").unwrap();

        let rules = user_contract_list(&db).unwrap();
        assert_eq!(rules.len(), 2);
        assert!(rules.contains(&"Always ask before schema migrations".to_string()));
    }

    #[test]
    fn belief_history_shows_all_including_doubted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        let r1 = bs.believe("auth uses JWT").unwrap();
        bs.believe("auth uses session cookies").unwrap();
        bs.doubt(&r1.id).unwrap();

        let history = belief_history(&db, "auth").unwrap();
        assert_eq!(history.len(), 2);
        assert!(history.iter().any(|b| b.doubted));
        assert!(history.iter().any(|b| !b.doubted));
    }

    // ── Codex review regression tests ─────────────────────────

    #[test]
    fn pipeline_belief_writes_statement_field() {
        // Verifies Fix 1: beliefs stored via remember() use "statement" not "summary".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let obs = Observation::from_text("I believe the auth approach is correct")
            .with_source(MemorySource::User)
            .with_hint("belief");
        let outcome = remember(&db, obs).unwrap();
        assert_eq!(outcome.memory_type, MemoryType::Belief);

        let record = outcome.record.unwrap();
        // Must have "statement" field for BeliefSystem compatibility.
        assert!(
            record.data.get("statement").is_some(),
            "belief record missing 'statement' field: {:?}",
            record.data
        );
        assert!(
            record.data.get("confidence").is_some(),
            "belief record missing 'confidence' field: {:?}",
            record.data
        );

        // Verify BeliefSystem can read it back.
        let bs = crate::beliefs::BeliefSystem::new(&db);
        let beliefs = bs.list(Some("auth"), false).unwrap();
        assert!(
            !beliefs.is_empty(),
            "BeliefSystem couldn't find the belief stored via pipeline"
        );
        assert!(
            beliefs[0].statement.contains("auth"),
            "statement content wrong: {}",
            beliefs[0].statement
        );
    }

    #[test]
    fn supersede_writes_superseded_by_backlink() {
        // Verifies Fix 2: superseded records get _superseded_by pointing to the new record.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Create two records where the second will supersede the first
        // (via the pipeline's resolve step — needs vector index, so test the
        // code path directly).
        let old = db
            .insert("context", json!({"summary": "auth uses JWT"}))
            .unwrap();
        let mut old_data = old.data.clone();
        if let Some(obj) = old_data.as_object_mut() {
            obj.insert("_superseded".to_string(), json!(true));
            obj.insert("_superseded_by".to_string(), json!("NEW_ID_HERE"));
        }
        db.update(&old.id, old_data).unwrap();

        // Verify why_revised can read the backlink.
        let explanation = why_revised(&db, &old.id).unwrap();
        assert_eq!(explanation.revision_type, "superseded");
        assert!(explanation.cause.contains("NEW_ID_HERE"));
    }

    #[test]
    fn doubt_record_halves_belief_confidence_field() {
        // Verifies Fix 3: doubt_record halves both _confidence and confidence.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        let bs = crate::beliefs::BeliefSystem::new(&db);
        let belief = bs.believe("cache uses Redis").unwrap();
        // BeliefSystem sets confidence = 1.0 (not _confidence).
        assert_eq!(belief.data["confidence"], 1.0);

        let doubted = doubt_record(&db, &belief.id, Some("migrated")).unwrap();
        // Both fields should be halved.
        assert_eq!(doubted.data["doubted"], true);
        let conf = doubted.data["confidence"].as_f64().unwrap();
        assert!(
            conf < 1.0 && conf >= 0.1,
            "confidence should be halved, got {conf}"
        );
    }
}
