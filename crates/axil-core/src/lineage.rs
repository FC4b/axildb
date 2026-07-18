//! Lineage-chain walk over `derived_from` (or any) graph edges.
//!
//! This is an adapter-tier feature (the CLI `axil lineage` command and the MCP
//! `lineage` tool) implemented as a breadth-first walk over [`Axil::edges`],
//! deliberately kept out of the recall core. It retains the *path* the standard
//! traversal API discards: each hop carries the record fields you select plus
//! the numeric delta against its parent hop, so a strategy-R&D loop can read
//! how a metric drifted across a chain of mutations.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{json, Map, Value};

use crate::db::Axil;
use crate::error::Result;
use crate::plugin::Direction;
use crate::record::RecordId;

/// Which way to follow lineage edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineageDirection {
    /// Follow OUT edges (child → parent): what each node was derived from.
    /// Yields the chain root-first.
    Ancestors,
    /// Follow IN edges (parent ← child): what was derived from the node.
    Descendants,
    /// Follow both directions from every node.
    Both,
}

impl LineageDirection {
    /// Parse a direction string (`ancestors` | `descendants` | `both`).
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ancestors" | "ancestor" => Ok(Self::Ancestors),
            "descendants" | "descendant" => Ok(Self::Descendants),
            "both" => Ok(Self::Both),
            other => Err(format!(
                "invalid direction '{other}' (ancestors | descendants | both)"
            )),
        }
    }

    /// Stable lowercase label for the JSON envelope.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ancestors => "ancestors",
            Self::Descendants => "descendants",
            Self::Both => "both",
        }
    }
}

/// One frontier item: the node to expand, its depth, the edge that led to it
/// (`None` for the root), and the numeric fields of the hop that discovered it
/// (empty for the root) — deltas are computed against that parent hop.
struct Pending {
    id: RecordId,
    depth: usize,
    edge: Option<Value>,
    parent_numeric: HashMap<String, f64>,
}

/// Walk a lineage chain from `root` over `edge_type` edges.
///
/// Returns a stable JSON envelope:
/// ```text
/// {"root", "direction", "edge_type", "hops":[
///   {"depth", "id", "table", "fields":{…}, "edge":{"edge_id","props"}, "delta":{…}}
/// ]}
/// ```
/// The walk is breadth-first with a global visited set (cycle-safe: a node is
/// emitted at most once). `fields` selects which `record.data` keys appear in
/// each hop's `fields` (all keys when `None`); `delta` holds, for the selected
/// keys numeric on both hops, `this hop minus its parent hop` — the node on the
/// other end of the edge that discovered it, so in a branching tree each
/// sibling's delta is measured against their shared parent, not each other.
/// `ancestors` is root-first ordering. A hop whose record is missing is emitted
/// with `"missing": true` (no fields/delta) rather than erroring.
pub fn walk(
    db: &Axil,
    root: &RecordId,
    edge_type: &str,
    direction: LineageDirection,
    max_depth: usize,
    fields: Option<&[String]>,
) -> Result<Value> {
    let mut visited: HashSet<RecordId> = HashSet::new();
    let mut queue: VecDeque<Pending> = VecDeque::new();
    let mut hops: Vec<Value> = Vec::new();

    visited.insert(root.clone());
    queue.push_back(Pending {
        id: root.clone(),
        depth: 0,
        edge: None,
        parent_numeric: HashMap::new(),
    });

    while let Some(item) = queue.pop_front() {
        let record = db.get(&item.id)?;

        // Selected fields + numeric map for this hop (empty when missing).
        let (fields_map, numeric) = match &record {
            Some(r) => {
                let selected = select_fields(&r.data, fields);
                let numeric = numeric_map(&selected);
                (selected, numeric)
            }
            None => (Map::new(), HashMap::new()),
        };

        let delta = compute_delta(&item.parent_numeric, &numeric);

        let mut hop = Map::new();
        hop.insert("depth".to_string(), json!(item.depth));
        hop.insert("id".to_string(), json!(item.id.to_string()));
        match &record {
            Some(r) => {
                hop.insert("table".to_string(), json!(r.table));
                hop.insert("fields".to_string(), Value::Object(fields_map));
            }
            None => {
                hop.insert("missing".to_string(), json!(true));
            }
        }
        hop.insert("edge".to_string(), item.edge.unwrap_or(Value::Null));
        hop.insert("delta".to_string(), Value::Object(delta));
        hops.push(Value::Object(hop));

        // Don't expand past the depth cap or through a missing record.
        if item.depth >= max_depth || record.is_none() {
            continue;
        }

        for (dir, take_to) in edge_dirs(direction) {
            for edge in db.edges(&item.id, Some(edge_type), dir)? {
                // The neighbor is the *other* endpoint of the edge.
                let next = if take_to { edge.to.clone() } else { edge.from.clone() };
                if visited.contains(&next) {
                    continue;
                }
                visited.insert(next.clone());
                let edge_json = json!({
                    "edge_id": edge.id.to_string(),
                    "props": edge.properties,
                });
                queue.push_back(Pending {
                    id: next,
                    depth: item.depth + 1,
                    edge: Some(edge_json),
                    parent_numeric: numeric.clone(),
                });
            }
        }
    }

    Ok(json!({
        "root": root.to_string(),
        "direction": direction.as_str(),
        "edge_type": edge_type,
        "hops": hops,
    }))
}

/// The `(direction, neighbor-is-`to`)` pairs to expand for a lineage direction.
///
/// Ancestors follow OUT edges (the neighbor is `edge.to`, the parent);
/// descendants follow IN edges (the neighbor is `edge.from`, the child).
fn edge_dirs(direction: LineageDirection) -> Vec<(Direction, bool)> {
    match direction {
        LineageDirection::Ancestors => vec![(Direction::Out, true)],
        LineageDirection::Descendants => vec![(Direction::In, false)],
        LineageDirection::Both => vec![(Direction::Out, true), (Direction::In, false)],
    }
}

/// Select the requested `record.data` keys (all keys when `fields` is `None`).
fn select_fields(data: &Value, fields: Option<&[String]>) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(obj) = data.as_object() else {
        return out;
    };
    match fields {
        Some(keys) => {
            for k in keys {
                if let Some(v) = obj.get(k) {
                    out.insert(k.clone(), v.clone());
                }
            }
            out
        }
        None => obj.clone(),
    }
}

/// Extract the numeric subset of a selected-fields map.
fn numeric_map(fields: &Map<String, Value>) -> HashMap<String, f64> {
    fields
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect()
}

/// Per-field `cur - prev` for keys numeric in both maps.
fn compute_delta(prev: &HashMap<String, f64>, cur: &HashMap<String, f64>) -> Map<String, Value> {
    let mut delta = Map::new();
    for (k, cv) in cur {
        if let Some(pv) = prev.get(k) {
            delta.insert(k.clone(), json!(cv - pv));
        }
    }
    delta
}
