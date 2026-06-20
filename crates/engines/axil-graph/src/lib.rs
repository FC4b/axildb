pub mod edge;
pub mod pagerank;
pub mod traverse;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde_json::Value;

use axil_core::plugin::{Capability, Direction, EdgeInfo, GraphIndex, Engine, TraversalStep};
use axil_core::record::{Record, RecordId};
use axil_core::{companion_path, AxilBuilder, AxilError, Result};

use crate::edge::Edge;

// ── redb table definitions ──────────────────────────────────────────

/// Edges table: edge_id -> serialized Edge.
const EDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("edges");

// ── In-memory adjacency index ───────────────────────────────────────

/// In-memory index for fast edge lookups.
///
/// Two maps: outgoing edges (from -> edges) and incoming edges (to -> edges).
/// Edges are stored by ID for quick removal; full Edge data is in the edges map.
struct AdjacencyIndex {
    /// All edges by ID.
    edges: HashMap<RecordId, Edge>,
    /// Outgoing: from_id -> set of edge IDs.
    outgoing: HashMap<RecordId, HashSet<RecordId>>,
    /// Incoming: to_id -> set of edge IDs.
    incoming: HashMap<RecordId, HashSet<RecordId>>,
}

impl AdjacencyIndex {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        }
    }

    fn add(&mut self, edge: Edge) {
        self.outgoing
            .entry(edge.from.clone())
            .or_default()
            .insert(edge.id.clone());
        self.incoming
            .entry(edge.to.clone())
            .or_default()
            .insert(edge.id.clone());
        self.edges.insert(edge.id.clone(), edge);
    }

    fn remove(&mut self, edge_id: &RecordId) -> Option<Edge> {
        if let Some(edge) = self.edges.remove(edge_id) {
            if let Some(set) = self.outgoing.get_mut(&edge.from) {
                set.remove(edge_id);
                if set.is_empty() {
                    self.outgoing.remove(&edge.from);
                }
            }
            if let Some(set) = self.incoming.get_mut(&edge.to) {
                set.remove(edge_id);
                if set.is_empty() {
                    self.incoming.remove(&edge.to);
                }
            }
            Some(edge)
        } else {
            None
        }
    }

    /// Remove all edges referencing the given record (as source or target).
    /// Returns the removed edge IDs for disk cleanup.
    fn remove_edges_for_record(&mut self, record_id: &RecordId) -> Vec<RecordId> {
        let mut removed_set: HashSet<RecordId> = HashSet::new();

        if let Some(edge_ids) = self.outgoing.remove(record_id) {
            removed_set.extend(edge_ids);
        }
        if let Some(edge_ids) = self.incoming.remove(record_id) {
            removed_set.extend(edge_ids);
        }

        for eid in &removed_set {
            if let Some(edge) = self.edges.remove(eid) {
                if edge.from != *record_id {
                    if let Some(set) = self.outgoing.get_mut(&edge.from) {
                        set.remove(eid);
                        if set.is_empty() {
                            self.outgoing.remove(&edge.from);
                        }
                    }
                }
                if edge.to != *record_id {
                    if let Some(set) = self.incoming.get_mut(&edge.to) {
                        set.remove(eid);
                        if set.is_empty() {
                            self.incoming.remove(&edge.to);
                        }
                    }
                }
            }
        }

        removed_set.into_iter().collect()
    }

    /// Get outgoing edges from a record, optionally filtered by edge type.
    fn get_outgoing(&self, from: &RecordId, edge_type: Option<&str>) -> Vec<&Edge> {
        self.outgoing
            .get(from)
            .map(|ids| {
                ids.iter()
                    .filter_map(|eid| self.edges.get(eid))
                    .filter(|e| edge_type.is_none() || Some(e.edge_type.as_str()) == edge_type)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming edges to a record, optionally filtered by edge type.
    fn get_incoming(&self, to: &RecordId, edge_type: Option<&str>) -> Vec<&Edge> {
        self.incoming
            .get(to)
            .map(|ids| {
                ids.iter()
                    .filter_map(|eid| self.edges.get(eid))
                    .filter(|e| edge_type.is_none() || Some(e.edge_type.as_str()) == edge_type)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get neighbor record IDs for a node in the given direction.
    fn neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<RecordId> {
        self.neighbor_ids_temporal(id, edge_type, direction, None)
    }

    /// Get neighbor record IDs with optional temporal filtering (8b.8).
    ///
    /// When `as_of` is Some, only edges valid at that point in time are traversed.
    fn neighbor_ids_temporal(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
        as_of: Option<&DateTime<Utc>>,
    ) -> Vec<RecordId> {
        let mut neighbors = Vec::new();
        let mut seen = HashSet::new();

        let is_valid = |edge: &Edge| -> bool { as_of.map_or(true, |t| edge.is_valid_at(t)) };

        if matches!(direction, Direction::Out | Direction::Both) {
            for edge in self.get_outgoing(id, edge_type) {
                if is_valid(edge) && seen.insert(edge.to.clone()) {
                    neighbors.push(edge.to.clone());
                }
            }
        }

        if matches!(direction, Direction::In | Direction::Both) {
            for edge in self.get_incoming(id, edge_type) {
                if is_valid(edge) && seen.insert(edge.from.clone()) {
                    neighbors.push(edge.from.clone());
                }
            }
        }

        neighbors
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ── Limits ──────────────────────────────────────────────────────────

/// Maximum number of edges allowed in a single graph store.
const MAX_EDGES: usize = 1_000_000;

/// Maximum byte size for edge property JSON.
const MAX_EDGE_PROPERTY_BYTES: usize = 65_536; // 64 KB

// ── GraphEngine ─────────────────────────────────────────────────────

/// Graph plugin for Axil — stores directed edges between records with
/// traversal and neighbor queries.
pub struct GraphEngine {
    graph_db: Database,
    index: RwLock<AdjacencyIndex>,
}

impl GraphEngine {
    /// Open or create a graph store at the companion path for the given database.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let graph_path = companion_path(db_path.as_ref(), ".graph");
        let graph_db = Database::create(&graph_path).map_err(|e| {
            AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "failed to open graph store at {}: {e}",
                graph_path.display()
            ))))
        })?;

        {
            let txn = graph_db.begin_write()?;
            let _ = txn.open_table(EDGES_TABLE)?;
            txn.commit()?;
        }

        // Load existing edges into memory, cleaning corrupt entries.
        let mut adj = AdjacencyIndex::new();
        let mut corrupt_keys: Vec<String> = Vec::new();
        {
            let txn = graph_db.begin_read()?;
            let table = txn.open_table(EDGES_TABLE)?;
            let iter = table.iter()?;
            for entry in iter {
                let entry = entry?;
                let key = entry.0.value().to_string();
                let bytes = entry.1.value();
                match Edge::from_bytes(bytes) {
                    Ok(edge) => adj.add(edge),
                    Err(e) => {
                        eprintln!("warning: removing corrupt edge {key}: {e}");
                        corrupt_keys.push(key);
                    }
                }
            }
        }

        // Remove corrupt entries from disk so they don't accumulate.
        if !corrupt_keys.is_empty() {
            let txn = graph_db.begin_write()?;
            {
                let mut table = txn.open_table(EDGES_TABLE)?;
                for key in &corrupt_keys {
                    table.remove(key.as_str())?;
                }
            }
            txn.commit()?;
        }

        if adj.edge_count() > MAX_EDGES {
            return Err(AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "graph store has {} edges, exceeding limit of {MAX_EDGES}",
                adj.edge_count()
            )))));
        }

        Ok(Self {
            graph_db,
            index: RwLock::new(adj),
        })
    }

    /// Create a directed edge between two records.
    ///
    /// Holds the write lock for the entire operation (count check, disk
    /// write, memory add) so that concurrent callers cannot exceed
    /// MAX_EDGES. Consistent with delete_edge/on_record_delete pattern.
    pub fn create_edge(
        &self,
        from: RecordId,
        edge_type: &str,
        to: RecordId,
        properties: Value,
    ) -> Result<Edge> {
        // Reject control characters in edge type to prevent terminal injection.
        if edge_type.bytes().any(|b| b < 0x20 || b == 0x7F) {
            return Err(AxilError::InvalidQuery(
                "edge type must not contain control characters".into(),
            ));
        }

        // Enforce property size limit before acquiring the lock.
        let prop_size = serde_json::to_string(&properties)
            .map(|s| s.len())
            .unwrap_or(0);
        if prop_size > MAX_EDGE_PROPERTY_BYTES {
            return Err(AxilError::InvalidQuery(format!(
                "edge properties exceed {MAX_EDGE_PROPERTY_BYTES} byte limit ({prop_size} bytes)",
            )));
        }

        let mut idx = self.index.write();

        // Check count under write lock to prevent concurrent overflow.
        if idx.edge_count() >= MAX_EDGES {
            return Err(AxilError::InvalidQuery(format!(
                "edge limit reached ({MAX_EDGES})"
            )));
        }

        let edge = Edge::new(from, edge_type, to, properties);
        self.persist_edge(&edge)?;
        idx.add(edge.clone());

        Ok(edge)
    }

    /// Create many directed edges in a single redb transaction.
    ///
    /// Drop-in batched equivalent of `create_edge` — same validation
    /// (control-char check, property size, MAX_EDGES) but folds the N
    /// per-edge `begin_write/commit` pairs into one. SCIP ingest spends
    /// >90% of its wall time in those commits; batching cuts an
    /// edge-heavy workload from minutes to seconds (see Phase 14
    /// dogfood friction #8).
    pub fn create_edges_batch(
        &self,
        specs: Vec<(RecordId, String, RecordId, Value)>,
    ) -> Result<Vec<Edge>> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }

        // Per-spec validation, mirroring create_edge.
        for (_, edge_type, _, props) in &specs {
            if edge_type.bytes().any(|b| b < 0x20 || b == 0x7F) {
                return Err(AxilError::InvalidQuery(
                    "edge type must not contain control characters".into(),
                ));
            }
            let prop_size = serde_json::to_string(props).map(|s| s.len()).unwrap_or(0);
            if prop_size > MAX_EDGE_PROPERTY_BYTES {
                return Err(AxilError::InvalidQuery(format!(
                    "edge properties exceed {MAX_EDGE_PROPERTY_BYTES} byte limit ({prop_size} bytes)",
                )));
            }
        }

        let mut idx = self.index.write();

        // Enforce MAX_EDGES under the write lock against the post-batch
        // count so a partial batch can't push the index over.
        if idx.edge_count() + specs.len() > MAX_EDGES {
            return Err(AxilError::InvalidQuery(format!(
                "edge limit reached ({MAX_EDGES}); batch of {} would exceed",
                specs.len()
            )));
        }

        // Materialize edges outside the redb txn so the txn time is
        // dominated by I/O, not CPU.
        let edges: Vec<Edge> = specs
            .into_iter()
            .map(|(from, edge_type, to, props)| Edge::new(from, &edge_type, to, props))
            .collect();

        // Disk before memory mirrors create_edge's invariant: a crash
        // mid-batch leaves the on-disk edges consistent with what's
        // not yet in memory.
        self.persist_edges_batch(&edges)?;

        for edge in &edges {
            idx.add(edge.clone());
        }

        Ok(edges)
    }

    /// Delete an edge by ID.
    ///
    /// Holds the write lock for the entire operation so that concurrent
    /// deleters cannot both observe the edge as present. Disk is updated
    /// before memory within the lock so a crash leaves a consistent state.
    pub fn delete_edge(&self, edge_id: &RecordId) -> Result<bool> {
        let mut idx = self.index.write();
        if !idx.edges.contains_key(edge_id) {
            return Ok(false);
        }
        self.remove_edge_from_disk(edge_id)?;
        idx.remove(edge_id);
        Ok(true)
    }

    /// Get an edge by ID.
    pub fn get_edge(&self, edge_id: &RecordId) -> Option<Edge> {
        self.index.read().edges.get(edge_id).cloned()
    }

    /// Get outgoing edges from a record.
    pub fn get_outgoing(&self, from: &RecordId, edge_type: Option<&str>) -> Vec<Edge> {
        self.index
            .read()
            .get_outgoing(from, edge_type)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get incoming edges to a record.
    pub fn get_incoming(&self, to: &RecordId, edge_type: Option<&str>) -> Vec<Edge> {
        self.index
            .read()
            .get_incoming(to, edge_type)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get all edges for a record in a given direction, optionally filtered by type.
    pub fn get_edges(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<Edge> {
        let idx = self.index.read();
        match direction {
            Direction::Out => idx
                .get_outgoing(id, edge_type)
                .into_iter()
                .cloned()
                .collect(),
            Direction::In => idx
                .get_incoming(id, edge_type)
                .into_iter()
                .cloned()
                .collect(),
            Direction::Both => {
                let mut edges: Vec<Edge> = idx
                    .get_outgoing(id, edge_type)
                    .into_iter()
                    .cloned()
                    .collect();
                let existing: HashSet<_> = edges.iter().map(|e| e.id.clone()).collect();
                for e in idx.get_incoming(id, edge_type) {
                    if !existing.contains(&e.id) {
                        edges.push(e.clone());
                    }
                }
                edges
            }
        }
    }

    /// Get neighbor record IDs reachable via edges in the given direction.
    pub fn neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<RecordId> {
        self.index.read().neighbor_ids(id, edge_type, direction)
    }

    /// Multi-hop traversal following a sequence of steps.
    /// Returns the record IDs at the end of the path. Holds a single read
    /// lock for the entire traversal to ensure a consistent snapshot.
    ///
    /// Each step expands the current frontier to its neighbors (filtered
    /// by edge type and direction), deduplicating within each step.
    /// Nodes may reappear across steps — a path `a ->knows-> b ->knows-> a`
    /// correctly returns `[a]`. Infinite loops are impossible because the
    /// number of steps is fixed (bounded by `MAX_DEPTH`).
    pub fn traverse_ids(&self, start: &RecordId, steps: &[TraversalStep]) -> Result<Vec<RecordId>> {
        self.traverse_ids_temporal(start, steps, None)
    }

    /// Multi-hop traversal with optional temporal filtering (8b.8).
    ///
    /// When `as_of` is Some, only edges valid at that point in time are traversed.
    /// This enables "what did the agent know at time T?" queries.
    pub fn traverse_ids_temporal(
        &self,
        start: &RecordId,
        steps: &[TraversalStep],
        as_of: Option<&DateTime<Utc>>,
    ) -> Result<Vec<RecordId>> {
        if steps.is_empty() {
            return Ok(vec![start.clone()]);
        }

        let idx = self.index.read();
        let mut current: Vec<RecordId> = vec![start.clone()];

        for step in steps {
            let mut next = Vec::new();
            let mut seen = HashSet::new();
            for node in &current {
                for n in
                    idx.neighbor_ids_temporal(node, Some(&step.edge_type), step.direction, as_of)
                {
                    if seen.insert(n.clone()) {
                        next.push(n);
                    }
                }
            }
            current = next;

            if current.is_empty() {
                break;
            }
        }

        Ok(current)
    }

    /// Total edge count.
    pub fn edge_count(&self) -> usize {
        self.index.read().edge_count()
    }

    // ── Persistence helpers ─────────────────────────────────────────

    fn persist_edge(&self, edge: &Edge) -> Result<()> {
        self.persist_edges_batch(std::slice::from_ref(edge))
    }

    /// Persist N edges in a single redb write transaction.
    ///
    /// Symmetric counterpart to `remove_edges_from_disk`. SCIP ingest
    /// uses this via `create_edges_batch` to fold ~130k per-edge
    /// commits into one — without it, edge-heavy workloads spent 99%
    /// of wall time in fsync.
    fn persist_edges_batch(&self, edges: &[Edge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let serialized: Vec<(String, Vec<u8>)> = edges
            .iter()
            .map(|e| {
                e.to_bytes()
                    .map(|b| (e.id.as_str().to_string(), b))
                    .map_err(|e| AxilError::Serialization(Box::new(e)))
            })
            .collect::<Result<Vec<_>>>()?;
        let txn = self.graph_db.begin_write()?;
        {
            let mut table = txn.open_table(EDGES_TABLE)?;
            for (id_str, bytes) in &serialized {
                table.insert(id_str.as_str(), bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    fn remove_edge_from_disk(&self, edge_id: &RecordId) -> Result<()> {
        let txn = self.graph_db.begin_write()?;
        {
            let mut table = txn.open_table(EDGES_TABLE)?;
            table.remove(edge_id.as_str())?;
        }
        txn.commit()?;
        Ok(())
    }

    fn remove_edges_from_disk(&self, edge_ids: &[RecordId]) -> Result<()> {
        if edge_ids.is_empty() {
            return Ok(());
        }
        let txn = self.graph_db.begin_write()?;
        {
            let mut table = txn.open_table(EDGES_TABLE)?;
            for eid in edge_ids {
                table.remove(eid.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }
}

// ── Engine trait ────────────────────────────────────────────────────

impl Engine for GraphEngine {
    fn name(&self) -> &str {
        "graph"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::GraphTraversal]
    }

    fn on_record_insert(&self, _record: &Record) -> Result<()> {
        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        // Hold the write lock for the entire operation so that no new
        // edges can be added for this record between collection and
        // removal. Disk is updated before memory within the lock.
        let mut idx = self.index.write();
        let to_remove = {
            let mut ids = HashSet::new();
            if let Some(set) = idx.outgoing.get(id) {
                ids.extend(set.iter().cloned());
            }
            if let Some(set) = idx.incoming.get(id) {
                ids.extend(set.iter().cloned());
            }
            ids.into_iter().collect::<Vec<_>>()
        };
        self.remove_edges_from_disk(&to_remove)?;
        idx.remove_edges_for_record(id);
        Ok(())
    }
}

// ── GraphIndex trait ────────────────────────────────────────────────

impl GraphIndex for GraphEngine {
    fn relate(
        &self,
        from: RecordId,
        edge_type: &str,
        to: RecordId,
        props: Value,
    ) -> Result<RecordId> {
        let edge = self.create_edge(from, edge_type, to, props)?;
        Ok(edge.id)
    }

    fn relate_batch(
        &self,
        edges: Vec<(RecordId, String, RecordId, Value)>,
    ) -> Result<Vec<RecordId>> {
        let materialized = self.create_edges_batch(edges)?;
        Ok(materialized.into_iter().map(|e| e.id).collect())
    }

    fn unrelate(&self, edge_id: &RecordId) -> Result<bool> {
        self.delete_edge(edge_id)
    }

    fn traverse(&self, start: RecordId, path: &[TraversalStep]) -> Result<Vec<RecordId>> {
        self.traverse_ids(&start, path)
    }

    fn neighbors(
        &self,
        id: RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<RecordId>> {
        Ok(self.neighbor_ids(&id, edge_type, direction))
    }

    fn edges(
        &self,
        id: RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<EdgeInfo>> {
        Ok(self
            .get_edges(&id, edge_type, direction)
            .into_iter()
            .map(|e| EdgeInfo {
                id: e.id,
                from: e.from,
                to: e.to,
                edge_type: e.edge_type,
                properties: e.properties,
                created_at: e.created_at.to_rfc3339(),
            })
            .collect())
    }

    fn edge_count(&self) -> usize {
        GraphEngine::edge_count(self)
    }

    fn all_edge_ids(&self) -> Result<Vec<(RecordId, RecordId, RecordId)>> {
        let idx = self.index.read();
        Ok(idx
            .edges
            .values()
            .map(|e| (e.id.clone(), e.from.clone(), e.to.clone()))
            .collect())
    }
}

// ── Builder extension ───────────────────────────────────────────────

/// Extension trait for adding graph support to `AxilBuilder`.
pub trait AxilBuilderGraphExt {
    /// Enable graph traversal with a companion `.graph` file.
    fn with_graph_engine(self) -> Result<Self>
    where
        Self: Sized;
}

impl AxilBuilderGraphExt for AxilBuilder {
    fn with_graph_engine(self) -> Result<Self> {
        let plugin = GraphEngine::open(self.path())?;
        let arc: Arc<dyn GraphIndex> = Arc::new(plugin);
        Ok(self.with_graph_index(arc))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Re-export for CLI: check if a graph store exists for the given database.
pub fn has_graph_store(db_path: &Path) -> bool {
    companion_path(db_path, ".graph").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_graph() -> (GraphEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let plugin = GraphEngine::open(&path).unwrap();
        (plugin, dir)
    }

    #[test]
    fn create_and_get_edge() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let edge = g
            .create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();

        let fetched = g.get_edge(&edge.id).unwrap();
        assert_eq!(fetched.from, a);
        assert_eq!(fetched.to, b);
        assert_eq!(fetched.edge_type, "knows");
    }

    #[test]
    fn delete_edge() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let edge = g.create_edge(a, "knows", b, json!({})).unwrap();

        assert!(g.delete_edge(&edge.id).unwrap());
        assert!(g.get_edge(&edge.id).is_none());
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn outgoing_incoming() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        g.create_edge(a.clone(), "likes", c.clone(), json!({}))
            .unwrap();

        let out = g.get_outgoing(&a, None);
        assert_eq!(out.len(), 2);

        let out_knows = g.get_outgoing(&a, Some("knows"));
        assert_eq!(out_knows.len(), 1);

        let incoming = g.get_incoming(&b, None);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].from, a);
    }

    #[test]
    fn neighbor_ids() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        g.create_edge(a.clone(), "knows", c.clone(), json!({}))
            .unwrap();

        let neighbors = g.neighbor_ids(&a, Some("knows"), Direction::Out);
        assert_eq!(neighbors.len(), 2);

        let neighbors_in = g.neighbor_ids(&b, Some("knows"), Direction::In);
        assert_eq!(neighbors_in.len(), 1);
        assert_eq!(neighbors_in[0], a);
    }

    #[test]
    fn cascade_delete_record() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        g.create_edge(c.clone(), "knows", a.clone(), json!({}))
            .unwrap();

        assert_eq!(g.edge_count(), 2);
        g.on_record_delete(&a).unwrap();
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn traverse_single_hop() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();

        let steps = crate::traverse::parse_path("->knows").unwrap();
        let result = g.traverse_ids(&a, &steps).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], b);
    }

    #[test]
    fn traverse_multi_hop() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();
        g.create_edge(a.clone(), "modified", b.clone(), json!({}))
            .unwrap();
        g.create_edge(b.clone(), "file", c.clone(), json!({}))
            .unwrap();

        let steps = crate::traverse::parse_path("->modified->file").unwrap();
        let result = g.traverse_ids(&a, &steps).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], c);
    }

    #[test]
    fn traverse_cycle_returns_start() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        g.create_edge(b.clone(), "knows", a.clone(), json!({}))
            .unwrap();

        // a ->knows-> b ->knows-> a: the start node is a valid result.
        let steps = crate::traverse::parse_path("->knows->knows").unwrap();
        let result = g.traverse_ids(&a, &steps).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], a);
    }

    #[test]
    fn traverse_cycle_oscillates() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        let b = RecordId::new();
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        g.create_edge(b.clone(), "knows", a.clone(), json!({}))
            .unwrap();

        // Three hops: a->b->a->b — nodes can reappear across steps.
        let steps = crate::traverse::parse_path("->knows->knows->knows").unwrap();
        let result = g.traverse_ids(&a, &steps).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], b);
    }

    #[test]
    fn traverse_empty_result() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();

        let steps = crate::traverse::parse_path("->nonexistent").unwrap();
        let result = g.traverse_ids(&a, &steps).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn self_loop_edge() {
        let (g, _dir) = temp_graph();
        let a = RecordId::new();
        g.create_edge(a.clone(), "self_ref", a.clone(), json!({}))
            .unwrap();
        assert_eq!(g.edge_count(), 1);

        let out = g.get_outgoing(&a, Some("self_ref"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to, a);

        // Cascade delete should clean up the self-loop.
        g.on_record_delete(&a).unwrap();
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let a = RecordId::new();
        let b = RecordId::new();

        // Create edges.
        {
            let g = GraphEngine::open(&path).unwrap();
            g.create_edge(a.clone(), "knows", b.clone(), json!({"weight": 1}))
                .unwrap();
            assert_eq!(g.edge_count(), 1);
        }

        // Reopen and verify.
        {
            let g = GraphEngine::open(&path).unwrap();
            assert_eq!(g.edge_count(), 1);
            let out = g.get_outgoing(&a, Some("knows"));
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].to, b);
            assert_eq!(out[0].properties["weight"], 1);
        }
    }
}
