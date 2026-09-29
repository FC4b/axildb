pub mod edge;
pub mod pagerank;
pub mod traverse;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use redb::{
    Database, MultimapTable, MultimapTableDefinition, ReadOnlyMultimapTable, ReadOnlyTable,
    ReadableDatabase, ReadableMultimapTable, ReadableTable, ReadableTableMetadata, Table,
    TableDefinition, WriteTransaction,
};
use serde_json::Value;

use axil_core::plugin::{Capability, Direction, EdgeInfo, GraphIndex, Engine, TraversalStep};
use axil_core::record::{Record, RecordId};
use axil_core::{companion_path, AxilBuilder, AxilError, Result};

use crate::edge::Edge;

// ── redb table definitions ──────────────────────────────────────────

/// Edges table: edge_id -> serialized Edge. The source of truth; every
/// other table in the store is derived from it.
const EDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("edges");

/// Outgoing adjacency: from-node id -> one [`AdjEntry`] per outgoing edge.
const OUT_TABLE: MultimapTableDefinition<&str, &[u8]> = MultimapTableDefinition::new("adj_out");

/// Incoming adjacency: to-node id -> one [`AdjEntry`] per incoming edge.
const IN_TABLE: MultimapTableDefinition<&str, &[u8]> = MultimapTableDefinition::new("adj_in");

/// Store metadata. Holds the adjacency stamp: the state of the edges table
/// the adjacency tables were last written against.
const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("graph_meta");

/// Key of the adjacency stamp in [`META_TABLE`].
const ADJACENCY_STAMP_KEY: &str = "adjacency";

/// Version of the adjacency tables' layout, the first byte of the stamp.
/// Bumping it makes every existing store rebuild its tables on first use.
const ADJACENCY_FORMAT: u8 = 1;

// ── On-disk adjacency ───────────────────────────────────────────────

/// One adjacency entry: an edge seen from one of its endpoints.
///
/// Encoded as `len(edge_type) edge_type len(other) other edge_id`, lengths
/// as big-endian u32. That answers neighbour queries and type filters
/// without reading or decoding the edge's JSON, which is what makes a
/// per-node lookup cost the node's degree instead of the whole graph.
struct AdjEntry<'a> {
    edge_type: &'a str,
    /// The endpoint that isn't the table key.
    other: &'a str,
    edge_id: &'a str,
}

impl<'a> AdjEntry<'a> {
    fn encode(edge_type: &str, other: &str, edge_id: &str) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + edge_type.len() + other.len() + edge_id.len());
        for part in [edge_type, other] {
            buf.extend_from_slice(&(part.len() as u32).to_be_bytes());
            buf.extend_from_slice(part.as_bytes());
        }
        buf.extend_from_slice(edge_id.as_bytes());
        buf
    }

    fn decode(bytes: &'a [u8]) -> Option<Self> {
        fn part(bytes: &[u8]) -> Option<(&str, &[u8])> {
            let len = u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
            let s = std::str::from_utf8(bytes.get(4..4 + len)?).ok()?;
            Some((s, &bytes[4 + len..]))
        }
        let (edge_type, rest) = part(bytes)?;
        let (other, rest) = part(rest)?;
        Some(Self {
            edge_type,
            other,
            edge_id: std::str::from_utf8(rest).ok()?,
        })
    }
}

/// The adjacency stamp for the current contents of the edges table: the
/// layout version, the row count and the greatest edge id.
///
/// The adjacency tables are trusted only while the stored stamp equals this
/// one. Every write through this engine updates both in one transaction, so
/// a mismatch means the edges table was changed by something that doesn't
/// maintain the adjacency tables — a store from before they existed, or an
/// older binary writing to it since. Edge ids are ULIDs, so an edge created
/// later sorts last and moves the stamp even when a deletion keeps the row
/// count unchanged.
fn adjacency_stamp(edges: &impl ReadableTable<&'static str, &'static [u8]>) -> Result<Vec<u8>> {
    let mut stamp = vec![ADJACENCY_FORMAT];
    stamp.extend_from_slice(&edges.len()?.to_le_bytes());
    if let Some((key, _)) = edges.last()? {
        stamp.extend_from_slice(key.value().as_bytes());
    }
    Ok(stamp)
}

/// Decode an edge fetched from disk. A corrupt one is reported, queued in
/// `corrupt` for removal, and treated as absent.
fn decode_edge(edge_id: &str, bytes: &[u8], corrupt: &mut Vec<String>) -> Option<Edge> {
    match Edge::from_bytes(bytes) {
        Ok(edge) => Some(edge),
        Err(e) => {
            eprintln!("warning: removing corrupt edge {edge_id}: {e}");
            corrupt.push(edge_id.to_string());
            None
        }
    }
}

/// A read-only snapshot of the store: every lookup through one view sees the
/// same committed state.
struct DiskView {
    edges: ReadOnlyTable<&'static str, &'static [u8]>,
    out: ReadOnlyMultimapTable<&'static str, &'static [u8]>,
    inc: ReadOnlyMultimapTable<&'static str, &'static [u8]>,
}

impl DiskView {
    fn open(db: &Database) -> Result<Self> {
        let txn = db.begin_read()?;
        Ok(Self {
            edges: txn.open_table(EDGES_TABLE)?,
            out: txn.open_multimap_table(OUT_TABLE)?,
            inc: txn.open_multimap_table(IN_TABLE)?,
        })
    }

    /// Call `f` for each edge of `node` on one side (`outgoing` or
    /// incoming) whose type matches `edge_type`, in stored order.
    fn for_each_entry(
        &self,
        node: &str,
        outgoing: bool,
        edge_type: Option<&str>,
        mut f: impl FnMut(AdjEntry<'_>) -> Result<()>,
    ) -> Result<()> {
        let table = if outgoing { &self.out } else { &self.inc };
        for guard in table.get(node)? {
            let guard = guard?;
            let Some(entry) = AdjEntry::decode(guard.value()) else {
                continue;
            };
            if edge_type.is_some_and(|t| t != entry.edge_type) {
                continue;
            }
            f(entry)?;
        }
        Ok(())
    }

    /// Fetch and decode one edge.
    fn edge(&self, edge_id: &str, corrupt: &mut Vec<String>) -> Result<Option<Edge>> {
        let Some(guard) = self.edges.get(edge_id)? else {
            return Ok(None);
        };
        Ok(decode_edge(edge_id, guard.value(), corrupt))
    }

    /// Same contract as [`AdjacencyIndex::neighbor_ids_bitemporal`]: out
    /// neighbours first, then in neighbours, each node once. Edge JSON is
    /// decoded only when a temporal filter needs its timestamps.
    fn neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
        event_time: Option<&DateTime<Utc>>,
        knowledge_time: Option<&DateTime<Utc>>,
        corrupt: &mut Vec<String>,
    ) -> Result<Vec<RecordId>> {
        let temporal = event_time.is_some() || knowledge_time.is_some();
        let mut neighbors = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for outgoing in [true, false] {
            let wanted = if outgoing {
                matches!(direction, Direction::Out | Direction::Both)
            } else {
                matches!(direction, Direction::In | Direction::Both)
            };
            if !wanted {
                continue;
            }
            self.for_each_entry(id.as_str(), outgoing, edge_type, |entry| {
                if temporal {
                    let Some(edge) = self.edge(entry.edge_id, corrupt)? else {
                        return Ok(());
                    };
                    let valid = event_time.map_or(true, |t| edge.is_valid_at(t))
                        && knowledge_time.map_or(true, |t| edge.known_at(t));
                    if !valid {
                        return Ok(());
                    }
                }
                if !seen.contains(entry.other) {
                    seen.insert(entry.other.to_string());
                    neighbors.push(RecordId(entry.other.to_string()));
                }
                Ok(())
            })?;
        }
        Ok(neighbors)
    }

    /// Edges of `id` in `direction`; for `Both`, outgoing first and a
    /// self-loop only once — the contract of [`GraphEngine::get_edges`].
    fn edges_of(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
        corrupt: &mut Vec<String>,
    ) -> Result<Vec<Edge>> {
        let mut edges: Vec<Edge> = Vec::new();
        if matches!(direction, Direction::Out | Direction::Both) {
            self.for_each_entry(id.as_str(), true, edge_type, |entry| {
                edges.extend(self.edge(entry.edge_id, corrupt)?);
                Ok(())
            })?;
        }
        if matches!(direction, Direction::In | Direction::Both) {
            let existing: HashSet<String> = if matches!(direction, Direction::Both) {
                edges.iter().map(|e| e.id.as_str().to_string()).collect()
            } else {
                HashSet::new()
            };
            self.for_each_entry(id.as_str(), false, edge_type, |entry| {
                if !existing.contains(entry.edge_id) {
                    edges.extend(self.edge(entry.edge_id, corrupt)?);
                }
                Ok(())
            })?;
        }
        Ok(edges)
    }
}

/// The tables of one write transaction. Every change to the edges table goes
/// through here so the adjacency tables and the stamp change with it,
/// atomically.
struct DiskWriter<'t> {
    edges: Table<'t, &'static str, &'static [u8]>,
    out: MultimapTable<'t, &'static str, &'static [u8]>,
    inc: MultimapTable<'t, &'static str, &'static [u8]>,
    meta: Table<'t, &'static str, &'static [u8]>,
}

impl<'t> DiskWriter<'t> {
    fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(Self {
            edges: txn.open_table(EDGES_TABLE)?,
            out: txn.open_multimap_table(OUT_TABLE)?,
            inc: txn.open_multimap_table(IN_TABLE)?,
            meta: txn.open_table(META_TABLE)?,
        })
    }

    fn link(&mut self, from: &str, edge_type: &str, to: &str, edge_id: &str) -> Result<()> {
        self.out
            .insert(from, AdjEntry::encode(edge_type, to, edge_id).as_slice())?;
        self.inc
            .insert(to, AdjEntry::encode(edge_type, from, edge_id).as_slice())?;
        Ok(())
    }

    fn unlink(&mut self, from: &str, edge_type: &str, to: &str, edge_id: &str) -> Result<()> {
        self.out
            .remove(from, AdjEntry::encode(edge_type, to, edge_id).as_slice())?;
        self.inc
            .remove(to, AdjEntry::encode(edge_type, from, edge_id).as_slice())?;
        Ok(())
    }

    /// Store an edge and its two adjacency entries.
    fn insert(&mut self, edge: &Edge, bytes: &[u8]) -> Result<()> {
        let id = edge.id.as_str();
        let replaced = self
            .edges
            .insert(id, bytes)?
            .map(|old| old.value().to_vec());
        // Ids are fresh ULIDs, so this never fires in practice; if it did,
        // the replaced edge's entries would otherwise point at the new one.
        if let Some(old) = replaced.and_then(|b| Edge::from_bytes(&b).ok()) {
            self.unlink(old.from.as_str(), &old.edge_type, old.to.as_str(), id)?;
        }
        self.link(edge.from.as_str(), &edge.edge_type, edge.to.as_str(), id)
    }

    /// Remove an edge and its adjacency entries. Returns whether it existed.
    fn remove(&mut self, edge_id: &str) -> Result<bool> {
        let Some(bytes) = self.edges.remove(edge_id)?.map(|b| b.value().to_vec()) else {
            return Ok(false);
        };
        match Edge::from_bytes(&bytes) {
            Ok(edge) => self.unlink(
                edge.from.as_str(),
                &edge.edge_type,
                edge.to.as_str(),
                edge_id,
            )?,
            // Without the edge's endpoints its entries can only be found by
            // scanning; corruption is rare enough for that to be fine.
            Err(_) => self.purge(&HashSet::from([edge_id.to_string()]))?,
        }
        Ok(true)
    }

    /// Remove every edge touching `node`, from both directions. Needs no
    /// edge JSON: the node's adjacency entries name both endpoints.
    fn remove_node(&mut self, node: &str) -> Result<()> {
        let collect =
            |table: &MultimapTable<'t, &'static str, &'static [u8]>| -> Result<Vec<Vec<u8>>> {
                let mut values = Vec::new();
                for guard in table.get(node)? {
                    values.push(guard?.value().to_vec());
                }
                Ok(values)
            };
        let outgoing = collect(&self.out)?;
        let incoming = collect(&self.inc)?;
        for bytes in &outgoing {
            let Some(entry) = AdjEntry::decode(bytes) else {
                self.out.remove(node, bytes.as_slice())?;
                continue;
            };
            self.edges.remove(entry.edge_id)?;
            self.unlink(node, entry.edge_type, entry.other, entry.edge_id)?;
        }
        for bytes in &incoming {
            let Some(entry) = AdjEntry::decode(bytes) else {
                self.inc.remove(node, bytes.as_slice())?;
                continue;
            };
            self.edges.remove(entry.edge_id)?;
            self.unlink(entry.other, entry.edge_type, node, entry.edge_id)?;
        }
        Ok(())
    }

    /// Remove the given edges and every adjacency entry naming them, by
    /// scanning both adjacency tables. Only for edges whose JSON can't be
    /// decoded, so their endpoints are unknown.
    fn purge(&mut self, edge_ids: &HashSet<String>) -> Result<()> {
        for id in edge_ids {
            self.edges.remove(id.as_str())?;
        }
        for outgoing in [true, false] {
            let table = if outgoing { &self.out } else { &self.inc };
            let mut doomed: Vec<(String, Vec<u8>)> = Vec::new();
            for row in table.iter()? {
                let (key, values) = row?;
                for guard in values {
                    let guard = guard?;
                    let bytes = guard.value();
                    if AdjEntry::decode(bytes).is_some_and(|e| edge_ids.contains(e.edge_id)) {
                        doomed.push((key.value().to_string(), bytes.to_vec()));
                    }
                }
            }
            let table = if outgoing {
                &mut self.out
            } else {
                &mut self.inc
            };
            for (key, bytes) in &doomed {
                table.remove(key.as_str(), bytes.as_slice())?;
            }
        }
        Ok(())
    }

    /// Record that the adjacency tables match the edges table. Called last
    /// in every transaction that writes through this writer.
    fn finish(mut self) -> Result<()> {
        let stamp = adjacency_stamp(&self.edges)?;
        self.meta.insert(ADJACENCY_STAMP_KEY, stamp.as_slice())?;
        Ok(())
    }
}

/// Invalidate the adjacency tables, for writes that change the edges table
/// without maintaining them (the in-memory fallback). The next open then
/// rebuilds them rather than trusting stale entries.
fn clear_adjacency_stamp(txn: &WriteTransaction) -> Result<()> {
    let mut meta = txn.open_table(META_TABLE)?;
    meta.remove(ADJACENCY_STAMP_KEY)?;
    Ok(())
}

// ── In-memory adjacency index ───────────────────────────────────────

/// In-memory index for fast edge lookups, used only when the on-disk
/// adjacency tables can't be built (a store that can't be written).
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

    /// All edges for a record in a given direction; for `Both`, a self-loop
    /// is listed once.
    fn get_edges(&self, id: &RecordId, edge_type: Option<&str>, direction: Direction) -> Vec<Edge> {
        match direction {
            Direction::Out => self
                .get_outgoing(id, edge_type)
                .into_iter()
                .cloned()
                .collect(),
            Direction::In => self
                .get_incoming(id, edge_type)
                .into_iter()
                .cloned()
                .collect(),
            Direction::Both => {
                let mut edges: Vec<Edge> = self
                    .get_outgoing(id, edge_type)
                    .into_iter()
                    .cloned()
                    .collect();
                let existing: HashSet<_> = edges.iter().map(|e| e.id.clone()).collect();
                for e in self.get_incoming(id, edge_type) {
                    if !existing.contains(&e.id) {
                        edges.push(e.clone());
                    }
                }
                edges
            }
        }
    }

    /// Get neighbor record IDs for a node in the given direction.
    #[cfg(test)]
    fn neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<RecordId> {
        self.neighbor_ids_bitemporal(id, edge_type, direction, None, None)
    }

    /// Neighbor IDs on both temporal axes: only edges valid at `event_time`
    /// (when the fact was true) and recorded by `knowledge_time` (what the
    /// agent had on disk by then) are traversed. `None` disables an axis.
    fn neighbor_ids_bitemporal(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
        event_time: Option<&DateTime<Utc>>,
        knowledge_time: Option<&DateTime<Utc>>,
    ) -> Vec<RecordId> {
        let mut neighbors = Vec::new();
        let mut seen = HashSet::new();

        let is_valid = |edge: &Edge| -> bool {
            event_time.map_or(true, |t| edge.is_valid_at(t))
                && knowledge_time.map_or(true, |t| edge.known_at(t))
        };

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

/// Where the engine answers graph queries from, settled by the first graph
/// operation.
enum Store {
    /// The on-disk adjacency tables are current: a per-node operation reads
    /// only the entries of the nodes it touches.
    Disk,
    /// The adjacency tables couldn't be brought up to date, so every edge
    /// is held in memory instead, as a store without them always was.
    Memory(RwLock<AdjacencyIndex>),
}

/// Graph plugin for Axil — stores directed edges between records with
/// traversal and neighbor queries.
pub struct GraphEngine {
    graph_db: Database,
    /// Settled by the first graph operation, not at open, so commands that
    /// open the database without using the graph pay nothing: checking the
    /// adjacency stamp, and on a store that lacks the tables (or was written
    /// by a binary that doesn't maintain them) building them — a one-time
    /// cost of a full edge scan. A failed setup is kept, so every later
    /// operation reports it.
    store: OnceLock<std::result::Result<Store, String>>,
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

        // Only a new store needs its tables created. Opening an existing one
        // commits nothing, so a read-only command leaves the file untouched.
        let has_table = {
            let txn = graph_db.begin_read()?;
            match txn.open_table(EDGES_TABLE) {
                Ok(_) => true,
                Err(redb::TableError::TableDoesNotExist(_)) => false,
                Err(e) => return Err(e.into()),
            }
        };
        if !has_table {
            let txn = graph_db.begin_write()?;
            DiskWriter::open(&txn)?.finish()?;
            txn.commit()?;
        }

        Ok(Self {
            graph_db,
            store: OnceLock::new(),
        })
    }

    /// The query backend, settling it on first use.
    fn store(&self) -> Result<&Store> {
        self.store
            .get_or_init(|| {
                self.prepare().map_err(|e| {
                    let msg = format!("graph store unavailable: {e}");
                    eprintln!("axil: {msg}");
                    msg
                })
            })
            .as_ref()
            .map_err(|msg| AxilError::Plugin(Box::new(std::io::Error::other(msg.clone()))))
    }

    /// Bring the adjacency tables up to date, or fall back to holding every
    /// edge in memory when they can't be written.
    fn prepare(&self) -> Result<Store> {
        let ready = match self.adjacency_is_current() {
            Ok(true) => Ok(()),
            Ok(false) => self.build_adjacency(),
            Err(e) => Err(e),
        };
        match ready {
            Ok(()) => {
                let count = DiskView::open(&self.graph_db)?.edges.len()? as usize;
                if count > MAX_EDGES {
                    return Err(AxilError::Plugin(Box::new(std::io::Error::other(format!(
                        "graph store has {count} edges, exceeding limit of {MAX_EDGES}"
                    )))));
                }
                Ok(Store::Disk)
            }
            Err(e) => {
                eprintln!("axil: graph adjacency tables unavailable ({e}); loading every edge into memory");
                Ok(Store::Memory(RwLock::new(self.load_index()?)))
            }
        }
    }

    /// Whether the adjacency tables were last written against the edges
    /// table as it is now (see [`adjacency_stamp`]).
    fn adjacency_is_current(&self) -> Result<bool> {
        let txn = self.graph_db.begin_read()?;
        let stored = match txn.open_table(META_TABLE) {
            Ok(meta) => meta.get(ADJACENCY_STAMP_KEY)?.map(|g| g.value().to_vec()),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(e.into()),
        };
        let Some(stored) = stored else {
            return Ok(false);
        };
        let edges = txn.open_table(EDGES_TABLE)?;
        Ok(stored == adjacency_stamp(&edges)?)
    }

    /// Rebuild the adjacency tables from the edges table, dropping corrupt
    /// edges, and stamp them — all in one transaction, so a crash leaves
    /// either the old state (rebuilt again next time) or a complete one.
    fn build_adjacency(&self) -> Result<()> {
        let txn = self.graph_db.begin_write()?;
        txn.delete_multimap_table(OUT_TABLE)?;
        txn.delete_multimap_table(IN_TABLE)?;
        {
            let mut w = DiskWriter::open(&txn)?;
            let mut corrupt: Vec<String> = Vec::new();
            for row in w.edges.iter()? {
                let (key, value) = row?;
                let id = key.value();
                let Some(edge) = decode_edge(id, value.value(), &mut corrupt) else {
                    continue;
                };
                let (from, to) = (edge.from.as_str(), edge.to.as_str());
                w.out
                    .insert(from, AdjEntry::encode(&edge.edge_type, to, id).as_slice())?;
                w.inc
                    .insert(to, AdjEntry::encode(&edge.edge_type, from, id).as_slice())?;
            }
            for id in &corrupt {
                w.edges.remove(id.as_str())?;
            }
            w.finish()?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read every edge into memory, removing corrupt entries from disk.
    fn load_index(&self) -> Result<AdjacencyIndex> {
        let graph_db = &self.graph_db;
        let mut adj = AdjacencyIndex::new();
        let mut corrupt_keys: Vec<String> = Vec::new();
        {
            let txn = graph_db.begin_read()?;
            let table = txn.open_table(EDGES_TABLE)?;
            let iter = table.iter()?;
            for entry in iter {
                let entry = entry?;
                let key = entry.0.value().to_string();
                if let Some(edge) = decode_edge(&key, entry.1.value(), &mut corrupt_keys) {
                    adj.add(edge);
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
            clear_adjacency_stamp(&txn)?;
            txn.commit()?;
        }

        if adj.edge_count() > MAX_EDGES {
            return Err(AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "graph store has {} edges, exceeding limit of {MAX_EDGES}",
                adj.edge_count()
            )))));
        }

        Ok(adj)
    }

    /// Remove edges found corrupt while reading, with their adjacency
    /// entries. Best-effort: a failure is reported and the edges stay
    /// skipped by every read that decodes them.
    fn purge_corrupt(&self, corrupt: Vec<String>) {
        if corrupt.is_empty() {
            return;
        }
        let ids: HashSet<String> = corrupt.into_iter().collect();
        let purge = || -> Result<()> {
            let txn = self.graph_db.begin_write()?;
            {
                let mut w = DiskWriter::open(&txn)?;
                w.purge(&ids)?;
                w.finish()?;
            }
            txn.commit()?;
            Ok(())
        };
        if let Err(e) = purge() {
            eprintln!("warning: failed to remove corrupt edges: {e}");
        }
    }

    /// Run a read against a snapshot of the on-disk store, then remove any
    /// corrupt edges it ran into.
    fn with_view<T>(
        &self,
        read: impl FnOnce(&DiskView, &mut Vec<String>) -> Result<T>,
    ) -> Result<T> {
        let mut corrupt = Vec::new();
        let result = {
            let view = DiskView::open(&self.graph_db)?;
            read(&view, &mut corrupt)
        };
        self.purge_corrupt(corrupt);
        result
    }

    /// Validate edge specs the way every edge-creating call does.
    fn validate_edge(edge_type: &str, properties: &Value) -> Result<()> {
        // Reject control characters in edge type to prevent terminal injection.
        if edge_type.bytes().any(|b| b < 0x20 || b == 0x7F) {
            return Err(AxilError::InvalidQuery(
                "edge type must not contain control characters".into(),
            ));
        }
        let prop_size = serde_json::to_string(properties)
            .map(|s| s.len())
            .unwrap_or(0);
        if prop_size > MAX_EDGE_PROPERTY_BYTES {
            return Err(AxilError::InvalidQuery(format!(
                "edge properties exceed {MAX_EDGE_PROPERTY_BYTES} byte limit ({prop_size} bytes)",
            )));
        }
        Ok(())
    }

    /// Store already-built edges, enforcing MAX_EDGES against the count at
    /// write time. `batch` only selects the limit error's wording.
    ///
    /// On disk the count check and the writes share one redb write
    /// transaction, which redb serializes, so concurrent callers can't
    /// exceed the limit. In memory the index's write lock is held across
    /// the check, the disk write and the index update for the same reason;
    /// disk is written before memory so a crash leaves them consistent.
    fn add_edges(&self, edges: &[Edge], batch: bool) -> Result<()> {
        let limit_error = || {
            if batch {
                AxilError::InvalidQuery(format!(
                    "edge limit reached ({MAX_EDGES}); batch of {} would exceed",
                    edges.len()
                ))
            } else {
                AxilError::InvalidQuery(format!("edge limit reached ({MAX_EDGES})"))
            }
        };
        // Serialize outside any transaction so its time is dominated by
        // I/O, not CPU.
        let serialized: Vec<Vec<u8>> = edges
            .iter()
            .map(|e| {
                e.to_bytes()
                    .map_err(|e| AxilError::Serialization(Box::new(e)))
            })
            .collect::<Result<Vec<_>>>()?;

        match self.store()? {
            Store::Disk => {
                let txn = self.graph_db.begin_write()?;
                {
                    let mut w = DiskWriter::open(&txn)?;
                    let count = w.edges.len()? as usize;
                    if count + edges.len() > MAX_EDGES {
                        return Err(limit_error());
                    }
                    for (edge, bytes) in edges.iter().zip(&serialized) {
                        w.insert(edge, bytes)?;
                    }
                    w.finish()?;
                }
                txn.commit()?;
            }
            Store::Memory(index) => {
                let mut idx = index.write();
                let count = idx.edge_count();
                if count + edges.len() > MAX_EDGES {
                    return Err(limit_error());
                }
                let txn = self.graph_db.begin_write()?;
                {
                    let mut table = txn.open_table(EDGES_TABLE)?;
                    for (edge, bytes) in edges.iter().zip(&serialized) {
                        table.insert(edge.id.as_str(), bytes.as_slice())?;
                    }
                }
                clear_adjacency_stamp(&txn)?;
                txn.commit()?;
                for edge in edges {
                    idx.add(edge.clone());
                }
            }
        }
        Ok(())
    }

    /// Create a directed edge between two records.
    ///
    /// The count check and the write are atomic, so concurrent callers
    /// cannot exceed MAX_EDGES.
    pub fn create_edge(
        &self,
        from: RecordId,
        edge_type: &str,
        to: RecordId,
        properties: Value,
    ) -> Result<Edge> {
        Self::validate_edge(edge_type, &properties)?;
        let edge = Edge::new(from, edge_type, to, properties);
        self.add_edges(std::slice::from_ref(&edge), false)?;
        Ok(edge)
    }

    /// Create many directed edges in a single redb transaction.
    ///
    /// Drop-in batched equivalent of `create_edge` — same validation
    /// (control-char check, property size, MAX_EDGES) but folds the N
    /// per-edge `begin_write/commit` pairs into one. SCIP ingest spends
    /// >90% of its wall time in those commits; batching cuts an
    /// edge-heavy workload from minutes to seconds (see dogfood friction #8).
    pub fn create_edges_batch(
        &self,
        specs: Vec<(RecordId, String, RecordId, Value)>,
    ) -> Result<Vec<Edge>> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        for (_, edge_type, _, props) in &specs {
            Self::validate_edge(edge_type, props)?;
        }
        let edges: Vec<Edge> = specs
            .into_iter()
            .map(|(from, edge_type, to, props)| Edge::new(from, &edge_type, to, props))
            .collect();
        // Enforced against the post-batch count so a partial batch can't
        // push the store over MAX_EDGES.
        self.add_edges(&edges, true)?;
        Ok(edges)
    }

    /// Delete an edge by ID.
    ///
    /// Atomic with respect to other writers: concurrent deleters cannot both
    /// observe the edge as present.
    pub fn delete_edge(&self, edge_id: &RecordId) -> Result<bool> {
        match self.store()? {
            Store::Disk => {
                let txn = self.graph_db.begin_write()?;
                let existed = {
                    let mut w = DiskWriter::open(&txn)?;
                    let existed = w.remove(edge_id.as_str())?;
                    if existed {
                        w.finish()?;
                    }
                    existed
                };
                if existed {
                    txn.commit()?;
                } else {
                    txn.abort()?;
                }
                Ok(existed)
            }
            Store::Memory(index) => {
                let mut idx = index.write();
                if !idx.edges.contains_key(edge_id) {
                    return Ok(false);
                }
                self.remove_edges_from_disk(std::slice::from_ref(edge_id))?;
                idx.remove(edge_id);
                Ok(true)
            }
        }
    }

    /// Get an edge by ID. `None` also when the store can't be loaded (the
    /// load error is printed once).
    pub fn get_edge(&self, edge_id: &RecordId) -> Option<Edge> {
        match self.store().ok()? {
            Store::Disk => self
                .with_view(|view, corrupt| view.edge(edge_id.as_str(), corrupt))
                .ok()
                .flatten(),
            Store::Memory(index) => index.read().edges.get(edge_id).cloned(),
        }
    }

    /// Get outgoing edges from a record (none when the store can't be loaded).
    pub fn get_outgoing(&self, from: &RecordId, edge_type: Option<&str>) -> Vec<Edge> {
        self.get_edges(from, edge_type, Direction::Out)
    }

    /// Get incoming edges to a record (none when the store can't be loaded).
    pub fn get_incoming(&self, to: &RecordId, edge_type: Option<&str>) -> Vec<Edge> {
        self.get_edges(to, edge_type, Direction::In)
    }

    /// Get all edges for a record in a given direction, optionally filtered by
    /// type (none when the store can't be loaded).
    pub fn get_edges(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<Edge> {
        self.try_get_edges(id, edge_type, direction)
            .unwrap_or_default()
    }

    fn try_get_edges(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<Edge>> {
        match self.store()? {
            Store::Disk => {
                self.with_view(|view, corrupt| view.edges_of(id, edge_type, direction, corrupt))
            }
            Store::Memory(index) => Ok(index.read().get_edges(id, edge_type, direction)),
        }
    }

    /// Get neighbor record IDs reachable via edges in the given direction
    /// (none when the store can't be loaded).
    pub fn neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<RecordId> {
        self.try_neighbor_ids(id, edge_type, direction)
            .unwrap_or_default()
    }

    fn try_neighbor_ids(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<RecordId>> {
        match self.store()? {
            Store::Disk => self.with_view(|view, corrupt| {
                view.neighbor_ids(id, edge_type, direction, None, None, corrupt)
            }),
            Store::Memory(index) => Ok(index
                .read()
                .neighbor_ids_bitemporal(id, edge_type, direction, None, None)),
        }
    }

    /// Multi-hop traversal following a sequence of steps.
    /// Returns the record IDs at the end of the path. The whole traversal
    /// reads one consistent snapshot.
    ///
    /// Each step expands the current frontier to its neighbors (filtered
    /// by edge type and direction), deduplicating within each step.
    /// Nodes may reappear across steps — a path `a ->knows-> b ->knows-> a`
    /// correctly returns `[a]`. Infinite loops are impossible because the
    /// number of steps is fixed (bounded by `MAX_DEPTH`).
    pub fn traverse_ids(&self, start: &RecordId, steps: &[TraversalStep]) -> Result<Vec<RecordId>> {
        self.traverse_ids_bitemporal(start, steps, None, None)
    }

    /// Multi-hop traversal on both temporal axes (bi-temporal query).
    ///
    /// `event_time` filters edges by when the fact was true (`valid_from` /
    /// `valid_until`); `knowledge_time` filters by when the edge had been
    /// recorded (`created_at`). Together they answer "what did the graph look
    /// like, as we knew it at K, for events at E?" — e.g. reviewing a
    /// decision made before a correction arrived. `None` on either axis
    /// disables that axis's filter.
    pub fn traverse_ids_bitemporal(
        &self,
        start: &RecordId,
        steps: &[TraversalStep],
        event_time: Option<&DateTime<Utc>>,
        knowledge_time: Option<&DateTime<Utc>>,
    ) -> Result<Vec<RecordId>> {
        if steps.is_empty() {
            return Ok(vec![start.clone()]);
        }

        let expand = |neighbors: &mut dyn FnMut(
            &RecordId,
            &TraversalStep,
        ) -> Result<Vec<RecordId>>|
         -> Result<Vec<RecordId>> {
            let mut current: Vec<RecordId> = vec![start.clone()];
            for step in steps {
                let mut next = Vec::new();
                let mut seen = HashSet::new();
                for node in &current {
                    for n in neighbors(node, step)? {
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
        };

        match self.store()? {
            Store::Disk => self.with_view(|view, corrupt| {
                expand(&mut |node, step| {
                    view.neighbor_ids(
                        node,
                        Some(&step.edge_type),
                        step.direction,
                        event_time,
                        knowledge_time,
                        corrupt,
                    )
                })
            }),
            Store::Memory(index) => {
                let idx = index.read();
                expand(&mut |node, step| {
                    Ok(idx.neighbor_ids_bitemporal(
                        node,
                        Some(&step.edge_type),
                        step.direction,
                        event_time,
                        knowledge_time,
                    ))
                })
            }
        }
    }

    /// Multi-hop traversal with optional temporal filtering.
    ///
    /// When `as_of` is Some, only edges valid at that point in time are traversed.
    /// This enables "what did the agent know at time T?" queries.
    pub fn traverse_ids_temporal(
        &self,
        start: &RecordId,
        steps: &[TraversalStep],
        as_of: Option<&DateTime<Utc>>,
    ) -> Result<Vec<RecordId>> {
        self.traverse_ids_bitemporal(start, steps, as_of, None)
    }

    /// Total edge count (0 when the store can't be loaded).
    pub fn edge_count(&self) -> usize {
        match self.store() {
            Ok(Store::Disk) => self
                .with_view(|view, _| Ok(view.edges.len()? as usize))
                .unwrap_or(0),
            Ok(Store::Memory(index)) => index.read().edge_count(),
            Err(_) => 0,
        }
    }

    /// Every edge as `(edge_id, from, to)`, read from the outgoing adjacency
    /// table, which carries all three without decoding edge JSON.
    fn try_all_edge_ids(&self) -> Result<Vec<(RecordId, RecordId, RecordId)>> {
        match self.store()? {
            Store::Disk => self.with_view(|view, _| {
                let mut all = Vec::new();
                for row in view.out.iter()? {
                    let (from, values) = row?;
                    for guard in values {
                        let guard = guard?;
                        if let Some(entry) = AdjEntry::decode(guard.value()) {
                            all.push((
                                RecordId(entry.edge_id.to_string()),
                                RecordId(from.value().to_string()),
                                RecordId(entry.other.to_string()),
                            ));
                        }
                    }
                }
                Ok(all)
            }),
            Store::Memory(index) => Ok(index
                .read()
                .edges
                .values()
                .map(|e| (e.id.clone(), e.from.clone(), e.to.clone()))
                .collect()),
        }
    }

    // ── Persistence helpers (in-memory fallback) ────────────────────

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
        clear_adjacency_stamp(&txn)?;
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
        match self.store()? {
            // One write transaction: no edge for this record can be added
            // between collecting its edges and removing them.
            Store::Disk => {
                let txn = self.graph_db.begin_write()?;
                {
                    let mut w = DiskWriter::open(&txn)?;
                    w.remove_node(id.as_str())?;
                    w.finish()?;
                }
                txn.commit()?;
                Ok(())
            }
            // Hold the write lock for the entire operation for the same
            // reason. Disk is updated before memory within the lock.
            Store::Memory(index) => {
                let mut idx = index.write();
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
        self.try_neighbor_ids(&id, edge_type, direction)
    }

    fn edges(
        &self,
        id: RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<EdgeInfo>> {
        Ok(self
            .try_get_edges(&id, edge_type, direction)?
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
        self.try_all_edge_ids()
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

#[cfg(test)]
mod bitemporal_tests {
    use super::*;
    use crate::edge::Edge;
    use chrono::{Duration, Utc};
    use serde_json::json;

    #[test]
    fn knowledge_time_predicate_excludes_future_created_edges() {
        let now = Utc::now();
        let mut edge = Edge::new(RecordId::new(), "depends_on", RecordId::new(), json!({}));
        edge.created_at = now - Duration::hours(1);
        assert!(edge.known_at(&now));
        assert!(!edge.known_at(&(now - Duration::hours(2))));

        // valid window is independent of knowledge time
        edge.valid_from = Some(now + Duration::hours(5));
        assert!(edge.known_at(&now));
        assert!(!edge.visible_at(&now, &now));
        assert!(edge.visible_at(&(now + Duration::hours(6)), &now));
    }

    #[test]
    fn bitemporal_traversal_filters_by_knowledge_time() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("bt.axil");
        let engine = GraphEngine::open(&db_path).unwrap();
        let from = RecordId::new();
        let to = RecordId::new();
        engine
            .create_edge(from.clone(), "depends_on", to.clone(), json!({}))
            .unwrap();
        drop(engine);

        let engine = GraphEngine::open(&db_path).unwrap();
        let steps = vec![TraversalStep {
            edge_type: "depends_on".to_string(),
            direction: Direction::Out,
        }];

        // Knowledge cutoff before the edge existed: not traversable.
        let before = Utc::now() - Duration::hours(1);
        let r = engine
            .traverse_ids_bitemporal(&from, &steps, None, Some(&before))
            .unwrap();
        assert!(!r.contains(&to));

        // Cutoff after creation: traversable.
        let after = Utc::now() + Duration::hours(1);
        let r = engine
            .traverse_ids_bitemporal(&from, &steps, None, Some(&after))
            .unwrap();
        assert!(r.contains(&to));

        // Single-axis (event-time-only) traversal still works unchanged.
        let r = engine.traverse_ids_temporal(&from, &steps, None).unwrap();
        assert!(r.contains(&to));
    }
}

/// Parity oracle for the on-disk adjacency tables: random operation
/// sequences are applied to a `GraphEngine` and mirrored into the in-memory
/// `AdjacencyIndex` (the pre-existing implementation), and every per-node
/// query is compared on every node.
#[cfg(test)]
mod adjacency_parity_tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    const TYPES: [&str; 3] = ["knows", "mentions", "depends_on"];
    const DIRECTIONS: [Direction; 3] = [Direction::Out, Direction::In, Direction::Both];

    /// xorshift64*: deterministic per seed, no dev-dependency needed.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn chance(&mut self, pct: usize) -> bool {
            self.below(100) < pct
        }
        fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
            &items[self.below(items.len())]
        }
    }

    fn nodes(n: usize) -> Vec<RecordId> {
        (0..n).map(|i| RecordId(format!("node{i:02}"))).collect()
    }

    fn base_time() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    /// An edge with a random event-time window and knowledge time.
    fn temporal_edge(rng: &mut Rng, nodes: &[RecordId]) -> Edge {
        let base = base_time();
        let mut edge = Edge::new(
            rng.pick(nodes).clone(),
            *rng.pick(&TYPES),
            rng.pick(nodes).clone(),
            json!({"w": rng.below(10)}),
        );
        edge.created_at = base + Duration::hours(rng.below(96) as i64 - 48);
        if rng.chance(40) {
            edge.valid_from = Some(base + Duration::hours(rng.below(96) as i64 - 48));
        }
        if rng.chance(40) {
            edge.valid_until = Some(base + Duration::hours(rng.below(96) as i64 - 48));
        }
        edge
    }

    fn query_times() -> Vec<DateTime<Utc>> {
        let base = base_time();
        [-60, -24, -1, 0, 7, 30, 60]
            .iter()
            .map(|h| base + Duration::hours(*h))
            .collect()
    }

    /// The pre-existing traversal algorithm, run on the in-memory index.
    fn oracle_traverse(
        idx: &AdjacencyIndex,
        start: &RecordId,
        steps: &[TraversalStep],
        event_time: Option<&DateTime<Utc>>,
        knowledge_time: Option<&DateTime<Utc>>,
    ) -> Vec<RecordId> {
        if steps.is_empty() {
            return vec![start.clone()];
        }
        let mut current = vec![start.clone()];
        for step in steps {
            let mut next = Vec::new();
            let mut seen = HashSet::new();
            for node in &current {
                for n in idx.neighbor_ids_bitemporal(
                    node,
                    Some(&step.edge_type),
                    step.direction,
                    event_time,
                    knowledge_time,
                ) {
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
        current
    }

    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort();
        v
    }

    fn ids(edges: &[Edge]) -> Vec<String> {
        sorted(edges.iter().map(|e| e.id.as_str().to_string()).collect())
    }

    fn rid_strings(v: Vec<RecordId>) -> Vec<String> {
        sorted(v.into_iter().map(|r| r.0).collect())
    }

    fn random_steps(rng: &mut Rng) -> Vec<TraversalStep> {
        (0..1 + rng.below(3))
            .map(|_| TraversalStep {
                edge_type: rng.pick(&TYPES).to_string(),
                direction: *rng.pick(&DIRECTIONS),
            })
            .collect()
    }

    /// Compare every query the engine answers against the oracle.
    fn assert_parity(g: &GraphEngine, oracle: &AdjacencyIndex, nodes: &[RecordId], rng: &mut Rng) {
        assert_eq!(g.edge_count(), oracle.edge_count(), "edge_count");
        assert_eq!(GraphIndex::edge_count(g), oracle.edge_count());

        let all: Vec<_> = sorted(
            g.all_edge_ids()
                .unwrap()
                .into_iter()
                .map(|(e, f, t)| (e.0, f.0, t.0))
                .collect(),
        );
        let want: Vec<_> = sorted(
            oracle
                .edges
                .values()
                .map(|e| (e.id.0.clone(), e.from.0.clone(), e.to.0.clone()))
                .collect(),
        );
        assert_eq!(all, want, "all_edge_ids");

        for (id, expected) in &oracle.edges {
            let got = g.get_edge(id).expect("edge present");
            assert_eq!(got.from, expected.from);
            assert_eq!(got.to, expected.to);
            assert_eq!(got.edge_type, expected.edge_type);
            assert_eq!(got.properties, expected.properties);
            assert_eq!(got.created_at, expected.created_at);
            assert_eq!(got.valid_from, expected.valid_from);
            assert_eq!(got.valid_until, expected.valid_until);
        }

        let mut probe: Vec<RecordId> = nodes.to_vec();
        probe.push(RecordId("never-linked".into()));
        let times = query_times();

        let mut filters: Vec<Option<&str>> = vec![None, Some("absent")];
        filters.extend(TYPES.iter().map(|t| Some(*t)));

        for node in &probe {
            for dir in DIRECTIONS {
                for et in &filters {
                    let ctx = format!("{node:?} {dir:?} {et:?}");
                    // Sorted vectors, not sets: multiplicity must match too.
                    let want = rid_strings(oracle.neighbor_ids(node, *et, dir));
                    assert_eq!(
                        rid_strings(g.neighbor_ids(node, *et, dir)),
                        want,
                        "neighbor_ids {ctx}"
                    );
                    assert_eq!(
                        rid_strings(GraphIndex::neighbors(g, node.clone(), *et, dir).unwrap()),
                        want,
                        "neighbors {ctx}"
                    );

                    let want = ids(&oracle.get_edges(node, *et, dir));
                    assert_eq!(ids(&g.get_edges(node, *et, dir)), want, "get_edges {ctx}");
                    let infos = GraphIndex::edges(g, node.clone(), *et, dir).unwrap();
                    assert_eq!(
                        sorted(infos.iter().map(|e| e.id.0.clone()).collect::<Vec<_>>()),
                        want,
                        "edges {ctx}"
                    );
                }
            }
            for et in &filters {
                let out: Vec<Edge> = oracle
                    .get_outgoing(node, *et)
                    .into_iter()
                    .cloned()
                    .collect();
                assert_eq!(ids(&g.get_outgoing(node, *et)), ids(&out));
                let inc: Vec<Edge> = oracle
                    .get_incoming(node, *et)
                    .into_iter()
                    .cloned()
                    .collect();
                assert_eq!(ids(&g.get_incoming(node, *et)), ids(&inc));
            }

            for _ in 0..4 {
                let steps = random_steps(rng);
                let ctx = format!("{node:?} {steps:?}");
                assert_eq!(
                    rid_strings(g.traverse_ids(node, &steps).unwrap()),
                    rid_strings(oracle_traverse(oracle, node, &steps, None, None)),
                    "traverse {ctx}"
                );
                let ev = *rng.pick(&times);
                assert_eq!(
                    rid_strings(g.traverse_ids_temporal(node, &steps, Some(&ev)).unwrap()),
                    rid_strings(oracle_traverse(oracle, node, &steps, Some(&ev), None)),
                    "temporal {ctx} {ev}"
                );
                let kn = *rng.pick(&times);
                let ev = if rng.chance(50) {
                    Some(*rng.pick(&times))
                } else {
                    None
                };
                assert_eq!(
                    rid_strings(
                        g.traverse_ids_bitemporal(node, &steps, ev.as_ref(), Some(&kn))
                            .unwrap()
                    ),
                    rid_strings(oracle_traverse(
                        oracle,
                        node,
                        &steps,
                        ev.as_ref(),
                        Some(&kn)
                    )),
                    "bitemporal {ctx} {ev:?} {kn}"
                );
            }
        }
    }

    /// Apply `ops` random operations to `g`, mirroring them into `oracle`,
    /// checking parity every `check_every` operations.
    fn run_ops(
        g: &GraphEngine,
        oracle: &mut AdjacencyIndex,
        nodes: &[RecordId],
        rng: &mut Rng,
        ops: usize,
        check_every: usize,
    ) {
        for step in 1..=ops {
            match rng.below(100) {
                0..=29 => {
                    let from = rng.pick(nodes).clone();
                    let edge_type = *rng.pick(&TYPES);
                    let to = rng.pick(nodes).clone();
                    let e = g.create_edge(from, edge_type, to, json!({})).unwrap();
                    oracle.add(e);
                }
                30..=44 => {
                    let n = 1 + rng.below(6);
                    let specs = (0..n)
                        .map(|_| {
                            (
                                rng.pick(nodes).clone(),
                                rng.pick(&TYPES).to_string(),
                                rng.pick(nodes).clone(),
                                json!({"batch": true}),
                            )
                        })
                        .collect();
                    for e in g.create_edges_batch(specs).unwrap() {
                        oracle.add(e);
                    }
                }
                45..=59 => {
                    // Temporal edges go through the path both creators use.
                    let n = 1 + rng.below(4);
                    let edges: Vec<Edge> = (0..n).map(|_| temporal_edge(rng, nodes)).collect();
                    g.add_edges(&edges, edges.len() > 1).unwrap();
                    for e in edges {
                        oracle.add(e);
                    }
                }
                60..=67 => {
                    // A parallel edge: same endpoints and type as an existing one.
                    let mut existing: Vec<Edge> = oracle.edges.values().cloned().collect();
                    existing.sort_by(|a, b| a.id.cmp(&b.id));
                    if !existing.is_empty() {
                        let e = rng.pick(&existing);
                        let dup = g
                            .create_edge(e.from.clone(), &e.edge_type, e.to.clone(), json!({}))
                            .unwrap();
                        oracle.add(dup);
                    }
                }
                68..=84 => {
                    let mut existing: Vec<RecordId> = oracle.edges.keys().cloned().collect();
                    existing.sort();
                    let target = if existing.is_empty() || rng.chance(10) {
                        RecordId::new()
                    } else {
                        rng.pick(&existing).clone()
                    };
                    let expected = oracle.remove(&target).is_some();
                    assert_eq!(g.delete_edge(&target).unwrap(), expected);
                }
                _ => {
                    let node = rng.pick(nodes).clone();
                    g.on_record_delete(&node).unwrap();
                    oracle.remove_edges_for_record(&node);
                }
            }
            if step % check_every == 0 {
                assert_parity(g, oracle, nodes, rng);
            }
        }
    }

    fn graph_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("parity.axil")
    }

    fn open_raw(dir: &tempfile::TempDir) -> Database {
        Database::create(companion_path(&graph_path(dir), ".graph")).unwrap()
    }

    /// Force the in-memory fallback, as when the store can't be written.
    fn force_memory(g: &GraphEngine) {
        let idx = g.load_index().unwrap();
        assert!(g.store.set(Ok(Store::Memory(RwLock::new(idx)))).is_ok());
    }

    fn is_disk(g: &GraphEngine) -> bool {
        matches!(g.store(), Ok(Store::Disk))
    }

    #[test]
    fn disk_adjacency_matches_in_memory_oracle() {
        for seed in 1..=8u64 {
            let dir = tempfile::tempdir().unwrap();
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            let mut oracle = AdjacencyIndex::new();
            let nodes = nodes(10);
            let mut rng = Rng::new(seed);
            run_ops(&g, &mut oracle, &nodes, &mut rng, 240, 40);
            assert!(is_disk(&g));
            assert!(g.adjacency_is_current().unwrap());
        }
    }

    #[test]
    fn memory_fallback_matches_oracle() {
        for seed in 11..=13u64 {
            let dir = tempfile::tempdir().unwrap();
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            force_memory(&g);
            let mut oracle = AdjacencyIndex::new();
            let nodes = nodes(8);
            let mut rng = Rng::new(seed);
            run_ops(&g, &mut oracle, &nodes, &mut rng, 150, 50);
        }
    }

    #[test]
    fn reopen_keeps_adjacency_without_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(10);
        let mut rng = Rng::new(21);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 200, 100);
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(g.adjacency_is_current().unwrap(), "stamp survives reopen");
        assert_parity(&g, &oracle, &nodes, &mut rng);
        // Writes after the reopen keep the tables current.
        run_ops(&g, &mut oracle, &nodes, &mut rng, 100, 50);
        drop(g);
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(g.adjacency_is_current().unwrap());
        assert_parity(&g, &oracle, &nodes, &mut rng);
    }

    #[test]
    fn store_with_only_edges_table_is_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(10);
        let mut rng = Rng::new(31);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 200, 200);
        }
        // Reduce the store to what an older binary wrote: just "edges".
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            assert!(txn.delete_multimap_table(OUT_TABLE).unwrap());
            assert!(txn.delete_multimap_table(IN_TABLE).unwrap());
            assert!(txn.delete_table(META_TABLE).unwrap());
            txn.commit().unwrap();
        }
        // Opening alone doesn't migrate: that waits for a graph operation.
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        drop(g);
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());

        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(is_disk(&g));
        assert!(g.adjacency_is_current().unwrap());

        // The build is idempotent: running it again changes nothing.
        g.build_adjacency().unwrap();
        g.build_adjacency().unwrap();
        assert_parity(&g, &oracle, &nodes, &mut rng);
        run_ops(&g, &mut oracle, &nodes, &mut rng, 80, 40);
    }

    #[test]
    fn edges_written_without_adjacency_trigger_rebuild() {
        // An older binary on the same store writes "edges" only; the stamp
        // no longer matches, so the next process rebuilds instead of
        // answering from stale adjacency.
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(6);
        let mut rng = Rng::new(41);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 60, 60);
        }
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            {
                let mut edges = txn.open_table(EDGES_TABLE).unwrap();
                // Delete one edge and add one: the row count is unchanged.
                let victim = oracle.edges.keys().min().cloned().unwrap();
                edges.remove(victim.as_str()).unwrap();
                oracle.remove(&victim);
                let added = Edge::new(nodes[0].clone(), "knows", nodes[1].clone(), json!({}));
                edges
                    .insert(added.id.as_str(), added.to_bytes().unwrap().as_slice())
                    .unwrap();
                oracle.add(added);
            }
            txn.commit().unwrap();
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        assert_parity(&g, &oracle, &nodes, &mut rng);
    }

    #[test]
    fn memory_fallback_writes_invalidate_adjacency() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(6);
        let mut rng = Rng::new(51);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 60, 60);
        }
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            force_memory(&g);
            run_ops(&g, &mut oracle, &nodes, &mut rng, 40, 40);
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
    }

    #[test]
    fn corrupt_edge_in_unmigrated_store_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(6);
        let mut rng = Rng::new(61);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 50, 50);
        }
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            {
                let mut edges = txn.open_table(EDGES_TABLE).unwrap();
                edges
                    .insert("corrupt-edge", b"{not json".as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert_parity(&g, &oracle, &nodes, &mut rng);
        let view = DiskView::open(&g.graph_db).unwrap();
        assert!(view.edges.get("corrupt-edge").unwrap().is_none());
    }

    #[test]
    fn corrupt_edge_after_migration_is_purged_with_its_adjacency() {
        let dir = tempfile::tempdir().unwrap();
        let a = RecordId("a".into());
        let b = RecordId("b".into());
        let c = RecordId("c".into());
        let (bad, bad2, good) = {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            let bad = g
                .create_edge(a.clone(), "knows", b.clone(), json!({}))
                .unwrap();
            let bad2 = g
                .create_edge(c.clone(), "knows", a.clone(), json!({}))
                .unwrap();
            let good = g
                .create_edge(a.clone(), "knows", c.clone(), json!({}))
                .unwrap();
            (bad, bad2, good)
        };
        // Overwrite two edges' bytes in place: count and last id unchanged,
        // so the stamp still matches and the adjacency tables are trusted.
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            {
                let mut edges = txn.open_table(EDGES_TABLE).unwrap();
                edges
                    .insert(bad.id.as_str(), b"garbage".as_slice())
                    .unwrap();
                edges
                    .insert(bad2.id.as_str(), b"garbage".as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(g.adjacency_is_current().unwrap());

        // Reading the edges skips the corrupt one and purges it.
        let out = g.get_edges(&a, None, Direction::Out);
        assert_eq!(ids(&out), vec![good.id.0.clone()]);
        assert_eq!(g.neighbor_ids(&a, None, Direction::Out), vec![c.clone()]);
        assert!(g.neighbor_ids(&b, None, Direction::In).is_empty());
        assert!(g.get_edge(&bad.id).is_none());

        // Deleting a corrupt edge by id removes its adjacency entries too.
        assert!(g.delete_edge(&bad2.id).unwrap());
        assert!(g.neighbor_ids(&a, None, Direction::In).is_empty());
        assert!(g.neighbor_ids(&c, None, Direction::Out).is_empty());

        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.all_edge_ids().unwrap().len(), 1);
        assert!(g.adjacency_is_current().unwrap());
    }

    #[test]
    fn adjacency_entry_round_trips() {
        let bytes = AdjEntry::encode("depends_on", "node:with\u{0}odd bytes", "01ABC");
        let e = AdjEntry::decode(&bytes).unwrap();
        assert_eq!(e.edge_type, "depends_on");
        assert_eq!(e.other, "node:with\u{0}odd bytes");
        assert_eq!(e.edge_id, "01ABC");
        assert!(AdjEntry::decode(&bytes[..3]).is_none());
    }
}
