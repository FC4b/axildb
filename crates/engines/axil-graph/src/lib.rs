pub mod edge;
pub mod pagerank;
pub mod traverse;

use std::collections::{HashMap, HashSet};
use std::ops::Bound;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use redb::{
    Database, MultimapTable, MultimapTableDefinition, ReadOnlyMultimapTable, ReadOnlyTable,
    ReadTransaction, ReadableDatabase, ReadableMultimapTable, ReadableTable, ReadableTableMetadata,
    Table, TableDefinition, WriteTransaction,
};
use serde_json::Value;

use axil_core::plugin::{Capability, Direction, EdgeInfo, Engine, GraphIndex, TraversalStep};
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

/// Most adjacency changes one write transaction of a sync commits (see
/// [`GraphEngine::sync_adjacency_in_chunks`]). Small enough that a killed
/// sync loses little of its inserts, large enough that the commits stay
/// few: a first build of 322k entries (two per edge on a 161k-edge store)
/// takes seven.
const SYNC_CHUNK: usize = 50_000;

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
///
/// What the stamp can't see: an older binary deleting some edges and adding
/// as many whose ids sort below the stamped greatest one, which takes a
/// clock that went backwards; or rewriting an edge in place, which no
/// binary does (every edge gets a fresh id).
fn adjacency_stamp(edges: &impl ReadableTable<&'static str, &'static [u8]>) -> Result<Vec<u8>> {
    let mut stamp = vec![ADJACENCY_FORMAT];
    stamp.extend_from_slice(&edges.len()?.to_le_bytes());
    if let Some((key, _)) = edges.last()? {
        stamp.extend_from_slice(key.value().as_bytes());
    }
    Ok(stamp)
}

/// A stored adjacency stamp, decoded: what the edges table held when the
/// adjacency tables last matched it exactly.
struct Stamp<'a> {
    rows: u64,
    /// The greatest edge id; `None` when the table was empty.
    last: Option<&'a str>,
}

impl<'a> Stamp<'a> {
    /// Decode a stamp written by [`adjacency_stamp`]; `None` for one of
    /// another layout version.
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        let (&format, rest) = bytes.split_first()?;
        if format != ADJACENCY_FORMAT {
            return None;
        }
        let rows = u64::from_le_bytes(rest.get(..8)?.try_into().ok()?);
        let last = std::str::from_utf8(&rest[8..]).ok()?;
        Some(Self {
            rows,
            last: (!last.is_empty()).then_some(last),
        })
    }
}

/// Open a table for reading, `None` when no transaction has created it yet.
fn open_existing(
    txn: &ReadTransaction,
    def: TableDefinition<&'static str, &'static [u8]>,
) -> Result<Option<ReadOnlyTable<&'static str, &'static [u8]>>> {
    match txn.open_table(def) {
        Ok(table) => Ok(Some(table)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// [`open_existing`] for a multimap table.
fn open_existing_multimap(
    txn: &ReadTransaction,
    def: MultimapTableDefinition<&'static str, &'static [u8]>,
) -> Result<Option<ReadOnlyMultimapTable<&'static str, &'static [u8]>>> {
    match txn.open_multimap_table(def) {
        Ok(table) => Ok(Some(table)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// One change to an adjacency table.
struct AdjOp {
    outgoing: bool,
    key: String,
    value: Vec<u8>,
    insert: bool,
}

/// The changes that make the adjacency tables match the edges table, and
/// the edges whose JSON can't be decoded (to be removed with them).
///
/// Only what is missing or wrong is written: every entry the first time,
/// and after an older binary wrote edges only the entries those edges
/// changed. A full delete-and-reinsert would rewrite every page of both
/// tables each time, and redb can't reuse the freed pages within that
/// transaction, so the file would grow by the size of the tables on every
/// rebuild.
struct AdjacencyPlan {
    /// Sorted by table, key and value, the order the tables store them in.
    /// A sync commits them in chunks, and redb copies every page a
    /// transaction changes; in this order each chunk covers one run of
    /// keys and copies only its pages, where unsorted every chunk would
    /// copy most of both tables.
    ops: Vec<AdjOp>,
    corrupt: Vec<String>,
}

impl AdjacencyPlan {
    /// The plan that brings the tables in line with `edges`, read from one
    /// snapshot: the stored stamp's catch-up when it applies, otherwise a
    /// full diff. `None` when the tables are already current.
    fn for_snapshot(txn: &ReadTransaction) -> Result<Option<(Vec<u8>, Self)>> {
        let edges = txn.open_table(EDGES_TABLE)?;
        let stored = match open_existing(txn, META_TABLE)? {
            Some(meta) => meta
                .get(ADJACENCY_STAMP_KEY)?
                .map(|guard| guard.value().to_vec()),
            None => None,
        };
        let target = adjacency_stamp(&edges)?;
        if stored.as_deref() == Some(target.as_slice()) {
            return Ok(None);
        }
        let caught_up = match stored.as_deref().and_then(Stamp::parse) {
            Some(stamp) => Self::catch_up(&edges, &stamp)?,
            None => None,
        };
        let plan = match caught_up {
            Some(plan) => plan,
            None => {
                let out = open_existing_multimap(txn, OUT_TABLE)?;
                let inc = open_existing_multimap(txn, IN_TABLE)?;
                Self::compute(&edges, out.as_ref(), inc.as_ref())?
            }
        };
        Ok(Some((target, plan)))
    }

    /// The plan for a store whose tables matched `stamp` and whose edges
    /// table has since only gained rows, which is what an older binary's
    /// `link`, `store` or SCIP ingest leaves behind. Reads just the rows
    /// after the stamped greatest id, instead of decoding every edge.
    ///
    /// New edges get fresh ULIDs, which sort after every edge that existed
    /// when the stamp was written. So when the row count grew by exactly the
    /// number of rows after the stamped greatest id, no stamped edge is
    /// gone and those rows are the whole change. Any other count means
    /// edges were deleted: `None`, and the caller runs the full diff.
    fn catch_up(
        edges: &impl ReadableTable<&'static str, &'static [u8]>,
        stamp: &Stamp<'_>,
    ) -> Result<Option<Self>> {
        let rows = edges.len()?;
        if rows < stamp.rows {
            return Ok(None);
        }
        let added = match stamp.last {
            Some(last) => edges.range::<&str>((Bound::Excluded(last), Bound::Unbounded))?,
            None => edges.iter()?,
        };
        let mut ops = Vec::new();
        let mut corrupt = Vec::new();
        let mut count = 0u64;
        for row in added {
            let (key, value) = row?;
            count += 1;
            if stamp.rows + count > rows {
                return Ok(None);
            }
            let id = key.value();
            let Some(edge) = decode_edge(id, value.value(), &mut corrupt) else {
                continue;
            };
            let out_entry = AdjEntry::encode(&edge.edge_type, edge.to.as_str(), id);
            let in_entry = AdjEntry::encode(&edge.edge_type, edge.from.as_str(), id);
            ops.push(AdjOp {
                outgoing: true,
                key: edge.from.0,
                value: out_entry,
                insert: true,
            });
            ops.push(AdjOp {
                outgoing: false,
                key: edge.to.0,
                value: in_entry,
                insert: true,
            });
        }
        if stamp.rows + count != rows {
            return Ok(None);
        }
        // Outgoing table first, then each table in its own order, as
        // `compute` emits them.
        ops.sort_unstable_by(|a, b| {
            (!a.outgoing, &a.key, &a.value).cmp(&(!b.outgoing, &b.key, &b.value))
        });
        Ok(Some(Self { ops, corrupt }))
    }

    /// The plan that turns whatever the tables hold into exactly what
    /// `edges` implies, by decoding every edge and diffing.
    fn compute(
        edges: &impl ReadableTable<&'static str, &'static [u8]>,
        out: Option<&impl ReadableMultimapTable<&'static str, &'static [u8]>>,
        inc: Option<&impl ReadableMultimapTable<&'static str, &'static [u8]>>,
    ) -> Result<Self> {
        let rows = edges.len()? as usize;
        let mut corrupt = Vec::new();
        let mut want_out: Vec<(String, Vec<u8>)> = Vec::with_capacity(rows);
        let mut want_in: Vec<(String, Vec<u8>)> = Vec::with_capacity(rows);
        for row in edges.iter()? {
            let (key, value) = row?;
            let id = key.value();
            let Some(edge) = decode_edge(id, value.value(), &mut corrupt) else {
                continue;
            };
            let out_entry = AdjEntry::encode(&edge.edge_type, edge.to.as_str(), id);
            let in_entry = AdjEntry::encode(&edge.edge_type, edge.from.as_str(), id);
            want_out.push((edge.from.0, out_entry));
            want_in.push((edge.to.0, in_entry));
        }
        let mut ops = Vec::new();
        Self::diff(true, want_out, out, &mut ops)?;
        Self::diff(false, want_in, inc, &mut ops)?;
        Ok(Self { ops, corrupt })
    }

    /// Append to `ops` the inserts and removals that turn `table` into
    /// exactly `want`, by merging the two in the table's sort order.
    fn diff(
        outgoing: bool,
        mut want: Vec<(String, Vec<u8>)>,
        table: Option<&impl ReadableMultimapTable<&'static str, &'static [u8]>>,
        ops: &mut Vec<AdjOp>,
    ) -> Result<()> {
        // redb orders `&str` keys and `&[u8]` values bytewise, as `Ord` on
        // `String` and `Vec<u8>` does.
        want.sort_unstable();
        let insert = |(key, value): (String, Vec<u8>)| AdjOp {
            outgoing,
            key,
            value,
            insert: true,
        };
        let mut want = want.into_iter().peekable();
        if let Some(table) = table {
            for row in table.iter()? {
                let (key, values) = row?;
                let key = key.value();
                for guard in values {
                    let guard = guard?;
                    let have = guard.value();
                    let mut wanted = false;
                    while let Some((k, v)) = want.peek() {
                        match (k.as_str(), v.as_slice()).cmp(&(key, have)) {
                            std::cmp::Ordering::Less => ops.push(insert(want.next().unwrap())),
                            std::cmp::Ordering::Equal => {
                                want.next();
                                wanted = true;
                                break;
                            }
                            std::cmp::Ordering::Greater => break,
                        }
                    }
                    if !wanted {
                        ops.push(AdjOp {
                            outgoing,
                            key: key.to_string(),
                            value: have.to_vec(),
                            insert: false,
                        });
                    }
                }
            }
        }
        ops.extend(want.map(insert));
        Ok(())
    }
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

    /// Whether `node` has any edge, in either direction.
    fn has_entries(&self, node: &str) -> Result<bool> {
        Ok(!self.out.get(node)?.is_empty() || !self.inc.get(node)?.is_empty())
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
                    let valid = event_time.is_none_or(|t| edge.is_valid_at(t))
                        && knowledge_time.is_none_or(|t| edge.known_at(t));
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
    /// Open the tables for writing. The adjacency tables are taken as they
    /// are: the engine brought them up to date on its first graph operation
    /// (see [`GraphEngine::prepare`]), and nothing else can write the file
    /// while the engine has it open.
    fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(Self {
            edges: txn.open_table(EDGES_TABLE)?,
            out: txn.open_multimap_table(OUT_TABLE)?,
            inc: txn.open_multimap_table(IN_TABLE)?,
            meta: txn.open_table(META_TABLE)?,
        })
    }

    fn apply(&mut self, ops: &[AdjOp]) -> Result<()> {
        for op in ops {
            let table = if op.outgoing {
                &mut self.out
            } else {
                &mut self.inc
            };
            if op.insert {
                table.insert(op.key.as_str(), op.value.as_slice())?;
            } else {
                table.remove(op.key.as_str(), op.value.as_slice())?;
            }
        }
        Ok(())
    }

    fn unlink(&mut self, from: &str, edge_type: &str, to: &str, edge_id: &str) -> Result<()> {
        self.out
            .remove(from, AdjEntry::encode(edge_type, to, edge_id).as_slice())?;
        self.inc
            .remove(to, AdjEntry::encode(edge_type, from, edge_id).as_slice())?;
        Ok(())
    }

    /// Store edges (each with its serialized bytes at the same index) and
    /// their adjacency entries.
    ///
    /// Each table is written in its own key order, so a large batch (a SCIP
    /// ingest adds tens of thousands of edges at once, their endpoints
    /// spread over the whole key space) walks each tree front to back
    /// instead of jumping between leaves. That is locality only: redb
    /// copies a page the first time a transaction changes it and edits the
    /// copy in place after that, so the order doesn't change how many
    /// pages the transaction copies, nor how much the file grows.
    fn insert_all(&mut self, edges: &[Edge], serialized: &[Vec<u8>]) -> Result<()> {
        // By id, and for a repeated id only its last occurrence, which is
        // what inserting one by one would leave.
        let mut order: Vec<usize> = (0..edges.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            (edges[a].id.as_str(), std::cmp::Reverse(a))
                .cmp(&(edges[b].id.as_str(), std::cmp::Reverse(b)))
        });
        order.dedup_by(|a, b| edges[*a].id == edges[*b].id);

        let mut out = Vec::with_capacity(order.len());
        let mut inc = Vec::with_capacity(order.len());
        for &i in &order {
            let edge = &edges[i];
            let id = edge.id.as_str();
            let replaced = self
                .edges
                .insert(id, serialized[i].as_slice())?
                .map(|old| old.value().to_vec());
            // Ids are fresh ULIDs, so this never fires in practice; if it
            // did, the replaced edge's entries would otherwise point at the
            // new one.
            if let Some(old) = replaced.and_then(|b| Edge::from_bytes(&b).ok()) {
                self.unlink(old.from.as_str(), &old.edge_type, old.to.as_str(), id)?;
            }
            out.push((
                edge.from.as_str(),
                AdjEntry::encode(&edge.edge_type, edge.to.as_str(), id),
            ));
            inc.push((
                edge.to.as_str(),
                AdjEntry::encode(&edge.edge_type, edge.from.as_str(), id),
            ));
        }
        out.sort_unstable();
        inc.sort_unstable();
        for (key, value) in &out {
            self.out.insert(*key, value.as_slice())?;
        }
        for (key, value) in &inc {
            self.inc.insert(*key, value.as_slice())?;
        }
        Ok(())
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
    /// edge JSON: the node's adjacency entries name both endpoints. Returns
    /// whether the node had any entry.
    fn remove_node(&mut self, node: &str) -> Result<bool> {
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
        Ok(!outgoing.is_empty() || !incoming.is_empty())
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
///
/// # Corrupt edges
///
/// An edge whose stored JSON can't be decoded is treated as absent and
/// removed from the store, but only once a read decodes it. Reads that
/// need only endpoints and types answer from the adjacency tables without
/// decoding edge JSON: [`neighbor_ids`](Self::neighbor_ids) and
/// `GraphIndex::neighbors`, [`traverse_ids`](Self::traverse_ids) without a
/// time filter, [`edge_count`](Self::edge_count), `GraphIndex::all_edge_ids`,
/// and a `delete_edge` of the id, which returns `true`. Until a decoding read
/// ([`get_edge`](Self::get_edge), [`get_edges`](Self::get_edges), a
/// time-filtered traversal) or a rebuild of the tables removes it, those
/// still count it. Its adjacency entry was written from a valid edge, so
/// what they report is the edge as it was before the damage. Decoding
/// every edge on those paths to catch this would cost what the adjacency
/// tables exist to save, and a stored edge only goes bad through a write
/// from outside this engine or damage to the file.
pub struct GraphEngine {
    graph_db: Database,
    /// Settled by the first graph operation, not at open, so commands that
    /// open the database without using the graph pay nothing (see
    /// [`Self::prepare`] for what settling costs). A failed setup is kept,
    /// so every later operation reports it.
    store: OnceLock<std::result::Result<Store, String>>,
}

impl GraphEngine {
    /// Open or create a graph store at the companion path for the given database.
    ///
    /// Returns [`AxilError::Busy`] when another handle, in this process or
    /// another, has the store open: redb gives one writable handle at a time
    /// the file's lock, and callers can wait for it as they do for the core
    /// database's.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::from_database(open_graph_file(&companion_path(
            db_path.as_ref(),
            ".graph",
        ))?)
    }

    /// Wrap an open store, creating its tables if it is new.
    fn from_database(graph_db: Database) -> Result<Self> {
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
    ///
    /// Runs once per engine, on its first graph operation, whether that is
    /// a read or a write. Once is enough: redb locks the file for as long
    /// as a writable handle has it open, and a second writable open, from
    /// this process or another, fails with `DatabaseAlreadyOpen` (on the
    /// platforms where redb can't lock, it requires that only one process
    /// opens the file). So every write the file sees while this engine is
    /// open goes through the engine and keeps the tables current; edges an
    /// older binary writes can only land between two opens, and the next
    /// open's first operation picks them up.
    ///
    /// A read syncs too rather than answer from memory: after an older
    /// binary's write, whichever process syncs pays once and every later
    /// one reads the tables, where answering from memory would have every
    /// read-only process decode every edge until some write came along.
    ///
    /// The cost depends on what changed since the tables were stamped:
    ///
    /// - nothing: three lookups;
    /// - an older binary only added edges (`link`, `store`, `ingest-scip`):
    ///   the added edges are decoded and their entries inserted;
    /// - an older binary deleted an edge, or the tables were never built:
    ///   every edge is decoded and diffed against the tables.
    ///
    /// benchmarks/results/graph-adjacency-size-2026-09-30.json has the CPU
    /// each case took on the 161k-edge dogfood snapshot.
    ///
    /// A failure falls back to memory for this engine and leaves the tables
    /// for the next open to fix, so a read never fails because of it.
    fn prepare(&self) -> Result<Store> {
        match self.sync_adjacency() {
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
    #[cfg(test)]
    fn adjacency_is_current(&self) -> Result<bool> {
        let txn = self.graph_db.begin_read()?;
        let Some(meta) = open_existing(&txn, META_TABLE)? else {
            return Ok(false);
        };
        let Some(stored) = meta.get(ADJACENCY_STAMP_KEY)? else {
            return Ok(false);
        };
        Ok(stored.value() == adjacency_stamp(&txn.open_table(EDGES_TABLE)?)?.as_slice())
    }

    /// Bring the adjacency tables in line with the edges table and stamp
    /// them; returns at once when the stamp already matches.
    fn sync_adjacency(&self) -> Result<()> {
        self.sync_adjacency_in_chunks(SYNC_CHUNK, &mut |_| true)
    }

    /// [`Self::sync_adjacency`], committing at most `chunk` adjacency
    /// changes per write transaction.
    ///
    /// Building the tables for a large store the first time takes seconds,
    /// and the process doing it may be a hook that gets killed at its
    /// timeout. Committing in chunks means a killed process still leaves
    /// most of its work behind: the next sync diffs against the partly
    /// built tables and writes only the rest.
    ///
    /// Only the last chunk stamps, so no process trusts the tables before
    /// they are complete. The first of several chunks also removes the old
    /// stamp: partly updated tables no longer match it, and an older binary
    /// could otherwise return the edges table to the stamped state (by
    /// deleting the edges it had added) and have them trusted again. Without
    /// a stamp the next sync runs the full diff, which is correct whatever
    /// the tables hold.
    ///
    /// The plan is computed once. Nothing else writes the file during the
    /// sync: other processes are locked out (see [`Self::prepare`]), and
    /// this engine's other operations wait for [`Self::store`], which runs
    /// the sync. Each chunk still checks that the edges table is the one
    /// the plan was made for and fails otherwise, since stamping tables
    /// built for another state would hide the difference for good.
    ///
    /// `before_chunk` is called with the number of chunks committed so far
    /// before each write transaction; returning `false` ends the sync there,
    /// as a killed process would.
    fn sync_adjacency_in_chunks(
        &self,
        chunk: usize,
        before_chunk: &mut dyn FnMut(usize) -> bool,
    ) -> Result<()> {
        let Some((target, plan)) = AdjacencyPlan::for_snapshot(&self.graph_db.begin_read()?)?
        else {
            return Ok(());
        };
        let chunks: Vec<&[AdjOp]> = if plan.ops.is_empty() {
            // Nothing to change, but the stamp still has to be written.
            vec![&[]]
        } else {
            plan.ops.chunks(chunk.max(1)).collect()
        };
        let last = chunks.len() - 1;
        for (i, ops) in chunks.into_iter().enumerate() {
            if !before_chunk(i) {
                return Ok(());
            }
            let txn = self.graph_db.begin_write()?;
            {
                let mut w = DiskWriter::open(&txn)?;
                if adjacency_stamp(&w.edges)? != target {
                    return Err(AxilError::Plugin(Box::new(std::io::Error::other(
                        "graph edges changed during the adjacency sync",
                    ))));
                }
                if i == 0 && i != last {
                    w.meta.remove(ADJACENCY_STAMP_KEY)?;
                }
                w.apply(ops)?;
                if i == last {
                    for id in &plan.corrupt {
                        w.edges.remove(id.as_str())?;
                    }
                    w.finish()?;
                }
            }
            txn.commit()?;
        }
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
    /// corrupt edges the read ran into.
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
                    w.insert_all(edges, &serialized)?;
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
                // A commit is a durable write of the graph file; skip it
                // when there is nothing to keep.
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
            Store::Disk => {
                // Most deleted records have no edges (recall chunks, doc
                // chunks, cache rows), and a commit is a durable write of
                // the graph file. A read settles that there is nothing to do
                // without taking the writer lock. An edge another thread
                // commits after this read is ordered after the delete, as
                // it would be had it committed after a write transaction.
                if !self.with_view(|view, _| view.has_entries(id.as_str()))? {
                    return Ok(());
                }
                // One write transaction: no edge for this record can be added
                // between collecting its edges and removing them.
                let txn = self.graph_db.begin_write()?;
                let changed = {
                    let mut w = DiskWriter::open(&txn)?;
                    let changed = w.remove_node(id.as_str())?;
                    if changed {
                        w.finish()?;
                    }
                    changed
                };
                // The edges may have gone between the read and the write.
                if changed {
                    txn.commit()?;
                } else {
                    txn.abort()?;
                }
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

/// Open the graph file at `graph_path` for writing, creating it if absent.
/// [`AxilError::Busy`] when another handle has it open.
fn open_graph_file(graph_path: &Path) -> Result<Database> {
    Database::create(graph_path).map_err(|e| match e {
        redb::DatabaseError::DatabaseAlreadyOpen => AxilError::Busy,
        e => AxilError::Plugin(Box::new(std::io::Error::other(format!(
            "failed to open graph store at {}: {e}",
            graph_path.display()
        )))),
    })
}

// ── File compaction ─────────────────────────────────────────────────

/// What [`compact_graph_store`] found in a `.graph` file and did to it.
///
/// `size_bytes` is the file's length, what `ls` and `axil info` show.
/// `disk_bytes` is the disk it takes up. redb lengthens a file without
/// writing the new part, so where the filesystem keeps unwritten ranges as
/// holes (APFS, ext4, XFS, btrfs) the two differ. It is the allocated
/// blocks on Unix and the length elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GraphFileCompaction {
    /// Whether the file was rewritten.
    pub compacted: bool,
    /// Why it was left as it was, when it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
    /// Bytes of pages redb had in use before compaction.
    pub in_use_bytes: u64,
    /// The file's length before.
    pub size_bytes_before: u64,
    /// The disk the file took up before.
    pub disk_bytes_before: u64,
    /// The file's length after, once redb closed it.
    pub size_bytes_after: u64,
    /// The disk the file took up after, once redb closed it.
    pub disk_bytes_after: u64,
}

/// The disk a file takes up: its allocated blocks where the platform
/// reports them, its length elsewhere.
fn disk_bytes(meta: &std::fs::Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.blocks() * 512
    }
    #[cfg(not(unix))]
    {
        meta.len()
    }
}

/// Most of a `.graph` file's disk use, in percent, that may hold data for
/// [`compact_graph_store`] to compact it.
const COMPACT_MAX_IN_USE_PERCENT: u64 = 40;

/// Rewrite the `.graph` file of the database at `db_path` with its pages
/// packed at the front, giving the free ones back to the filesystem, when
/// at most 40% of the disk it takes up holds data. `Ok(None)` when the
/// database has no graph store.
///
/// The store must be closed: redb compacts only through the one writable
/// handle, so drop every engine on it first. [`AxilError::Busy`] when
/// another handle, in this process or another, has it open.
///
/// # Why only when mostly free
///
/// redb 3.1.3 writes its allocator state on every close, and that commit
/// needs fresh pages. A compacted file has none left, and redb grows a
/// file smaller than one region (4 GiB at the default 4 KiB page) by
/// doubling it, so the file comes out of the close about twice as long as
/// the data it holds. The new half is never written: on filesystems that
/// keep unwritten ranges as holes it takes no disk, but elsewhere, and in
/// a copy that doesn't keep holes, it takes its full length. Below half,
/// the result is no larger than before by length or disk use on any
/// filesystem. The margin down to 40% keeps a file just compacted on a
/// filesystem without holes, which then holds data in about half its
/// disk, from being compacted again by every run.
///
/// The files this engine writes are mostly full until many edges are
/// deleted: on copies of the 161,373-edge dogfood snapshot, 96.9-98.7% of
/// their disk use held data. Compacting them anyway gave back 1.3-3.1% of
/// the disk and left them 16-97% longer, after about 10 s of work each on
/// a busy machine (benchmarks/results/graph-adjacency-size-2026-09-30.json).
///
/// # If it is interrupted
///
/// Compaction moves pages in ordinary atomic commits, so a killed
/// compaction loses no data; as after any killed writer, redb repairs the
/// file on its next open. Compaction changes where the tables' pages sit,
/// not what they hold, so the adjacency tables stay current.
pub fn compact_graph_store(db_path: impl AsRef<Path>) -> Result<Option<GraphFileCompaction>> {
    let path = companion_path(db_path.as_ref(), ".graph");
    let io = |e: std::io::Error| {
        AxilError::Plugin(Box::new(std::io::Error::new(
            e.kind(),
            format!("graph store at {}: {e}", path.display()),
        )))
    };
    let before = match std::fs::metadata(&path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let (size_bytes_before, disk_bytes_before) = (before.len(), disk_bytes(&before));

    let mut db = open_graph_file(&path)?;
    let in_use_bytes = {
        let txn = db.begin_write()?;
        let stats = txn.stats()?;
        txn.abort()?;
        stats.allocated_pages() * stats.page_size() as u64
    };
    let skipped = if in_use_bytes.saturating_mul(100)
        <= disk_bytes_before.saturating_mul(COMPACT_MAX_IN_USE_PERCENT)
    {
        db.compact().map_err(|e| {
            AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "failed to compact graph store at {}: {e}",
                path.display()
            ))))
        })?;
        None
    } else {
        Some(format!(
            "{in_use_bytes} of the file's {disk_bytes_before} bytes of disk hold data; \
             compaction runs only when at most {COMPACT_MAX_IN_USE_PERCENT}% do"
        ))
    };
    // The size only settles once redb's closing commit has run.
    drop(db);

    let after = std::fs::metadata(&path).map_err(io)?;
    Ok(Some(GraphFileCompaction {
        compacted: skipped.is_none(),
        skipped,
        in_use_bytes,
        size_bytes_before,
        disk_bytes_before,
        size_bytes_after: after.len(),
        disk_bytes_after: disk_bytes(&after),
    }))
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

    fn clear_stamp(g: &GraphEngine) {
        let txn = g.graph_db.begin_write().unwrap();
        clear_adjacency_stamp(&txn).unwrap();
        txn.commit().unwrap();
    }

    /// How many adjacency changes a sync would make now.
    fn pending_ops(g: &GraphEngine) -> usize {
        let txn = g.graph_db.begin_read().unwrap();
        let edges = txn.open_table(EDGES_TABLE).unwrap();
        let out = open_existing_multimap(&txn, OUT_TABLE).unwrap();
        let inc = open_existing_multimap(&txn, IN_TABLE).unwrap();
        AdjacencyPlan::compute(&edges, out.as_ref(), inc.as_ref())
            .unwrap()
            .ops
            .len()
    }

    /// Write an edge the way an older binary does: to "edges" alone.
    fn write_edge_without_adjacency(db: &Database, edge: &Edge) {
        let txn = db.begin_write().unwrap();
        {
            let mut edges = txn.open_table(EDGES_TABLE).unwrap();
            edges
                .insert(edge.id.as_str(), edge.to_bytes().unwrap().as_slice())
                .unwrap();
        }
        txn.commit().unwrap();
    }

    /// A new edge whose id sorts after every edge in the store, as a later
    /// process's ULID does. `Edge::new` alone can't promise that: ULIDs made
    /// in the same millisecond as the current greatest are in random order.
    fn edge_after_last(db: &Database, from: &RecordId, edge_type: &str, to: &RecordId) -> Edge {
        let last = {
            let txn = db.begin_read().unwrap();
            let edges = txn.open_table(EDGES_TABLE).unwrap();
            let last = edges.last().unwrap().map(|(k, _)| k.value().to_string());
            last
        };
        let mut edge = Edge::new(from.clone(), edge_type, to.clone(), json!({}));
        if let Some(last) = last {
            edge.id = RecordId(format!("{last}Z"));
        }
        edge
    }

    /// Delete edges from "edges" alone, as an older binary's record delete
    /// does.
    fn remove_edges_without_adjacency(db: &Database, ids: &[RecordId]) {
        let txn = db.begin_write().unwrap();
        {
            let mut edges = txn.open_table(EDGES_TABLE).unwrap();
            for id in ids {
                edges.remove(id.as_str()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    /// The catch-up plan for the store as it is, if the stored stamp allows
    /// one; `None` when a sync would need the full diff.
    fn catch_up_ops(db: &Database) -> Option<usize> {
        let txn = db.begin_read().unwrap();
        let edges = txn.open_table(EDGES_TABLE).unwrap();
        let meta = txn.open_table(META_TABLE).unwrap();
        let stored = meta.get(ADJACENCY_STAMP_KEY).unwrap()?;
        let stamp = Stamp::parse(stored.value()).unwrap();
        AdjacencyPlan::catch_up(&edges, &stamp)
            .unwrap()
            .map(|plan| plan.ops.len())
    }

    /// A file backend that counts redb's writes and syncs, so a test can
    /// tell exactly whether an operation committed: every commit writes
    /// the file header and syncs, and nothing else writes.
    #[derive(Debug)]
    struct CountingBackend {
        inner: redb::backends::FileBackend,
        writes: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingBackend {
        fn bump(&self) {
            self.writes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl redb::StorageBackend for CountingBackend {
        fn len(&self) -> std::result::Result<u64, std::io::Error> {
            self.inner.len()
        }
        fn read(&self, offset: u64, out: &mut [u8]) -> std::result::Result<(), std::io::Error> {
            self.inner.read(offset, out)
        }
        fn set_len(&self, len: u64) -> std::result::Result<(), std::io::Error> {
            self.bump();
            self.inner.set_len(len)
        }
        fn sync_data(&self) -> std::result::Result<(), std::io::Error> {
            self.bump();
            self.inner.sync_data()
        }
        fn write(&self, offset: u64, data: &[u8]) -> std::result::Result<(), std::io::Error> {
            self.bump();
            self.inner.write(offset, data)
        }
    }

    /// An engine on `path`'s graph store whose file writes are counted.
    fn counting_engine(path: &Path) -> (GraphEngine, Arc<std::sync::atomic::AtomicUsize>) {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(companion_path(path, ".graph"))
            .unwrap();
        let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend = CountingBackend {
            inner: redb::backends::FileBackend::new(file).unwrap(),
            writes: writes.clone(),
        };
        let db = redb::Builder::new().create_with_backend(backend).unwrap();
        (GraphEngine::from_database(db).unwrap(), writes)
    }

    fn count(writes: &std::sync::atomic::AtomicUsize) -> usize {
        writes.load(std::sync::atomic::Ordering::SeqCst)
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

        // The build is idempotent: with the stamp gone, a sync finds nothing
        // to change and only stamps again.
        clear_stamp(&g);
        assert_eq!(pending_ops(&g), 0);
        g.sync_adjacency().unwrap();
        assert!(g.adjacency_is_current().unwrap());
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
            // Delete one edge and add one: the row count is unchanged, and
            // only the new greatest id tells the stamp something happened.
            let victim = oracle.edges.keys().min().cloned().unwrap();
            let added = edge_after_last(&db, &nodes[0], "knows", &nodes[1]);
            let txn = db.begin_write().unwrap();
            {
                let mut edges = txn.open_table(EDGES_TABLE).unwrap();
                edges.remove(victim.as_str()).unwrap();
                edges
                    .insert(added.id.as_str(), added.to_bytes().unwrap().as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
            oracle.remove(&victim);
            oracle.add(added);
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
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
    fn corrupt_edge_counts_until_a_decoding_read() {
        // The contract documented on `GraphEngine`: reads that don't decode
        // edge JSON still see a corrupt edge until one that does removes it.
        let dir = tempfile::tempdir().unwrap();
        let (a, b, c) = (
            RecordId("a".into()),
            RecordId("b".into()),
            RecordId("c".into()),
        );
        let bad = {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            g.create_edge(a.clone(), "knows", c.clone(), json!({}))
                .unwrap();
            g.create_edge(a.clone(), "knows", b.clone(), json!({}))
                .unwrap()
        };
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            txn.open_table(EDGES_TABLE)
                .unwrap()
                .insert(bad.id.as_str(), b"garbage".as_slice())
                .unwrap();
            txn.commit().unwrap();
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert_eq!(
            rid_strings(g.neighbor_ids(&a, None, Direction::Out)),
            vec!["b".to_string(), "c".to_string()]
        );
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.all_edge_ids().unwrap().len(), 2);
        let knows = crate::traverse::parse_path("->knows").unwrap();
        assert_eq!(rid_strings(g.traverse_ids(&a, &knows).unwrap()).len(), 2);

        assert_eq!(g.get_edges(&a, None, Direction::Out).len(), 1);

        assert_eq!(g.neighbor_ids(&a, None, Direction::Out), vec![c.clone()]);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.all_edge_ids().unwrap().len(), 1);
        assert!(!g.delete_edge(&bad.id).unwrap());
    }

    #[test]
    fn edgeless_record_deletes_write_nothing() {
        // 200 deletes of records with no edges in one process, as a bulk
        // delete makes them. Each used to commit, a durable write of the
        // graph file.
        let dir = tempfile::tempdir().unwrap();
        let (g, writes) = counting_engine(&graph_path(&dir));
        let (a, b) = (RecordId("a".into()), RecordId("b".into()));
        g.create_edge(a.clone(), "knows", b.clone(), json!({}))
            .unwrap();
        assert!(is_disk(&g));

        let before = count(&writes);
        for i in 0..200 {
            g.on_record_delete(&RecordId(format!("lonely{i}"))).unwrap();
        }
        assert_eq!(
            count(&writes),
            before,
            "an edgeless record delete wrote to the graph file"
        );
        // Deleting an edge that isn't there writes nothing either.
        assert!(!g.delete_edge(&RecordId("no-such-edge".into())).unwrap());
        assert_eq!(count(&writes), before);

        // A record with edges still loses them, and that commits.
        g.on_record_delete(&a).unwrap();
        assert!(count(&writes) > before);
        assert_eq!(g.edge_count(), 0);
        assert!(g.neighbor_ids(&b, None, Direction::In).is_empty());
        assert!(g.adjacency_is_current().unwrap());
    }

    #[test]
    fn axil_deletes_of_edgeless_records_write_nothing_to_the_graph() {
        // 100 `Axil::delete` calls on records with no edges, with the graph
        // attached: every delete fans out to the graph's on_record_delete.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.axil");
        let (engine, writes) = counting_engine(&path);
        let engine = Arc::new(engine);
        let db = axil_core::Axil::open(&path)
            .with_graph_index(engine.clone() as Arc<dyn GraphIndex>)
            .build()
            .unwrap();
        let notes: Vec<RecordId> = (0..100)
            .map(|i| db.insert("notes", json!({ "n": i })).unwrap().id)
            .collect();
        let a = db.insert("notes", json!({ "n": 1000 })).unwrap().id;
        let b = db.insert("notes", json!({ "n": 1001 })).unwrap().id;
        db.relate(&a, "knows", &b, None).unwrap();
        assert_eq!(engine.edge_count(), 1, "the notes carry no edges");

        let before = count(&writes);
        for id in &notes {
            assert!(db.delete(id).unwrap());
        }
        assert_eq!(
            count(&writes),
            before,
            "an edgeless Axil::delete wrote to the graph file"
        );

        assert!(db.delete(&a).unwrap());
        assert!(count(&writes) > before);
        assert_eq!(engine.edge_count(), 0);
    }

    #[test]
    fn edges_added_by_an_older_binary_are_caught_up_without_a_full_diff() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(6);
        let mut rng = Rng::new(71);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 60, 60);
        }
        // Between two opens, an older binary adds three edges, one of which
        // can't be decoded.
        {
            let db = open_raw(&dir);
            for (from, to) in [(0, 1), (2, 3)] {
                let e = edge_after_last(&db, &nodes[from], "knows", &nodes[to]);
                write_edge_without_adjacency(&db, &e);
                oracle.add(e);
            }
            let bad = edge_after_last(&db, &nodes[4], "knows", &nodes[5]);
            let txn = db.begin_write().unwrap();
            txn.open_table(EDGES_TABLE)
                .unwrap()
                .insert(bad.id.as_str(), b"garbage".as_slice())
                .unwrap();
            txn.commit().unwrap();
            assert_eq!(
                catch_up_ops(&db),
                Some(4),
                "two entries for each readable added edge"
            );
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), 0);
        drop(g);

        // Then it deletes one edge and adds one: the row count no longer
        // adds up, so the next sync diffs the whole table.
        {
            let db = open_raw(&dir);
            let victim = oracle.edges.keys().min().cloned().unwrap();
            remove_edges_without_adjacency(&db, std::slice::from_ref(&victim));
            oracle.remove(&victim);
            let e = edge_after_last(&db, &nodes[1], "mentions", &nodes[0]);
            write_edge_without_adjacency(&db, &e);
            oracle.add(e);
            assert_eq!(catch_up_ops(&db), None);
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), 0);
    }

    #[test]
    fn interrupted_catch_up_is_not_trusted_when_the_edges_table_goes_back() {
        // A partial sync must not leave the old stamp behind: an older
        // binary could return the edges table to the stamped state, and the
        // half-updated tables would then be trusted.
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(6);
        let mut rng = Rng::new(101);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 60, 60);
        }
        let added = {
            let db = open_raw(&dir);
            let e = edge_after_last(&db, &nodes[0], "knows", &nodes[5]);
            write_edge_without_adjacency(&db, &e);
            e
        };
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            // Killed after the first of the catch-up's two one-entry chunks,
            // which wrote the outgoing entry.
            g.sync_adjacency_in_chunks(1, &mut |done| done < 1).unwrap();
            assert!(!g.adjacency_is_current().unwrap());
        }
        {
            let db = open_raw(&dir);
            remove_edges_without_adjacency(&db, std::slice::from_ref(&added.id));
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(
            !g.adjacency_is_current().unwrap(),
            "the partial sync removed the old stamp"
        );
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), 0);
    }

    #[test]
    fn interrupted_sync_resumes_where_it_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(10);
        let mut rng = Rng::new(81);
        let mut oracle = AdjacencyIndex::new();
        {
            let g = GraphEngine::open(graph_path(&dir)).unwrap();
            run_ops(&g, &mut oracle, &nodes, &mut rng, 200, 200);
        }
        {
            let db = open_raw(&dir);
            let txn = db.begin_write().unwrap();
            txn.delete_multimap_table(OUT_TABLE).unwrap();
            txn.delete_multimap_table(IN_TABLE).unwrap();
            txn.delete_table(META_TABLE).unwrap();
            txn.commit().unwrap();
        }
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        let total = pending_ops(&g);
        assert_eq!(total, 2 * oracle.edge_count());

        // Killed after two chunks: their entries are on disk, unstamped.
        g.sync_adjacency_in_chunks(10, &mut |done| done < 2)
            .unwrap();
        assert!(!g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), total - 20);
        drop(g);

        // An older binary then removes edges whose entries are already
        // written, and adds one.
        let written: Vec<RecordId> = oracle
            .edges
            .values()
            .filter(|e| e.from == nodes[0])
            .map(|e| e.id.clone())
            .collect();
        assert!(!written.is_empty());
        {
            let db = open_raw(&dir);
            remove_edges_without_adjacency(&db, &written);
            for id in &written {
                oracle.remove(id);
            }
            let added = edge_after_last(&db, &nodes[0], "knows", &nodes[9]);
            write_edge_without_adjacency(&db, &added);
            oracle.add(added);
        }

        // The next process finishes the job from there.
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        assert!(pending_ops(&g) < total);
        assert_parity(&g, &oracle, &nodes, &mut rng);
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), 0);
    }

    #[test]
    fn sync_stops_when_the_edges_table_changes_between_chunks() {
        // Can't happen while redb locks the file. If it did, stamping tables
        // built for another state would hide the difference for good, so
        // the sync fails and the next one starts over.
        let dir = tempfile::tempdir().unwrap();
        let nodes = nodes(8);
        let mut rng = Rng::new(91);
        let mut oracle = AdjacencyIndex::new();
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        run_ops(&g, &mut oracle, &nodes, &mut rng, 120, 120);
        clear_stamp(&g);
        // Damage the tables: drop the outgoing entries of half the nodes and
        // add a stray.
        {
            let txn = g.graph_db.begin_write().unwrap();
            {
                let mut out = txn.open_multimap_table(OUT_TABLE).unwrap();
                for node in &nodes[..4] {
                    out.remove_all(node.as_str()).unwrap();
                }
                out.insert(
                    nodes[2].as_str(),
                    AdjEntry::encode("knows", "ghost", "no-such-edge").as_slice(),
                )
                .unwrap();
            }
            txn.commit().unwrap();
        }
        assert!(pending_ops(&g) > 3);
        let added = edge_after_last(&g.graph_db, &nodes[3], "knows", &nodes[4]);
        let err = g
            .sync_adjacency_in_chunks(1, &mut |done| {
                if done == 1 {
                    write_edge_without_adjacency(&g.graph_db, &added);
                }
                true
            })
            .unwrap_err();
        assert!(err.to_string().contains("changed during"), "{err}");
        oracle.add(added);
        assert!(!g.adjacency_is_current().unwrap());

        g.sync_adjacency().unwrap();
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(pending_ops(&g), 0);
        assert_parity(&g, &oracle, &nodes, &mut rng);
    }

    #[test]
    fn opening_a_store_another_handle_holds_is_busy() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        match GraphEngine::open(graph_path(&dir)) {
            Err(e) => assert!(e.is_busy(), "expected Busy, got {e}"),
            Ok(_) => panic!("a second handle opened a store that is held"),
        }
        drop(g);
        assert!(GraphEngine::open(graph_path(&dir)).is_ok());
    }

    #[test]
    fn batch_with_a_repeated_id_keeps_the_last_edge() {
        // What inserting the batch one edge at a time leaves.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphEngine::open(graph_path(&dir)).unwrap();
        let (a, b, c) = (
            RecordId("a".into()),
            RecordId("b".into()),
            RecordId("c".into()),
        );
        let first = Edge::new(a.clone(), "knows", b.clone(), json!({}));
        let mut second = Edge::new(a.clone(), "likes", c.clone(), json!({}));
        second.id = first.id.clone();
        g.add_edges(&[first, second.clone()], true).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.neighbor_ids(&a, None, Direction::Out), vec![c]);
        assert!(g.neighbor_ids(&b, None, Direction::In).is_empty());
        assert_eq!(g.get_edge(&second.id).unwrap().edge_type, "likes");
        assert_eq!(pending_ops(&g), 0);
    }

    /// `n` edges `e000000..` from `a<i>` to `b<i>`, each with `pad` bytes of
    /// properties, so their ids and both endpoints sort in creation order.
    fn padded_edges(n: usize, pad: usize) -> Vec<Edge> {
        (0..n)
            .map(|i| {
                let mut e = Edge::new(
                    RecordId(format!("a{i:06}")),
                    "rel",
                    RecordId(format!("b{i:06}")),
                    json!({ "pad": "x".repeat(pad) }),
                );
                e.id = RecordId(format!("e{i:06}"));
                e
            })
            .collect()
    }

    #[test]
    fn compacting_a_store_with_most_edges_deleted_gives_back_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = graph_path(&dir);
        let edges = padded_edges(3000, 400);
        let (gone, kept) = edges.split_at(2700);
        {
            let g = GraphEngine::open(&path).unwrap();
            g.add_edges(&edges, true).unwrap();
            let txn = g.graph_db.begin_write().unwrap();
            {
                let mut w = DiskWriter::open(&txn).unwrap();
                for e in gone {
                    assert!(w.remove(e.id.as_str()).unwrap());
                }
                w.finish().unwrap();
            }
            txn.commit().unwrap();
        }

        let r = compact_graph_store(&path).unwrap().unwrap();
        assert!(r.compacted, "{r:?}");
        assert!(r.skipped.is_none());
        assert!(r.disk_bytes_after < r.disk_bytes_before, "{r:?}");
        assert!(r.size_bytes_after <= r.size_bytes_before, "{r:?}");
        // What the closing commit leaves is not worth compacting again.
        let again = compact_graph_store(&path).unwrap().unwrap();
        assert!(!again.compacted, "{again:?}");

        let g = GraphEngine::open(&path).unwrap();
        assert!(g.adjacency_is_current().unwrap());
        assert_eq!(g.edge_count(), kept.len());
        for e in kept {
            let stored = g.get_edge(&e.id).unwrap();
            assert_eq!(stored.properties, e.properties);
            assert_eq!(
                g.neighbor_ids(&e.from, None, Direction::Out),
                vec![e.to.clone()]
            );
            assert_eq!(
                g.neighbor_ids(&e.to, None, Direction::In),
                vec![e.from.clone()]
            );
        }
        for e in gone {
            assert!(g.get_edge(&e.id).is_none());
            assert!(g.neighbor_ids(&e.from, None, Direction::Both).is_empty());
        }
        assert_eq!(pending_ops(&g), 0);
    }

    #[test]
    fn compaction_leaves_a_mostly_full_store_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = graph_path(&dir);
        let edges = padded_edges(3000, 400);
        GraphEngine::open(&path)
            .unwrap()
            .add_edges(&edges, true)
            .unwrap();

        let r = compact_graph_store(&path).unwrap().unwrap();
        assert!(!r.compacted, "{r:?}");
        assert!(r.skipped.is_some());
        assert!(
            r.in_use_bytes * 100 > r.disk_bytes_before * COMPACT_MAX_IN_USE_PERCENT,
            "{r:?}"
        );

        let g = GraphEngine::open(&path).unwrap();
        assert_eq!(g.edge_count(), edges.len());
        assert!(g.adjacency_is_current().unwrap());
    }

    #[test]
    fn compaction_of_an_open_store_is_busy_and_of_a_missing_one_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = graph_path(&dir);
        assert_eq!(compact_graph_store(&path).unwrap(), None);
        assert!(!has_graph_store(&path), "compaction created a store");

        let g = GraphEngine::open(&path).unwrap();
        match compact_graph_store(&path) {
            Err(e) => assert!(e.is_busy(), "expected Busy, got {e}"),
            Ok(r) => panic!("compacted a store another handle holds: {r:?}"),
        }
        drop(g);
        assert!(compact_graph_store(&path).unwrap().is_some());
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
