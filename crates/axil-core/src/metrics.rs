use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Number of latency samples to keep in the circular buffer.
const LATENCY_WINDOW: usize = 1000;

/// Operation types tracked by the metrics collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpType {
    Insert,
    Get,
    Delete,
    Update,
    VectorSearch,
    FtsSearch,
    Traversal,
    Recall,
}

impl OpType {
    pub fn all() -> &'static [OpType] {
        &[
            OpType::Insert,
            OpType::Get,
            OpType::Delete,
            OpType::Update,
            OpType::VectorSearch,
            OpType::FtsSearch,
            OpType::Traversal,
            OpType::Recall,
        ]
    }

    pub fn counter_name(&self) -> &'static str {
        match self {
            OpType::Insert => "inserts_total",
            OpType::Get => "gets_total",
            OpType::Delete => "deletes_total",
            OpType::Update => "updates_total",
            OpType::VectorSearch => "vector_searches_total",
            OpType::FtsSearch => "fts_searches_total",
            OpType::Traversal => "traversals_total",
            OpType::Recall => "recalls_total",
        }
    }
}

/// Atomic counters for each operation type.
struct Counters {
    inserts: AtomicU64,
    gets: AtomicU64,
    deletes: AtomicU64,
    updates: AtomicU64,
    vector_searches: AtomicU64,
    fts_searches: AtomicU64,
    traversals: AtomicU64,
    recalls: AtomicU64,
}

impl Counters {
    fn new() -> Self {
        Self {
            inserts: AtomicU64::new(0),
            gets: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            vector_searches: AtomicU64::new(0),
            fts_searches: AtomicU64::new(0),
            traversals: AtomicU64::new(0),
            recalls: AtomicU64::new(0),
        }
    }

    fn get(&self, op: OpType) -> &AtomicU64 {
        match op {
            OpType::Insert => &self.inserts,
            OpType::Get => &self.gets,
            OpType::Delete => &self.deletes,
            OpType::Update => &self.updates,
            OpType::VectorSearch => &self.vector_searches,
            OpType::FtsSearch => &self.fts_searches,
            OpType::Traversal => &self.traversals,
            OpType::Recall => &self.recalls,
        }
    }
}

/// Circular buffer for recording operation latencies.
struct LatencyBuffer {
    samples: Vec<f64>,
    pos: usize,
    count: usize,
}

impl LatencyBuffer {
    fn new() -> Self {
        Self {
            samples: vec![0.0; LATENCY_WINDOW],
            pos: 0,
            count: 0,
        }
    }

    fn record(&mut self, ms: f64) {
        self.samples[self.pos] = ms;
        self.pos = (self.pos + 1) % LATENCY_WINDOW;
        if self.count < LATENCY_WINDOW {
            self.count += 1;
        }
    }
}

/// Latency percentiles for an operation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Snapshot of all metrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub counters: std::collections::HashMap<String, u64>,
    pub latencies: std::collections::HashMap<String, LatencyPercentiles>,
    pub db_created_at: Option<String>,
    pub last_write_at: Option<String>,
    pub last_read_at: Option<String>,
}

/// Lightweight, always-on metrics collector.
///
/// Counters use atomics (lock-free). Latency recording uses a parking_lot
/// mutex protecting a circular buffer — the critical section is <100ns so
/// it never meaningfully contends with actual operations.
pub struct Metrics {
    counters: Counters,
    latencies: parking_lot::Mutex<std::collections::HashMap<OpType, LatencyBuffer>>,
    db_created_at: parking_lot::Mutex<Option<DateTime<Utc>>>,
    last_write_at: AtomicU64,
    last_read_at: AtomicU64,
}

impl Metrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        let mut latencies = std::collections::HashMap::new();
        for op in OpType::all() {
            latencies.insert(*op, LatencyBuffer::new());
        }
        Self {
            counters: Counters::new(),
            latencies: parking_lot::Mutex::new(latencies),
            db_created_at: parking_lot::Mutex::new(None),
            last_write_at: AtomicU64::new(0),
            last_read_at: AtomicU64::new(0),
        }
    }

    /// Set the database creation timestamp.
    pub fn set_created_at(&self, dt: DateTime<Utc>) {
        *self.db_created_at.lock() = Some(dt);
    }

    /// Increment the counter for an operation type.
    pub fn inc(&self, op: OpType) {
        self.counters.get(op).fetch_add(1, Ordering::Relaxed);
    }

    /// Update the last read/write timestamp for an operation type.
    fn touch(&self, op: OpType, now_us: u64) {
        match op {
            OpType::Insert | OpType::Delete | OpType::Update => {
                self.last_write_at.store(now_us, Ordering::Relaxed);
            }
            OpType::Get
            | OpType::VectorSearch
            | OpType::FtsSearch
            | OpType::Traversal
            | OpType::Recall => {
                self.last_read_at.store(now_us, Ordering::Relaxed);
            }
        }
    }

    /// Record a latency sample for an operation type.
    pub fn record_latency(&self, op: OpType, ms: f64) {
        if let Some(buf) = self.latencies.lock().get_mut(&op) {
            buf.record(ms);
        }
    }

    /// Get the current counter value for an operation type.
    pub fn counter(&self, op: OpType) -> u64 {
        self.counters.get(op).load(Ordering::Relaxed)
    }

    /// Start timing an operation. Call `.finish()` on the returned guard.
    pub fn start_timer(&self, op: OpType) -> Timer<'_> {
        Timer {
            metrics: self,
            op,
            start: Instant::now(),
        }
    }

    /// Take a snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut counters = std::collections::HashMap::new();
        for op in OpType::all() {
            counters.insert(
                op.counter_name().to_string(),
                self.counters.get(*op).load(Ordering::Relaxed),
            );
        }

        // Copy raw samples under the lock, then sort outside it.
        let raw_latencies: Vec<(OpType, Vec<f64>)> = {
            let lat_map = self.latencies.lock();
            OpType::all()
                .iter()
                .filter_map(|op| {
                    lat_map
                        .get(op)
                        .filter(|b| b.count > 0)
                        .map(|b| (*op, b.samples[..b.count].to_vec()))
                })
                .collect()
        };
        let mut latencies = std::collections::HashMap::new();
        for (op, mut sorted) in raw_latencies {
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p = |pct: f64| -> f64 {
                let idx = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };
            latencies.insert(
                op.counter_name().replace("_total", ""),
                LatencyPercentiles {
                    p50: p(50.0),
                    p95: p(95.0),
                    p99: p(99.0),
                },
            );
        }

        let last_write = self.last_write_at.load(Ordering::Relaxed);
        let last_read = self.last_read_at.load(Ordering::Relaxed);

        MetricsSnapshot {
            counters,
            latencies,
            db_created_at: self.db_created_at.lock().map(|dt| dt.to_rfc3339()),
            last_write_at: if last_write > 0 {
                DateTime::from_timestamp_micros(last_write as i64).map(|dt| dt.to_rfc3339())
            } else {
                None
            },
            last_read_at: if last_read > 0 {
                DateTime::from_timestamp_micros(last_read as i64).map(|dt| dt.to_rfc3339())
            } else {
                None
            },
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII timer that records latency on drop.
pub struct Timer<'a> {
    metrics: &'a Metrics,
    op: OpType,
    start: Instant,
}

impl<'a> Timer<'a> {
    /// Finish timing and record the latency. Also increments the counter.
    pub fn finish(self) -> f64 {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.inc(self.op);
        self.metrics.record_latency(self.op, ms);
        self.metrics
            .touch(self.op, Utc::now().timestamp_micros() as u64);
        ms
    }
}

/// Slow query entry stored in the internal log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryEntry {
    pub timestamp: String,
    pub command: String,
    pub duration_ms: f64,
    pub result_count: usize,
}

/// Audit log entry for write operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub operation: String,
    pub record_id: String,
    pub table: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment() {
        let m = Metrics::new();
        assert_eq!(m.counter(OpType::Insert), 0);
        m.inc(OpType::Insert);
        m.inc(OpType::Insert);
        assert_eq!(m.counter(OpType::Insert), 2);
    }

    #[test]
    fn latency_recording() {
        let m = Metrics::new();
        m.record_latency(OpType::Get, 1.0);
        m.record_latency(OpType::Get, 2.0);
        m.record_latency(OpType::Get, 3.0);

        let snap = m.snapshot();
        let lat = snap.latencies.get("gets").unwrap();
        assert!(lat.p50 >= 1.0 && lat.p50 <= 3.0);
    }

    #[test]
    fn timer_records() {
        let m = Metrics::new();
        let timer = m.start_timer(OpType::Insert);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let ms = timer.finish();
        assert!(ms > 0.0);
        assert_eq!(m.counter(OpType::Insert), 1);
    }

    #[test]
    fn snapshot_is_complete() {
        let m = Metrics::new();
        m.inc(OpType::Insert);
        m.inc(OpType::Get);
        let snap = m.snapshot();
        assert_eq!(*snap.counters.get("inserts_total").unwrap(), 1);
        assert_eq!(*snap.counters.get("gets_total").unwrap(), 1);
    }
}
