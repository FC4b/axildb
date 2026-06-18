//! Deferred indexing write buffer (8b.11).
//!
//! Accumulates inserted records and defers plugin indexing (vector, FTS, timeseries)
//! until a flush trigger: buffer full, read query, or explicit flush().
//! Records are persisted to storage immediately for durability — only plugin
//! hooks are deferred.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::record::Record;

/// Default maximum records before auto-flush.
pub const DEFAULT_MAX_RECORDS: usize = 1000;

/// Default maximum estimated bytes before auto-flush (10 MB).
pub const DEFAULT_MAX_BYTES: usize = 10 * 1024 * 1024;

/// Configuration for the write buffer.
#[derive(Debug, Clone)]
pub struct WriteBufferConfig {
    /// Maximum records before auto-flush.
    pub max_records: usize,
    /// Maximum estimated bytes before auto-flush.
    pub max_bytes: usize,
    /// Whether deferred indexing is enabled.
    pub enabled: bool,
}

impl Default for WriteBufferConfig {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_MAX_RECORDS,
            max_bytes: DEFAULT_MAX_BYTES,
            enabled: false, // opt-in
        }
    }
}

/// Write buffer that accumulates records for deferred plugin indexing.
///
/// Thread-safe: internal Mutex protects the record list.
pub struct WriteBuffer {
    records: Mutex<Vec<Record>>,
    byte_estimate: AtomicUsize,
    config: WriteBufferConfig,
}

impl WriteBuffer {
    /// Create a new write buffer with the given configuration.
    pub fn new(config: WriteBufferConfig) -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            byte_estimate: AtomicUsize::new(0),
            config,
        }
    }

    /// Push a record into the buffer. Returns the buffered records if the
    /// buffer is full and needs flushing.
    pub fn push(&self, record: Record) -> Option<Vec<Record>> {
        let est_size = estimate_record_size(&record);
        let mut buf = self.records.lock();
        buf.push(record);
        let new_bytes = self.byte_estimate.fetch_add(est_size, Ordering::Relaxed) + est_size;

        if buf.len() >= self.config.max_records || new_bytes >= self.config.max_bytes {
            let drained = std::mem::take(&mut *buf);
            self.byte_estimate.store(0, Ordering::Relaxed);
            Some(drained)
        } else {
            None
        }
    }

    /// Drain all buffered records for flushing.
    pub fn drain(&self) -> Vec<Record> {
        let mut buf = self.records.lock();
        self.byte_estimate.store(0, Ordering::Relaxed);
        std::mem::take(&mut *buf)
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.records.lock().is_empty()
    }

    /// Number of buffered records.
    pub fn len(&self) -> usize {
        self.records.lock().len()
    }

    /// Whether deferred indexing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Estimate the byte size of a record (JSON serialization approximation).
fn estimate_record_size(record: &Record) -> usize {
    // ID + table + data JSON + timestamps ≈ actual serialized size
    record.id.to_string().len()
        + record.table.len()
        + serde_json::to_string(&record.data)
            .map(|s| s.len())
            .unwrap_or(100)
        + 80 // timestamps, field names overhead
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn push_and_drain() {
        let buf = WriteBuffer::new(WriteBufferConfig {
            max_records: 3,
            max_bytes: DEFAULT_MAX_BYTES,
            enabled: true,
        });

        assert!(buf.push(Record::new("t", json!({"a": 1}))).is_none());
        assert!(buf.push(Record::new("t", json!({"b": 2}))).is_none());
        assert_eq!(buf.len(), 2);

        // Third push triggers flush
        let flushed = buf.push(Record::new("t", json!({"c": 3})));
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().len(), 3);
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_returns_all() {
        let buf = WriteBuffer::new(WriteBufferConfig::default());
        buf.push(Record::new("t", json!({"x": 1})));
        buf.push(Record::new("t", json!({"y": 2})));
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn empty_buffer() {
        let buf = WriteBuffer::new(WriteBufferConfig::default());
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.drain().is_empty());
    }
}
