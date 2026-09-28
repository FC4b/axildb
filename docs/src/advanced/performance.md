# Performance

## Benchmarks

Axil includes Criterion benchmarks for all hot paths:

```bash
cargo bench -p axil-core
cargo bench -p axil-vector
cargo bench -p axil-graph
cargo bench -p axil-fts
```

For per-release numbers and the regression-tracking workflow, see the [Evaluation Log](./eval-log.md).

## Key optimizations

### Cascaded filtering
Queries apply cheap filters first (table, time range) before expensive operations (vector search).

### Adaptive RRF
Reciprocal Rank Fusion weights are automatically tuned based on which signals are available.

### Batch embedding
Multiple texts can be embedded in a single ONNX inference call.

### Int8 embedding model
Use `bge-small-int8` (`bge-small-en-v1.5-int8`, an int8-quantized ONNX model) for faster
embedding at a small quality cost. No committed benchmark measures the trade-off yet, so
check it on your own data before switching.

Stored vectors are full `f32`, and the vector store is loaded into memory when the database
opens. Vector quantization and memory-mapped vectors exist as code in `axil-vector` but are
not wired in yet.

### Deferred indexing
Write buffer batches index updates for high-throughput insert workloads.

### Tiered memory
Records are classified into Hot/Warm/Cold/Archived tiers for efficient retrieval.

## Retrieval quality

| Benchmark | Score |
|-----------|-------|
| LoCoMo | 99% hit rate, 94.4% recall |
| LongMemEval | Competitive with Hindsight (91.4%) |

## Binary size

Target: 5-10MB with all features. Use feature flags to reduce size:

```bash
# Minimal (core only)
cargo build --release -p axildb

# Full (all engines)
cargo build --release -p axildb --features full,memory
```

## Configuration for performance

```toml
[healing]
compact_expired_threshold = 1000     # doctor flags cleanup pressure after N expired
compact_superseded_threshold = 500   # ...or N superseded records

[index]
embedding_model = "bge-small-en-v1.5-int8"  # Faster embeddings
```
