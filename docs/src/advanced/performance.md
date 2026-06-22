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

### Int8 quantization
Use `bge-small-en-v1.5-int8` for ~3x faster embedding with minimal quality loss.

### Mmap vectors
Vector index is memory-mapped for zero-copy access on large datasets.

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

# Full (all plugins)
cargo build --release -p axildb --features full,memory
```

## Configuration for performance

```toml
[healing]
auto_compact_threshold = 1000  # Compact after N deletes

[index]
embedding_model = "bge-small-en-v1.5-int8"  # Faster embeddings
```
