## Axil vs SQLite+sqlite-vec (100k vectors, 384 dims, top_k=10)

Re-run 2026-04-20 after CUDA + ONNX batch-embedding.

| metric | axil-vector (HNSW) | sqlite-vec (brute force) |
|--------|---------------------:|-------------------------:|
| insert time | 189955 ms | 1804 ms |
| insert throughput | 526 vec/s | 55432 vec/s |
| index build time | 49518 ms | 0 ms (flat — no build) |
| search mean | 619.1 us | 105862.6 us |
| search p50 | 608.9 us | 105459.9 us |
| search p95 | 723.3 us | 108850.4 us |
| search p99 | 921.8 us | 111943.5 us |
| qps | 1615 | 9 |
| disk usage | 275.0 MB | 156.6 MB |

_Axil uses HNSW (approximate); sqlite-vec's vec0 is exact brute force. Recall differs — this compares raw latency + insert cost on equivalent workload, not equal algorithms._

Previous (2026-04-19 baseline): Axil insert 181 vec/s, search p50 686 µs, 1450 qps.
