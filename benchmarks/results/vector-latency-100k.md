## Vector latency benchmark (100k vectors, 384 dims)

Re-run 2026-04-20 after CUDA + ONNX batch-embedding + `insert_batch_raw`.

**Insert:** 100000 vectors in 187421 ms (533 vec/s)
**HNSW rebuild:** 47718 ms

| top_k | queries | mean (us) | p50 (us) | p95 (us) | p99 (us) | max (us) | qps |
|------:|--------:|----------:|---------:|---------:|---------:|---------:|-----:|
|     1 |    1000 |     623.8 |    611.5 |    733.3 |   1141.4 |   1824.9 | 1603 |
|    10 |    1000 |     619.0 |    607.8 |    716.2 |   1045.1 |   1589.8 | 1615 |
|   100 |    1000 |     643.7 |    632.7 |    750.6 |   1054.3 |   1705.9 | 1553 |

Previous (2026-04-19 baseline): insert 180 vec/s, rebuild 91 s, search p50 681 µs, 1441 qps.
