---
name: axil-optimize
description: "Optimize Axil performance, binary size, and dependencies based on field reports and benchmarks"
trigger: when user mentions optimize axil, axil performance, axil binary size, axil benchmarks
---

# Axil Optimize — Performance & Size Optimization

Deep performance analysis and optimization of the Axil codebase. Goes beyond bug fixing to make Axil faster and smaller. **This skill is for use in the Axil source repository only.**

## Workflow

### 1. Read Field Reports for Performance Patterns

```bash
# Check for performance-related problems in reports
cat reports/incoming/*.json 2>/dev/null | jq '.problems[] | select(.type == "performance")'

# Check sibling projects
for d in ../*/; do
  cat "$d/.axil-reports/"*.json 2>/dev/null | jq '.problems[] | select(.type == "performance")'
done
```

### 2. Run Benchmarks (BEFORE)

Always benchmark before making changes:

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific component benchmarks
cargo bench -p axil-core
cargo bench -p axil-vector

# Save baseline
mkdir -p benches/baselines
cargo bench --workspace 2>&1 | tee benches/baselines/$(date +%Y%m%d).txt
```

### 3. Analyze Hot Paths

**Algorithmic complexity:**
- Vector search: O(log n) expected for HNSW — check if degrading
- Graph traversal: check for unnecessary full scans
- FTS: tantivy should handle this, but check query parsing overhead

**Serialization overhead:**
- `serde_json` is used everywhere — check for unnecessary serialize/deserialize cycles
- Consider `rmp-serde` (MessagePack) for internal storage if JSON overhead is significant
- Check if `Value::clone()` is happening where references would work

**Memory allocation patterns:**
- Search for `Vec::new()` in loops — replace with `Vec::with_capacity()`
- Check for `String::clone()` where `&str` references work
- Look for `collect()` → immediate iteration patterns (use iterators instead)

### 4. Check Dependency Health

```bash
# Find duplicate dependencies
cargo tree -d

# Check for outdated crates
cargo outdated

# Check unused dependencies (requires cargo-udeps)
cargo +nightly udeps --workspace

# Check feature flag bloat — are we pulling in features we don't use?
cargo tree --edges features
```

### 5. Check Binary Size

```bash
# Build release binary
cargo build --release -p axil-cli --features full

# Check size
ls -lh target/release/axil

# Detailed size analysis (requires cargo-bloat)
cargo bloat --release -p axil-cli --features full

# Check which crates contribute most to size
cargo bloat --release -p axil-cli --features full --crates
```

Target: under 10MB for the release binary (configurable in `axil.toml` → `optimize.binary_size_target_mb`).

### 6. Apply Optimizations

Common optimizations:

**Reduce allocations:**
```rust
// Before
let mut results = Vec::new();
for item in items {
    results.push(process(item));
}

// After
let mut results = Vec::with_capacity(items.len());
for item in items {
    results.push(process(item));
}
```

**Avoid unnecessary clones:**
```rust
// Before
let name = record.table.clone();
do_something(&name);

// After
do_something(&record.table);
```

**Use iterators instead of collecting:**
```rust
// Before
let ids: Vec<_> = records.iter().map(|r| r.id.clone()).collect();
for id in &ids { ... }

// After
for id in records.iter().map(|r| &r.id) { ... }
```

### 7. Re-Benchmark (AFTER)

```bash
# Run benchmarks again
cargo bench --workspace 2>&1 | tee benches/baselines/$(date +%Y%m%d)-after.txt

# Compare
diff benches/baselines/$(date +%Y%m%d).txt benches/baselines/$(date +%Y%m%d)-after.txt
```

Never claim improvement without measurement.

### 8. Verify

```bash
cargo test --workspace
cargo clippy --workspace --all-features -- -D warnings
```

## Optimization Checklist

- [ ] Read field reports for real-world performance patterns
- [ ] Benchmark BEFORE changes
- [ ] Check algorithmic complexity of hot paths
- [ ] Check serialization overhead
- [ ] Check memory allocation patterns
- [ ] Check dependency health (duplicates, outdated, unused)
- [ ] Check binary size
- [ ] Apply targeted optimizations
- [ ] Benchmark AFTER changes
- [ ] Run full test suite
- [ ] Document performance deltas
