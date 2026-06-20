# Engine Traits (Tier 1)

> **Tier note (Phase 17):** This page documents the `Engine` trait — the surface every Tier-1 **Engine** implements. See [Extending Axil](../extending/overview.md) for the three-tier model and [Authoring Engines](../extending/engines.md) for the full authoring walkthrough. For Tier-2 (`Extension`) and Tier-3 (`Adapter`) trait references, see the corresponding authoring guides.

Axil's Engine tier is built on Rust traits. Each Engine implements `Engine` (the base trait) and optionally one of the index traits (`VectorIndex`, `GraphIndex`, `SearchIndex`, `TimeSeriesIndex`). Engines are registered at build time via the `AxilBuilder` extension-trait pattern.

## Core traits

See [Engines (Storage Plugins)](../concepts/plugins.md) for the full trait definitions and [Authoring Engines](../extending/engines.md) for usage walkthrough.

## Implementing a custom Engine

```rust
use axil_core::{Engine, Capability, Record, RecordId, Result};

struct MyEngine;

impl Engine for MyEngine {
    fn name(&self) -> &str { "my-plugin" }
    fn capabilities(&self) -> Vec<Capability> { vec![] }
    fn on_record_insert(&self, record: &Record) -> Result<()> {
        println!("Record inserted: {}", record.id);
        Ok(())
    }
    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        println!("Record deleted: {}", id);
        Ok(())
    }
}
```

## Built-in plugin implementations

| Engine | Crate | Storage |
|--------|-------|---------|
| `HnswIndex` | `axil-vector` | `*.axil.vec` (mmap) |
| `OnnxEmbedder` | `axil-vector` | ONNX model files |
| `GraphStorage` | `axil-graph` | `*.axil.graph` (redb) |
| `TantivyIndex` | `axil-fts` | `*.axil.fts/` (directory) |
| `TimeSeriesEngine` | `axil-core` | `*.axil.ts` |
