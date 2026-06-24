# Embedded Usage

Use Axil as a Rust library for direct embedding in your application.

## Basic usage

```rust
use axil_core::{Axil, Op};
use axil_vector::{models::EmbeddingModel, AxilBuilderVectorExt};
use axil_graph::AxilBuilderGraphExt;
use axil_fts::AxilBuilderFtsExt;
use serde_json::json;

// Open with engines
let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?
    .with_graph_engine()?
    .with_fts_engine()?
    .build()?;

// Insert
let record = db.insert("sessions", json!({
    "summary": "Fixed auth timeout bug",
    "project": "my-app",
}))?;

// Embed for vector search
db.embed_field(&record.id, "summary")?;

// Graph relationships
db.relate(&record.id, "modified", &file_id.id, None)?;

// Vector search
let results = db.similar_to("auth error", 5)?;

// Full-text search
let results = db.search_text("timeout", 10)?;

// Combined query
let results = db.query()
    .similar_to("auth error", 5)
    .where_field("project", Op::Eq, json!("my-app"))
    .exec()?;
```

## With agent memory

```rust
use axil_memory::AgentMemory;

let mem = AgentMemory::new(&db);

// Semantic memory
mem.semantic().know("auth-module", "Uses JWT tokens", None)?;
let knowledge = mem.semantic().about("auth-module")?;

// Sessions
let session = mem.working().start_session(None)?;
mem.working().end_session(&session.id, Some("Fixed bug"), None, None, None)?;

// Multi-agent
let alice_mem = AgentMemory::for_agent(&db, "alice");
let session = alice_mem.working().start_session(None)?;
```

## With LLM enhancement

```rust
let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?
    .with_llm(Arc::new(HttpLlm::new(endpoint, api_key, model)))
    .build()?;

// LLM-enhanced entity extraction
let entities = db.extract_entities_enhanced("Fixed the auth timeout bug")?;
```
