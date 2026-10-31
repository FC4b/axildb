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

// The vector store records the model that built it. A store that holds
// another model's vectors is not attached: `db.degraded_engines()` says why,
// and vector search fails with that reason. A store opened raw and given a
// model (`VectorEngine::open(path, dims)?.with_model(model)`) is checked the
// same way when it is attached. A store written before stores recorded their
// model is taken for the model passed in when that model has the store's
// width and the database's axil.toml names no model: any built-in but granite
// (which came after model records), or a custom model. When axil.toml names a
// model, only that one takes such a store over, and only a built-in other
// than granite; any other model passed in is refused (see "Switching
// embedding models" in the installation guide). The next vector written
// records the model.

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
