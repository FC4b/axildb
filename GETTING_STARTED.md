# Axil - Getting Started with Claude Code

## Quick Start

### 1. Create your project

```bash
mkdir axil && cd axil
```

### 2. Download CLAUDE.md from this chat and move it to the project root

```bash
mv ~/Downloads/CLAUDE.md .
```

### 3. Initialize git

```bash
git init
echo "target/\nmodels/*.onnx\n*.axil\n" > .gitignore
git add . && git commit -m "init: Axil project spec"
```

### 4. Start Claude Code

```bash
claude
```

### 5. Give Claude Code the first prompt

**Scaffold everything + start Phase 1:**
```
Read CLAUDE.md and scaffold the complete Cargo workspace with all 
crates. Implement Phase 1 (core + document store) with redb storage, 
Record/RecordId types, and basic CRUD operations. Include a working 
CLI with `axil open`, `axil insert`, `axil get`, `axil list`, 
`axil delete` commands. Add unit tests.
```

**Or start smaller:**
```
Read CLAUDE.md. Let's start with just axil-core. Create the Cargo 
workspace, implement the Record type, RecordId, storage traits, 
and a redb-backed storage engine with insert/get/delete/list. 
Add comprehensive tests.
```

**Or discuss architecture first:**
```
Read CLAUDE.md. Before coding, I want to discuss the plugin trait 
design. How should the query builder combine results from vector, 
graph, and FTS plugins in a single query? What's the best way to 
handle plugin registration at compile time vs runtime?
```

## Tips

### Keep CLAUDE.md updated
As you make decisions, tell Claude Code:
```
Update CLAUDE.md to reflect that we chose X over Y because Z
```

### Commit often
```
Git commit after each milestone with a descriptive message
```

### Test-driven
```
Write tests first for Record CRUD, then implement
```

### Phase 2 prep (vector search)
You'll need an ONNX model (~22MB). Claude Code can help set up the download script:
```
Set up a build script that downloads all-MiniLM-L6-v2 ONNX model 
to the models/ directory if not present
```

## Alternative Approach: Build on SurrealDB

If you decide SurrealDB 3.0's features are sufficient and you just 
want the agent memory layer on top:

```
Read CLAUDE.md. Instead of building storage from scratch with redb, 
let's build Axil as a thin MCP server + Rust library on top of 
SurrealDB 3.0. It already has document, graph, vector, and FTS. 
We just need the agent memory patterns (auto-embed, TTL, superseding, 
recall) and MCP interface. This trades independence for speed.
```
