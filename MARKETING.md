# Axil — Marketing & Launch Plan

> Status: draft v1 · 2026-07-10 · owner: FC4b
> Goal: take github.com/FC4b/axildb from 0 stars to a visible, credible project with real users.
>
> **Message discipline (inherits the repo's Numbers-integrity policy):** every number in
> outbound copy must trace to a committed benchmark, a named baseline, or be labeled an
> estimate. Never call Axil "open source" — it is **source-available, free for noncommercial
> use** (PolyForm NC). Getting this wrong on HN/Reddit costs more credibility than any
> feature earns.

---

## 1. TL;DR — the one-page plan

1. **Polish first (Week 0):** demo GIF, GitHub About + topics, social preview, Cargo.toml
   `keywords`/`categories`, a pinned "Star if this saved you tokens" CTA. Launching with 0
   stars is fine; launching with a bare repo page is not.
2. **Launch wave (Weeks 1–2), staggered not simultaneous:** Show HN → r/rust → r/LocalLLaMA →
   r/ClaudeAI → X thread → This Week in Rust → Lobste.rs. One channel per day, learn and
   adjust between posts.
3. **Distribution surfaces (Weeks 1–4):** MCP directories (mcp.so, PulseMCP, Glama, Smithery),
   awesome-lists PRs (awesome-mcp-servers, awesome-ai-agents, awesome-claude-code,
   awesome-rust), crates.io metadata.
4. **Content engine (Weeks 2–12):** 3 technical blog posts + 4 comparison pages
   (vs Mem0 / Memvid / Zep / Redis Iris) that capture "agent memory" search traffic.
5. **Momentum mechanics (ongoing):** answer every launch-thread comment fast, ship visible
   weekly releases, keep `good first issue` labels stocked, post build-in-public updates.

The single highest-leverage asset is a **30-second terminal GIF** showing an agent resume a
session from `axil boot` and answer "where is X?" in ~100 tokens. Every channel below links
to it.

---

## 2. Situation snapshot

**Assets we already have**
- A differentiated, defensible position: the only embeddable, Rust-native agent memory with
  graph + 5 memory types + decay/beliefs/consolidation, **no LLM required**.
- Real, committed benchmarks: up to ~80% fewer context tokens in an equal-correctness A/B on
  a large repo (≈ parity on tiny repos — we say so), 93.5% Recall-QTC on the LongMemEval
  500-question baseline (`benchmarks/results/qtc-500.json`), <100 ms CLI commands, ~5–10 MB
  binary.
- Working distribution: crates.io publish via release-plz, prebuilt binaries + cargo-binstall
  (Windows ONNX bundled — a real pain point solved), MCP server, `axil install` one-command
  agent wiring.
- A README that already tells the story well.

**Honest blockers (see §9)**
- PolyForm NC license: excluded from "open source only" lists, corporate users hesitate.
- Single maintainer, zero social proof (0 stars, no testimonials).
- Crowded, noisy category ("agent memory" has Mem0 at 51k stars).

**Implication:** we can't out-shout Mem0. We win on a sharp wedge — *local-first, no-LLM,
token-frugal memory for coding agents* — aimed at communities that value exactly that
(r/LocalLLaMA, Rust devs, Claude Code / Cursor power users).

---

## 3. Goals & metrics

| Horizon | Stars | Other signals |
|---|---|---|
| 30 days | 100+ | 1 HN front-page or 200+-upvote Reddit post; 500+ unique repo visitors/wk; first 3 external issues |
| 60 days | 300+ | 1k crates.io downloads; listed in 4+ MCP directories & 2+ awesome lists; first external PR |
| 90 days | 750+ | 3 published deep-dive posts; comparison pages ranking for "mem0 alternative"-type queries; 5+ unsolicited mentions |

Track weekly: GitHub Insights (visitors, clones, referrers), star history, crates.io
downloads, MCP directory installs. Referrer data tells you which channel worked — double
down there.

---

## 4. Positioning & core messages

### Audiences (in priority order)

1. **Coding-agent power users** (Claude Code, Cursor, Codex CLI, Copilot CLI) — feel the
   amnesia + token-burn pain daily. Hook: *money and repeated context*.
2. **Local-first / privacy devs** (r/LocalLLaMA, self-hosters) — allergic to cloud memory
   services. Hook: *one file, fully offline, no LLM, no telemetry*.
3. **Rust developers** — appreciate the engineering (redb, tantivy, HNSW, ONNX in one
   binary). Hook: *the architecture itself*.
4. **Agent-framework builders** — need an embeddable memory layer, not another SaaS. Hook:
   *`Axil::open(path)` and you're done*.

### One-liners (pick per channel)

- **Primary:** *Agent memory in one local file. No server, no cloud, no LLM.*
- **Compounding angle:** *A memory that compounds: your agent starts tomorrow smarter than it ended today.*
- **Money angle:** *Your agent re-reads the same files every session — and burns your money doing it. Axil fixes that.*
- **Rust angle:** *Vector + knowledge graph + full-text + time-series in a single ~5–10 MB Rust binary.*
- **Anti-stack angle:** *Mem0 needs an LLM plus three databases to store a memory. Axil needs a file path.*

### Elevator pitches

**10 seconds:** Axil is cognitive memory for AI agents in a single local file — vector
search, knowledge graph, full-text and time-series in one Rust binary, with an MCP server.
No LLM required, nothing to host.

**30 seconds:** Coding agents are brilliant and amnesiac: every session they re-read the
same files, re-learn the same architecture, repeat the same mistakes — and burn tokens doing
it. Axil is the second brain that fixes this. It stores decisions, errors, and code
structure across sessions and hands the agent the *right* memory at the right moment — a
pointer in ~100 tokens instead of a stack of file dumps. In our equal-correctness A/B test
that meant up to ~80% fewer context tokens on a large repo. It's one Rust binary, one
`.axil` file, works with any MCP client, and needs no LLM: importance scoring, decay,
consolidation, and beliefs are all rule-based.

**2 minutes:** add the competitive framing — Mem0 requires an LLM + external databases; Zep
requires Neo4j; Hindsight requires PostgreSQL + an LLM; Memvid is the closest (Rust,
single-file) but has no knowledge graph, no memory types, no consolidation. Axil is the only
one that is embeddable, graph-native, cognitive (decay/beliefs/consolidation), and
LLM-free — plus code-aware: a SCIP code graph and version-pinned dependency docs, offline.

### Proof points (each traces to a source — do not improvise new numbers)

| Claim | Source |
|---|---|
| Up to ~80% fewer context tokens (equal correctness, large repo, semantic queries; ≈ parity on tiny repos) | README A/B methodology section |
| 93.5% Recall-QTC on LongMemEval 500-question subset | `benchmarks/results/qtc-500.json` + CI gate |
| LoCoMo 99% hit rate / 94.4% recall (historical run) | `benchmarks/locomo/` — label "historical" |
| <100 ms CLI commands, ~5–10 MB binary | README; bench-check CI |
| "Where is X?" answered in ~100 tokens | `axil code-search` output, README |

---

## 5. Pre-launch checklist (Week 0 — do before any post)

**P0 — repo surface (an hour of work, permanent payoff)**
- [ ] **GitHub About:** `Agent memory in one local file — vector + knowledge graph +
  full-text + time-series in a single Rust binary. MCP server included. No LLM, no cloud,
  no server.` Website field → docs site or releases page.
- [ ] **GitHub topics:** `ai-agents`, `agent-memory`, `vector-database`, `knowledge-graph`,
  `embedded-database`, `rust`, `mcp`, `mcp-server`, `claude-code`, `local-first`,
  `semantic-search`, `rag`, `llm-memory`, `hnsw`, `developer-tools`
- [ ] **Social preview image** (1280×640): logo + "Agent memory in one local file. No
  server, no cloud, no LLM." — this is what every shared link renders as.
- [ ] **Cargo.toml `keywords` + `categories`** — currently missing from every published
  crate, so Axil is invisible to crates.io/lib.rs browsing. Add to the workspace-inherited
  package metadata, e.g. `keywords = ["ai", "agent", "memory", "vector", "embedded"]`,
  `categories = ["database", "database-implementations", "caching"]` (crates.io allows 5
  keywords / 5 categories; ships with the next release-plz version).
- [ ] **Pin the repo** on the FC4b profile.
- [ ] **Enable GitHub Discussions** (Q&A + Show-and-tell categories) — gives lurkers a
  low-friction way to engage before filing issues.
- [ ] Seed **5–8 `good first issue` labels** (docs fixes, small CLI ergonomics) — signals a
  living project.

**P0 — the demo GIF (the launch asset)**
- 30–45 s terminal recording (use `vhs` or asciinema→agg), embedded at the top of the README:
  1. `axil install --claude-code --bootstrap` (one command, project wired)
  2. an agent session ends → checkpoint written
  3. new session: `axil boot` → "## Resume Here" block appears
  4. `axil code-search "login handler"` → pointer in ~100 tokens vs a screenful of grep
- Also export a 60–90 s MP4 of the same flow for X/LinkedIn (autoplay video outperforms links).

**P1 — pre-launch content**
- [ ] FAQ section or `docs/faq.md` covering the questions HN *will* ask: Why not
  SQLite+pgvector? Why no LLM — doesn't that limit quality? What exactly does the license
  allow? How is this different from Mem0/Memvid? Is my data sent anywhere? (answer: nothing
  ever leaves the machine)
- [ ] A "Star the repo" CTA at the end of the README quick-start ("If Axil saved your agent
  tokens, a ⭐ helps others find it").
- [ ] Verify install path end-to-end on a clean machine (macOS + Windows) — launch traffic
  that hits a broken `cargo binstall` is worse than no traffic.

---

## 6. Channel playbook — ready-to-paste copy

> Cadence: **one primary channel per day**, engage in comments for 24 h before the next.
> Best posting windows: HN Tue–Thu 8–10 am ET; Reddit weekday mornings US time.

### 6.1 Show HN (the anchor launch)

**Title options (≤ 80 chars, no hype words):**
1. `Show HN: Axil – cognitive memory for AI agents in one file (Rust, no LLM)`
2. `Show HN: Axil – agent memory in a single local file, no server or LLM`
3. `Show HN: I built an agent memory engine that needs no LLM and no server`

**Body (first-person, honest, technical):**

> My coding agent kept re-reading the same files and re-learning my architecture every
> session — at my token cost. Markdown notes files didn't scale, and every "memory layer" I
> found wanted an LLM plus a database server (Mem0 needs an LLM + external stores; Zep
> needs Neo4j; Hindsight needs PostgreSQL + an LLM).
>
> So I built Axil: cognitive memory for AI agents in a single local file. One Rust binary
> (~5–10 MB) with vector search (HNSW + local ONNX embeddings), a knowledge graph, full-text
> search (tantivy) and time-series — fused into one ranked recall. It ships an MCP server,
> so it plugs into Claude Code, Cursor, Codex, or anything MCP-speaking; hooks auto-capture
> decisions/errors and write a structured checkpoint the next session resumes from.
>
> The part I'm most proud of: the "cognitive" layer needs no LLM. Importance scoring, decay
> with reinforcement (active forgetting), contradiction detection/consolidation, and a
> belief system are all rule-based. It's also code-aware — a SCIP code graph (real
> callers/callees) and version-pinned docs for your exact dependency versions, all offline.
>
> Numbers (methodology in the repo): in an equal-correctness A/B on a large repo, agents
> answered the same coding questions with up to ~80% fewer context tokens (≈ parity on a
> tiny repo where grep already wins). 93.5% Recall-QTC on a 500-question LongMemEval subset,
> gated in CI.
>
> Honest caveats: it's source-available under PolyForm Noncommercial (free for personal/
> research use; commercial needs a license) — I know how HN feels about that and I'm happy
> to discuss the reasoning. Single maintainer. Rule-based extraction won't match an LLM's
> extraction quality on messy prose; it trades that for determinism, speed, and $0.
>
> Repo: https://github.com/FC4b/axildb — I'd love feedback on the recall-quality
> methodology most of all.

**HN survival rules:** reply to every comment in the first 3 h; never argue about the
license, just explain the reasoning and note you're open to feedback; concede valid
criticism immediately; have the FAQ ready to link.

### 6.2 Reddit (one sub per day, tailored — never cross-post identical text)

**r/rust — angle: the engineering.**
Title: `Axil: vector + knowledge graph + FTS + time-series in one embedded Rust file (redb, tantivy, HNSW, ONNX)`
Body: architecture-first — redb core with companion files per engine (like SQLite WAL/SHM),
the `Engine` trait / plugin tiers, why hand-rolled prost messages instead of build.rs, how
the workspace keeps a ~5–10 MB binary. End with "built as memory for AI agents, but the
storage layer stands alone." r/rust upvotes craft, not AI hype.

**r/LocalLLaMA — angle: local-first, no cloud, no LLM.**
Title: `Agent memory that runs 100% offline — one file, local embeddings, no LLM required`
Body: lead with "nothing leaves your machine": local BGE embeddings via ONNX, rule-based
consolidation instead of LLM calls, works with any local agent stack via MCP or CLI.
Contrast with memory layers that phone an API for every extraction. This community will
ask about GGUF/embedding-model choices — mention bge-small default, configurable models.

**r/ClaudeAI (+ r/ChatGPTCoding later) — angle: the workflow payoff.**
Title: `I gave Claude Code persistent memory — it resumes sessions and answers "where is X?" in ~100 tokens`
Body: show the loop concretely: `axil install --claude-code --bootstrap`, hooks capture
decisions/errors, `axil boot` injects "Resume Here" next session. GIF up front. Mention the
~80% token A/B result with the caveat sentence intact.

**r/selfhosted — angle: it's a file, not a service.**
Short post: "Memory for AI agents you don't have to host — it's literally one file next to
your repo." This sub loves anti-SaaS framing.

### 6.3 Lobste.rs
Submit the architecture blog post (§7, post #3) rather than the repo — Lobste.rs prefers
write-ups. Tags: `rust`, `ai`, `databases`.

### 6.4 X/Twitter launch thread (also the LinkedIn source material)

1. Your coding agent is brilliant — and amnesiac. Every session it re-reads the same files,
   re-learns your architecture, repeats the same mistakes. And you pay for every token of it.
2. I built Axil to fix that: cognitive memory for AI agents in ONE local file. No server.
   No cloud. No LLM. [demo video]
3. One ~5–10 MB Rust binary: vector search + knowledge graph + full-text + time-series,
   fused into one ranked recall. MCP server built in — works with Claude Code, Cursor,
   Codex, anything.
4. The memory *compounds*. Importance scoring, decay + reinforcement, contradiction
   detection, beliefs — all rule-based. Your agent starts tomorrow smarter than it ended
   today.
5. It's code-aware: a SCIP code graph knows your real callers/callees, and dependency docs
   are pinned to your exact lockfile versions. "Where is X?" → a pointer in ~100 tokens.
6. Measured, equal-correctness A/B on a large repo: up to ~80% fewer context tokens for the
   same answers. (Tiny repos: parity — grep is already great there. Methodology in the repo.)
7. Setup is one command: `axil install --claude-code --bootstrap`. Hooks capture decisions
   and errors as you work; next session resumes from a structured checkpoint.
8. Free for personal & research use. Built in Rust, runs fully offline, nothing ever leaves
   your machine. → github.com/FC4b/axildb  ⭐ if your agent deserves a memory.

### 6.5 This Week in Rust
Submit to the "Project Updates / New Crates" section via PR to `rust-lang/this-week-in-rust`
the week of the r/rust post. One sentence + link. Free, high-signal Rust audience.

### 6.6 MCP directories & awesome lists (compounding, zero-cost distribution)

| Surface | Action |
|---|---|
| mcp.so, PulseMCP, Glama, Smithery | Submit axil-mcp with the About text + tool list (recall/store/link/search/query_history/code_search/dep_docs/checkpoint) |
| `punkpeye/awesome-mcp-servers` | PR under Knowledge & Memory — follow their format exactly |
| `e2b-dev/awesome-ai-agents` | PR under memory/tools |
| awesome-claude-code (+ cursor equivalents) | PR the skills/hooks integration |
| `rust-unofficial/awesome-rust` | PR under Database — note: some awesome lists require OSI licenses; check CONTRIBUTING first, don't burn goodwill |
| crates.io / lib.rs | Fixed by the keywords/categories item in §5 |

### 6.7 Discords / communities (participate, don't spam)
Rust Community Discord (#showcase), Claude Developers Discord, MCP community spaces,
Cursor forum. Rule: answer memory/context questions helpfully and link Axil only when it
genuinely answers the question. One good answer > ten drive-by links.

### 6.8 Product Hunt — defer to Phase 2 (after ~300 stars + testimonials), launching there
with zero social proof wastes the one-shot.

---

## 7. Content engine (Weeks 2–12)

Three pillar posts (publish on a blog or dev.to, syndicate links to HN/Reddit/X):

1. **"Your agent is brilliant and amnesiac"** — the problem essay. Token-burn math of
   re-reading context every session; why markdown memory files stop scaling; what
   "compounding memory" means. Ends at Axil. (Broadest reach; HN-friendly.)
2. **"How we measured ~80% context-token savings without cherry-picking"** — the A/B
   methodology post: equal-correctness design, large vs tiny repo results, where Axil does
   *not* help. Radical honesty is the differentiator; this post earns trust and citations.
3. **"One file, four indexes: building an embedded vector+graph+FTS+time-series engine in
   Rust"** — the architecture deep-dive for r/rust and Lobste.rs: redb + companion files,
   Engine/Extension/Adapter tiers, HNSW + ONNX embedding pipeline, int8/binary quantization.

Four comparison pages in `docs/` (SEO — these capture buyer-intent searches):
- **Axil vs Mem0** ("mem0 alternative", "mem0 without LLM") — no LLM, no external DBs, embeddable.
- **Axil vs Memvid** — closest rival; graph + memory types + consolidation vs doc store.
- **Axil vs Zep/Graphiti** — no Neo4j, Community Edition deprecation angle.
- **Axil vs Redis Iris / agent-memory-server** — local file vs cloud bundle; no-LLM vs LLM-required (reuse the Phase-25-era research already in Axil memory).

Each page: honest feature table (reuse CLAUDE.md's), "when to choose them instead" section
(credibility), quick-start CTA.

---

## 8. Momentum mechanics (ongoing)

- **Ship visibly weekly:** release-plz already cuts versions — write 3-line human release
  notes on each GitHub Release; releases feed and Watch subscribers are free re-engagement.
- **Respond fast:** every issue/discussion within 24 h during the first 90 days. Early
  adopters become evangelists precisely when maintainer response is instant.
- **Build in public on X:** short weekly "what shipped" notes with a GIF. Benchmarks and
  perf graphs outperform feature lists.
- **Convert users to proof:** when someone reports success, ask permission to quote them in
  the README ("What users say").
- **Star etiquette:** the README CTA plus, optionally, one tasteful line at the end of
  `axil install` output ("Wired ✓ — if Axil helps, a star helps others find it:
  <repo url>"). No nags, no popups, shown once.

---

## 9. Honest blockers & strategic risks

1. **License (the big one).** PolyForm NC excludes Axil from OSI-only awesome lists and
   makes companies hesitate — while Memvid (Apache-2.0) sits at 13.7k stars. Options, in
   increasing boldness: (a) crystal-clear LICENSING FAQ — "free for individuals, hobbyists,
   research; commercial = paid" front and center; (b) add a free small-business tier
   (PolyForm Small Business or custom grant) to defuse the "can my startup even try this?"
   objection; (c) dual-license selected crates (e.g. axil-core) Apache-2.0 and keep the
   cognitive extensions NC — open-core, maximum reach. **Decision owner: you.** Marketing
   can work with (a) alone, but expect the license to be the top HN comment.
2. **Solo-maintainer risk perception.** Mitigate with visible CI, weekly releases, fast
   issue responses, and a public roadmap (GitHub Projects board).
3. **Category noise.** Don't fight "best agent memory" head-on; own the wedge — *local-first,
   no-LLM, token-frugal memory for coding agents* — and let comparisons pull the rest in.
4. **Claim skepticism.** The ~80% figure will be challenged; always ship it with the
   equal-correctness + large-repo caveat attached, and link the methodology. The caveat *is*
   the credibility.

---

## 10. 30 / 60 / 90-day calendar

**Weeks 0–1 (Jul 13–24): polish + anchor launch**
- Week 0: §5 checklist complete (About/topics/social preview/keywords/GIF/FAQ), clean-machine
  install verified, PR #17 merged so the release is current.
- Week 1: Show HN (Tue–Thu am ET) → r/rust (+ This Week in Rust PR same week) → r/LocalLLaMA.

**Weeks 2–4: second wave + surfaces**
- r/ClaudeAI, r/selfhosted, X thread, LinkedIn.
- All MCP directory submissions + awesome-list PRs.
- Publish post #1 (problem essay); submit to HN.

**Weeks 5–8: content + comparisons**
- Publish post #2 (methodology) and the Mem0 + Memvid comparison pages.
- Lobste.rs with post #3 (architecture).
- Start weekly build-in-public cadence on X.

**Weeks 9–12: consolidate**
- Zep + Redis Iris comparison pages; post #3 syndication.
- First "What users say" README section; evaluate Product Hunt readiness.
- Retro: which referrers drove stars (GitHub Insights) → double down on the top two channels.

---

## Appendix: copy-paste snippets

**GitHub About:**
`Agent memory in one local file — vector + knowledge graph + full-text + time-series in a single Rust binary. MCP server included. No LLM, no cloud, no server.`

**One-line directory blurb:**
`Axil — cognitive memory for AI agents: one local file, one Rust binary, vector + graph + FTS + time-series, MCP server, no LLM required. Free for noncommercial use.`

**Signature CTA (issues/posts):**
`If Axil saved your agent tokens, a ⭐ on github.com/FC4b/axildb helps others find it.`
