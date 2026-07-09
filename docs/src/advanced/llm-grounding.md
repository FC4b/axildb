# Optional LLM Grounding

Axil's cognition runs **without an LLM**. Entity extraction, consolidation,
importance scoring, decay, recall, and superseding are all algorithmic —
no API key, no network, no model to host. An LLM is a *sharpener*, not a
dependency: plug one in and a few specific steps get higher-quality
output; leave it out and everything still works.

This page documents exactly what the algorithmic default does, exactly
what an LLM upgrades, how to configure a provider, and the graceful-
fallback behavior that guarantees an LLM outage never breaks a write.

> **Two ways an LLM enters the picture.** This page is about *Path B* —
> handing Axil an LLM callback so the library can call it internally. The
> primary integration for Claude Code and agent frameworks is *Path A*,
> where the agent itself is the LLM and orchestrates Axil via the CLI. In
> Path A you never configure an `[llm]` block at all. See
> [Claude Code Integration](../agents/claude-code.md).

## The algorithmic default (no LLM)

With no provider configured, Axil uses pattern-based algorithms for the
two steps an LLM would otherwise assist:

- **Entity extraction** — regex-based extraction pulls code identifiers,
  file paths, and structural tokens out of stored text. This is precise
  on the things agents most need linked (symbols, paths) and needs no
  model.
- **Consolidation** — merging multiple facts about an entity uses a
  template: it surfaces the latest value and a count of prior facts
  (e.g. *"Uses JWT with 1h expiry (latest: 2026-06-01). 2 prior
  facts."*). Deterministic and free.

Every other cognitive feature — importance scoring, recency-weighted
recall, superseding, decay, beliefs — is algorithmic in *all*
configurations; an LLM never touches them. As the
[project overview](../introduction.md) puts it, entity extraction,
consolidation, and inference all run algorithmically with no LLM
required; the two steps below are simply where a provider can lift the
output quality.

## What an LLM upgrades

When a provider is configured and available, exactly these code paths
change. Each one **starts from the algorithmic result and improves on
it** — the LLM never fully replaces the baseline:

| Step | Without LLM | With LLM |
|------|-------------|----------|
| Entity extraction | Regex: code identifiers, file paths | Regex baseline **plus** LLM-found people, concepts, and names; results merged (LLM wins on conflicts) |
| Consolidation | Template: latest value + prior-fact count | Prose summary noting what changed and when |
| Recall rerank | RRF fusion order (or cross-encoder) | `axil recall --rerank llm` re-scores the top candidates through the provider |

The extraction merge is additive by design: the algorithmic entities are
always computed first as a baseline, then LLM-discovered entities that
aren't already present are added on top. So enabling an LLM can only
*add* recall signal for person/concept entities that regex can't catch —
it never drops the code identifiers the regex path already found. The
extraction and consolidation logic is in
[`crates/axil-core/src/db.rs`](../../../crates/axil-core/src/db.rs)
(`extract_entities_enhanced`, `consolidate_entity_enhanced`).

## Configuring a provider

The provider is an OpenAI-compatible chat-completions endpoint (OpenAI,
Anthropic via a compatible endpoint, Ollama, OpenRouter, or any service
speaking that format). Configure it in the `[llm]` block of `axil.toml`:

```toml
[llm]
endpoint = "https://api.openai.com/v1/chat/completions"
model = "gpt-4o-mini"
# api_key = "sk-..."   # prefer the env var below over committing a key
```

Provide the key via environment variable — it takes precedence over the
config file, so secrets stay out of version control:

```bash
export AXIL_LLM_API_KEY="sk-..."
```

A provider is considered configured only when endpoint, model, and a
resolved API key are all present. See
[Configuration](../getting-started/configuration.md) for the full
`axil.toml` reference.

### Cost limits (fallback triggers)

The `[llm]` block also carries usage limits. When any limit is reached,
Axil falls back to algorithmic rather than blocking or erroring. The
defaults are conservative:

| Limit | Default | Behavior when hit |
|-------|---------|-------------------|
| Calls per minute | `10` | Fall back to algorithmic |
| Tokens per session | `50000` | Fall back to algorithmic |
| Budget per day (USD) | `1.0` | Fall back to algorithmic |

A value of `0` (or `0.0` for the budget) disables that particular limit.
Per-1M-token input/output costs are also configurable for accurate cost
tracking. The limit and tracking types live in
[`crates/axil-core/src/llm.rs`](../../../crates/axil-core/src/llm.rs).

## Graceful fallback

The fallback contract is the whole point of the design: **an LLM path
that fails behaves exactly as if no LLM were configured.** Concretely:

- If no provider is set, or the provider reports itself unavailable, the
  guarded call returns an error and the caller uses its algorithmic
  path. No feature is gated behind the LLM.
- Before every call, a rate limiter checks the per-minute, per-session,
  and daily-budget limits in a single lock (so concurrent calls can't
  race past the ceiling). If a limit is exceeded, the call is refused and
  counted as a fallback.
- Successful calls record their input/output token usage; refused calls
  increment a fallback counter. Both feed the usage snapshot below.

Extraction always computes its algorithmic baseline first and only
*adds* LLM-found entities on top; consolidation falls back to its
template on any error, empty response, or unparseable output. Either
way, a mid-write LLM outage can only cost you the *quality upgrade* for
that record — never the record itself.

## CLI

Three subcommands manage and inspect the provider:

```bash
axil llm test      # test connectivity to the configured provider
axil llm config    # show the resolved LLM configuration
axil llm usage     # show call count, tokens, estimated cost, and fallbacks
```

`axil llm usage` reports in-memory counters for the current process (they
are not persisted across invocations), including how many calls fell back
to algorithmic — a quick way to see whether your limits are being hit.

## See also

- [Configuration](../getting-started/configuration.md) — the `[llm]` block and environment variables
- [Claude Code Integration](../agents/claude-code.md) — Path A, where the agent is the LLM
- [Cognitive Memory](./cognitive.md) — the algorithmic cognition an LLM sharpens
- [Session Compaction & Token Budgets](./session-compaction.md) — consolidation and superseding in context
- [Retrieval Pipeline](./retrieval-pipeline.md) — where `--rerank llm` fits the recall flow
