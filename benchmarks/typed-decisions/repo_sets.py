#!/usr/bin/env python3
"""Emit the hand-labeled U1/U2 pairs drawn from Axil's own dogfood memory.

Texts are shortened from real `decisions` / `errors` / `context` records in
this repo's `.axil` (record-id suffix in `source`), in the terse style Axil
stores. Labels were authored by Claude on 2026-09-27 and still need a human
spot-check; see README.md.

  U1 (supersession): label 1 = NEW makes OLD no longer current (status change,
      reversal, correction). label 0 = both still hold (restatement,
      refinement, addition, same topic, or a negation about something else).
  U2 (error resolved): label 1 = LATER shows the ERROR is actually fixed.
      label 0 = still broken, only planned, merged-but-not-shipped, or a fix
      for something else.

Usage: python repo_sets.py   (rewrites sets/u1_repo.jsonl and sets/u2_repo.jsonl)
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (id, old, new, label, kind, source)
U1 = [
    ("r1-01", "Initial crates.io publish is PARTIAL: 5 of 17 crates live, 12 pending on the new-crate rate limit.",
     "axildb is fully published to crates.io: all 17 crates live at v1.0.0.", 1, "update", "3KEQ61->1X1R6R"),
    ("r1-02", "axildb 2.3.0 is still unpublished: the release-plz release job fails with crates.io 403.",
     "axildb 2.3.0 published 2026-09-25 after rotating CARGO_REGISTRY_TOKEN and rerunning release-plz.", 1, "update", "K8J5AE->B8BSYS"),
    ("r1-03", "WASM plugin support (Part B) is toolchain-blocked and cannot be built in this environment.",
     "Full WASM plugin round-trip works end to end: a real .wasm component loads and runs sandboxed.", 1, "contradiction", "PDRTWX->SHPAJ1"),
    ("r1-04", "Top improvement from the CodeGraph analysis: build a one-call verbatim-source explore tool.",
     "Revised verdict: the explore tool is not worth building as specified; it reverses the compact-pointer design.", 1, "reversal", "BBA93V->HZQSA3"),
    ("r1-05", "Phase 21 plan lives in tasks/phase-21-extensibility-1.0.md.",
     "Merged Phase 21 and Phase 22 into one doc, tasks/phase-21-22-extensibility-and-wasm.md, and deleted the originals.", 1, "update", "HJRS5R->QR2QKQ"),
    ("r1-06", "brain::MemoryType and memory_type_to_table() are dead code and safe to delete.",
     "Do not delete brain::MemoryType: remember() is wired to Command::Remember and the belief commands are live.", 1, "contradiction", "TSQGC6"),
    ("r1-07", "The benchmark harnesses were untracked from git and cannot be regenerated.",
     "74 files are tracked under benchmarks/, including every harness's Cargo.toml and src/main.rs.", 1, "contradiction", "2K01HP"),
    ("r1-08", "The project has no CI.",
     "CI runs build, tests and the recall-quality gate on every PR via .github/workflows/ci.yml.", 1, "contradiction", "6XZG4S"),
    ("r1-09", "The backlog features are on main but not published: crates.io is still at 1.1.1.",
     "axildb 1.2.0 published to crates.io with the open-backlog features.", 1, "update", "XT8M93->JWJR8J"),
    ("r1-10", "README tagline: 'Cognitive memory for AI agents. One binary. No LLM required.'",
     "README tagline is now 'Agent memory in one local file. No server, no cloud, no LLM.'", 1, "update", "VDN1GD"),
    ("r1-11", "The publishable CLI crate is axil-cli.",
     "Renamed the publishable CLI crate axil-cli to axildb; cargo install axildb installs the axil binary.", 1, "update", "TATQBX"),
    ("r1-12", "Plan: genericize the Engine lifecycle hooks into a single loop (21.6).",
     "Decision on 21.6: do not genericize the Engine lifecycle hooks; the typed hooks have non-uniform semantics.", 1, "reversal", "WBDE7Q->Z9WZ0P"),
    ("r1-13", "All workspace crates are at version 1.0.0.",
     "axildb 1.1.0 published: all 17 crates are now at 1.1.0.", 1, "update", "1X1R6R->2RNZJC"),
    ("r1-14", "Releases ship six targets, including x86_64-apple-darwin for Intel Macs.",
     "Dropped the x86_64-apple-darwin release target; releases now ship five triples.", 1, "update", "1FB33G"),
    ("r1-15", "Local cargo build --release links Homebrew onnxruntime dynamically.",
     "The repo's .cargo/config.toml sets LIBONNXRUNTIME_NO_PKG_CONFIG=1, so every local build links onnxruntime statically.", 1, "update", "R1SFJX->Z972VD"),
    ("r1-16", "Open backlog progress: 5 of 8 items done and committed on feat/open-backlog.",
     "Open backlog shipped to main: PR #5 merged with all backlog items.", 1, "update", "CEVPQT->JKR10E"),
    ("r1-17", "Phase 29 plan: integrate the Laya typed-decision model, Python spike then a Rust ONNX TypedDecider.",
     "Phase 29 reframed as a per-decision bake-off; small NLI and reranker models are the main candidates, Laya is the baseline.", 1, "reversal", "VVQZM3->W1RR1K"),
    ("r1-18", "The dogfood vector store is dead; recall runs FTS-only.",
     "The dogfood vector store was repaired and 193 missing embeddings restored; recall uses vectors again.", 1, "update", "GMS3Q7->MTAZWX"),
    ("r1-19", "Cross-encoder reranking is the recommended way to lift recall quality.",
     "Do not wire either reranker into db.recall(): both regressed LongMemEval recall by 17-27 points.", 1, "contradiction", "lme-rerank-investigation"),
    ("r1-20", "Deploys happen at 5pm.", "Deploys now happen at 6pm.", 1, "update", "consolidation.rs doc example"),

    ("r0-01", "Phase 22 WASM runtime plugins are functionally complete and shipped (19 commits).",
     "Phase 22 is fully complete: the compiled-module cache and ABI negotiation polish landed.", 0, "refinement", "AFQ292/WTWCBM"),
    ("r0-02", "Phases 21, 22 and 23 are confirmed shipped in 1.0.0.",
     "Phase 22 WASM runtime plugins shipped and documented.", 0, "restatement", "1ZBMJM/AFQ292"),
    ("r0-03", "Fixed numbers-integrity violations in README, docs and CLI output.",
     "Wired the offline recall-quality gate into CI.", 0, "both_hold", "HDAVME/GWQPSJ"),
    ("r0-04", "The HNSW graph is built lazily on the first search above the 128-vector exact-scan threshold.",
     "The HNSW graph is not persisted, so every CLI process rebuilds it on first search.", 0, "refinement", "QRRXHC/FNZHF0"),
    ("r0-05", "One supersede path: can_supersede and mark_superseded enforce table, policy, pin and recency.",
     "axil.toml supersede threshold and lifecycle settings are applied in AxilBuilder::build.", 0, "both_hold", "EHA4KW/TVNH0N"),
    ("r0-06", "Release pipeline fully working: 2.3.0 and 2.3.1 are on crates.io.",
     "First fully automatic release: axildb 2.3.1 was published, tagged and built for 5 platforms.", 0, "restatement", "5KN8XT/3BWDRX"),
    ("r0-07", "WASM is in scope for Axil 1.0; Phase 21 is its foundation.",
     "Do not genericize the Engine lifecycle hooks in Phase 21.6.", 0, "negation_other_subject", "ZYGJN4/Z9WZ0P"),
    ("r0-08", "Do not promote the ask strategy: it scores 26.7% on LongMemEval.",
     "Do not enable reranking by default: every tested reranker regressed recall.", 0, "both_hold", "lme-rerank-investigation"),
    ("r0-09", "Compared ponytail and axildb benchmark harnesses: a category mismatch, not head-to-head.",
     "Ponytail gap analysis: axildb is more mature on every axis except benchmark reproducibility.", 0, "both_hold", "DF9Q8D/VNE51D"),
    ("r0-10", "Releases v2.1.0 to v2.2.1 have zero archives.",
     "PR #34 opened; after merge, backfill archives for v2.1.0 to v2.2.1.", 0, "same_topic", "DHHY69/HZQG0Q"),
    ("r0-11", "Renamed the publishable CLI crate to axildb.",
     "Set up crates.io publishing: added version to all 42 internal path deps.", 0, "both_hold", "TATQBX/Y0X2KQ"),
    ("r0-12", "MCP stdio server runs read-only tools concurrently; other tools are barriers.",
     "MCP initialize now returns a server instructions string.", 0, "both_hold", "Z0ZTBZ/B5J46K"),
    ("r0-13", "The dogfood vector store is dead; recall runs FTS-only.",
     "Research found three divergent scorers in the recall path.", 0, "same_topic", "GMS3Q7/FNZHF0"),
    ("r0-14", "Data categorization: table is the canonical category; type becomes a first-class facet.",
     "Categorize by function, not topic: the table is the record's kind.", 0, "restatement", "J6G41E/memory-taxonomy"),
    ("r0-15", "axildb 2.3.0 published after rotating CARGO_REGISTRY_TOKEN.",
     "Dropped the Intel Mac release target in PR #39.", 0, "same_topic", "B8BSYS/1FB33G"),
    ("r0-16", "Parallel multi-agent fixes use one git worktree per agent.",
     "Parallel agents share the build cache via an APFS copy-on-write clone of target/debug.", 0, "refinement", "KXRKKJ"),
    ("r0-17", "Phase 25 P0 shipped: benchmark provenance fixed and sqlite-compare CI-gated.",
     "Nightly Criterion benchmarks run informationally via nightly-bench.yml.", 0, "both_hold", "3D8AE5"),
    ("r0-18", "Deploys happen at 5pm.", "Deploys happen at 5pm on weekdays; hotfixes can ship any time.", 0, "refinement", "synthetic, consolidation style"),
    ("r0-19", "The user prefers dark mode in the editor.", "The user prefers light mode for printed docs.", 0, "negation_other_subject", "synthetic, scope trap"),
    ("r0-20", "Recall is not reliable when the vector store is down.", "Recall uses vectors plus FTS fusion.", 0, "negation_other_subject", "synthetic, negation trap"),
]

# (id, error, later, label, kind, source)
U2 = [
    ("e1-01", "release-plz failed publishing axil-core 2.3.0: crates.io 403 Forbidden.",
     "axildb 2.3.0 published 2026-09-25 after rotating CARGO_REGISTRY_TOKEN and rerunning release-plz.", 1, "fixed", "XYHXMD->B8BSYS"),
    ("e1-02", "Nightly ql_parse fuzz panicked: lexer string escape hit a byte index inside a multi-byte char.",
     "PR #36 (ql lexer multi-byte escape fix) merged; main CI green.", 1, "fixed", "73BGJT->K8J5AE"),
    ("e1-03", "Releases v2.1.0 to v2.2.1 have zero archives; the one-line installer 404s.",
     "Archives backfilled for v2.1.0 to v2.2.1 and v2.3.0 on all 5 platforms.", 1, "fixed", "DHHY69->5KN8XT"),
    ("e1-04", "Local cargo build linked Homebrew onnxruntime dynamically, unlike the self-contained release binaries.",
     "The repo's .cargo/config.toml now sets LIBONNXRUNTIME_NO_PKG_CONFIG=1, so local builds link onnxruntime statically.", 1, "fixed", "R1SFJX->Z972VD"),
    ("e1-05", "Dogfood vector store silently dead; recall runs FTS-only.",
     "Vector store repaired with a writable open; 193 missing embeddings restored and recall uses vectors again.", 1, "fixed", "GMS3Q7->MTAZWX"),
    ("e1-06", "Backlog features merged to main but not published; crates.io is still at 1.1.1.",
     "axildb 1.2.0 published to crates.io.", 1, "fixed", "XT8M93->JWJR8J"),
    ("e1-07", "README leads with a 74-80% token-savings headline that has no committed source.",
     "Fixed the benchmark numbers-integrity violations in README, docs and CLI; resolves the flagged credibility risk.", 1, "fixed", "NP78A5->HDAVME"),
    ("e1-08", "Docs claim the benchmark harnesses are untracked and cannot be regenerated.",
     "Phase 25 P0 shipped: fixed the stale provenance in the Cargo.toml exclude comment and the CLAUDE.md note.", 1, "fixed", "2K01HP->3D8AE5"),
    ("e1-09", "benchmarks/sqlite-compare did not build against the current tree.",
     "sqlite-compare now runs as a CI gate at n=10000 on every PR.", 1, "fixed_implicit", "6Z19NF->3D8AE5"),
    ("e1-10", "axil-cli failed to compile with --no-default-features --features wasm-host.",
     "Removed the stray deps cfg from ExtCommand; all 4 feature combinations build.", 1, "fixed", "5K8GPY fix"),
    ("e1-11", "code_graph_hint used _entities as its signal and gave false negatives.",
     "code_graph_hint now keys off _scip_aliases, which only SCIP ingest writes; verified.", 1, "fixed", "0E1EM8 fix"),
    ("e1-12", "brain::tests::pipeline_overhead_under_5ms fails under a full workspace test run.",
     "De-flaked the overhead test with best-of-5 timing; it passes under load.", 1, "fixed", "8AK49P fix"),
    ("e1-13", "README advertised a broken cargo install command that omitted the core feature.",
     "Added core as the first feature in the README install command.", 1, "fixed", "KF6FQ8 fix"),
    ("e1-14", "Docs claim the project has no CI.",
     "Reconciled the docs: CLAUDE.md now describes the CI workflow and the recall gate.", 1, "fixed", "6XZG4S fix"),
    ("e1-15", "Phase 21.3 incomplete: deps and checkpoint commands not migrated to generic dispatch.",
     "Dropped the typed DepsCommand and CheckpointCommand variants; both now ride run_extension_dispatch.", 1, "fixed", "A9VJYE fix"),
    ("e1-16", "/code-review reviewed 187 commits instead of the 6-commit PR diff.",
     "Now fast-forward main before reviewing and check the commit count matches the PR.", 1, "fixed", "81VQ8S fix"),

    ("e0-01", "release-plz failed publishing axil-core 2.3.0: crates.io 403 Forbidden.",
     "axildb 2.3.0 is still unpublished: the release job keeps failing with 403.", 0, "still_broken", "XYHXMD/K8J5AE"),
    ("e0-02", "release-plz failed publishing axil-core 2.3.0: crates.io 403 Forbidden.",
     "User must mint a new crates.io token and set CARGO_REGISTRY_TOKEN.", 0, "planned", "XYHXMD fix-instruction"),
    ("e0-03", "Releases v2.1.0 to v2.2.1 have zero archives; the one-line installer 404s.",
     "PR #34 opened; after merge, backfill archives for v2.1.0 to v2.2.1.", 0, "planned", "DHHY69/HZQG0Q"),
    ("e0-04", "Dogfood vector store silently dead; recall runs FTS-only.",
     "Research found three divergent scorers and a CLI weight sum of 1.45.", 0, "unrelated_same_area", "GMS3Q7/FNZHF0"),
    ("e0-05", "Dogfood vector store silently dead; recall runs FTS-only.",
     "Next step: a trust PR to fix the vector probe and add a doctor check.", 0, "planned", "GMS3Q7/checkpoint"),
    ("e0-06", "Nightly ql_parse fuzz panicked on a multi-byte string escape.",
     "First fully automatic release: axildb 2.3.1 published.", 0, "unrelated_same_area", "73BGJT/3BWDRX"),
    ("e0-07", "Backlog features merged to main but not published; crates.io is still at 1.1.1.",
     "Open backlog shipped to main: PR #5 merged.", 0, "merged_not_shipped", "XT8M93/JKR10E"),
    ("e0-08", "Local cargo build linked Homebrew onnxruntime dynamically.",
     "Dropped the Intel Mac release target.", 0, "unrelated_same_area", "R1SFJX/1FB33G"),
    # e0-09 dropped 2026-09-27: both blind annotators read "rewrote the README hero"
    # as removing the unsourced headline; the text alone supports that reading
    # (the real rewrite predates the error, which the pair cannot show).
    ("e0-10", "Docs claim the benchmark harnesses are untracked and cannot be regenerated.",
     "Verified the Criterion micro-benchmark claim is reproducible from tracked sources.", 0, "unrelated_same_area", "2K01HP/4MTKQX"),
    ("e0-11", "/code-review reviewed 187 commits instead of the 6-commit PR diff.",
     "Fixed all 15 code-review findings on the WASM plugin work.", 0, "lexical_trap", "81VQ8S/9MYYBA"),
    ("e0-12", "Docs claim the project has no CI.",
     "Wired the offline recall-quality gate into the CI workflow.", 0, "other_fix_same_area", "6XZG4S/GWQPSJ"),
    ("e0-13", "axil-cli failed to compile with --no-default-features --features wasm-host.",
     "Plugin to Engine code rename complete across the workspace.", 0, "unrelated_same_area", "5K8GPY/8QWC6D"),
    ("e0-14", "Saved git diff as a patch, then git apply failed after checkout wiped the change.",
     "Parallel multi-agent fixes use one git worktree per agent.", 0, "unrelated_same_area", "JXJEZY/KXRKKJ"),
    ("e0-15", "code_graph_hint used _entities as its signal and gave false negatives.",
     "Phase 23 implemented: MCP initialize now returns an instructions string.", 0, "other_fix_same_phase", "0E1EM8/B5J46K"),
    ("e0-16", "brain::tests::pipeline_overhead_under_5ms fails under a full workspace test run.",
     "The flaky overhead test is still failing intermittently in CI.", 0, "still_broken", "synthetic, 8AK49P"),
]


def split_of(pair_id: str) -> str:
    h = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 4 else "test"


def write(path, rows, a, b):
    with path.open("w") as f:
        for pid, x, y, label, kind, src in rows:
            f.write(json.dumps({"id": pid, a: x, b: y, "label": label, "kind": kind,
                                "source": f"axil-dogfood:{src}", "split": split_of(pid)},
                               ensure_ascii=False) + "\n")
    pos = sum(r[3] for r in rows)
    print(f"wrote {len(rows)} ({pos} positive) to {path.relative_to(HERE)}")


if __name__ == "__main__":
    (HERE / "sets").mkdir(exist_ok=True)
    write(HERE / "sets" / "u1_repo.jsonl", U1, "old", "new")
    write(HERE / "sets" / "u2_repo.jsonl", U2, "error", "later")
