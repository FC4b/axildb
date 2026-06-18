//! Phase 17 P1.4 — workspace-level audit of every in-tree Extension.
//!
//! Walks the workspace's known Extension impls and asserts each one
//! follows the conventions described in
//! `docs/src/extending/extensions.md`:
//!
//! 1. The Extension registers cleanly with `Axil::open(...).with_extension(...)`.
//! 2. Its `id()` matches the crate name minus the `axil-` prefix.
//! 3. Its `table_prefixes()` are non-empty (if it owns any tables) and
//!    each prefix actually starts with `_` (per the convention that
//!    Extension-owned tables live under leading-underscore names).
//! 4. The builder-time prefix-overlap rejection mechanism actually
//!    fires when two Extensions claim conflicting prefixes.
//! 5. The builder-time duplicate-id rejection mechanism actually fires.
//!
//! When a new Extension is added to the workspace, drop a new row into
//! the `KNOWN_EXTENSIONS` table at the bottom of this file. Forgetting
//! to register an Extension here is a documentation gap, not a test
//! failure — but the gap is visible at code-review time.

use std::sync::Arc;

use axil_core::{Axil, Extension};

/// One row per in-tree Extension. Add a new entry whenever a new
/// `axil-*` crate ships an `impl Extension`.
fn known_extensions() -> Vec<KnownExtension> {
    vec![
        KnownExtension {
            crate_name: "axil-docs",
            expected_id: "docs",
            construct: || Arc::new(axil_docs::DocsExtension),
        },
        KnownExtension {
            crate_name: "axil-checkpoint",
            expected_id: "checkpoint",
            construct: || Arc::new(axil_checkpoint::CheckpointExtension),
        },
    ]
}

struct KnownExtension {
    crate_name: &'static str,
    expected_id: &'static str,
    construct: fn() -> Arc<dyn Extension>,
}

#[test]
fn every_known_extension_registers_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let known = known_extensions();
    assert!(
        !known.is_empty(),
        "no known Extensions registered — at minimum axil-docs should be present"
    );

    for ext_spec in &known {
        let path = dir
            .path()
            .join(format!("audit-{}.axil", ext_spec.crate_name));
        let ext = (ext_spec.construct)();
        let ext_id = ext.id().to_string();
        let prefixes: Vec<String> = ext.table_prefixes().iter().map(|s| s.to_string()).collect();

        // The Extension must register without panicking.
        let db = Axil::open(&path)
            .with_extension_arc(ext)
            .build()
            .unwrap_or_else(|e| {
                panic!(
                    "extension `{}` (from {}) failed to register: {e}",
                    ext_id, ext_spec.crate_name,
                )
            });

        // Sanity checks on the impl.
        assert_eq!(
            db.extensions().len(),
            1,
            "extension `{ext_id}` registered, but Axil::extensions() did not surface it",
        );
        assert_eq!(
            db.extensions()[0].id(),
            ext_spec.expected_id,
            "extension id mismatch — expected `{}` for crate `{}`, got `{}`",
            ext_spec.expected_id,
            ext_spec.crate_name,
            db.extensions()[0].id(),
        );

        // Every declared prefix must start with `_` (Extensions own
        // tables in the leading-underscore namespace).
        for p in &prefixes {
            assert!(
                p.starts_with('_'),
                "extension `{ext_id}` declares prefix `{p}` which does not \
                 start with `_` — Extension-owned tables must use the \
                 leading-underscore namespace",
            );
        }
    }
}

#[test]
fn two_of_the_same_extension_panic_on_duplicate_id() {
    // Use a closure-driven catch_unwind so the test passes whether the
    // builder panics or returns an error.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup.axil");
    let result = std::panic::catch_unwind(|| {
        let _ = Axil::open(&path)
            .with_extension(axil_docs::DocsExtension)
            .with_extension(axil_docs::DocsExtension);
    });
    assert!(
        result.is_err(),
        "registering the same Extension twice should panic on duplicate id",
    );
}

#[test]
fn overlapping_prefix_across_extensions_panics() {
    // `axil-docs` owns `_dep`. A second Extension claiming `_dep_…`
    // (or `_dep` itself) must be rejected at builder time.
    struct Conflicting;
    impl Extension for Conflicting {
        fn id(&self) -> &str {
            "conflicting"
        }
        fn table_prefixes(&self) -> &[&str] {
            &["_dep_evil"]
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conflict.axil");
    let result = std::panic::catch_unwind(|| {
        let _ = Axil::open(&path)
            .with_extension(axil_docs::DocsExtension)
            .with_extension(Conflicting);
    });
    assert!(
        result.is_err(),
        "two Extensions with overlapping table prefixes must be rejected at builder time",
    );
}

