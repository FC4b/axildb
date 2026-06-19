//! Workspace-level audit of every in-tree Extension.
//!
//! The Extension set is sourced from [`axil_bundle::builtin_extensions`] — the
//! same central registry the CLI and MCP server use — so this audit
//! automatically covers any new built-in Extension the moment it is added to
//! the bundle. There is no hand-maintained list to keep in sync.
//!
//! Asserts each Extension follows the conventions in
//! `docs/src/extending/extensions.md`:
//!
//! 1. registers cleanly with `Axil::open(...).with_extension_arc(...)`;
//! 2. has a non-empty, kebab-case `id()`;
//! 3. declares only leading-underscore `table_prefixes()` (Extension-owned
//!    tables live under the leading-underscore namespace);
//! 4. the builder-time prefix-overlap rejection fires on conflicting prefixes;
//! 5. the builder-time duplicate-id rejection fires.

use std::sync::Arc;

use axil_core::{Axil, AxilConfig, Extension};

#[test]
fn every_bundle_extension_registers_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    // Derived from the central registry — adding a built-in Extension to the
    // bundle automatically extends this audit's coverage.
    let exts = axil_bundle::builtin_extensions(&AxilConfig::default());
    assert!(
        !exts.is_empty(),
        "no built-in Extensions in the bundle — under `full` at least axil-docs should be present",
    );

    for ext in &exts {
        let ext_id = ext.id().to_string();
        let prefixes: Vec<String> = ext.table_prefixes().iter().map(|s| s.to_string()).collect();

        // id() convention: non-empty kebab-case.
        assert!(!ext_id.is_empty(), "an Extension reported an empty id()");
        assert!(
            ext_id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "extension id `{ext_id}` is not kebab-case",
        );

        // The Extension must register without panicking, into a fresh DB.
        let path = dir.path().join(format!("audit-{ext_id}.axil"));
        let db = Axil::open(&path)
            .with_extension_arc(Arc::clone(ext))
            .build()
            .unwrap_or_else(|e| panic!("extension `{ext_id}` failed to register: {e}"));

        assert_eq!(
            db.extensions().len(),
            1,
            "extension `{ext_id}` registered, but Axil::extensions() did not surface it",
        );
        assert_eq!(db.extensions()[0].id(), ext_id);

        // Every declared prefix must start with `_`.
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

