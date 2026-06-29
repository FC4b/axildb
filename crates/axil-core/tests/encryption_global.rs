//! Process-wide default-cipher wiring (the `encryption` feature).
//!
//! This is an integration test in its own binary on purpose: it installs a
//! process-wide default cipher via `set_default_cipher`, which is a `OnceLock`
//! set-once. Running it in its own crate keeps that global out of the library
//! unit-test binary (which opens hundreds of cleartext databases).
//!
//! What it guards: that a *raw* `Axil::open(path).build()` — with no explicit
//! `with_cipher` — picks up the installed default cipher. That is the exact open
//! shape used by core-internal multi-database operations (`branch_merge`, the
//! workspace federation fan-out) that the CLI cannot reach with an explicit
//! cipher. The CLI installs the default once at startup; these opens inherit it.
#![cfg(feature = "encryption")]

use axil_core::crypto::Cipher;
use axil_core::Axil;
use serde_json::json;

/// Every test in this binary shares ONE key — `set_default_cipher` is set-once,
/// so the first install wins and all opens in the process use it. Tests must not
/// assume a different global key.
fn install_global_key() {
    let _ = axil_core::crypto::set_default_cipher(Cipher::from_key_bytes(&[5u8; 32]).unwrap());
}

/// A raw `Axil::open(path).build()` (no explicit cipher) seals on write and
/// unseals on read via the installed process default — proving branch/workspace
/// style opens are covered without threading a key to each call site.
#[test]
fn global_cipher_round_trips_raw_builder_opens() {
    install_global_key();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.axil");

    // Scoped so the writer lock releases before reopening.
    let id = {
        let db = Axil::open(&path).build().unwrap(); // NB: no .with_cipher
        db.insert("secrets", json!({"summary": "global-needle"}))
            .unwrap()
            .id
    };

    let db = Axil::open(&path).build().unwrap();
    let fetched = db.get(&id).unwrap().unwrap();
    assert_eq!(fetched.data["summary"], "global-needle");
}

/// The body a raw (global-cipher) open writes is genuinely sealed: a handle that
/// supplies a *different* explicit key cannot read it. This also confirms an
/// explicit `with_cipher` overrides the process default for that handle.
#[test]
fn global_sealed_body_rejects_wrong_explicit_key() {
    install_global_key();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g2.axil");

    let id = {
        let db = Axil::open(&path).build().unwrap(); // sealed under the global key
        db.insert("secrets", json!({"summary": "sealed"}))
            .unwrap()
            .id
    };

    // Explicit wrong key overrides the global default and fails to decrypt.
    let wrong = Cipher::from_key_bytes(&[9u8; 32]).unwrap();
    let db = Axil::open(&path).with_cipher(wrong).build().unwrap();
    assert!(
        db.get(&id).is_err(),
        "a raw global-cipher write must be sealed, not cleartext"
    );
}
