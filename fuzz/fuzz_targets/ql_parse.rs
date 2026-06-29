#![no_main]

//! Fuzz target for `axil_ql::parse` — the one genuinely untrusted byte surface
//! in Axil.
//!
//! An AxilQL query string can arrive verbatim from an external client (the
//! query console, an MCP/HTTP caller, a `axil ql '<...>'` argv). Everything
//! else Axil parses is either internal serialization it wrote itself or a file
//! the operator already trusts. The parser must therefore be panic-free and
//! bounded on *arbitrary* bytes: it may return a `ParseError`, but it must
//! never panic, abort, or run away on memory.
//!
//! The harness only feeds bytes to `parse`. A returned `Ok`/`Err` is equally
//! fine — libFuzzer flags a finding only on a panic, an abort, an OOM, or a
//! timeout (the run-away cases). Seed corpus lives in `fuzz/corpus/ql_parse/`,
//! mirroring the adversarial inputs already asserted panic-free in
//! `crates/adapters/axil-ql/tests/comprehensive.rs` (the `f0*` fuzz-safety
//! tests).
//!
//! Run (nightly + libFuzzer required):
//! `cargo +nightly fuzz run ql_parse`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The parser's contract is over `&str`, so only exercise it on valid UTF-8
    // (invalid UTF-8 never reaches `parse` — callers hand it a Rust `String`).
    // libFuzzer mutates raw bytes; the lossy reject keeps us on the real
    // surface instead of fuzzing `from_utf8` itself.
    if let Ok(input) = std::str::from_utf8(data) {
        // Discard the result: the property under test is "does not panic / OOM
        // / hang", not the parse outcome. Both `Ok` and `Err` are correct.
        let _ = axil_ql::parse(input);
    }
});
