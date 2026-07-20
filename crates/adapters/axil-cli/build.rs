//! Embed the git state of the source tree into the binary.
//!
//! `axil --version` prints `axil <semver> (<git describe>)`, where the
//! describe output carries a `-dirty` suffix for uncommitted-source builds.
//! Two same-version binaries are otherwise indistinguishable ("did I actually
//! build what I think I built?"), because the crate version only moves at
//! release time.

use std::path::Path;
use std::process::Command;

fn main() {
    // Force this script to rerun on EVERY build. Watching .git/HEAD + .git/index
    // (the usual approach) misses the case that motivates the feature: an
    // unstaged source edit recompiles the crate without rerunning the build
    // script, so the binary would carry a stale "clean" hash. Pointing
    // rerun-if-changed at a path that never exists makes cargo rerun the script
    // each build (~10ms for git describe); the crate itself is only recompiled
    // when the emitted value actually changes.
    println!("cargo:rerun-if-changed=build/nonexistent-force-rerun");

    // Emitted as a ready-to-concat suffix (" (<describe>)" or "") so main.rs
    // can build the version string at compile time with concat!.
    let describe = git_describe();
    let suffix = if describe.is_empty() {
        String::new()
    } else {
        format!(" ({describe})")
    };
    println!("cargo:rustc-env=AXIL_VERSION_SUFFIX={suffix}");
}

/// `git describe --always --dirty` for the workspace this crate is built from,
/// or "" when that is not a git checkout (crates.io / vendored tarballs — the
/// guard also keeps an unrelated enclosing repo from stamping bogus hashes).
fn git_describe() -> String {
    let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(d) => d,
        Err(_) => return String::new(),
    };
    let repo_root = Path::new(&manifest_dir).join("../../..");
    if !repo_root.join(".git").exists() {
        return String::new();
    }

    Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .current_dir(&repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}
