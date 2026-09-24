//! Embed the git state of the source tree into the binary.
//!
//! `axil --version` prints `axil <semver> (<git describe>)`, where the
//! describe output carries a `-dirty` suffix for uncommitted-source builds.
//! Two same-version binaries are otherwise indistinguishable ("did I actually
//! build what I think I built?"), because the crate version only moves at
//! release time.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let describe = match repo_root() {
        Some(root) => {
            watch_git_inputs(&root);
            git_describe(&root)
        }
        None => {
            // Not a git checkout: there is no stamp to keep fresh, so the
            // script's only input is itself.
            println!("cargo:rerun-if-changed=build.rs");
            String::new()
        }
    };

    // Emitted as a ready-to-concat suffix (" (<describe>)" or "") so main.rs
    // can build the version string at compile time with concat!.
    let suffix = if describe.is_empty() {
        String::new()
    } else {
        format!(" ({describe})")
    };
    println!("cargo:rustc-env=AXIL_VERSION_SUFFIX={suffix}");
}

/// The workspace root when this crate is built from a git checkout of it, or
/// `None` otherwise (crates.io / vendored tarballs — the guard also keeps an
/// unrelated enclosing repo from stamping bogus hashes). `.git` is a directory
/// in a plain clone and a file in a linked worktree; either counts.
fn repo_root() -> Option<PathBuf> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let root = Path::new(&manifest_dir).join("../../..");
    if !root.join(".git").exists() {
        return None;
    }
    Some(root.canonicalize().unwrap_or(root))
}

/// Rerun this script only when the `git describe --always --dirty` output can
/// have changed.
///
/// A `rerun-if-changed` path that does not exist makes cargo rerun the script
/// — and recompile the whole crate — on every build, so only paths that exist
/// are emitted. They are resolved with `git rev-parse --git-path`, which knows
/// the worktree layout: HEAD, its reflog and the index are per-worktree, while
/// refs and `packed-refs` live in the common git dir.
///
/// - `HEAD` moves on checkout; the branch ref it names moves on commit/reset.
///   A branch that only exists in `packed-refs` has no loose file to watch,
///   but every move of HEAD is also appended to `logs/HEAD`.
/// - `refs/tags` and `packed-refs`: a new tag changes the describe name.
/// - `index`: staging changes the `-dirty` state.
/// - `src`: an unstaged edit to this crate flips `-dirty` without touching any
///   git file. Edits elsewhere in the workspace re-stamp at the next index or
///   HEAD change.
fn watch_git_inputs(root: &Path) {
    println!("cargo:rerun-if-changed=src");

    let head_ref = git(root, &["symbolic-ref", "-q", "HEAD"]);
    let mut wanted = vec!["HEAD", "logs/HEAD", "index", "packed-refs", "refs/tags"];
    if let Some(ref r) = head_ref {
        wanted.push(r);
    }
    let mut args = vec!["rev-parse"];
    for path in wanted {
        args.extend(["--git-path", path]);
    }
    let Some(resolved) = git(root, &args) else {
        return;
    };
    for line in resolved.lines() {
        // Relative output is relative to `root`; joining an absolute path
        // replaces `root` entirely.
        let path = root.join(line);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// `git describe --always --dirty` for the checkout at `root`, or "" when git
/// is unavailable or fails.
fn git_describe(root: &Path) -> String {
    git(root, &["describe", "--always", "--dirty"]).unwrap_or_default()
}

/// Run git in `root`, returning its trimmed stdout on success.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}
