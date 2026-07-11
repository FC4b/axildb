#!/usr/bin/env bash
# align-version.sh — realign every publishable workspace crate onto one version.
#
# The workspace is versioned INDEPENDENTLY: release-plz bumps only the crates a
# PR changed plus their dependents, so crate versions drift apart over time.
# For a coordinated "big release" you sometimes want every published crate on
# the same number again. This does exactly that — and nothing else.
#
# Usage:
#   scripts/align-version.sh 2.2.0
#
# What it does:
#   • sets `[package] version` of every PUBLISHABLE crate (derived from
#     `cargo metadata`; publish = false crates are skipped) to <version>
#   • rewrites the matching `version = "…"` pins in the root
#     [workspace.dependencies] table for those same crates
#   • refuses to run on a dirty tree (so the resulting diff is only the bump)
#   • refuses a downgrade (a <version> below any current crate version)
#
# It does NOT commit. Review the diff, then commit with a conventional message,
# e.g.  `feat: align workspace to 2.2.0`  (a `feat:` makes release-plz cut a
# minor for the changed crates on the next run).
set -euo pipefail

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    echo "usage: scripts/align-version.sh <version>   (e.g. 2.2.0)" >&2
    exit 2
fi

if ! [[ "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: '<version>' must be a plain semver like 2.2.0 (got '$TARGET')" >&2
    exit 2
fi

# Run from the workspace root regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# Uncommitted edits to tracked files would ride along with the version bump and
# muddy the release commit — refuse until they're committed or stashed. Untracked
# files are ignored on purpose: they never enter a `git commit -am` bump, so they
# can't contaminate the release commit (and this keeps the tool usable while its
# own script is still untracked).
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "error: tracked files have uncommitted changes — commit or stash first, then re-run." >&2
    exit 1
fi

# All parsing + editing lives in one python3 worker (same pattern as the other
# scripts/*-gate.sh). python3 is the only non-cargo dependency.
TARGET="$TARGET" python3 <<'PY'
import json, os, re, subprocess, sys, pathlib

target = os.environ["TARGET"]

def release_tuple(v):
    # Compare on the numeric release part only; the repo doesn't use
    # pre-release/build metadata, but strip it defensively if present.
    core = re.split(r'[-+]', v, 1)[0]
    return tuple(int(x) for x in core.split('.'))

md = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--no-deps"]))
root = pathlib.Path(md["workspace_root"])

# Publishable = NOT publish = false. `publish` is null for "anywhere",
# [] for `publish = false`, or a registry list — only [] is excluded.
crates = []
for p in md["packages"]:
    if p.get("publish") == []:
        continue
    crates.append((p["name"], p["version"], pathlib.Path(p["manifest_path"])))
crates.sort()

# Downgrade guard: refuse if any crate currently sits above the target.
tgt = release_tuple(target)
downgrades = [(n, v) for (n, v, _) in crates if release_tuple(v) > tgt]
if downgrades:
    sys.stderr.write(
        "error: %s is lower than these current crate versions:\n" % target)
    for n, v in downgrades:
        sys.stderr.write("  %s %s\n" % (n, v))
    sys.stderr.write("nothing changed.\n")
    sys.exit(1)


def set_package_version(text, newver):
    """Replace the `version = "…"` line inside the [package] table only."""
    out = []
    in_pkg = False
    hits = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('['):
            in_pkg = (stripped == '[package]')
            out.append(line)
            continue
        if in_pkg and re.match(r'version\s*=\s*"[^"]*"\s*$', stripped):
            indent = line[:len(line) - len(line.lstrip())]
            out.append('%sversion = "%s"\n' % (indent, newver))
            hits += 1
            in_pkg = False  # only the first version line in [package]
            continue
        out.append(line)
    return ''.join(out), hits


def set_dep_pin(text, crate, newver):
    """Rewrite the version in a `crate = { path = …, version = "…" }` pin."""
    pat = re.compile(
        r'^(?P<pre>' + re.escape(crate) + r'\s*=\s*\{[^\n]*?version\s*=\s*")'
        r'[^"]*(?P<post>"[^\n]*)$', re.M)
    return pat.subn(lambda m: m.group('pre') + newver + m.group('post'), text)


# 1) Each publishable crate's own [package] version.
changed = []
for name, ver, mpath in crates:
    text = mpath.read_text(encoding="utf-8")
    new, hits = set_package_version(text, target)
    if hits != 1:
        sys.stderr.write(
            "error: %s: expected 1 [package] version line, changed %d\n"
            % (name, hits))
        sys.exit(1)
    if new != text:
        mpath.write_text(new, encoding="utf-8")
        changed.append((name, ver, target))
    else:
        changed.append((name, ver, target))  # already at target

# 2) The internal version pins in root [workspace.dependencies].
root_manifest = root / "Cargo.toml"
rtext = root_manifest.read_text(encoding="utf-8")
pins = []
for name, _ver, _mpath in crates:
    rtext, n = set_dep_pin(rtext, name, target)
    if n:
        pins.append(name)
root_manifest.write_text(rtext, encoding="utf-8")

# Summary.
print("Aligned %d publishable crate(s) to %s:" % (len(changed), target))
for name, old, new in changed:
    note = "" if old != new else "  (unchanged)"
    print("  %-20s %s -> %s%s" % (name, old, new, note))
print("Rewrote %d version pin(s) in root [workspace.dependencies]: %s"
      % (len(pins), ", ".join(pins)))

skipped = [p["name"] for p in md["packages"] if p.get("publish") == []]
if skipped:
    print("Skipped (publish = false): %s" % ", ".join(sorted(skipped)))
PY

echo
echo "Done. Review with:  git diff"
echo "Then commit with a conventional message, e.g.:"
echo "  git commit -am \"feat: align workspace to ${TARGET}\""
echo "(feat: makes release-plz cut a minor for the changed crates.)"
