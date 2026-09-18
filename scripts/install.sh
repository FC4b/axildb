#!/bin/sh
# Axil one-line installer (Linux/macOS) — POSIX sh, needs only curl + tar.
#
#   curl -fsSL https://raw.githubusercontent.com/FC4b/axildb/main/scripts/install.sh | sh
#
# Downloads the archive for this platform from the latest GitHub release and
# installs the `axil` binary into $AXIL_HOME/bin (default ~/.axil/bin).
# The release binaries statically link onnxruntime, so nothing else is needed.
set -eu

REPO="FC4b/axildb"
PREFIX="${AXIL_HOME:-$HOME/.axil}"

info() { printf '%s\n' "axil-install: $*"; }
err()  { printf '%s\n' "axil-install: ERROR: $*" >&2; exit 1; }

main() {
    uname_s="$(uname -s)"
    uname_m="$(uname -m)"

    case "$uname_s" in
        Linux)  os="unknown-linux-gnu" ;;
        Darwin) os="apple-darwin" ;;
        *) err "unsupported OS '$uname_s' — on Windows use scripts/install.ps1 (irm | iex)" ;;
    esac
    case "$uname_m" in
        x86_64|amd64)   arch="x86_64" ;;
        aarch64|arm64)  arch="aarch64" ;;
        *) err "unsupported architecture '$uname_m'" ;;
    esac

    triple="$arch-$os"
    archive="axildb-$triple.tar.gz"
    url="https://github.com/$REPO/releases/latest/download/$archive"

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    info "downloading $url"
    curl -fsSL "$url" | tar xzf - -C "$tmp"

    install_dir="$PREFIX/bin"
    mkdir -p "$install_dir"
    cp "$tmp/axildb-$triple/axil" "$install_dir/axil"
    chmod 0755 "$install_dir/axil"

    case ":$PATH:" in
        *":$install_dir:"*) ;;
        *) info "add to PATH:  export PATH=\"$install_dir:\$PATH\"" ;;
    esac

    info "installed -> $install_dir/axil"
    "$install_dir/axil" --version
}

main "$@"
