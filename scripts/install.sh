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
        Darwin)
            os="apple-darwin"
            # A shell running under Rosetta reports x86_64 on Apple Silicon;
            # the native arm64 build is the right one there.
            if [ "$uname_m" = "x86_64" ] \
                && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = "1" ]; then
                uname_m="arm64"
            fi
            ;;
        *) err "unsupported OS '$uname_s' — on Windows use scripts/install.ps1 (irm | iex)" ;;
    esac
    case "$uname_m" in
        x86_64|amd64)   arch="x86_64" ;;
        aarch64|arm64)  arch="aarch64" ;;
        *) err "unsupported architecture '$uname_m'" ;;
    esac
    if [ "$os" = "apple-darwin" ] && [ "$arch" = "x86_64" ]; then
        err "no prebuilt binary for Intel Macs — build from source instead: cargo install axildb"
    fi

    triple="$arch-$os"
    archive="axildb-$triple.tar.gz"
    url="https://github.com/$REPO/releases/latest/download/$archive"

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    info "downloading $url"
    if ! curl -fsSL "$url" -o "$tmp/$archive"; then
        # A new Release exists before its archives are uploaded (they build for
        # ~20 minutes after the tag), so `latest` can 404 for a while. Fall back
        # to the newest release that already carries this platform's archive.
        info "latest release has no $archive yet; trying the newest release that does"
        url="$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=20" \
            | grep -o "\"browser_download_url\": *\"[^\"]*/$archive\"" \
            | head -n 1 \
            | sed 's/.*"\(https[^"]*\)"$/\1/')" || url=""
        [ -n "$url" ] || err "no published release has $archive yet"
        info "downloading $url"
        curl -fsSL "$url" -o "$tmp/$archive"
    fi
    tar xzf "$tmp/$archive" -C "$tmp"

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
