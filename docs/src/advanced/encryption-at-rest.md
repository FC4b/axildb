# Encryption at Rest

Axil can encrypt **core record bodies** at rest with an authenticated cipher.
This is an **off-by-default, opt-in** capability behind the `encryption` Cargo
feature — a default build is byte-identical and carries none of the crypto
dependencies.

## Honest scope (v1)

This feature encrypts the **serialized record body only** — the JSON payload
that each record stores. It deliberately does **not** encrypt:

- **`.vec` companion embeddings.** A vector is a lossy reconstruction channel of
  the source text; in v1 it stays cleartext.
- **`.fts` companion tokens.** The full-text index stores tokenized terms in the
  clear.
- **Table names and record IDs.** These remain visible in the core file's key
  space and table index.

The honest pitch is therefore *"encrypted record bodies"*, not *"encrypted
memory"*. An operator who needs the embeddings and FTS index encrypted as well
should layer full-disk or filesystem encryption underneath Axil.

## Cipher and wire format

The cipher is **XChaCha20-Poly1305** (pure-Rust, no AES-NI assumption). Each
stored body is laid out as:

```
XNonce (24 bytes) || ciphertext + Poly1305 tag
```

The nonce is freshly random per write (sourced from the OS CSPRNG). The AEAD
associated data (AAD) is bound to the **record ID** (the redb key), so a
ciphertext authenticated for one record cannot be replayed into a different
record's slot — decryption fails cleanly. The record's `table` is part of the
authenticated plaintext body, so it is integrity-protected as well (a flipped
table byte fails the tag).

## Key management

Keys are **32 bytes** and **never touch the `.axil` file**. Two sources are
supported, in priority order:

1. **`AXIL_ENC_KEY` environment variable** — 32 raw key bytes encoded as hex
   (64 chars) or standard base64.
2. **A key file** — its contents are parsed as hex/base64 first, falling back to
   raw 32 bytes.

Generate a key, for example:

```bash
# 32 random bytes as hex
export AXIL_ENC_KEY=$(head -c 32 /dev/urandom | xxd -p -c 64)
```

## Failure behavior

Opening an encrypted database with the **wrong key**, or with **no key**, fails
cleanly with a typed storage error rather than returning corrupt or partial
data. A full scan under a mismatched key surfaces the error instead of silently
returning an empty result set.

## Library usage

```rust
# #[cfg(feature = "encryption")]
# fn demo() -> axil_core::error::Result<()> {
use axil_core::crypto::Cipher;
use axil_core::storage::Storage;

let cipher = Cipher::from_env()?; // reads AXIL_ENC_KEY
let storage = Storage::open("./memory.axil")?.with_cipher(cipher);
// inserts/gets now transparently seal/unseal record bodies
# Ok(())
# }
```

## Performance

Measured with `cargo bench -p axil-core --features encryption --bench
encryption_benchmarks` (criterion 0.5) on the project's dev host (Windows,
software XChaCha20-Poly1305 — no AES-NI assumption). Body = a JSON record with an
N-byte string field. Treat the absolute numbers as host-specific; the *shape* of
the result is the robust takeaway.

| Operation | Cleartext | Encrypted | Delta |
|---|---|---|---|
| AEAD seal, 256 B body | — | ~1.55 µs | (fixed cost ≈ nonce draw) |
| AEAD seal, 4 KB body | — | ~4.1 µs | ≈0.96 GiB/s |
| AEAD seal, 16 KB body | — | ~12.4 µs | ≈1.2 GiB/s |
| `Storage::insert`, 256 B (incl. redb commit) | ~582 µs | ~589 µs | **+~7 µs (~1.2 %)** |
| `Storage::insert`, 4 KB | ~780 µs | ~680 µs | within measurement noise |
| `Storage::get`, 256 B | ~1.50 µs | ~3.38 µs | +~1.9 µs |
| `Storage::get`, 4 KB | ~2.09 µs | ~6.35 µs | +~4.3 µs |

**Takeaways:**

- **Writes are barely affected (~1 % or less).** Each insert is dominated by
  redb's transaction commit (the fsync, ~0.6 ms), which encryption doesn't touch
  — the ~1.5–4 µs of crypto is lost in that.
- **Reads add a few microseconds per record body.** A `get` is cheap (no fsync),
  so the per-record decrypt is a larger *relative* share (256 B: 1.5 → 3.4 µs),
  but the absolute cost is still single-digit microseconds.
- **Recall / search latency is unchanged.** The `.vec` and `.fts` companion
  indexes stay cleartext, so the search itself (millisecond-scale) is unaffected;
  only the record bodies actually returned pay the per-record decrypt. For a
  top-k of 10 that is roughly +20–40 µs total — negligible next to the search.
- The small-body cost is dominated by a fixed per-call overhead (~1.3 µs, mostly
  the per-write `OsRng` nonce draw); throughput approaches ~1.2 GiB/s on larger
  bodies.
