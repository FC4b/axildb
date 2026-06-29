//! Opt-in encryption-at-rest for core record bodies (XChaCha20-Poly1305 AEAD).
//!
//! This module is compiled only under the off-by-default `encryption` Cargo
//! feature, so a default build is byte-identical and carries none of these
//! dependencies. When the feature is on *and* a key is supplied, the storage
//! layer encrypts each record's serialized body before writing it to the core
//! `.axil` file and decrypts on read.
//!
//! ## Honest scope (v1)
//!
//! This encrypts **core record bodies only** — the JSON payload that
//! [`Record::to_bytes`](crate::Record::to_bytes) produces. It deliberately does
//! **not** encrypt:
//!
//! - `.vec` companion embeddings (a vector is a lossy reconstruction channel of
//!   the source text — it stays cleartext in v1),
//! - `.fts` companion tokens (the full-text index stores tokenized terms in the
//!   clear),
//! - table names and record IDs (these remain visible in the core file's key
//!   space and table index).
//!
//! The honest pitch is therefore *"encrypted record bodies"*, not *"encrypted
//! memory"*. An operator who needs the embeddings and FTS index encrypted as
//! well should layer full-disk / filesystem encryption underneath Axil.
//!
//! ## Wire format
//!
//! Each encrypted body is `XNonce (24 bytes) || ciphertext+tag`. The nonce is
//! freshly random per write. The AEAD associated data (AAD) is bound to the
//! record's ID (the redb key), so a ciphertext authenticated for one record
//! cannot be replayed into a different record's slot — decryption fails cleanly.
//! The record's `table` is part of the authenticated plaintext body, so it is
//! integrity-protected too (a flipped table byte fails the Poly1305 tag).
//!
//! ## Key management
//!
//! Keys are 32 bytes and **never** touch the `.axil` file. Two sources are
//! supported, in priority order:
//!
//! 1. `AXIL_ENC_KEY` environment variable — 32 raw key bytes encoded as hex
//!    (64 chars) or standard base64.
//! 2. A key file — its contents parsed as hex/base64 first, falling back to
//!    raw 32 bytes.
//!
//! Opening an encrypted database with the wrong key, or with no key, fails
//! cleanly with [`CryptoError`] rather than returning corrupt or partial data.

use chacha20poly1305::{
    aead::{rand_core::RngCore, Aead, KeyInit, OsRng, Payload},
    XChaCha20Poly1305, XNonce,
};

use crate::error::AxilError;

/// Length in bytes of the AEAD key.
const KEY_LEN: usize = 32;

/// Length in bytes of the XChaCha20-Poly1305 extended nonce.
const NONCE_LEN: usize = 24;

/// Errors raised while loading a key or encrypting/decrypting a record body.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    /// The supplied key material was not exactly 32 bytes after decoding.
    #[error("encryption key must be 32 bytes (got {0} after decoding)")]
    BadKeyLength(usize),

    /// The key string was neither valid hex nor valid base64.
    #[error("encryption key is not valid hex or base64")]
    BadKeyEncoding,

    /// No key source was configured (env var unset and no key file given).
    #[error("encryption enabled but no key configured (set AXIL_ENC_KEY or pass a key file)")]
    MissingKey,

    /// Reading the key file failed.
    #[error("failed to read key file: {0}")]
    KeyFile(#[source] std::io::Error),

    /// A stored body was shorter than the nonce prefix — truncated or corrupt.
    #[error("ciphertext too short to contain a nonce ({0} bytes)")]
    Truncated(usize),

    /// AEAD verification failed: wrong key, tampered ciphertext, or a body that
    /// was written without encryption (or under a different key).
    #[error("decryption failed: wrong key or corrupt/tampered record body")]
    DecryptFailed,
}

impl From<CryptoError> for AxilError {
    fn from(e: CryptoError) -> Self {
        // Crypto failures are surfaced as storage errors so callers that open a
        // DB with the wrong key get a clean, typed failure instead of a panic.
        AxilError::Storage(Box::new(e))
    }
}

/// A loaded 32-byte AEAD key plus its initialized cipher.
///
/// Construct via [`Cipher::from_env`] or [`Cipher::from_key_file`]. The key
/// bytes live only in process memory; they are never serialized to disk.
#[derive(Clone)]
pub struct Cipher {
    inner: XChaCha20Poly1305,
}

impl std::fmt::Debug for Cipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material.
        f.write_str("Cipher(<redacted>)")
    }
}

impl Cipher {
    /// Build a cipher from raw 32-byte key material.
    pub fn from_key_bytes(key: &[u8]) -> Result<Self, CryptoError> {
        if key.len() != KEY_LEN {
            return Err(CryptoError::BadKeyLength(key.len()));
        }
        let inner = XChaCha20Poly1305::new(key.into());
        Ok(Self { inner })
    }

    /// Load a key from the `AXIL_ENC_KEY` environment variable.
    ///
    /// The value is decoded as hex (64 chars) or standard base64 to 32 bytes.
    /// Returns [`CryptoError::MissingKey`] if the variable is unset or empty.
    pub fn from_env() -> Result<Self, CryptoError> {
        let raw = std::env::var("AXIL_ENC_KEY").ok().filter(|s| !s.is_empty());
        match raw {
            Some(s) => {
                let key = decode_key_str(&s)?;
                Self::from_key_bytes(&key)
            }
            None => Err(CryptoError::MissingKey),
        }
    }

    /// Load a key from a file. The file contents are parsed as hex/base64
    /// first, then as raw 32 bytes if neither decoding yields 32 bytes.
    pub fn from_key_file(path: impl AsRef<std::path::Path>) -> Result<Self, CryptoError> {
        let bytes = std::fs::read(path.as_ref()).map_err(CryptoError::KeyFile)?;

        // Try text encodings first (trim whitespace/newlines a keyfile often has).
        if let Ok(text) = std::str::from_utf8(&bytes) {
            let trimmed = text.trim();
            if let Ok(key) = decode_key_str(trimmed) {
                return Self::from_key_bytes(&key);
            }
        }
        // Fall back to raw bytes.
        Self::from_key_bytes(&bytes)
    }

    /// Resolve a cipher from the standard sources in priority order: the
    /// `AXIL_ENC_KEY` env var, then `key_file` if provided. Returns
    /// [`CryptoError::MissingKey`] if neither yields a key.
    pub fn resolve(key_file: Option<&std::path::Path>) -> Result<Self, CryptoError> {
        match Self::from_env() {
            Ok(c) => Ok(c),
            Err(CryptoError::MissingKey) => match key_file {
                Some(path) => Self::from_key_file(path),
                None => Err(CryptoError::MissingKey),
            },
            Err(e) => Err(e),
        }
    }

    /// Encrypt a plaintext record body. AAD binds the ciphertext to `record_id`.
    ///
    /// Output is `XNonce (24 bytes) || ciphertext+tag`.
    pub fn encrypt(&self, plaintext: &[u8], record_id: &str) -> Result<Vec<u8>, CryptoError> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);

        let ct = self
            .inner
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: record_id.as_bytes(),
                },
            )
            .map_err(|_| CryptoError::DecryptFailed)?;

        let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ct);
        Ok(out)
    }

    /// Decrypt a stored body produced by [`Cipher::encrypt`]. AAD must match the
    /// `record_id` the ciphertext was sealed under, or decryption fails cleanly.
    pub fn decrypt(&self, stored: &[u8], record_id: &str) -> Result<Vec<u8>, CryptoError> {
        if stored.len() < NONCE_LEN {
            return Err(CryptoError::Truncated(stored.len()));
        }
        let (nonce_bytes, ct) = stored.split_at(NONCE_LEN);
        let nonce = XNonce::from_slice(nonce_bytes);
        self.inner
            .decrypt(
                nonce,
                Payload {
                    msg: ct,
                    aad: record_id.as_bytes(),
                },
            )
            .map_err(|_| CryptoError::DecryptFailed)
    }
}

/// Decode a hex (64 chars) or standard base64 string into exactly 32 bytes.
fn decode_key_str(s: &str) -> Result<Vec<u8>, CryptoError> {
    let s = s.trim();
    // Try hex first (a 32-byte key is 64 hex chars).
    if s.len() == KEY_LEN * 2 {
        if let Some(bytes) = decode_hex(s) {
            return Ok(bytes);
        }
    }
    // Then base64 (standard alphabet, with or without padding).
    if let Some(bytes) = decode_base64(s) {
        return Ok(bytes);
    }
    // If hex was the right length but invalid chars, report encoding error.
    if s.len() == KEY_LEN * 2 {
        return Err(CryptoError::BadKeyEncoding);
    }
    Err(CryptoError::BadKeyEncoding)
}

/// Minimal hex decoder (avoids pulling in a hex crate for one helper).
fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_val(bytes[i])?;
        let lo = hex_val(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Minimal standard-base64 decoder (accepts optional `=` padding). Returns
/// `None` on any invalid character.
fn decode_base64(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in s.as_bytes() {
        let v = base64_val(b)?;
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

fn base64_val(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        for (i, b) in k.iter_mut().enumerate() {
            *b = i as u8;
        }
        k
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let cipher = Cipher::from_key_bytes(&test_key()).unwrap();
        let pt = b"hello secret body";
        let ct = cipher.encrypt(pt, "01ABC").unwrap();
        // Nonce-prefixed and longer than plaintext (tag + nonce).
        assert!(ct.len() > pt.len() + NONCE_LEN);
        let recovered = cipher.decrypt(&ct, "01ABC").unwrap();
        assert_eq!(recovered, pt);
    }

    #[test]
    fn nonces_are_unique_per_write() {
        let cipher = Cipher::from_key_bytes(&test_key()).unwrap();
        let a = cipher.encrypt(b"same", "id").unwrap();
        let b = cipher.encrypt(b"same", "id").unwrap();
        // Random nonce → different ciphertexts for identical plaintext.
        assert_ne!(a, b);
    }

    #[test]
    fn wrong_key_fails_cleanly() {
        let cipher = Cipher::from_key_bytes(&test_key()).unwrap();
        let ct = cipher.encrypt(b"data", "id").unwrap();
        let mut other = test_key();
        other[0] ^= 0xff;
        let wrong = Cipher::from_key_bytes(&other).unwrap();
        let err = wrong.decrypt(&ct, "id").unwrap_err();
        assert!(matches!(err, CryptoError::DecryptFailed));
    }

    #[test]
    fn aad_mismatch_fails() {
        let cipher = Cipher::from_key_bytes(&test_key()).unwrap();
        let ct = cipher.encrypt(b"data", "id-A").unwrap();
        // Moving the ciphertext to a different record id breaks the AAD bind.
        let err = cipher.decrypt(&ct, "id-B").unwrap_err();
        assert!(matches!(err, CryptoError::DecryptFailed));
    }

    #[test]
    fn truncated_body_fails_cleanly() {
        let cipher = Cipher::from_key_bytes(&test_key()).unwrap();
        let err = cipher.decrypt(&[0u8; 4], "id").unwrap_err();
        assert!(matches!(err, CryptoError::Truncated(4)));
    }

    #[test]
    fn bad_key_length_rejected() {
        let err = Cipher::from_key_bytes(&[0u8; 16]).unwrap_err();
        assert!(matches!(err, CryptoError::BadKeyLength(16)));
    }

    #[test]
    fn hex_key_decodes() {
        let hex = "00".repeat(32);
        let bytes = decode_key_str(&hex).unwrap();
        assert_eq!(bytes, vec![0u8; 32]);
    }

    #[test]
    fn base64_key_decodes() {
        // 32 zero bytes base64-encoded.
        let b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        let bytes = decode_key_str(b64).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes, vec![0u8; 32]);
    }

    #[test]
    fn from_env_round_trip() {
        // Use a guarded env mutation; serial within this test only.
        let hex = "ab".repeat(32);
        std::env::set_var("AXIL_ENC_KEY", &hex);
        let cipher = Cipher::from_env().unwrap();
        std::env::remove_var("AXIL_ENC_KEY");
        let ct = cipher.encrypt(b"x", "id").unwrap();
        assert_eq!(cipher.decrypt(&ct, "id").unwrap(), b"x");
    }

    #[test]
    fn from_key_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key.hex");
        std::fs::write(&path, format!("{}\n", "cd".repeat(32))).unwrap();
        let cipher = Cipher::from_key_file(&path).unwrap();
        let ct = cipher.encrypt(b"y", "id").unwrap();
        assert_eq!(cipher.decrypt(&ct, "id").unwrap(), b"y");
    }
}
